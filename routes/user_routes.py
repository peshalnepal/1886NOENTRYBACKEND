from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
import uuid
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from application.services.user_snapshot_cache import UserSnapshotCache

from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from application.repositories.user_repository import UserRepository
from application.dtos import UserProfileUpdateDTO
from core.database_orm import Notification, User
from core.database import AsyncSessionLocal
from core.security.hashing import get_password_hash, verify_password
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import (
    EventClipService,
    extract_notification_clip_storage_keys as _extract_notification_clip_storage_keys,
)
import asyncio
import logging
from typing import Any, Dict, List, Optional

from dependencies import get_async_db, get_current_user, get_manager
from application.services.manager import Manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


def _invalidate_user_snapshot_cache(request: Request, user_id: int) -> None:
    cache = getattr(request.app.state, "user_snapshot_cache", None)
    if cache is None:
        return
    cache.invalidate(int(user_id))
    
class UserOut(BaseModel):
    id: int
    user_name: str
    user_email: EmailStr


class UserProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str | None = Field(default=None, min_length=1, max_length=255)
    user_email: EmailStr | None = None

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr | None) -> str | None:
        if value is None:
            return None
        return str(value).lower().strip()

    @model_validator(mode="after")
    def validate_has_fields(self):
        if self.user_name is None and self.user_email is None:
            raise ValueError("Provide at least one field to update")
        return self


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: SecretStr
    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 8:
            raise ValueError("New password must be at least 8 characters long")
        return value


class DeleteAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: SecretStr


def _to_user_out(user: User) -> UserOut:
    return UserOut(
        id=int(user.id),
        user_name=user.user_name,
        user_email=user.email,
    )


@router.get("/me", response_model=UserOut)
async def get_me(
    current_user: User = Depends(get_current_user),
):
    return _to_user_out(current_user)


@router.patch("/me", response_model=UserOut)
async def update_me(
    payload: UserProfileUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    user_repo = UserRepository()

    next_user_name: Optional[str] = None
    next_email: Optional[str] = None

    if payload.user_name is not None and payload.user_name != current_user.user_name:
        next_user_name = payload.user_name

    if payload.user_email is not None:
        candidate_email = str(payload.user_email).lower().strip()
        current_email = str(current_user.email).lower().strip()

        if candidate_email != current_email:
            existing = await user_repo.get_by_email(db, candidate_email)
            if existing is not None and int(existing.id) != int(current_user.id):
                raise HTTPException(status_code=409, detail="Email is already in use")

            next_email = candidate_email

    if next_user_name is None and next_email is None:
        return _to_user_out(current_user)

    try:
        await user_repo.update_profile(
            db,
            int(current_user.id),
            UserProfileUpdateDTO(user_name=next_user_name, email=next_email),
        )
        await db.commit()
        _invalidate_user_snapshot_cache(request, int(current_user.id))
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email is already in use")
    except Exception:
        await db.rollback()
        raise

    await db.refresh(current_user)
    return _to_user_out(current_user)


@router.patch("/me/password")
async def change_my_password(
    payload: ChangePasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    current_password = payload.current_password.get_secret_value()
    new_password = payload.new_password.get_secret_value()

    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if verify_password(new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="New password must be different from current password",
        )

    try:
        await UserRepository().update_password_hash(
            db, int(current_user.id), get_password_hash(new_password)
        )
        await db.commit()
        _invalidate_user_snapshot_cache(request, int(current_user.id))
    except Exception:
        await db.rollback()
        raise

    return {"message": "Password updated successfully"}


from routes._background import _delete_blobs_background, _spawn_bg_task


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_account(
    payload: DeleteAccountRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    """
    Full account deletion:
    1. Snapshot camera/site info, disable cameras in DB
    2. Stop cameras on edge/WebRTC/pipeline + purge notification service state
    3a. Extract video record blob keys (foreground)
    3b. Delete camera rows + relationships (foreground, makes UI clean)
    4. Invalidate caches
    5. Background: batch-delete notifications (extract blob keys),
       schedule blob deletion, delete sites, delete user row
    """
    from routes.notifications_routes import invalidate_camera_mode_cache

    if not verify_password(payload.password.get_secret_value(), current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    user_id = int(current_user.id)
    logger.info(f"[User Delete] Starting deletion of user={user_id}")

    # ========================================
    # PHASE 1: Snapshot camera + site info BEFORE any changes
    # ========================================
    channel_repo = ChannelRepository()
    site_repo = SiteRepository()

    camera_rows = await channel_repo.list_cameras(db, user_id=user_id)
    camera_uuids = [cam.camera_uuid for cam in camera_rows]

    site_uuids = await site_repo.list_site_uuids(db, user_id=user_id)

    logger.info(f"[User Delete] Snapshotted {len(camera_uuids)} cameras, {len(site_uuids)} sites")

    # ========================================
    # PHASE 1b: Disable all cameras in DB BEFORE edge/MediaMTX cleanup.
    # Reconcile reads is_enabled/is_detection_enabled from DB; if it fires
    # between our edge cleanup and DB deletion it re-adds the cameras.
    # ========================================
    if camera_uuids:
        logger.info(f"[User Delete] Phase 1b: Disabling {len(camera_uuids)} cameras in DB to prevent reconcile re-adds")
        await channel_repo.disable_cameras(db, user_id=user_id)
        await db.commit()

    # ========================================
    # PHASE 2: Stop cameras BEFORE any DB changes
    # ========================================
    logger.info(f"[User Delete] Phase 2: Stopping cameras on edge/WebRTC/pipeline")

    # 2a: Purge notification service in-memory state
    notif_svc = getattr(manager, "_notification_service", None) if manager is not None else None
    if notif_svc is not None:
        try:
            purge_fn = getattr(notif_svc, "purge_deleted_site_runtime_state", None)
            for site_uuid_val in site_uuids:
                if callable(purge_fn):
                    await purge_fn(user_id=user_id, site_uuid=site_uuid_val, camera_uuids=camera_uuids)
                else:
                    notif_svc.invalidate_recipient_cache(user_id=user_id, site_uuid=site_uuid_val)
            for cam_uuid in camera_uuids:
                inv_fn = getattr(notif_svc, "invalidate_camera_roi_state", None)
                if callable(inv_fn):
                    inv_fn(str(cam_uuid))
        except Exception as exc:
            logger.warning(f"[User Delete] Notification service purge failed: {exc}", exc_info=True)

    # 2b: Edge devices, WebRTC streams, in-memory pipeline
    if manager is not None:
        try:
            cleanup = await asyncio.wait_for(
                manager.cleanup_user_resources(db, user_id=user_id),
                timeout=60.0,
            )
            if cleanup.get("errors"):
                logger.warning("[User Delete] Partial cleanup errors: %s", cleanup["errors"])
        except asyncio.TimeoutError:
            logger.warning(f"[User Delete] Manager cleanup timed out after 60s — proceeding")
        except Exception as exc:
            logger.warning(f"[User Delete] Manager cleanup failed — proceeding: {exc}", exc_info=True)

    # ========================================
    # PHASE 3a: Extract video record blob keys BEFORE site/camera
    # deletion.  Site deletion CASCADE-deletes cameras, which
    # CASCADE-deletes VideoRecords, losing their storage_key values.
    # ========================================
    video_clip_keys: List[str] = []
    if camera_uuids:
        video_clip_keys = await site_repo.list_video_record_keys_for_cameras(
            AsyncSessionLocal, camera_uuids=camera_uuids
        )
        logger.info(f"[User Delete] Phase 3a: Extracted {len(video_clip_keys)} video record blob keys")

    # ========================================
    # PHASE 3b: Fast Foreground DB Cleanup
    # Delete camera relationships and camera rows so the UI is clean.
    # Do NOT delete site or user rows yet — their FK CASCADEs would
    # wipe Notification rows before the background task can extract
    # blob storage keys.
    # Cameras are disabled (Phase 1b) and runtime-stopped (Phase 2),
    # so no new data arrives.
    # Camera deletion SET NULLs Notification.camera_uuid and
    # CASCADE-deletes VideoRecords (keys saved in Phase 3a).
    # ========================================
    logger.info(f"[User Delete] Phase 3b: Deleting camera rows (Foreground)")
    try:
        if camera_uuids:
            await site_repo.delete_cameras_for_user(
                AsyncSessionLocal, user_id=user_id, camera_uuids=camera_uuids
            )

        # Mark all sites as soft-deleted so they disappear from queries immediately
        if site_uuids:
            await site_repo.soft_delete_sites_for_user(AsyncSessionLocal, user_id=user_id)

        _invalidate_user_snapshot_cache(request, user_id)
    except Exception:
        raise

    # ========================================
    # PHASE 4: Invalidate caches
    # ========================================
    for cam_uuid in camera_uuids:
        try:
            await invalidate_camera_mode_cache(cam_uuid)
        except Exception:
            pass

    # ========================================
    # PHASE 5: Background Heavy Table Cleanup (Notifications, Blobs, Sites, User)
    # User and site rows are still alive, so Notification rows with
    # user_id / site_uuid FKs have NOT been cascade-deleted.
    # After batch-deleting notifications (extracting blob keys) and
    # scheduling blob deletion, the background task deletes the
    # site rows and user row.
    # ========================================
    logger.info(f"[User Delete] Phase 5: Spawning background task to clean up heavy tables (Notifications/Videos)")

    async def _heavy_table_cleanup_task(uid: int, s_uuids: List, pre_video_keys: List[str]):
        logger.info(f"[User Cleanup Task] Starting background heavy cleanup for user={uid}")
        from application.repositories.site_repository import SiteRepository
        repo_for_delete = SiteRepository()
        alert_blob_keys: List[str] = []
        clip_blob_keys: List[str] = list(pre_video_keys)

        try:
            # Batch delete notifications and extract keys
            await repo_for_delete._batch_delete(
                AsyncSessionLocal,
                table=Notification,
                where_clause=Notification.user_id == uid,
                batch_size=2000,
                label="user_notifications",
                extract_col=Notification.payload,
                extract_alert_fn=extract_image_storage_key,
                extract_clip_fn=_extract_notification_clip_storage_keys,
                alert_keys_out=alert_blob_keys,
                clip_keys_out=clip_blob_keys,
            )

            # Schedule the blob deletions
            if alert_blob_keys:
                _spawn_bg_task(
                    _delete_blobs_background(alert_blob_keys, service_cls=AlertImageStorageService, label="alert image"),
                    name=f"delete_user_alert_blobs:{uid}",
                )

            if clip_blob_keys:
                _spawn_bg_task(
                    _delete_blobs_background(clip_blob_keys, service_cls=EventClipService, label="clip"),
                    name=f"delete_user_clip_blobs:{uid}",
                )

            # Delete sites (CASCADE cleans up SiteSettings, SiteDevices, NotificationEmails)
            if s_uuids:
                await repo_for_delete.delete_sites_for_user(AsyncSessionLocal, user_id=uid)

            # Finally delete the user row
            async with AsyncSessionLocal() as del_session:
                deleted = await UserRepository().delete_by_id(del_session, uid)
                if deleted:
                    await del_session.commit()

            logger.info(f"[User Cleanup Task] Background heavy cleanup COMPLETE for user={uid}")

        except Exception as e:
            logger.error(f"[User Cleanup Task] Failed heavy cleanup for user={uid}: {e}", exc_info=True)

    _spawn_bg_task(
        _heavy_table_cleanup_task(user_id, site_uuids, video_clip_keys),
        name=f"delete_user_heavy_tables:{user_id}",
    )

    logger.info(f"[User Delete] HTTP response ready: user={user_id} effectively deleted from UI")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
