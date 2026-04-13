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
from sqlalchemy import delete as sql_delete, select, update
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from application.services.user_snapshot_cache import UserSnapshotCache

from core.database_orm import Camera, CameraDevice, Device, Notification, Site, User, VideoRecord
from core.database import AsyncSessionLocal
from core.security.hashing import get_password_hash, verify_password
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import EventClipService
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
    changed = False

    if payload.user_name is not None and payload.user_name != current_user.user_name:
        current_user.user_name = payload.user_name
        changed = True

    if payload.user_email is not None:
        next_email = str(payload.user_email).lower().strip()
        current_email = str(current_user.email).lower().strip()

        if next_email != current_email:
            exists = (
                await db.execute(
                    select(User.id).where(
                        User.email == next_email,
                        User.id != int(current_user.id),
                    ).limit(1)
                )
            ).scalar_one_or_none()
            if exists is not None:
                raise HTTPException(status_code=409, detail="Email is already in use")

            current_user.email = next_email
            changed = True

    if not changed:
        return _to_user_out(current_user)

    try:
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

    current_user.hashed_password = get_password_hash(new_password)

    try:
        await db.commit()
        _invalidate_user_snapshot_cache(request, int(current_user.id))
    except Exception:
        await db.rollback()
        raise

    return {"message": "Password updated successfully"}


def _spawn_bg_task(coro, *, name: str) -> None:
    task = asyncio.create_task(coro, name=name)

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"[Background Task] {name}: FAILED with error: {e}")

    task.add_done_callback(_on_done)


async def _delete_blobs_background(keys: List[str], *, service_cls: type, label: str) -> None:
    unique = list(dict.fromkeys(k for k in keys if k))
    if not unique:
        return
    svc = service_cls()
    try:
        for i in range(0, len(unique), 10):
            batch = unique[i: i + 10]
            results = await asyncio.gather(*[svc.delete_blob(blob_name=k) for k in batch], return_exceptions=True)
            for key, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(f"[Blob Cleanup] Failed to delete {label} blob {key}: {result}")
    finally:
        try:
            await svc.close()
        except Exception:
            pass


def _extract_notification_clip_storage_keys(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    keys: List[str] = []

    def _collect(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        key = str(raw.get("storage_key") or "").strip()
        if key:
            keys.append(key)

    msg = payload.get("msg")
    if isinstance(msg, dict):
        _collect(msg)
        _collect(msg.get("clip"))

    extra = payload.get("extra")
    if isinstance(extra, dict):
        _collect(extra)
        _collect(extra.get("clip"))
        for item in list(extra.get("multi_camera_prerecordings") or []):
            _collect(item)

    _collect(payload.get("clip"))
    return list(dict.fromkeys(keys))


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
    1. Snapshot all camera info for this user
    2. Stop cameras (edge/WebRTC/pipeline) + purge notification service state
    3. Extract all blob keys (notification images + clips, video record clips)
    4. Delete user from DB (DB cascade removes sites, cameras, notifications, video records, etc.)
    5. Async blob deletion
    6. Invalidate caches
    """
    from routes.notifications_routes import invalidate_camera_mode_cache

    if not verify_password(payload.password.get_secret_value(), current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    user_id = int(current_user.id)
    logger.info(f"[User Delete] Starting deletion of user={user_id}")

    # ========================================
    # PHASE 1: Snapshot camera + site info BEFORE any changes
    # ========================================
    camera_rows = (
        await db.execute(
            select(Camera.camera_uuid, Camera.camera_code).where(Camera.user_id == user_id)
        )
    ).all()
    camera_uuids = [row[0] for row in camera_rows]

    site_uuids = (
        await db.execute(select(Site.site_uuid).where(Site.user_id == user_id))
    ).scalars().all()

    logger.info(f"[User Delete] Snapshotted {len(camera_uuids)} cameras, {len(site_uuids)} sites")

    # ========================================
    # PHASE 1b: Disable all cameras in DB BEFORE edge/MediaMTX cleanup.
    # Reconcile reads is_enabled/is_detection_enabled from DB; if it fires
    # between our edge cleanup and DB deletion it re-adds the cameras.
    # ========================================
    if camera_uuids:
        logger.info(f"[User Delete] Phase 1b: Disabling {len(camera_uuids)} cameras in DB to prevent reconcile re-adds")
        await db.execute(
            update(Camera)
            .where(Camera.user_id == user_id)
            .values(is_enabled=False, is_detection_enabled=False)
            .execution_options(synchronize_session=False)
        )
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
    # PHASE 3: Fast Foreground DB Cleanup (Sites & User)
    # ========================================
    logger.info(f"[User Delete] Phase 3: Deleting DB records (sites, user, cascades) (Foreground)")
    try:
        if site_uuids:
            # Explicitly delete sites; cameras and relationships should cascade 
            # based on how your DB is configured.
            await db.execute(sql_delete(Site).where(Site.user_id == user_id))

        await db.delete(current_user)
        await db.commit()
        _invalidate_user_snapshot_cache(request, user_id)
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
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
    # PHASE 5: Background Database Cleanup (Notifications & Videos)
    # ========================================
    logger.info(f"[User Delete] Phase 5: Spawning background task to clean up heavy tables (Notifications/Videos)")

    async def _heavy_table_cleanup_task(uid: int, cam_uuids: List[uuid.UUID]):
        logger.info(f"[User Cleanup Task] Starting background heavy cleanup for user={uid}")
        from application.repositories.site_repository import SiteRepository
        repo_for_delete = SiteRepository()
        alert_blob_keys: List[str] = []
        clip_blob_keys: List[str] = []

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

            if cam_uuids:
                # Batch delete video records and extract keys
                await repo_for_delete._batch_delete(
                    AsyncSessionLocal,
                    table=VideoRecord,
                    where_clause=VideoRecord.camera_uuid.in_(cam_uuids),
                    batch_size=2000,
                    label="user_video_records",
                    extract_col=VideoRecord.storage_key,
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
            logger.info(f"[User Cleanup Task] Background heavy cleanup COMPLETE for user={uid}")

        except Exception as e:
            logger.error(f"[User Cleanup Task] Failed heavy cleanup for user={uid}: {e}", exc_info=True)

    _spawn_bg_task(
        _heavy_table_cleanup_task(user_id, camera_uuids),
        name=f"delete_user_heavy_tables:{user_id}",
    )

    logger.info(f"[User Delete] HTTP response ready: user={user_id} effectively deleted from UI")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
