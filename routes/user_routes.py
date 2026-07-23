from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import uuid
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from application.services.user_snapshot_cache import UserSnapshotCache

from application.repositories.channel_repository import ChannelRepository
from application.repositories.organization_repository import OrganizationRepository
from application.repositories.site_repository import SiteRepository
from application.repositories.user_repository import UserRepository
from application.repositories.verify_repository import EmailVerificationRepository
from application.dtos import UserProfileUpdateDTO
from core.schemas import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    UserOut,
    UserProfileUpdateRequest,
)
from core.database_orm import Notification, User
from application.repositories._helpers import normalize_uuid_list
from core.database import AsyncSessionLocal
from core.security.hashing import get_password_hash, verify_password
from core.security.roles import OrgRole
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


async def perform_user_deletion(
    *,
    user_id: int,
    request: Request,
    db: AsyncSession,
    manager: Optional[Manager],
) -> None:
    """Run the full account-deletion pipeline for ``user_id``.

    Shared by ``DELETE /users/me`` (self-service),
    ``DELETE /platform/users/{id}`` (platform-admin god mode) and the
    org-admin member delete. Password verification is the caller's
    responsibility.

    **Ownership rule.** Sites, cameras, devices, notifications and their
    blobs belong to the *organization*, not to the member who happened to
    create them. So this runs in one of two modes:

    * The user is the **last member** of their org — the org is being
      dissolved, so its resources are torn down and the org row is dropped.
    * Other members remain — only the user's personal rows go. The org's
      sites/cameras/devices stay exactly where they are (their `user_id`
      FK is SET NULL, so they survive) and keep serving the remaining team.

    Personal rows (`notification`, `notification_emails`, `site_settings`,
    `pipelines`, `access_grants`) are per-user by definition and always
    CASCADE away with the user.
    """
    from routes.notifications_routes import invalidate_camera_mode_cache

    logger.info(f"[User Delete] Starting deletion of user={user_id}")

    # ========================================
    # PHASE 0: Decide the mode. An org is dissolved only when this user is
    # its last member; otherwise its resources must be left untouched.
    # ========================================
    org_repo = OrganizationRepository()
    org_ids = await org_repo.list_org_ids_for_user(db, user_id=user_id)
    dissolving_org_ids = [
        oid
        for oid in org_ids
        if await org_repo.count_org_members(db, org_id=oid, exclude_user_id=user_id) == 0
    ]
    logger.info(
        f"[User Delete] user={user_id} orgs={org_ids} "
        f"dissolving={dissolving_org_ids} "
        f"(orgs with remaining members keep all their resources)"
    )

    # ========================================
    # PHASE 1: Snapshot camera + site info BEFORE any changes.
    # Scoped to the orgs actually being dissolved — NOT to the departing
    # user. Keying this off `user_id` is what used to destroy a co-worker's
    # sites just because the creator left.
    # ========================================
    channel_repo = ChannelRepository()
    site_repo = SiteRepository()

    camera_uuids: List = []
    site_uuids: List = []
    for oid in dissolving_org_ids:
        camera_uuids.extend(
            cam.camera_uuid for cam in await channel_repo.list_cameras(db, org_id=oid)
        )
        site_uuids.extend(await site_repo.list_site_uuids(db, org_id=oid))

    logger.info(f"[User Delete] Snapshotted {len(camera_uuids)} cameras, {len(site_uuids)} sites")

    # ========================================
    # PHASE 1b: Disable all cameras in DB BEFORE edge/MediaMTX cleanup.
    # Reconcile reads is_enabled/is_detection_enabled from DB; if it fires
    # between our edge cleanup and DB deletion it re-adds the cameras.
    # ========================================
    if camera_uuids:
        logger.info(f"[User Delete] Phase 1b: Disabling {len(camera_uuids)} cameras in DB to prevent reconcile re-adds")
        await channel_repo.disable_cameras(db, camera_uuids=camera_uuids)
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
                # Scoped to dissolving orgs' cameras: the rest belong to
                # co-members and must keep streaming.
                manager.cleanup_user_resources(
                    db, user_id=user_id, camera_uuids=camera_uuids
                ),
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

        # Mark the dissolving orgs' sites soft-deleted so they leave queries now
        if site_uuids:
            await site_repo.soft_delete_sites(AsyncSessionLocal, site_uuids=site_uuids)

        _invalidate_user_snapshot_cache(request, user_id)
    except Exception:
        raise

    # ========================================
    # PHASE 3c: Delete the user row NOW, in the foreground.
    # This used to be the last step of the background task, which meant a
    # crash/restart mid-cleanup — or any exception in the notification
    # batch-delete, which is caught and logged — left the `users` row alive
    # forever. The account could still log in while all of its cameras and
    # sites were gone. The user row is the thing that must never linger, so
    # it is deleted synchronously and committed before we return.
    #
    # Notification rows survive this because their user_id FK is CASCADE:
    # deleting the user wipes them, so blob storage keys must be harvested
    # BEFORE this point (see Phase 3d) or the blobs leak.
    # ========================================
    user_email: Optional[str] = None
    org_ids: List[int] = []

    async with AsyncSessionLocal() as pre_session:
        user_row = await UserRepository().get_by_id(pre_session, user_id)
        if user_row is not None:
            user_email = str(user_row.email)
        # Grants are CASCADE-wiped with the user, so snapshot the orgs first.
        org_ids = await OrganizationRepository().list_org_ids_for_user(
            pre_session, user_id=user_id
        )

    # ========================================
    # PHASE 3d: Harvest notification blob keys BEFORE the CASCADE removes
    # the rows, so the background task can still delete the blobs.
    #
    # Two sources: the departing user's own notification rows (always), plus
    # every row belonging to a dissolving org's sites — those siblings belong
    # to co-members whose accounts may outlive this one, and their images and
    # clips would otherwise be orphaned in blob storage once the site's
    # CASCADE removes the rows that referenced them.
    # ========================================
    alert_blob_keys: List[str] = []
    clip_blob_keys: List[str] = list(video_clip_keys)

    scan_filter = Notification.user_id == user_id
    if site_uuids:
        scan_filter = scan_filter | Notification.site_uuid.in_(
            normalize_uuid_list(site_uuids)
        )

    # Paged by id so a heavy account never loads every payload at once.
    _SCAN_BATCH = 2000
    last_id = 0
    async with AsyncSessionLocal() as scan_session:
        while True:
            rows = (
                await scan_session.execute(
                    select(Notification.id, Notification.payload)
                    .where(
                        scan_filter,
                        Notification.id > last_id,
                    )
                    .order_by(Notification.id)
                    .limit(_SCAN_BATCH)
                )
            ).all()
            if not rows:
                break

            for notif_id, payload in rows:
                last_id = int(notif_id)
                try:
                    alert_key = extract_image_storage_key(payload)
                    if alert_key:
                        alert_blob_keys.append(alert_key)
                    clip_blob_keys.extend(
                        _extract_notification_clip_storage_keys(payload) or []
                    )
                except Exception:
                    continue

    logger.info(
        f"[User Delete] Phase 3d: Harvested {len(alert_blob_keys)} alert / "
        f"{len(clip_blob_keys)} clip blob keys"
    )

    async with AsyncSessionLocal() as del_session:
        try:
            deleted = await UserRepository().delete_by_id(del_session, user_id)
            # Purge OTP + pending-signup rows: keyed by email, no FK to users,
            # so nothing else would ever reap them.
            if user_email:
                await EmailVerificationRepository().purge_for_email(
                    del_session, user_email
                )
            # Drop any org this user was the last member of.
            dropped = await OrganizationRepository().delete_if_abandoned(
                del_session, org_ids=org_ids
            )
            await del_session.commit()
            logger.info(
                f"[User Delete] Phase 3c/3d: user row deleted={bool(deleted)}, "
                f"abandoned orgs dropped={dropped}"
            )
        except Exception:
            await del_session.rollback()
            raise

    _invalidate_user_snapshot_cache(request, user_id)

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

    async def _heavy_table_cleanup_task(
        uid: int, s_uuids: List, alert_keys: List[str], clip_keys: List[str]
    ):
        """Best-effort cleanup of things that only cost storage, never identity.

        The user row, its grants, notifications and any abandoned org are
        already gone (Phase 3c/3d) — everything here is safe to retry or lose
        without leaving a usable account behind.
        """
        logger.info(f"[User Cleanup Task] Starting background heavy cleanup for user={uid}")
        from application.repositories.site_repository import SiteRepository
        repo_for_delete = SiteRepository()

        try:
            if alert_keys:
                _spawn_bg_task(
                    _delete_blobs_background(alert_keys, service_cls=AlertImageStorageService, label="alert image"),
                    name=f"delete_user_alert_blobs:{uid}",
                )

            if clip_keys:
                _spawn_bg_task(
                    _delete_blobs_background(clip_keys, service_cls=EventClipService, label="clip"),
                    name=f"delete_user_clip_blobs:{uid}",
                )

            # Sites are soft-deleted and the user row is gone; hard-delete the
            # rows (CASCADE cleans up SiteSettings, SiteDevices, NotificationEmails).
            # Only the dissolving orgs' sites are in this list.
            if s_uuids:
                await repo_for_delete.delete_sites(AsyncSessionLocal, site_uuids=s_uuids)

            logger.info(f"[User Cleanup Task] Background heavy cleanup COMPLETE for user={uid}")

        except Exception as e:
            logger.error(f"[User Cleanup Task] Failed heavy cleanup for user={uid}: {e}", exc_info=True)

    _spawn_bg_task(
        _heavy_table_cleanup_task(user_id, site_uuids, alert_blob_keys, clip_blob_keys),
        name=f"delete_user_heavy_tables:{user_id}",
    )

    logger.info(f"[User Delete] HTTP response ready: user={user_id} effectively deleted from UI")


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_account(
    payload: DeleteAccountRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    """Self-service account deletion. See :func:`perform_user_deletion`.

    Succession rule: an Org Admin may leave whenever another admin is left
    to run the org. The *last* admin may only leave once the org is empty —
    otherwise their departure would strand the remaining members in a tenant
    nobody can administer. They must delete the other members first, at which
    point deleting their own account takes the (now empty) org with it.
    """
    if not verify_password(payload.password.get_secret_value(), current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    org_repo = OrganizationRepository()
    for org_id in await org_repo.list_org_ids_for_user(db, user_id=int(current_user.id)):
        role_name = await org_repo.get_org_role_name(
            db, user_id=int(current_user.id), org_id=int(org_id)
        )
        if role_name != OrgRole.ADMIN.value:
            continue
        if await org_repo.count_admins(db, org_id=int(org_id)) > 1:
            continue  # another admin remains — safe to leave

        others = await org_repo.count_org_members(
            db, org_id=int(org_id), exclude_user_id=int(current_user.id)
        )
        if others:
            org = await org_repo.get_by_id(db, int(org_id))
            raise HTTPException(
                status_code=409,
                detail=(
                    f"You are the only admin of \"{org.name if org else 'your organization'}\" "
                    f"and it still has {others} other member(s). Promote another admin, or "
                    f"remove the remaining members first, then delete your account."
                ),
            )

    await perform_user_deletion(
        user_id=int(current_user.id),
        request=request,
        db=db,
        manager=manager,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
