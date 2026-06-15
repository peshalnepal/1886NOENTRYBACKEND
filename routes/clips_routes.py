import asyncio
import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, desc, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.clip_storage import EventClipService
from core.database import AsyncSessionLocal
from core.database_orm import Camera, Notification, Site, VideoRecord, User
from core.schemas import BulkClipDeleteRequest, BulkClipDeleteResponse, ClipOut
from dependencies import get_async_db, get_current_user


router = APIRouter(prefix="/clips", tags=["clips"])
logger = logging.getLogger(__name__)


def _normalize_clip_ids(raw_ids: List[int]) -> List[int]:
    ordered_ids: List[int] = []
    seen: set[int] = set()

    for raw_id in raw_ids:
        clip_id = int(raw_id)
        if clip_id <= 0 or clip_id in seen:
            continue
        seen.add(clip_id)
        ordered_ids.append(clip_id)

    return ordered_ids


async def _fetch_owned_clips(
    *,
    db: AsyncSession,
    user_id: int,
    clip_ids: List[int],
    is_platform_admin: bool = False,
) -> List[VideoRecord]:
    ordered_ids = _normalize_clip_ids(clip_ids)
    if not ordered_ids:
        return []

    conds = [VideoRecord.id.in_(ordered_ids)]
    if not is_platform_admin:
        conds.append(Camera.user_id == int(user_id))

    rows = (
        await db.execute(
            select(VideoRecord)
            .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
            .where(*conds)
        )
    ).scalars().all()

    clips_by_id = {int(clip.id): clip for clip in rows}
    return [clips_by_id[clip_id] for clip_id in ordered_ids if clip_id in clips_by_id]


async def _delete_clip_blobs_background(blob_keys: List[str]) -> None:
    """Background task: delete Azure Blob Storage objects for removed clips.

    Runs AFTER db.commit() so the HTTP response is never blocked by Azure
    Storage round-trips. Uses parallel deletion (10 concurrent) instead of
    sequential to maximize throughput: 5000 blobs in 50s vs 500s.
    """
    unique_keys = list(dict.fromkeys(k for k in blob_keys if k))
    if not unique_keys:
        logger.info("[Clip Blob Cleanup] no blob keys to delete")
        return

    svc = EventClipService()
    batch_size = 10
    successfully_deleted = 0
    failed_count = 0

    try:
        logger.info("[Clip Blob Cleanup] starting parallel deletion of %d blobs", len(unique_keys))
        for i in range(0, len(unique_keys), batch_size):
            batch = unique_keys[i : i + batch_size]
            # Run up to 10 concurrent blob deletes
            tasks = [svc.delete_blob(blob_name=k) for k in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for key, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning("[Clip Blob Cleanup] blob delete failed %s: %s", key, str(result))
                    failed_count += 1
                else:
                    successfully_deleted += 1

            logger.info(
                "[Clip Blob Cleanup] batch %d/%d complete: %d deleted, %d failed",
                (i // batch_size) + 1,
                (len(unique_keys) + batch_size - 1) // batch_size,
                sum(1 for r in results if not isinstance(r, Exception)),
                sum(1 for r in results if isinstance(r, Exception)),
            )
    except Exception:
        logger.exception("[Clip Blob Cleanup] unexpected error during parallel deletion")
    finally:
        try:
            await svc.close()
        except Exception:
            pass
        logger.info(
            "[Clip Blob Cleanup] complete: %d successfully deleted, %d failed",
            successfully_deleted,
            failed_count,
        )


async def _batch_delete_clips(
    *,
    clip_ids: List[int],
    batch_size: int = 500,
) -> Tuple[int, List[str]]:
    """Delete clips in batches to prevent table lock exhaustion.

    Each batch is a separate transaction, releasing locks between commits.
    Returns: (total_deleted, blob_storage_keys_to_cleanup)
    """
    if not clip_ids:
        return 0, []

    total_deleted = 0
    all_blob_keys: List[str] = []
    sorted_ids = sorted(set(int(cid) for cid in clip_ids if cid > 0))

    logger.info("[Clip Batch Delete] starting deletion of %d clips in batches of %d", len(sorted_ids), batch_size)

    for batch_num, i in enumerate(range(0, len(sorted_ids), batch_size), start=1):
        batch_ids = sorted_ids[i : i + batch_size]

        try:
            async with AsyncSessionLocal() as db:
                # Fetch blob keys for this batch BEFORE deletion
                rows = (
                    await db.execute(
                        select(VideoRecord.storage_key).where(
                            VideoRecord.id.in_(batch_ids),
                        )
                    )
                ).scalars().all()

                batch_blob_keys = [
                    str(key or "").strip() for key in rows if str(key or "").strip()
                ]
                all_blob_keys.extend(batch_blob_keys)

                # Delete this batch
                result = await db.execute(
                    delete(VideoRecord).where(VideoRecord.id.in_(batch_ids))
                )
                await db.commit()

                deleted_count = result.rowcount or len(batch_ids)
                total_deleted += deleted_count

                logger.info(
                    "[Clip Batch Delete] batch %d: deleted %d clips, total=%d, remaining=%d",
                    batch_num,
                    deleted_count,
                    total_deleted,
                    len(sorted_ids) - total_deleted,
                )
        except Exception as exc:
            logger.exception(
                "[Clip Batch Delete] batch %d failed for %d clips: %s",
                batch_num,
                len(batch_ids),
                str(exc),
            )
            raise

    logger.info("[Clip Batch Delete] complete: %d clips deleted, %d blob keys to cleanup", total_deleted, len(all_blob_keys))
    return total_deleted, all_blob_keys


async def _delete_clip_records(
    *,
    db: AsyncSession,
    clips: List[VideoRecord],
) -> int:
    if not clips:
        return 0

    # Extract clip IDs for batched deletion
    clip_ids = [int(clip.id) for clip in clips if hasattr(clip, "id")]

    # Delete clips in batches to prevent table lock exhaustion
    # This returns immediately after all DB commits are done
    try:
        total_deleted, blob_keys = await _batch_delete_clips(clip_ids=clip_ids)
    except Exception as exc:
        logger.exception("[Clip Delete] batch deletion failed: %s", str(exc))
        raise

    # Delete Azure blobs in the background after the 200/204 is already sent.
    if blob_keys:
        asyncio.create_task(
            _delete_clip_blobs_background(blob_keys),
            name="clip_blob_cleanup",
        )
        logger.info("[Clip Delete] spawned background blob cleanup task for %d keys", len(blob_keys))

    return total_deleted


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


from application.services.overlay_normalize import (
    _append_overlay_detection,
)


def _overlay_payload_for_clip_camera(
    raw_payload: Any,
    *,
    camera_uuid: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_payload, dict):
        return None

    clip_camera_uuid = str(camera_uuid or "").strip()
    payload_camera_uuid = str(raw_payload.get("camera_uuid") or "").strip()
    if payload_camera_uuid and clip_camera_uuid and payload_camera_uuid != clip_camera_uuid:
        return None

    return raw_payload


def _iter_overlay_detection_candidates(*sources: Any):
    for source in sources:
        if isinstance(source, list):
            for item in source:
                yield item
        elif isinstance(source, dict):
            yield source


def _extract_clip_overlay_from_notification_payload(
    payload: Any,
    *,
    fallback_camera_uuid: Optional[str] = None,
) -> Optional[Tuple[Tuple[str, str], Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None

    msg = payload.get("msg") if isinstance(payload.get("msg"), dict) else {}
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    extra_clip = extra.get("clip") if isinstance(extra.get("clip"), dict) else {}

    camera_uuid = str(msg.get("camera_uuid") or fallback_camera_uuid or "").strip()
    clip_url = str(msg.get("clip_url") or extra_clip.get("recording_url") or "").strip()
    if not camera_uuid or not clip_url:
        return None

    detections: List[Dict[str, Any]] = []
    seen_exact: set[Tuple[Any, ...]] = set()
    tracked_bases: set[Tuple[Any, ...]] = set()
    untracked_indexes: Dict[Tuple[Any, ...], int] = {}
    detection_sources = (
        msg.get("detections"),
        msg.get("track"),
        msg.get("alert"),
        extra.get("detections"),
        extra.get("track"),
        extra.get("alert"),
    )
    for raw_detection in _iter_overlay_detection_candidates(*detection_sources):
        _append_overlay_detection(
            detections,
            raw_detection,
            seen_exact=seen_exact,
            tracked_bases=tracked_bases,
            untracked_indexes=untracked_indexes,
        )

    frame_w = _coerce_positive_int(
        extra.get("frame_w")
        or extra.get("frameWidth")
        or msg.get("frame_w")
        or msg.get("frameWidth")
    )
    frame_h = _coerce_positive_int(
        extra.get("frame_h")
        or extra.get("frameHeight")
        or msg.get("frame_h")
        or msg.get("frameHeight")
    )

    frame_ts_ms = _coerce_positive_int(
        extra.get("frame_ts_ms")
        or msg.get("frame_ts_ms")
        or msg.get("ts_ms")
    )
    frame_seq = _coerce_positive_int(extra.get("frame_seq") or msg.get("frame_seq")) or 0

    if not detections and frame_w is None and frame_h is None:
        return None

    return (
        (camera_uuid, clip_url),
        {
            "camera_uuid": camera_uuid,
            "frame_ts_ms": frame_ts_ms,
            "frame_seq": frame_seq,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "detections": detections,
        },
    )


async def _load_clip_overlay_payloads(
    *,
    db: AsyncSession,
    user_id: int,
    rows: List[Tuple[VideoRecord, Optional[str], Optional[str], Optional[uuid.UUID], Optional[str], Optional[str]]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    clip_keys: Dict[Tuple[str, str], None] = {}
    camera_uuids: List[uuid.UUID] = []
    seen_camera_uuids: set[uuid.UUID] = set()
    earliest_detected_at: Optional[datetime] = None
    # Track clip time windows so we can add clip_start_time/clip_end_time to fallback overlays.
    clip_time_windows: Dict[Tuple[str, str], Tuple[Optional[datetime], Optional[datetime]]] = {}

    for record, *_rest in rows:
        clip_url = str(getattr(record, "recording_url", "") or "").strip()
        if not clip_url:
            continue

        camera_uuid = getattr(record, "camera_uuid", None)
        if camera_uuid is None:
            continue

        key = (str(camera_uuid), clip_url)
        clip_keys[key] = None
        clip_time_windows[key] = (
            getattr(record, "start_time", None),
            getattr(record, "end_time", None),
        )
        if camera_uuid not in seen_camera_uuids:
            seen_camera_uuids.add(camera_uuid)
            camera_uuids.append(camera_uuid)

        candidate_dt = getattr(record, "start_time", None) or getattr(record, "created_at", None)
        if candidate_dt is not None and (
            earliest_detected_at is None or candidate_dt < earliest_detected_at
        ):
            earliest_detected_at = candidate_dt

    if not clip_keys or not camera_uuids:
        return {}

    stmt = (
        select(Notification)
        .where(
            Notification.user_id == int(user_id),
            Notification.camera_uuid.in_(camera_uuids),
        )
        .order_by(desc(Notification.detected_at), desc(Notification.id))
    )

    # Keep the lookup bounded to the time window represented by this clip page.
    if earliest_detected_at is not None:
        stmt = stmt.where(Notification.detected_at >= earliest_detected_at - timedelta(days=2))
    max_scan = min(len(clip_keys) * 10, 2000)
    stmt = stmt.limit(max_scan)

    notifications = (await db.execute(stmt)).scalars().all()
    overlays: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for notification in notifications:
        match = _extract_clip_overlay_from_notification_payload(
            getattr(notification, "payload", None),
            fallback_camera_uuid=str(getattr(notification, "camera_uuid", "") or ""),
        )
        if match is None:
            continue

        key, overlay_payload = match
        if key not in clip_keys or key in overlays:
            continue

        # Attach clip time window so the frontend can position overlay frames on the timeline.
        window = clip_time_windows.get(key)
        if window:
            clip_start, clip_end = window
            if clip_start is not None:
                overlay_payload["clip_start_time"] = clip_start.astimezone(timezone.utc).isoformat()
            if clip_end is not None:
                overlay_payload["clip_end_time"] = clip_end.astimezone(timezone.utc).isoformat()

        overlays[key] = overlay_payload
        if len(overlays) >= len(clip_keys):
            break

    return overlays

@router.get("", response_model=List[ClipOut])
async def list_clips(
    site_uuid: Optional[uuid.UUID] = None,
    camera_uuid: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")

    stmt = (
        select(
            VideoRecord,
            Camera.name,
            Camera.camera_code,
            Site.site_uuid,
            Site.name,
            Site.site_code,
        )
        .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
        .join(Site, Site.site_uuid == Camera.site_uuid)
        .order_by(desc(VideoRecord.created_at), desc(VideoRecord.id))
        .offset(int(offset))
        .limit(min(int(limit), 200))
    )

    # Platform admins see every clip; regular users see only their own, and only
    # once they're visible. When the site's org has an operator, freshly captured
    # clips are held invisible until the operator approves the matching alert, so
    # end users never get the playback before review.
    if not bool(getattr(user, "is_platform_admin", False)):
        stmt = stmt.where(Camera.user_id == int(user.id))
        stmt = stmt.where(VideoRecord.visible.is_(True))

    if site_uuid is not None:
        stmt = stmt.where(Camera.site_uuid == site_uuid)

    if camera_uuid is not None:
        stmt = stmt.where(VideoRecord.camera_uuid == camera_uuid)

    if status:
        stmt = stmt.where(VideoRecord.status == str(status).strip())

    rows = (await db.execute(stmt)).all()
    fallback_overlays: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if any(
        _overlay_payload_for_clip_camera(
            getattr(record, "overlay_payload", None),
            camera_uuid=getattr(record, "camera_uuid", None),
        ) is None
        for record, *_rest in rows
    ):
        fallback_overlays = await _load_clip_overlay_payloads(
            db=db,
            user_id=int(user.id),
            rows=rows,
        )

    clips: List[ClipOut] = []
    for record, camera_name, camera_code, record_site_uuid, site_name, site_code in rows:
        overlay_payload = _overlay_payload_for_clip_camera(
            getattr(record, "overlay_payload", None),
            camera_uuid=getattr(record, "camera_uuid", None),
        )
        if overlay_payload is None:
            clip_key = (
                str(getattr(record, "camera_uuid", "") or ""),
                str(getattr(record, "recording_url", "") or "").strip(),
            )
            overlay_payload = _overlay_payload_for_clip_camera(
                fallback_overlays.get(clip_key),
                camera_uuid=getattr(record, "camera_uuid", None),
            )

        # Ensure clip_start_time/clip_end_time are present so the frontend can
        # time-align detection overlays during video playback.  The fallback
        # overlays (from notifications) already carry these fields, but overlays
        # stored directly in VideoRecord.overlay_payload may not.
        if overlay_payload is not None and "clip_start_time" not in overlay_payload:
            overlay_payload = dict(overlay_payload)
            start = record.start_time
            end = record.end_time
            if start is not None:
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                overlay_payload["clip_start_time"] = start.isoformat()
            if end is not None:
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                overlay_payload["clip_end_time"] = end.isoformat()

        clips.append(
            ClipOut(
                id=int(record.id),
                camera_uuid=str(record.camera_uuid),
                camera_name=camera_name,
                camera_code=camera_code,
                site_uuid=str(record_site_uuid) if record_site_uuid is not None else None,
                site_name=site_name,
                site_code=site_code,
                external_id=str(record.external_id),
                start_time=record.start_time,
                end_time=record.end_time,
                duration=int(record.duration) if record.duration is not None else None,
                status=str(record.status),
                recording_url=record.recording_url,
                overlay_payload=overlay_payload,
                storage_key=record.storage_key,
                error=record.error,
                created_at=record.created_at,
            )
        )

    return clips
    
@router.delete("", response_model=BulkClipDeleteResponse)
async def delete_clips(
    payload: BulkClipDeleteRequest,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    clip_ids = _normalize_clip_ids([int(value) for value in payload.clip_ids])
    if not clip_ids:
        raise HTTPException(status_code=422, detail="Select at least one clip to delete")

    clips = await _fetch_owned_clips(
        db=db,
        user_id=int(user.id),
        clip_ids=clip_ids,
        is_platform_admin=bool(getattr(user, "is_platform_admin", False)),
    )
    deleted = await _delete_clip_records(db=db, clips=clips)
    return BulkClipDeleteResponse(
        requested=len(clip_ids),
        deleted=deleted,
        deleted_ids=[int(clip.id) for clip in clips],
    )


@router.delete("/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_clip(
    clip_id: int,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    clips = await _fetch_owned_clips(db=db, user_id=int(user.id), clip_ids=[clip_id])
    if not clips:
        raise HTTPException(status_code=404, detail="Clip not found")

    await _delete_clip_records(db=db, clips=clips)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
