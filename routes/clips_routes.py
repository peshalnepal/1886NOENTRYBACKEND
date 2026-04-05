import uuid
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, PositiveInt
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.clip_storage import EventClipService
from core.database_orm import Camera, Notification, Site, VideoRecord, User
from dependencies import get_async_db, get_current_user


router = APIRouter(prefix="/clips", tags=["clips"])
logger = logging.getLogger(__name__)


class ClipOut(BaseModel):
    id: int
    camera_uuid: str
    camera_name: Optional[str] = None
    camera_code: Optional[str] = None
    site_uuid: Optional[str] = None
    site_name: Optional[str] = None
    site_code: Optional[str] = None
    external_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    status: str
    recording_url: Optional[str] = None
    overlay_payload: Optional[Dict[str, Any]] = None
    storage_key: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime


class BulkClipDeleteRequest(BaseModel):
    clip_ids: List[PositiveInt] = Field(..., min_length=1)


class BulkClipDeleteResponse(BaseModel):
    requested: int
    deleted: int
    deleted_ids: List[int] = Field(default_factory=list)


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
) -> List[VideoRecord]:
    ordered_ids = _normalize_clip_ids(clip_ids)
    if not ordered_ids:
        return []

    rows = (
        await db.execute(
            select(VideoRecord)
            .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
            .where(
                VideoRecord.id.in_(ordered_ids),
                Camera.user_id == int(user_id),
            )
        )
    ).scalars().all()

    clips_by_id = {int(clip.id): clip for clip in rows}
    return [clips_by_id[clip_id] for clip_id in ordered_ids if clip_id in clips_by_id]


async def _delete_clip_records(
    *,
    db: AsyncSession,
    clips: List[VideoRecord],
) -> int:
    if not clips:
        return 0

    clip_service = EventClipService()
    try:
        for clip in clips:
            storage_key = str(clip.storage_key or "").strip()
            if storage_key:
                try:
                    await clip_service.delete_blob(blob_name=storage_key)
                except Exception:
                    logger.warning(
                        "Failed deleting clip blob %s; removing DB record anyway",
                        storage_key,
                        exc_info=True,
                    )

            await db.delete(clip)

        await db.commit()
    finally:
        await clip_service.close()

    return len(clips)


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _normalize_overlay_box(raw_box: Any) -> Optional[Dict[str, int]]:
    if isinstance(raw_box, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(key in raw_box for key in keys):
            return None
        try:
            return {
                "x1": int(raw_box["x1"]),
                "y1": int(raw_box["y1"]),
                "x2": int(raw_box["x2"]),
                "y2": int(raw_box["y2"]),
            }
        except (TypeError, ValueError):
            return None

    if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        try:
            return {
                "x1": int(raw_box[0]),
                "y1": int(raw_box[1]),
                "x2": int(raw_box[2]),
                "y2": int(raw_box[3]),
            }
        except (TypeError, ValueError):
            return None

    return None


def _normalize_overlay_detection(raw_detection: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_detection, dict):
        return None

    box = _normalize_overlay_box(raw_detection.get("box") or raw_detection.get("bbox"))
    if box is None:
        return None

    try:
        conf = float(raw_detection.get("conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    return {
        "cls_name": str(raw_detection.get("cls_name") or raw_detection.get("class") or "obj"),
        "conf": conf,
        "box": box,
    }


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
    seen = set()
    for raw_detections in (msg.get("detections"), extra.get("detections")):
        if not isinstance(raw_detections, list):
            continue
        for raw_detection in raw_detections:
            normalized = _normalize_overlay_detection(raw_detection)
            if normalized is None:
                continue
            key = (
                normalized["cls_name"],
                normalized["conf"],
                normalized["box"]["x1"],
                normalized["box"]["y1"],
                normalized["box"]["x2"],
                normalized["box"]["y2"],
            )
            if key in seen:
                continue
            seen.add(key)
            detections.append(normalized)

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
    ) or 0
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

    for record, *_rest in rows:
        clip_url = str(getattr(record, "recording_url", "") or "").strip()
        if not clip_url:
            continue

        camera_uuid = getattr(record, "camera_uuid", None)
        if camera_uuid is None:
            continue

        clip_keys[(str(camera_uuid), clip_url)] = None
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
        .where(Camera.user_id == int(user.id))
        .order_by(desc(VideoRecord.created_at), desc(VideoRecord.id))
        .offset(int(offset))
        .limit(min(int(limit), 200))
    )

    if site_uuid is not None:
        stmt = stmt.where(Camera.site_uuid == site_uuid)

    if camera_uuid is not None:
        stmt = stmt.where(VideoRecord.camera_uuid == camera_uuid)

    if status:
        stmt = stmt.where(VideoRecord.status == str(status).strip())

    rows = (await db.execute(stmt)).all()
    clip_overlay_payloads = await _load_clip_overlay_payloads(
        db=db,
        user_id=int(user.id),
        rows=rows,
    )

    return [
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
            overlay_payload=clip_overlay_payloads.get(
                (str(record.camera_uuid), str(record.recording_url or "").strip())
            ),
            storage_key=record.storage_key,
            error=record.error,
            created_at=record.created_at,
        )
        for record, camera_name, camera_code, record_site_uuid, site_name, site_code in rows
    ]


@router.delete("", response_model=BulkClipDeleteResponse)
async def delete_clips(
    payload: BulkClipDeleteRequest,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    clip_ids = _normalize_clip_ids([int(value) for value in payload.clip_ids])
    if not clip_ids:
        raise HTTPException(status_code=422, detail="Select at least one clip to delete")

    clips = await _fetch_owned_clips(db=db, user_id=int(user.id), clip_ids=clip_ids)
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
