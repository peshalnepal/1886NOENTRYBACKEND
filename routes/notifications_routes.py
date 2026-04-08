import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import StreamingResponse

from application.services.alert_image_storage import (
    AlertImageStorageService,
    extract_image_storage_key,
)
from application.services.notification import WebNotificationHub
from application.services.user_snapshot_cache import (
    CachedUserSnapshot,
    UserSnapshotCache,
    UserSnapshotLookupError,
)
from core.database_orm import Notification, User
from core.security.tokens import decode_access_token
from dependencies import (
    get_alert_blob_cleanup_tasks,
    get_async_db,
    get_current_user,
    get_notification_hub,
    get_session_factory,
    get_user_snapshot_cache,
    get_notification_service
)
from application.services.notification import NotificationService

router = APIRouter(prefix="/notifications")
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# stream helpers
# -------------------------------------------------------------------
def _sse(data: str, event: Optional[str] = None) -> str:
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def _extract_bearer_token(auth_header: str, access_token: Optional[str]) -> str:
    token = ""

    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()

    if not token:
        token = str(access_token or "").strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    return token


async def _resolve_stream_user(
    *,
    auth_header: str,
    access_token: Optional[str],
    cache: UserSnapshotCache,
    session_factory: async_sessionmaker[AsyncSession],
) -> CachedUserSnapshot:
    token = _extract_bearer_token(auth_header, access_token)

    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    raw_user_id = payload.get("user_id") or payload.get("sub")
    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    try:
        user = await cache.get(session_factory=session_factory, user_id=user_id)
    except UserSnapshotLookupError:
        raise HTTPException(status_code=503, detail="Database not available")

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# -------------------------------------------------------------------
# response / request models
# -------------------------------------------------------------------
class NotificationOut(BaseModel):
    id: int
    user_id: int
    site_uuid: str
    camera_uuid: Optional[str] = None
    site_name: Optional[str] = None
    camera_name: Optional[str] = None
    device_uuid: Optional[str] = None
    event_type: str
    title: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    image_storage_key: Optional[str] = None
    clip_url: Optional[str] = None
    clip_status: Optional[str] = None
    detected_at: datetime
    created_at: datetime
    read_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str


class ChartPoint(BaseModel):
    bucket_start: datetime
    count: int


class DetectionsOverTimeOut(BaseModel):
    user_id: int
    site_uuid: Optional[str] = None
    hours: int
    object_class: Optional[str] = None
    roi_only: bool
    bucket_minutes: int
    from_time: datetime = Field(alias="from")
    to: datetime
    total: int
    points: List[ChartPoint]

    class Config:
        populate_by_name = True


class ClearNotificationsRequest(BaseModel):
    site_uuid: Optional[str] = None
    camera_uuid: Optional[str] = None


class DeleteNotificationsRequest(BaseModel):
    site_uuid: Optional[str] = None
    camera_uuid: Optional[str] = None
    notification_ids: Optional[List[int]] = None


# -------------------------------------------------------------------
# shared helpers
# -------------------------------------------------------------------
def _parse_optional_uuid(value: Optional[str], field_name: str) -> Optional[uuid.UUID]:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}")


def _validate_pagination(limit: int, offset: int) -> Tuple[int, int]:
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")
    return min(int(limit), 500), int(offset)


def _payload_msg(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    msg = payload.get("msg")
    if isinstance(msg, dict):
        return msg
    return payload


def _to_out(n: Notification) -> NotificationOut:
    msg = _payload_msg(n.payload)
    extra = n.payload.get("extra") if isinstance(n.payload, dict) else None

    image_url = ""
    image_storage_key = ""
    clip_url = ""
    clip_status = ""

    if isinstance(extra, dict):
        image_url = str(extra.get("image_url") or "").strip()
        image_storage_key = str(extra.get("image_storage_key") or "").strip()

    if not image_url:
        image_url = str(msg.get("image_url") or "").strip()
    if not image_storage_key:
        image_storage_key = str(msg.get("image_storage_key") or "").strip()

    clip_url = str(msg.get("clip_url") or "").strip()
    clip_status = str(msg.get("clip_status") or "").strip()

    if isinstance(extra, dict) and not clip_url:
        clip_payload = extra.get("clip")
        if isinstance(clip_payload, dict):
            clip_url = str(clip_payload.get("recording_url") or "").strip()
            clip_status = str(clip_payload.get("status") or "").strip()

    return NotificationOut(
        id=int(n.id),
        user_id=int(n.user_id),
        site_uuid=str(n.site_uuid),
        camera_uuid=str(n.camera_uuid) if n.camera_uuid else None,
        site_name=str(msg.get("site_name") or "") or None,
        camera_name=str(msg.get("camera_name") or "") or None,
        device_uuid=str(n.device_uuid) if n.device_uuid else None,
        event_type=str(n.event_type),
        title=n.title,
        message=n.message,
        payload=n.payload,
        image_url=image_url or None,
        image_storage_key=image_storage_key or None,
        clip_url=clip_url or None,
        clip_status=clip_status or None,
        detected_at=n.detected_at,
        created_at=n.created_at,
        read_at=n.read_at,
        sent_at=n.sent_at,
        status=str(n.status),
    )


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_OBJECT_CLASS_ALIASES: Dict[str, str] = {
    "person": "person",
    "people": "person",
    "persons": "person",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motor cycle": "motorcycle",
    "motor bike": "motorcycle",
    "motorbike": "motorcycle",
    "motor-bike": "motorcycle",
}


def _canonical_object_class(raw: Any) -> Optional[str]:
    if raw is None:
        return None

    key = str(raw).strip().lower()
    if not key or key == "all":
        return None

    key = key.replace("_", " ").replace("-", " ")
    key = " ".join(key.split())

    normalized = _OBJECT_CLASS_ALIASES.get(key)
    if normalized:
        return normalized

    if key.endswith("s"):
        normalized = _OBJECT_CLASS_ALIASES.get(key[:-1])
        if normalized:
            return normalized

    return None


def _normalize_object_class(raw: Optional[str]) -> Optional[str]:
    normalized = _canonical_object_class(raw)
    if normalized:
        return normalized
    if raw is None or str(raw).strip().lower() in {"", "all"}:
        return None

    raise HTTPException(
        status_code=422,
        detail="Invalid object_class. Use one of: person, car, truck, motorcycle",
    )


def _extract_object_classes(event_type: Any, title: Any, message: Any, payload: Any) -> Set[str]:
    msg = _payload_msg(payload)
    classes: Set[str] = set()

    for source in (
        msg.get("cls_names"),
        payload.get("cls_names") if isinstance(payload, dict) else None,
    ):
        if isinstance(source, (list, tuple)):
            for item in source:
                normalized = _canonical_object_class(item)
                if normalized:
                    classes.add(normalized)

    for source in (
        msg.get("cls_name"),
        payload.get("cls_name") if isinstance(payload, dict) else None,
    ):
        normalized = _canonical_object_class(source)
        if normalized:
            classes.add(normalized)

    hint = " ".join(
        [
            str(event_type or ""),
            str(title or ""),
            str(message or ""),
        ]
    ).lower()

    if re.search(r"\bperson\b|\bpeople\b", hint):
        classes.add("person")
    if re.search(r"\bcar\b|\bcars\b", hint):
        classes.add("car")
    if re.search(r"\btruck\b|\btrucks\b", hint):
        classes.add("truck")
    if re.search(r"\bmotor[\s-]?cycle\b|\bmotor bike\b|\bmotorbike\b", hint):
        classes.add("motorcycle")

    return classes


def _is_roi_notification(event_type: Any, title: Any, message: Any, payload: Any) -> bool:
    hint = " ".join([str(event_type or ""), str(title or ""), str(message or "")]).lower()
    if "roi" in hint:
        return True

    msg = _payload_msg(payload)
    if msg.get("roi_id") not in (None, "", 0):
        return True

    if isinstance(payload, dict) and payload.get("roi_id") not in (None, "", 0):
        return True

    return False


# -------------------------------------------------------------------
# blob cleanup helpers
# -------------------------------------------------------------------
async def _delete_alert_blob_keys(storage_keys: List[str]) -> None:
    unique_keys = [key for key in dict.fromkeys(str(key or "").strip() for key in storage_keys) if key]
    if not unique_keys:
        return

    image_service = AlertImageStorageService()
    try:
        for storage_key in unique_keys:
            try:
                await image_service.delete_blob(blob_name=storage_key)
            except Exception:
                logger.warning(
                    "Failed deleting alert image blob %s after alert removal",
                    storage_key,
                    exc_info=True,
                )
    finally:
        await image_service.close()


def _schedule_alert_blob_cleanup(tasks: set, storage_keys: List[str]) -> None:
    unique_keys = [key for key in dict.fromkeys(str(key or "").strip() for key in storage_keys) if key]
    if not unique_keys:
        return

    task = asyncio.create_task(
        _delete_alert_blob_keys(unique_keys),
        name="alert_blob_cleanup",
    )
    tasks.add(task)

    def _on_done(done_task: asyncio.Task) -> None:
        tasks.discard(done_task)
        try:
            done_task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Alert blob cleanup task failed")

    task.add_done_callback(_on_done)


# -------------------------------------------------------------------
# delete implementation
# -------------------------------------------------------------------
async def _delete_notifications_impl(
    *,
    payload: DeleteNotificationsRequest,
    db: AsyncSession,
    current_user: User,
    cleanup_tasks: set,
):
    su = _parse_optional_uuid(payload.site_uuid, "site_uuid")
    cu = _parse_optional_uuid(payload.camera_uuid, "camera_uuid")

    conds = [
        Notification.user_id == int(current_user.id),
        Notification.visible.is_(True),
    ]

    if su:
        conds.append(Notification.site_uuid == su)
    if cu:
        conds.append(Notification.camera_uuid == cu)

    if payload.notification_ids:
        ids: List[int] = []
        for raw_id in payload.notification_ids:
            try:
                parsed = int(raw_id)
            except Exception:
                continue
            if parsed > 0:
                ids.append(parsed)

        ids = sorted(set(ids))
        if not ids:
            return {"ok": True, "deleted": 0}

        conds.append(Notification.id.in_(ids))

    rows = (
        await db.execute(
            select(Notification.id, Notification.payload).where(and_(*conds))
        )
    ).all()

    matched_ids: List[int] = []
    storage_keys: List[str] = []

    for notification_id, notification_payload in rows:
        try:
            parsed_id = int(notification_id)
        except Exception:
            continue

        if parsed_id <= 0:
            continue

        matched_ids.append(parsed_id)

        storage_key = extract_image_storage_key(notification_payload)
        if storage_key:
            storage_keys.append(storage_key)

    matched_ids = sorted(set(matched_ids))
    if not matched_ids:
        return {"ok": True, "deleted": 0}

    await db.execute(
        update(Notification)
        .where(Notification.id.in_(matched_ids))
        .values(visible=False)
        .execution_options(synchronize_session=False)
    )
    await db.commit()

    _schedule_alert_blob_cleanup(cleanup_tasks, storage_keys)
    return {"ok": True, "deleted": len(matched_ids)}


# -------------------------------------------------------------------
# routes
# -------------------------------------------------------------------
@router.get("/stream")
async def notifications_stream(
    request: Request,
    access_token: Optional[str] = None,
    hub: WebNotificationHub = Depends(get_notification_hub),
    user_snapshot_cache: UserSnapshotCache = Depends(get_user_snapshot_cache),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """
    User-scoped notifications SSE stream.
    Authentication can be provided via Authorization header or access_token query param.
    """
    user = await _resolve_stream_user(
        auth_header=request.headers.get("authorization", ""),
        access_token=access_token,
        cache=user_snapshot_cache,
        session_factory=session_factory,
    )
    user_id = int(user.id)
    q = await hub.subscribe(user_id=user_id)

    async def gen():
        last_ping = time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    break

                now = time.monotonic()
                if (now - last_ping) > 20.0:
                    last_ping = now
                    yield _sse(json.dumps({"ts": int(time.time() * 1000)}), event="ping")

                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                yield _sse(msg.model_dump_json(), event="notification")
        finally:
            await hub.unsubscribe(user_id=user_id, q=q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@router.get("", response_model=List[NotificationOut])
async def list_notifications(
    site_uuid: Optional[str] = None,
    camera_uuid: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    limit, offset = _validate_pagination(limit, offset)
    su = _parse_optional_uuid(site_uuid, "site_uuid")
    cu = _parse_optional_uuid(camera_uuid, "camera_uuid")

    stmt = select(Notification).where(
        Notification.user_id == int(current_user.id),
        Notification.visible.is_(True),
    )

    if su:
        stmt = stmt.where(Notification.site_uuid == su)
    if cu:
        stmt = stmt.where(Notification.camera_uuid == cu)
    if unread_only:
        stmt = stmt.where(Notification.read_at.is_(None))

    stmt = stmt.order_by(desc(Notification.detected_at)).offset(offset).limit(limit)

    rows = (await db.execute(stmt)).scalars().all()
    return [_to_out(n) for n in rows]


@router.post("/delete")
async def delete_notifications_post(
    payload: DeleteNotificationsRequest,
    current_user: User = Depends(get_current_user),
    notification_service: NotificationService = Depends(get_notification_service),
):
    return await notification_service.handle_deletion_event(
        user_id=int(current_user.id),
        notification_ids=payload.notification_ids or [],
    )

@router.delete("")
async def delete_notifications(
    payload: DeleteNotificationsRequest,
    current_user: User = Depends(get_current_user),
    notification_service: NotificationService = Depends(get_notification_service),
):
    return await notification_service.handle_deletion_event(
        user_id=int(current_user.id),
        notification_ids=payload.notification_ids or [],
    )
    
@router.get("/detections-over-time", response_model=DetectionsOverTimeOut)
async def detections_over_time(
    site_uuid: Optional[str] = None,
    hours: int = 24,
    object_class: Optional[str] = None,
    roi_only: bool = False,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    su = _parse_optional_uuid(site_uuid, "site_uuid")
    class_filter = _normalize_object_class(object_class)

    hours_i = max(1, min(int(hours), 24 * 7))
    bucket_minutes = 60 if hours_i <= 24 else 24 * 60

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_i)
    bucket_ms = bucket_minutes * 60 * 1000
    start_ms = int(start.timestamp() * 1000)
    aligned_start_ms = start_ms - (start_ms % bucket_ms)
    now_ms = int(now.timestamp() * 1000)

    stmt = select(
        Notification.detected_at,
        Notification.event_type,
        Notification.title,
        Notification.message,
        Notification.payload,
    ).where(
        Notification.user_id == int(current_user.id),
        Notification.visible.is_(True),
        Notification.detected_at >= start,
    )

    if su:
        stmt = stmt.where(Notification.site_uuid == su)

    rows = (await db.execute(stmt)).all()

    counts: Dict[int, int] = {}
    for raw_dt, event_type, title, message, payload in rows:
        if roi_only and not _is_roi_notification(event_type, title, message, payload):
            continue

        if class_filter:
            classes = _extract_object_classes(event_type, title, message, payload)
            if class_filter not in classes:
                continue

        dt = _as_utc(raw_dt)
        ts_ms = int(dt.timestamp() * 1000)
        bucket = ts_ms - (ts_ms % bucket_ms)
        counts[bucket] = counts.get(bucket, 0) + 1

    points: List[ChartPoint] = []
    cursor = aligned_start_ms
    while cursor <= now_ms:
        points.append(
            ChartPoint(
                bucket_start=datetime.fromtimestamp(cursor / 1000.0, tz=timezone.utc),
                count=int(counts.get(cursor, 0)),
            )
        )
        cursor += bucket_ms

    total = sum(p.count for p in points)

    return DetectionsOverTimeOut(
        user_id=int(current_user.id),
        site_uuid=str(su) if su else None,
        hours=hours_i,
        object_class=class_filter,
        roi_only=bool(roi_only),
        bucket_minutes=bucket_minutes,
        **{"from": datetime.fromtimestamp(aligned_start_ms / 1000.0, tz=timezone.utc)},
        to=now,
        total=int(total),
        points=points,
    )


@router.get("/unread-count")
async def unread_count(
    site_uuid: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    su = _parse_optional_uuid(site_uuid, "site_uuid")

    stmt = select(func.count(Notification.id)).where(
        Notification.user_id == int(current_user.id),
        Notification.read_at.is_(None),
        Notification.visible.is_(True),
    )

    if su:
        stmt = stmt.where(Notification.site_uuid == su)

    count = (await db.execute(stmt)).scalar_one()

    return {
        "user_id": int(current_user.id),
        "site_uuid": str(su) if su else None,
        "unread": int(count),
    }