# routes/notifications_routes.py (or wherever your router lives)

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, func, select, update, delete
from starlette.responses import StreamingResponse

from core.database_orm import Camera, Notification
from application.services.notification import WebNotificationHub, CameraMode
from domain.events import DetectionBox, DetectionItem, DetectionsProducedEvent

router = APIRouter(prefix="/notifications")


# ---------------------------
# SSE helpers
# ---------------------------
def _sse(data: str, event: Optional[str] = None) -> str:
    # SSE format: optional event + data
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


@router.get("/stream")
async def notifications_stream(request: Request):
    """
    Global notifications SSE stream.
    Reads from in-memory hub (WebNotificationHub).
    """
    hub: WebNotificationHub = getattr(request.app.state, "notification_hub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="Notification hub not available")

    q = await hub.subscribe()

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
            await hub.unsubscribe(q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # If you ever run behind nginx:
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ---------------------------
# Alert ingest (/alert)
# ---------------------------
class AlertRequest(BaseModel):
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    detections: List[Dict[str, Any]] = Field(default_factory=list)


def _parse_box(raw_box: Any) -> Optional[DetectionBox]:
    if isinstance(raw_box, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(k in raw_box for k in keys):
            return None
        try:
            return DetectionBox(
                x1=int(raw_box["x1"]),
                y1=int(raw_box["y1"]),
                x2=int(raw_box["x2"]),
                y2=int(raw_box["y2"]),
            )
        except Exception:
            return None

    if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        try:
            return DetectionBox(
                x1=int(raw_box[0]),
                y1=int(raw_box[1]),
                x2=int(raw_box[2]),
                y2=int(raw_box[3]),
            )
        except Exception:
            return None

    return None


def _session_factory_from_app(request: Request):
    """
    Prefer app.state.session_factory if you store it there.
    Fallback to notification_service._session_factory.
    """
    sf = getattr(request.app.state, "session_factory", None)
    if sf is not None:
        return sf
    svc = getattr(request.app.state, "notification_service", None)
    return getattr(svc, "_session_factory", None)


@router.post("/alert")
async def receive_alert(payload: AlertRequest, request: Request):
    """
    Ingest detections from edge devices (Jetson).
    """
    svc = getattr(request.app.state, "notification_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Notification service not available")

    # validate camera_uuid as UUID string
    try:
        cam_uuid_obj = uuid.UUID(str(payload.camera_uuid))
        camera_uuid = str(cam_uuid_obj)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid camera_uuid")

    # Convert payload detections -> domain DetectionItem list
    det_items: List[DetectionItem] = []
    for d in payload.detections or []:
        box = _parse_box(d.get("box"))
        if box is None:
            continue
        det_items.append(
            DetectionItem(
                cls_name=str(d.get("cls_name", "unknown")),
                conf=float(d.get("conf", 0.0) or 0.0),
                box=box,
            )
        )

    ev = DetectionsProducedEvent(
        camera_uuid=camera_uuid,
        model_id="remote-jetson",
        frame_ts_ms=int(payload.frame_ts_ms),
        frame_seq=int(payload.frame_seq),
        detections=det_items,
    )

    mode = CameraMode(detection_enabled=True, notification_enabled=True)
    sf = _session_factory_from_app(request)
    if sf is not None:
        try:
            async with sf() as session:
                res = await session.execute(
                    select(
                        Camera.is_enabled,
                        Camera.is_detection_enabled,
                        Camera.is_notification_enabled,
                    ).where(Camera.camera_uuid == cam_uuid_obj)  # FIX: UUID-to-UUID compare
                )
                row = res.first()
                if row:
                    enabled, det_enabled, notif_enabled = row
                    mode = CameraMode(
                        detection_enabled=bool(enabled and det_enabled),
                        notification_enabled=bool(enabled and notif_enabled),
                    )
        except Exception:
            # If DB lookup fails, don’t hard fail ingest; just default to True.
            # (You can flip this to fail-closed if you prefer.)
            pass

    await svc.handle_detection_event(
        ev,
        camera_mode=mode,
        frame_w=payload.frame_w,
        frame_h=payload.frame_h,
    )

    return {"ok": True}


class NotificationOut(BaseModel):
    id: int
    user_id: int
    site_uuid: str
    camera_uuid: Optional[str] = None
    device_uuid: Optional[str] = None
    event_type: str
    title: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    detected_at: datetime
    created_at: datetime
    read_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str


def _to_out(n: Notification) -> NotificationOut:
    return NotificationOut(
        id=int(n.id),
        user_id=int(n.user_id),
        site_uuid=str(n.site_uuid),
        camera_uuid=str(n.camera_uuid) if n.camera_uuid else None,
        device_uuid=str(n.device_uuid) if n.device_uuid else None,
        event_type=str(n.event_type),
        title=n.title,
        message=n.message,
        payload=n.payload,
        detected_at=n.detected_at,
        created_at=n.created_at,
        read_at=n.read_at,
        sent_at=n.sent_at,
        status=str(n.status),
    )


@router.get("", response_model=List[NotificationOut])
async def list_notifications(
    request: Request,
    user_id: int,
    site_uuid: Optional[str] = None,
    camera_uuid: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 100,
):
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    su = None
    cu = None
    try:
        su = uuid.UUID(site_uuid) if site_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid site_uuid")

    try:
        cu = uuid.UUID(camera_uuid) if camera_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid camera_uuid")

    async with sf() as db:
        stmt = select(Notification).where(Notification.user_id == int(user_id),
                                          Notification.visible==True)
        if su:
            stmt = stmt.where(Notification.site_uuid == su)
        if cu:
            stmt = stmt.where(Notification.camera_uuid == cu)
        if unread_only:
            stmt = stmt.where(Notification.read_at.is_(None))

        stmt = stmt.order_by(desc(Notification.detected_at)).limit(max(1, min(int(limit), 500)))
        rows = (await db.execute(stmt)).scalars().all()

    return [_to_out(n) for n in rows]


class ClearNotificationsRequest(BaseModel):
    user_id: int
    site_uuid: Optional[str] = None
    camera_uuid: Optional[str] = None


@router.post("/clear")
async def clear_notifications(payload: ClearNotificationsRequest, request: Request):
    """
    Recommended behavior: do NOT delete history.
    Mark as read (read_at + status='read').
    """
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    su = None
    cu = None
    try:
        su = uuid.UUID(payload.site_uuid) if payload.site_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid site_uuid")
    try:
        cu = uuid.UUID(payload.camera_uuid) if payload.camera_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid camera_uuid")

    now = datetime.now(timezone.utc)

    async with sf() as db:
        conds = [Notification.user_id == int(payload.user_id)]
        if su:
            conds.append(Notification.site_uuid == su)
        if cu:
            conds.append(Notification.camera_uuid == cu)

        stmt = (
            update(Notification)
            .where(and_(*conds))
            .values(read_at=now, status="read")
        )
        res = await db.execute(stmt)
        await db.commit()

    return {"ok": True, "updated": int(getattr(res, "rowcount", 0) or 0)}


class DeleteNotificationsRequest(BaseModel):
    user_id: int
    site_uuid: Optional[str] = None
    camera_uuid: Optional[str] = None
    notification_ids: Optional[List[int]] = None


@router.delete("")
async def delete_notifications(payload: DeleteNotificationsRequest, request: Request):
    """
    Hard delete (use sparingly).
    """
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    su = None
    cu = None
    try:
        su = uuid.UUID(payload.site_uuid) if payload.site_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid site_uuid")
    try:
        cu = uuid.UUID(payload.camera_uuid) if payload.camera_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid camera_uuid")

    async with sf() as db:
        conds = [Notification.user_id == int(payload.user_id)]
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

        stmt = (
            update(Notification)
            .where(and_(*conds))
            .values(visible=False)
        )
        res = await db.execute(stmt)        
        await db.commit()

    return {"ok": True, "deleted": int(getattr(res, "rowcount", 0) or 0)}


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


def _payload_msg(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    msg = payload.get("msg")
    if isinstance(msg, dict):
        return msg
    return payload


def _extract_object_classes(event_type: Any, title: Any, message: Any, payload: Any) -> Set[str]:
    msg = _payload_msg(payload)
    classes: Set[str] = set()

    for source in (msg.get("cls_names"), payload.get("cls_names") if isinstance(payload, dict) else None):
        if isinstance(source, (list, tuple)):
            for item in source:
                normalized = _canonical_object_class(item)
                if normalized:
                    classes.add(normalized)

    for source in (msg.get("cls_name"), payload.get("cls_name") if isinstance(payload, dict) else None):
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


@router.get("/detections-over-time")
async def detections_over_time(
    request: Request,
    user_id: int,
    site_uuid: Optional[str] = None,
    hours: int = 24,
    object_class: Optional[str] = None,
    roi_only: bool = False,
):
    """
    Returns simple time buckets for dashboard charts.
    """
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    su = None
    try:
        su = uuid.UUID(site_uuid) if site_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid site_uuid")

    class_filter = _normalize_object_class(object_class)

    hours_i = max(1, min(int(hours), 24 * 90))
    # Keep output compact and readable:
    # - 1 day or less => hourly buckets
    # - over 1 day     => daily buckets
    bucket_minutes = 60 if hours_i <= 24 else 24 * 60

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_i)
    bucket_ms = bucket_minutes * 60 * 1000
    start_ms = int(start.timestamp() * 1000)
    aligned_start_ms = start_ms - (start_ms % bucket_ms)
    now_ms = int(now.timestamp() * 1000)

    async with sf() as db:
        stmt = select(
            Notification.detected_at,
            Notification.event_type,
            Notification.title,
            Notification.message,
            Notification.payload,
        ).where(
            Notification.user_id == int(user_id),
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

    points = []
    cursor = aligned_start_ms
    while cursor <= now_ms:
        points.append(
            {
                "bucket_start": datetime.fromtimestamp(cursor / 1000.0, tz=timezone.utc),
                "count": int(counts.get(cursor, 0)),
            }
        )
        cursor += bucket_ms

    total = sum(p["count"] for p in points)
    return {
        "user_id": int(user_id),
        "site_uuid": str(su) if su else None,
        "hours": hours_i,
        "object_class": class_filter,
        "roi_only": bool(roi_only),
        "bucket_minutes": bucket_minutes,
        "from": datetime.fromtimestamp(aligned_start_ms / 1000.0, tz=timezone.utc),
        "to": now,
        "total": int(total),
        "points": points,
    }


@router.get("/unread-count")
async def unread_count(
    request: Request,
    user_id: int,
    site_uuid: Optional[str] = None,
):
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    su = None
    try:
        su = uuid.UUID(site_uuid) if site_uuid else None
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid site_uuid")

    async with sf() as db:
        stmt = select(func.count(Notification.id)).where(
            Notification.user_id == int(user_id),
            Notification.read_at.is_(None),
            Notification.visible==True
        )
        if su:
            stmt = stmt.where(Notification.site_uuid == su)

        n = (await db.execute(stmt)).scalar_one()

    return {"user_id": int(user_id), "site_uuid": str(su) if su else None, "unread": int(n)}
