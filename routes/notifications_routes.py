# routes/notifications_routes.py (or wherever your router lives)

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

                # Keepalive ping every 20s so proxies don’t kill the connection
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
    # Accept dict {"x1":..,"y1":..,"x2":..,"y2":..} or list [x1,y1,x2,y2]
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

    # Fetch camera mode from DB (correct UUID compare)
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


# ---------------------------
# DB-backed notifications (list / clear / delete)
# ---------------------------
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

    # parse filters
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
        stmt = select(Notification).where(Notification.user_id == int(user_id))
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

        stmt = delete(Notification).where(and_(*conds))
        res = await db.execute(stmt)
        await db.commit()

    return {"ok": True, "deleted": int(getattr(res, "rowcount", 0) or 0)}


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
        )
        if su:
            stmt = stmt.where(Notification.site_uuid == su)

        n = (await db.execute(stmt)).scalar_one()

    return {"user_id": int(user_id), "site_uuid": str(su) if su else None, "unread": int(n)}
