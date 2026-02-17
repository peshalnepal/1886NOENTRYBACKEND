import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from core.database_orm import Camera
from sqlalchemy import select

from application.services.notification import WebNotificationHub, CameraMode
from domain.events import DetectionBox, DetectionItem, DetectionsProducedEvent

router = APIRouter(prefix="/notifications")

def _sse(data: str) -> str:
    return f"data: {data}\n\n"

@router.get("/stream")
async def notifications_stream(request: Request):
    hub: WebNotificationHub = request.app.state.notification_hub
    q = await hub.subscribe()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await q.get()
                yield _sse(msg.model_dump_json())
        finally:
            await hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


class AlertRequest(BaseModel):
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    detections: List[Dict[str, Any]]

@router.post("/alert")
async def receive_alert(payload: AlertRequest, request: Request):
    """
    Ingest detections from edge devices (Jetson).
    """
    if not hasattr(request.app.state, "notification_service"):
        raise HTTPException(status_code=503, detail="Notification service not available")
    
    svc = request.app.state.notification_service
    try:
        camera_uuid = str(uuid.UUID(str(payload.camera_uuid)))
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid camera_uuid")

    # Convert payload -> DetectionsProducedEvent
    det_items = []
    for d in payload.detections:
        box_arr = d.get("box")
        if not box_arr or len(box_arr) < 4:
            continue
        
        # safely parse box
        box = DetectionBox(x1=int(box_arr[0]), y1=int(box_arr[1]), x2=int(box_arr[2]), y2=int(box_arr[3]))
        det_items.append(DetectionItem(
            cls_name=str(d.get("cls_name", "unknown")),
            conf=float(d.get("conf", 0.0)),
            box=box
        ))

    ev = DetectionsProducedEvent(
        camera_uuid=camera_uuid,
        model_id="remote-jetson",
        frame_ts_ms=payload.frame_ts_ms,
        frame_seq=payload.frame_seq,
        detections=det_items
    )

    # Fetch camera mode from DB to check detection/notification status
    mode = CameraMode(detection_enabled=True, notification_enabled=True)
    session_factory = getattr(svc, "_session_factory", None)
    if session_factory is not None:
        async with session_factory() as session:
            res = await session.execute(
                select(Camera.is_enabled, Camera.is_detection_enabled, Camera.is_notification_enabled)
                .where(Camera.camera_uuid == camera_uuid)
            )
            row = res.first()
            if row:
                enabled, det_enabled, notif_enabled = row
                # If camera itself is disabled, treat detection/notification as disabled
                mode = CameraMode(
                    detection_enabled=bool(enabled and det_enabled),
                    notification_enabled=bool(enabled and notif_enabled)
                )

    await svc.handle_detection_event(
        ev,
        camera_mode=mode,
        frame_w=payload.frame_w,
        frame_h=payload.frame_h,
    )    
    return {"ok": True}
