import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, status
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from application.services.notification import WebNotificationHub
from domain.events import DetectionsProducedEvent, DetectionItem, DetectionBox
from application.services.notification import  CameraMode

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
        camera_uuid=payload.camera_uuid,
        model_id="remote-jetson",
        frame_ts_ms=payload.frame_ts_ms,
        frame_seq=payload.frame_seq,
        detections=det_items
    )

    # For now, assume notifications are enabled for all incoming alerts
    # In a real app, you might fetch camera config from DB to check `notification_enabled`
    mode = CameraMode(detection_enabled=True, notification_enabled=True)

    await svc.handle_detection_event(
        ev,
        camera_mode=mode,
        frame_w=payload.frame_w,
        frame_h=payload.frame_h,
    )    
    return {"ok": True}
