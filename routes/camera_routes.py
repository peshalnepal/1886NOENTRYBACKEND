import asyncio
import time
import uuid
from typing import AsyncGenerator, Optional, Tuple, Literal
import logging
import cv2

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from dependencies import get_manager
from application.channels.channel_config import VideoChannelConfig
from application.services.manager import Manager
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent
from utils import render_frame_with_overlays

logger = logging.getLogger(__name__)

# -----------------------
# DTOs
# -----------------------

class CameraUpdateRequest(BaseModel):
    rtsp_url: Optional[str] = None
    enabled: Optional[bool] = None
    detection_enabled: Optional[bool] = None
    notification_enabled: Optional[bool] = None

    sample_fps: Optional[float] = None
    decode_backend: Optional[str] = None  # "gstreamer" | "opencv"
    resize: Optional[Tuple[int, int]] = None
    reconnect_base_ms: Optional[int] = None
    reconnect_max_ms: Optional[int] = None
    emit_format: Optional[str] = None  # "raw" | "jpeg"
    jpeg_quality: Optional[int] = None


class CameraCreateRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL")
    enabled: bool = True
    detection_enabled: bool = True
    notification_enabled: bool = True

    sample_fps: Optional[float] = None
    decode_backend: Optional[Literal["gstreamer", "opencv"]] = None
    resize: Optional[Tuple[int, int]] = None
    emit_format: Optional[Literal["raw", "jpeg"]] = None
    jpeg_quality: Optional[int] = None


router = APIRouter()


# -----------------------
# Utils
# -----------------------

def _make_mjpeg_part(jpeg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
        jpeg_bytes + b"\r\n"
    )


def _event_to_jpeg(ev) -> Optional[bytes]:
    enc = getattr(ev, "encoded", None)
    fmt = getattr(ev, "format", None)
    if enc is not None and fmt == "jpeg":
        return enc

    frame = getattr(ev, "frame", None)
    if frame is None:
        return None

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buf.tobytes()


async def get_current_user_id() -> int:
    # replace with real auth
    return 1


def _auto_defaults(num_active: int) -> dict:
    if num_active <= 2:
        return dict(sample_fps=10.0, resize=(640, 360), emit_format="jpeg", jpeg_quality=65, decode_backend="gstreamer")
    if num_active <= 4:
        return dict(sample_fps=7.0, resize=(640, 360), emit_format="jpeg", jpeg_quality=60, decode_backend="gstreamer")
    return dict(sample_fps=5.0, resize=(640, 360), emit_format="jpeg", jpeg_quality=55, decode_backend="gstreamer")


def _choose(v, fallback):
    return fallback if v is None else v


# -----------------------
# Routes
# -----------------------
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent

@router.post("/cameras", response_class=JSONResponse)
async def add_camera(
    payload: CameraCreateRequest,
    pipeline_id: Optional[uuid.UUID] = Query(default=None),
    manager: Manager = Depends(get_manager),
    user_id: int = Depends(get_current_user_id),
):
    # resolve pid and active count safely
    active = await manager.get_activepipeline()
    if pipeline_id is None:
        if active is None:
            raise HTTPException(status_code=409, detail="No active pipeline. Provide pipeline_id.")
        pid = active.pipeline_id
        num_active = len(active.list_channel_ids()) if hasattr(active, "list_channel_ids") else 0
    else:
        pid = pipeline_id
        # if not active pipeline, num_active=0
        num_active = len(active.list_channel_ids()) if (active and hasattr(active, "list_channel_ids")) else 0

    auto = _auto_defaults(num_active)

    cfg = {
        "rtsp_url": payload.rtsp_url,
        "enabled": payload.enabled,
        "detection_enabled": payload.detection_enabled,
        "notification_enabled": payload.notification_enabled,
        "sample_fps": payload.sample_fps or auto["sample_fps"],
        "decode_backend": payload.decode_backend or auto["decode_backend"],
        "resize": payload.resize or auto["resize"],
        "emit_format": payload.emit_format or auto["emit_format"],
        "jpeg_quality": payload.jpeg_quality or auto["jpeg_quality"],
    }

    ev = ChannelCreateEvent(configs=cfg)

    result = await manager.update_pipeline(
        pipeline_id=pid,
        channel_events=[ev],
        user_id=user_id,
        camera_code_prefix="cam",
    )

    if result is None or not result.cameras:
        raise HTTPException(status_code=404, detail="Pipeline not found or camera create failed.")

    cam = result.cameras[0]
    return {
        "pipeline_id": str(result.pipeline_id),
        "active_in_memory": bool(result.active_in_memory),
        "camera_uuid": str(cam.camera_uuid),
        "channel_id": cam.channel_id,
        "rtsp_url": cam.rtsp_url,
        "enabled": bool(cam.enabled),
        "detection_enabled": bool(cam.detection_enabled),
        "notification_enabled": bool(cam.notification_enabled),
        "config": {
            "sample_fps": cam.sample_fps,
            "decode_backend": cam.decode_backend,
            "resize": cam.resize,
            "emit_format": cam.emit_format,
            "jpeg_quality": cam.jpeg_quality,
        },
    }
@router.put("/cameras/{camera_uuid}", response_class=JSONResponse)
async def edit_camera(
    camera_uuid: uuid.UUID,
    payload: CameraUpdateRequest,
    pipeline_id: uuid.UUID = Query(...),
    manager: Manager = Depends(get_manager),
    user_id: int = Depends(get_current_user_id),
):
    patch = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update.")

    ev = ChannelEditEvent(channel_id=camera_uuid, configs=patch)

    try:
        result = await manager.update_pipeline(
            pipeline_id=pipeline_id,
            channel_events=[ev],
            user_id=user_id,
            camera_code_prefix="cam",
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if result is None or not result.cameras:
        raise HTTPException(status_code=404, detail="Pipeline/camera not found")

    cam = result.cameras[0]
    return {
        "pipeline_id": str(result.pipeline_id),
        "active_in_memory": bool(result.active_in_memory),
        "camera_uuid": str(cam.camera_uuid),
        "channel_id": cam.channel_id,
        "rtsp_url": cam.rtsp_url,
        "enabled": bool(cam.enabled),
        "detection_enabled": bool(cam.detection_enabled),
        "notification_enabled": bool(cam.notification_enabled),
    }
@router.delete("/cameras/{camera_uuid}", response_class=JSONResponse)
async def delete_camera(
    camera_uuid: uuid.UUID,
    pipeline_id: Optional[uuid.UUID] = Query(default=None),
    manager: Manager = Depends(get_manager),
):
    if pipeline_id is None:
        active = await manager.get_activepipeline()
        pipeline_id = getattr(active, "pipeline_id", None) if active else None
        if pipeline_id is None:
            raise HTTPException(status_code=409, detail="No active pipeline. Provide pipeline_id.")

    ev = ChannelRemoveEvent(channel_id=camera_uuid)

    result = await manager.update_pipeline(
        pipeline_id=pipeline_id,
        channel_events=[ev],
        user_id=None,
        camera_code_prefix="cam",
    )

    return {
        "deleted": True,
        "camera_uuid": str(camera_uuid),
        "channel_id": str(camera_uuid),
        "pipeline_id": str(pipeline_id),
        "active_in_memory": bool(result.active_in_memory) if result else False,
    }


@router.get("/cameras/{camera_uuid}/snapshot.jpg")
async def snapshot_jpg(
    pipeline_id: uuid.UUID,
    camera_uuid: uuid.UUID,
    request: Request,
    timeout_s: float = Query(2.0, ge=0.1, le=10.0),
    manager: Manager = Depends(get_manager),
):
    pipeline = await manager.get_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    async def _get_one():
        agen = pipeline.stream(str(camera_uuid))
        try:
            return await agen.__anext__()
        finally:
            await agen.aclose()

    try:
        ev = await asyncio.wait_for(_get_one(), timeout=timeout_s)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="No frame received (timeout)")

    jpeg = _event_to_jpeg(ev)
    if jpeg is None:
        raise HTTPException(status_code=502, detail="Failed to encode frame")

    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/cameras/{camera_uuid}/stream.mjpg")
async def stream_mjpeg(
    camera_uuid: str,
    manager: Manager = Depends(get_manager),
    fps: float = Query(5.0, ge=0.1, le=60.0),
    overlay: bool = Query(True),
):
    pipeline = await manager.get_activepipeline()
    if pipeline is None:
        raise HTTPException(status_code=409, detail="No active pipeline running")

    if hasattr(pipeline, "list_channel_ids"):
        if str(camera_uuid) not in set(pipeline.list_channel_ids()):
            raise HTTPException(status_code=404, detail="Camera not active in pipeline runtime")

    if overlay and fps > 3.0:
        fps = 3.0

    frame_interval = 1.0 / float(fps)
    last_sent = 0.0

    async def gen() -> AsyncGenerator[bytes, None]:
        nonlocal last_sent
        try:
            async for ev, det in pipeline.stream_with_detections(str(camera_uuid)):
                now = time.time()
                if last_sent and (now - last_sent) < frame_interval:
                    await asyncio.sleep(frame_interval - (now - last_sent))

                if overlay:
                    jpeg = render_frame_with_overlays(ev, det)
                else:
                    jpeg = _event_to_jpeg(ev)

                if jpeg is None:
                    continue

                last_sent = time.time()
                yield _make_mjpeg_part(jpeg)

        except asyncio.CancelledError:
            return

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )
