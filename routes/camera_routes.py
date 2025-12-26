
import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Optional, Tuple,Literal
import logging 
import cv2
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession
from dependencies import get_manager
from application.channels.channel import VideoChannel
from application.channels.channel_config import VideoChannelConfig
from application.repositories.channel_repository import ChannelRepository
from application.repositories.pipeline_repository import PipelineRepository
from application.services.manager import Manager

logger = logging.getLogger(__name__)

async def get_db() -> AsyncSession:
    """
    Replace with your own AsyncSession dependency.
    Example:
      async with async_session() as db: yield db
    """
    raise NotImplementedError



# -----------------------
# DTOs
# -----------------------

class CameraCreateRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL")
    enabled: bool = Field(default=True)


class CameraUpdateRequest(BaseModel):
    # camera fields (what frontend might edit)
    rtsp_url: Optional[str] = None
    enabled: Optional[bool] = None

    # optional overrides to config knobs (keep optional)
    sample_fps: Optional[float] = None
    decode_backend: Optional[str] = None  # "gstreamer" | "opencv"
    resize: Optional[Tuple[int, int]] = None
    reconnect_base_ms: Optional[int] = None
    reconnect_max_ms: Optional[int] = None
    emit_format: Optional[str] = None  # "raw" | "jpeg"
    jpeg_quality: Optional[int] = None


# -----------------------
# Router
# -----------------------

router = APIRouter()


def _make_mjpeg_part(jpeg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
        jpeg_bytes + b"\r\n"
    )


def _event_to_jpeg(ev) -> Optional[bytes]:
    """
    RTSPEvent -> jpeg bytes.
    - if ev.encoded exists (emit_format=jpeg), use it
    - else encode raw frame with cv2.imencode
    """
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
    return 1


class CameraCreateRequest(BaseModel):
    rtsp_url: str
    enabled: bool = True

    # optional overrides (client can omit all)
    sample_fps: float = 5.0
    decode_backend: Literal["gstreamer", "opencv"] = "gstreamer"
    resize: Optional[Tuple[int, int]] = None
    emit_format: Literal["raw", "jpeg"] = "raw"
    jpeg_quality: int = 80


@router.post("/cameras", response_class=JSONResponse)
async def add_camera(
    payload: CameraCreateRequest,
    pipeline_id: Optional[uuid.UUID] = Query(default=None),
    manager: Manager = Depends(get_manager),
    user_id: int = Depends(get_current_user_id),
):
    """
    Frontend sends: rtsp_url (+enabled optional).

    Server:
      - Upserts camera + channel config in DB
      - Hot-adds channel if pipeline is active
      - Returns camera_uuid, camera_code, rtsp_url, enabled, pipeline_id, channel_id, config defaults
    """

    # 1) Resolve pipeline id: query param wins, else active pipeline
    pid: Optional[uuid.UUID] = pipeline_id
    if pid is None:
        active = await manager.get_activepipeline()
        if active is None:
            raise HTTPException(status_code=409, detail="No active pipeline. Provide pipeline_id.")
        pid = active.pipeline_id

    # 2) Build API input config (defaults are in model)
    cfg_in = VideoChannelConfig(
        rtsp_url=payload.rtsp_url,
        enabled=payload.enabled,
        sample_fps=payload.sample_fps,
        decode_backend=payload.decode_backend,
        resize=payload.resize,
        emit_format=payload.emit_format,
        jpeg_quality=payload.jpeg_quality,
    )

    result= await manager.update_pipeline(
        pipeline_id=pid,
        channel_configs=[cfg_in],  # must be a list
        user_id=user_id,
        camera_code_prefix="cam",
        replace_runtime=False,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="Pipeline not found or invalid pipeline_id.")

    if not result.cameras:
        raise HTTPException(status_code=500, detail="Camera upsert succeeded but no camera returned.")

    cam = result.cameras[0]  # we submitted one config => one camera in same order

    # 4) Return required info (+ config applied)
    return {
        "pipeline_id": str(result.pipeline_id),
        "active_in_memory": bool(result.active_in_memory),
        "camera_uuid": str(cam.camera_uuid),
        "channel_id": cam.channel_id,          # recommended: str(camera_uuid)
        "rtsp_url": cam.rtsp_url,
        "enabled": bool(cam.enabled),

        # helpful for frontend/debug (optional)
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
    pipeline_id: uuid.UUID,
    camera_uuid: uuid.UUID,
    payload: CameraUpdateRequest,
    db: AsyncSession = Depends(get_db),
    manager:Manager=Depends(get_manager)
):
    """
    Edit camera:
      - updates Camera table (rtsp_url / enabled)
      - updates ChannelConfiguration (optional knobs)
      - ensures membership to pipeline_id
    If pipeline is active, swaps the runtime channel (remove + add with updated config).
    """
    pipe_repo = PipelineRepository()
    chan_repo = ChannelRepository()

    if not await pipe_repo.pipeline_exists(db, pipeline_id):
        raise HTTPException(status_code=404, detail="Pipeline not found")

    full = await chan_repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam, chan_cfg, _existing_pid = full

    # Reconstruct config from DB (configuration JSON + required camera fields)
    data: dict[str, Any] = {}
    if chan_cfg and getattr(chan_cfg, "configuration", None):
        data.update(chan_cfg.configuration)

    # must exist
    data["camera_uuid"] = cam.camera_uuid
    data["rtsp_url"] = payload.rtsp_url if payload.rtsp_url is not None else cam.rtsp_url
    data["enabled"] = payload.enabled if payload.enabled is not None else bool(cam.is_enabled)

    # keep channel_id stable (needed for runtime remove)
    data.setdefault("channel_id", getattr(cam, "camera_code", None) or str(cam.camera_uuid))

    # overlay optional tuning if provided
    for k in (
        "sample_fps",
        "decode_backend",
        "resize",
        "reconnect_base_ms",
        "reconnect_max_ms",
        "emit_format",
        "jpeg_quality",
    ):
        v = getattr(payload, k)
        if v is not None:
            data[k] = v

    # validate
    try:
        cfg = VideoChannelConfig(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid channel config: {e}")

    await chan_repo.upsert_camera_from_channel_config(
        db,
        pipeline_id=pipeline_id,
        channel_config=cfg,  # update flow (camera_uuid present)
    )
    await db.commit()

    # runtime swap if active
    if getattr(manager, "_active_id", None) == pipeline_id and getattr(manager, "_active_pipeline", None) is not None:
        active = manager._active_pipeline
        await active.start()
        await active.remove_channel(cfg.channel_id)
        await active.add_channel(VideoChannel(config=cfg))

    return {
        "camera_uuid": str(cam.camera_uuid),
        "rtsp_url": data["rtsp_url"],
        "enabled": bool(data["enabled"]),
        "pipeline_id": str(pipeline_id),
        "channel_id": cfg.channel_id,
    }

@router.delete("/cameras/{camera_uuid}", response_class=JSONResponse)
async def delete_camera(
    camera_uuid: uuid.UUID,
    manager: Manager = Depends(get_manager),
):
    # if pipeline_id not provided, use active 
    logger.info("#####################################333")
    logger.info(camera_uuid)

    pipeline_id=None
    if pipeline_id is None:
        active = await manager.get_activepipeline()
        pipeline_id = getattr(active, "id", None) or getattr(active, "pipeline_id", None)
        if pipeline_id is None:
            raise HTTPException(status_code=409, detail="No active pipeline. Provide pipeline_id.")
    logger.info(pipeline_id)

    ok = await manager.remove_camera_from_pipeline(pipeline_id=pipeline_id, camera_uuid=camera_uuid)
    logger.info("#####################################333")
    return {
        "deleted": bool(ok),
        "camera_uuid": str(camera_uuid),
        "channel_id": str(camera_uuid),  # ✅ runtime channel id
        "pipeline_id": str(pipeline_id),
    }
    
@router.get("/cameras/{camera_uuid}/snapshot.jpg")
async def snapshot_jpg(
    pipeline_id: uuid.UUID,
    camera_uuid: uuid.UUID,
    request: Request,
    timeout_s: float = Query(2.0, ge=0.1, le=10.0),
    manager:Manager=Depends(get_manager)
):
    """
    One-shot snapshot: waits for the next frame from ModelPipeline.stream().
    """

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
    fps: float = Query(10.0, ge=0.1, le=60.0),
):
    pipeline = await manager.get_activepipeline()
    if pipeline is None:
        raise HTTPException(status_code=409, detail="No active pipeline running")

    # optional: validate camera exists in runtime
    if hasattr(pipeline, "list_channel_ids"):
        if str(camera_uuid) not in set(pipeline.list_channel_ids()):
            raise HTTPException(status_code=404, detail="Camera not active in pipeline runtime")

    frame_interval = 1.0 / float(fps)
    last_sent = 0.0

    async def gen() -> AsyncGenerator[bytes, None]:
        nonlocal last_sent
        try:
            async for ev in pipeline.stream(str(camera_uuid)):
                now = time.time()

                # throttle output to requested fps
                dt = now - last_sent
                if last_sent and dt < frame_interval:
                    await asyncio.sleep(frame_interval - dt)

                jpeg = _event_to_jpeg(ev)
                if jpeg is None:
                    continue

                last_sent = time.time()
                yield _make_mjpeg_part(jpeg)

        except asyncio.CancelledError:
            # client disconnected
            return

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )
