# agents/api/routes/camera_routes.py

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_async_db, get_manager
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent
from core.schemas import (
    CameraSchema,
    CameraCreateSchema,
    CameraEditSchema,
    CameraWithConfigSchema,
)  # type: ignore
from application.services.manager import Manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


# -------------------------
# Detection Schemas
# -------------------------
class BoxPx(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class BoxNorm(BaseModel):
    x: float  # 0..1 left
    y: float  # 0..1 top
    w: float  # 0..1 width
    h: float  # 0..1 height


class DetectionItemOut(BaseModel):
    box: BoxPx
    cls_name: str
    conf: float
    box_norm: Optional[BoxNorm] = None


class DetectionOut(BaseModel):
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int
    inference_ms: Optional[int] = None
    model_id: Optional[str] = None

    frame_w: Optional[int] = None
    frame_h: Optional[int] = None

    detections: List[DetectionItemOut] = Field(default_factory=list)
    pose: Optional[Any] = None


def _normalize_box_px(box: Dict[str, Any], frame_w: Optional[int], frame_h: Optional[int]) -> Optional[BoxNorm]:
    if not frame_w or not frame_h:
        return None
    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return BoxNorm(
        x=max(0.0, min(1.0, x1 / frame_w)),
        y=max(0.0, min(1.0, y1 / frame_h)),
        w=max(0.0, min(1.0, w / frame_w)),
        h=max(0.0, min(1.0, h / frame_h)),
    )


def _resp_to_detection_out(resp: Any, *, normalize: bool) -> DetectionOut:
    fw = getattr(resp, "frame_w", None)
    fh = getattr(resp, "frame_h", None)

    items: List[DetectionItemOut] = []
    for d in (list(resp.detections) if getattr(resp, "detections", None) else []):
        box = d.get("box") or {}
        if not all(k in box for k in ("x1", "y1", "x2", "y2")):
            continue

        box_px = BoxPx(x1=box["x1"], y1=box["y1"], x2=box["x2"], y2=box["y2"])
        items.append(
            DetectionItemOut(
                box=box_px,
                cls_name=str(d.get("cls_name") or ""),
                conf=float(d.get("conf") or 0.0),
                box_norm=_normalize_box_px(box, fw, fh) if normalize else None,
            )
        )

    return DetectionOut(
        camera_uuid=str(resp.camera_uuid),
        frame_ts_ms=int(resp.frame_ts_ms),
        frame_seq=int(resp.frame_seq),
        inference_ms=int(resp.inference_ms) if getattr(resp, "inference_ms", None) is not None else None,
        model_id=str(getattr(resp, "model_id", None)) if getattr(resp, "model_id", None) is not None else None,
        frame_w=int(fw) if fw is not None else None,
        frame_h=int(fh) if fh is not None else None,
        detections=items,
        pose=getattr(resp, "pose", None),
    )


# -------------------------
# Cameras
# -------------------------
@router.get("", response_model=List[CameraSchema])
async def list_cameras(
    db: AsyncSession = Depends(get_async_db),
    site_uuid: uuid.UUID = None,  # keep query param; if None -> 422 below
):
    """
    List cameras from Azure DB for a site.
    Uses camera.webrtc_url for playback in the frontend.
    """
    if site_uuid is None:
        raise HTTPException(status_code=422, detail="site_uuid query param is required")

    repo = ChannelRepository()
    site_repo=SiteRepository()
    cams = await site_repo.list_cameras_by_site(db, site_uuid)

    out: List[CameraSchema] = []
    for cam in cams:
        # New rule: exactly 1 device per camera, but for list we won't hard-fail old data
        dev = await repo.get_device(db, camera_uuid=cam.camera_uuid, required=False)

        out.append(
            CameraSchema(
                camera_uuid=cam.camera_uuid,
                camera_code=cam.camera_code,
                site_uuid=cam.site_uuid,
                device_uuid=(dev.device_uuid if dev else None),
                rtsp_url=cam.rtsp_url,
                webrtc_url=getattr(cam, "webrtc_url", None),
                is_enabled=cam.is_enabled,
                is_detection_enabled=cam.is_detection_enabled,
                is_notification_enabled=cam.is_notification_enabled,
                roi=getattr(cam, "roi", None),
                created_at=cam.created_at,
                updated_at=cam.updated_at,
            )
        )
    return out


@router.get("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def get_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, cfg, _pid = full

    # New rule: must have exactly 1 device
    try:
        dev = await repo.get_device(db, camera_uuid=cam.camera_uuid, required=True)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return CameraWithConfigSchema(
        camera_uuid=cam.camera_uuid,
        camera_code=cam.camera_code,
        site_uuid=cam.site_uuid,
        device_uuid=dev.device_uuid,
        rtsp_url=cam.rtsp_url,
        webrtc_url=getattr(cam, "webrtc_url", None),
        is_enabled=cam.is_enabled,
        is_detection_enabled=cam.is_detection_enabled,
        is_notification_enabled=cam.is_notification_enabled,
        roi=getattr(cam, "roi", None),
        configuration=(cfg.configuration if cfg else {}),
        timezone=(cfg.timezone if cfg else None),
        created_at=cam.created_at,
        updated_at=cam.updated_at,
    )


@router.get("/{camera_uuid}/playback")
async def get_camera_playback(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
):
    """Returns the WebRTC playback URL for this camera."""
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    if not getattr(cam, "webrtc_url", None):
        raise HTTPException(status_code=409, detail="WebRTC URL not provisioned yet")

    return {"camera_uuid": str(cam.camera_uuid), "webrtc_url": cam.webrtc_url}


# -------------------------
# Detection endpoints
# -------------------------
@router.get("/{camera_uuid}/detections/latest", response_model=DetectionOut)
async def get_latest_detection(
    camera_uuid: uuid.UUID,
    refresh: bool = False,
    normalize: bool = False,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    pipeline = await manager.get_activepipeline()

    resp = await pipeline.get_latest_detection(str(camera_uuid))
    if resp is None and refresh:
        resp = await pipeline.refresh_detection_once(str(camera_uuid))

    if resp is None:
        raise HTTPException(status_code=404, detail="No detection available yet")

    return _resp_to_detection_out(resp, normalize=normalize)


@router.get("/{camera_uuid}/detections/stream")
async def stream_detections_sse(
    request: Request,
    camera_uuid: uuid.UUID,
    after_seq: int = 0,
    timeout_ms: int = 30000,
    normalize: bool = False,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    pipeline = await manager.get_activepipeline()
    cam_key = str(camera_uuid)

    async def gen():
        last = int(after_seq)

        initial = await pipeline.get_latest_detection(cam_key)
        if initial is not None and int(initial.frame_seq) > last:
            last = int(initial.frame_seq)
            payload = _resp_to_detection_out(initial, normalize=normalize).model_dump()
            yield f"event: detection\ndata: {json.dumps(payload)}\n\n"

        while True:
            if await request.is_disconnected():
                return

            resp = await pipeline.detect_store.wait_new(
                cam_key,
                after_seq=last,
                timeout_ms=int(timeout_ms),
            )

            if resp is None:
                yield "event: heartbeat\ndata: {}\n\n"
                continue

            last = int(resp.frame_seq)
            payload = _resp_to_detection_out(resp, normalize=normalize).model_dump()
            yield f"event: detection\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/detections/stream")
async def stream_all_detections_sse(
    request: Request,
    timeout_ms: int = 30000,
    normalize: bool = False,
    manager: Manager = Depends(get_manager),
):
    pipeline = await manager.get_activepipeline()
    hub = pipeline.detection_hub

    async def gen():
        q = await hub.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    resp = await asyncio.wait_for(q.get(), timeout=float(timeout_ms) / 1000.0)
                    payload = _resp_to_detection_out(resp, normalize=normalize).model_dump()
                    yield f"event: detection\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            await hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/detections/latest")
async def latest_detections_for_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
):
    repo = ChannelRepository()
    site_repo=SiteRepository()
    cams = await site_repo.list_cameras_by_site(db, site_uuid)

    pipeline = await manager.get_activepipeline()

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for cam in cams:
        cam_id = str(cam.camera_uuid)
        resp = await pipeline.get_latest_detection(cam_id)
        out[cam_id] = _resp_to_detection_out(resp, normalize=False).model_dump() if resp is not None else None

    return {"site_uuid": str(site_uuid), "latest": out}


# -------------------------
# Create / Edit / Delete
# -------------------------
@router.post("", response_model=CameraWithConfigSchema)
async def create_camera(
    payload: CameraCreateSchema,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
):
    try:
        data = payload.model_dump(exclude_none=True)

        # New rule: device_uuid is required
        if not data.get("device_uuid"):
            raise HTTPException(status_code=422, detail="device_uuid is required (each camera must have a device).")

        pipeline = await manager.get_activepipeline()

        ev = ChannelCreateEvent(
            channel_id=None,
            configs=data,
            created_at=datetime.now(timezone.utc),
        )

        result = await manager.update_pipeline(
            pipeline.pipeline_id,
            [ev],
            user_id=data.get("user_id"),
            camera_code_prefix="cam",
        )

        if not result or not result.cameras:
            raise HTTPException(status_code=500, detail="Operation failed to create camera record")

        cam_out = result.cameras[0]
        return CameraWithConfigSchema(
            camera_uuid=cam_out.camera_uuid,
            camera_code=cam_out.camera_code,
            site_uuid=cam_out.site_uuid,
            device_uuid=cam_out.device_uuid,
            rtsp_url=cam_out.rtsp_url,
            webrtc_url=cam_out.webrtc_url,
            is_enabled=cam_out.enabled,
            is_detection_enabled=cam_out.detection_enabled,
            is_notification_enabled=cam_out.notification_enabled,
            roi=cam_out.roi,
            configuration=cam_out.configuration,
            timezone=cam_out.timezone,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating camera")
        raise HTTPException(status_code=500, detail=f"Failed to create camera: {str(e)}")


@router.patch("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def edit_camera(
    camera_uuid: uuid.UUID,
    payload: CameraEditSchema,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
):
    pipeline = await manager.get_activepipeline()

    ev = ChannelEditEvent(
        channel_id=camera_uuid,
        configs=payload.model_dump(exclude_none=True),
        created_at=datetime.now(timezone.utc),
    )

    result = await manager.update_pipeline(pipeline.pipeline_id, [ev])
    if not result or not result.cameras:
        raise HTTPException(status_code=500, detail="Failed to edit camera")

    cam_out = result.cameras[0]
    return CameraWithConfigSchema(
        camera_uuid=cam_out.camera_uuid,
        camera_code=cam_out.camera_code,
        site_uuid=cam_out.site_uuid,
        device_uuid=cam_out.device_uuid,
        rtsp_url=cam_out.rtsp_url,
        webrtc_url=cam_out.webrtc_url,
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        roi=cam_out.roi,
        configuration=cam_out.configuration,
        timezone=cam_out.timezone,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@router.delete("/{camera_uuid}")
async def delete_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
):
    pipeline = await manager.get_activepipeline()

    ev = ChannelRemoveEvent(
        channel_id=camera_uuid,
        configs={},
        created_at=datetime.now(timezone.utc),
    )

    await manager.update_pipeline(pipeline.pipeline_id, [ev])
    return {"ok": True}


# --------------------------------------------------------------------
# Legacy endpoints (MJPEG/snapshot) intentionally disabled
# --------------------------------------------------------------------
@router.get("/{camera_uuid}/snapshot.jpg")
async def snapshot_jpg(camera_uuid: uuid.UUID):
    raise HTTPException(
        status_code=410,
        detail="Snapshot disabled on Azure service. Use WebRTC playback URL for video.",
    )


@router.get("/{camera_uuid}/stream.mjpg")
async def stream_mjpeg(camera_uuid: uuid.UUID):
    raise HTTPException(
        status_code=410,
        detail="MJPEG streaming disabled. Use WebRTC playback URL for video.",
    )
