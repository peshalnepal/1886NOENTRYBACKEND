# agents/api/routes/camera_routes.py

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db, get_manager
from application.repositories.channel_repository import ChannelRepository
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent

# NOTE: Schema imports depend on your project structure; keep your existing ones.
from core.schemas import CameraSchema, CameraCreateSchema, CameraEditSchema, CameraWithConfigSchema  # type: ignore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("/", response_model=List[CameraSchema])
async def list_cameras(
    db: AsyncSession = Depends(get_db),
):
    """
    List cameras from Azure DB.

    IMPORTANT:
      - Do NOT return MJPEG stream URLs from this service.
      - Use camera.webrtc_url for playback in the frontend.
    """
    repo = ChannelRepository()
    cams = await repo.list_cameras(db)  # keep your existing repo method
    out: List[CameraSchema] = []
    for cam in cams:
        out.append(
            CameraSchema(
                camera_uuid=cam.camera_uuid,
                camera_code=cam.camera_code,
                site_uuid=cam.site_uuid,
                rtsp_url=cam.rtsp_url,
                webrtc_url=getattr(cam, "webrtc_url", None),
                is_enabled=cam.is_enabled,
                is_detection_enabled=cam.is_detection_enabled,
                is_notification_enabled=cam.is_notification_enabled,
                created_at=cam.created_at,
                updated_at=cam.updated_at,
            )
        )
    return out


@router.get("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def get_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, cfg, _pid = full
    return CameraWithConfigSchema(
        camera_uuid=cam.camera_uuid,
        camera_code=cam.camera_code,
        site_uuid=cam.site_uuid,
        rtsp_url=cam.rtsp_url,
        webrtc_url=getattr(cam, "webrtc_url", None),
        is_enabled=cam.is_enabled,
        is_detection_enabled=cam.is_detection_enabled,
        is_notification_enabled=cam.is_notification_enabled,
        configuration=(cfg.configuration if cfg else {}),
        timezone=(cfg.timezone if cfg else None),
        created_at=cam.created_at,
        updated_at=cam.updated_at,
    )


@router.get("/{camera_uuid}/playback")
async def get_camera_playback(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the WebRTC playback URL for this camera.
    """
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    if not getattr(cam, "webrtc_url", None):
        raise HTTPException(status_code=409, detail="WebRTC URL not provisioned yet")
    return {"camera_uuid": str(cam.camera_uuid), "webrtc_url": cam.webrtc_url}


@router.post("/", response_model=CameraWithConfigSchema)
async def create_camera(
    payload: CameraCreateSchema,
    db: AsyncSession = Depends(get_db),
    manager=Depends(get_manager),
):
    """
    Create camera:
      1) Provision WebRTC (RTSP->WebRTC) on Azure gateway
      2) Persist camera + config in Azure DB
      3) Push camera config to Jetson TensorRT service (device assignment required)
    """
    pipeline = await manager.get_activepipeline()
    ev = ChannelCreateEvent(
        channel_id=None,
        configs=payload.model_dump(exclude_none=True),
        created_at=datetime.now(timezone.utc),
    )

    result = await manager.update_pipeline(
        pipeline.pipeline_id,
        [ev],
        user_id=payload.user_id,
        camera_code_prefix="cam",
    )

    if not result or not result.cameras:
        raise HTTPException(status_code=500, detail="Failed to create camera")

    cam_out = result.cameras[0]
    return CameraWithConfigSchema(
        camera_uuid=cam_out.camera_uuid,
        camera_code=cam_out.camera_code,
        site_uuid=cam_out.site_uuid,
        rtsp_url=cam_out.rtsp_url,
        webrtc_url=cam_out.webrtc_url,
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        configuration={},  # your ChannelConfiguration JSON is in DB; fetch if you need it
        timezone=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@router.patch("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def edit_camera(
    camera_uuid: uuid.UUID,
    payload: CameraEditSchema,
    db: AsyncSession = Depends(get_db),
    manager=Depends(get_manager),
):
    """
    Edit camera:
      - updates Azure DB + pushes changes to Jetson
      - WebRTC URL is immutable (cannot be edited here)
      - if rtsp_url changes, gateway stream mapping is updated but playback URL stays the same
    """
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
        rtsp_url=cam_out.rtsp_url,
        webrtc_url=cam_out.webrtc_url,
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        configuration={},
        timezone=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@router.delete("/{camera_uuid}")
async def delete_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    manager=Depends(get_manager),
):
    """
    Remove camera:
      - removes from Jetson inference
      - removes WebRTC mapping (best effort)
      - deletes from Azure DB
    """
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
