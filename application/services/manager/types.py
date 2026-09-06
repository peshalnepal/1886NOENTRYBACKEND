"""DTOs and exceptions for the manager package."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class CameraOut(BaseModel):
    camera_uuid: uuid.UUID
    camera_code: Optional[str] = None
    name: Optional[str] = None
    location: Optional[str] = None
    site_uuid: uuid.UUID

    source_url: str
    webrtc_url: Optional[str] = None

    enabled: bool
    detection_enabled: bool
    notification_enabled: bool
    roi: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)
    timezone: Optional[str] = None
    notification_trigger_mode: str = "inherit"
    camera_playback_enabled: str = "inherit"
    use_site_schedule: Optional[bool] = None

    device_uuid: uuid.UUID
    device_url: str

    sample_fps: float = Field(default=5.0, ge=0.1)
    decode_backend: str = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = None
    emit_format: str = Field(default="raw")
    jpeg_quality: int = Field(default=80, ge=1, le=100)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_camera(
        cls,
        cam: Any,
        *,
        device_uuid: uuid.UUID,
        device_url: str,
        cfg_json: Dict[str, Any],
        capture_cfg: Dict[str, Any],
        timezone: Optional[str],
        use_site_schedule: Optional[bool],
    ) -> "CameraOut":
        """Project a persisted `Camera` plus its resolved config into the DTO.

        `capture_cfg` is the config blob the capture defaults are read from —
        the create path passes the request patch, the edit path the merged
        config — which is the only thing that differs between call sites.
        """
        return cls(
            camera_uuid=cam.camera_uuid,
            camera_code=getattr(cam, "camera_code", None),
            name=getattr(cam, "name", None),
            location=getattr(cam, "location", None),
            site_uuid=cam.site_uuid,
            source_url=cam.source_url,
            webrtc_url=cam.webrtc_url,
            enabled=bool(cam.is_enabled),
            detection_enabled=bool(cam.is_detection_enabled),
            notification_enabled=bool(cam.is_notification_enabled),
            device_uuid=device_uuid,
            device_url=device_url,
            sample_fps=float(capture_cfg.get("sample_fps", 5.0)),
            decode_backend=str(capture_cfg.get("decode_backend", "gstreamer")),
            resize=capture_cfg.get("resize"),
            emit_format=str(capture_cfg.get("emit_format", "raw")),
            jpeg_quality=int(capture_cfg.get("jpeg_quality", 80)),
            roi=cam.roi,
            configuration=cfg_json or {},
            timezone=timezone,
            notification_trigger_mode=str(
                getattr(cam, "notification_trigger_mode", "inherit") or "inherit"
            ),
            camera_playback_enabled=str(
                getattr(cam, "camera_playback_enabled", "inherit") or "inherit"
            ),
            use_site_schedule=use_site_schedule,
            created_at=getattr(cam, "created_at", None),
            updated_at=getattr(cam, "updated_at", None),
        )


class PipelineUpdateResult(BaseModel):
    pipeline_id: uuid.UUID
    active_in_memory: bool = False
    cameras: List[CameraOut] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class EdgeDeviceUnavailableError(RuntimeError):
    def __init__(self, device_url: str, cause: Exception):
        self.device_url = device_url
        self.cause = cause
        cause_name = type(cause).__name__
        cause_msg = str(cause).strip()
        detail = f"{cause_name}: {cause_msg}" if cause_msg else cause_name
        super().__init__(f"Edge device unreachable ({device_url}): {detail}")
