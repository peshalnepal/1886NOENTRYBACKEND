"""DTOs and exceptions for the manager package.

Extracted verbatim from the former monolithic application/services/manager.py.
"""

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

    rtsp_url: str
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
