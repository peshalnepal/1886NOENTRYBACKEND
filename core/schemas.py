from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -------------------------
# ROI Schema
# -------------------------
class ROISchema(BaseModel):
    """Region of Interest polygon for detection alerts."""
    model_config = ConfigDict(extra="forbid")
    
    points: List[List[float]] = Field(..., description="List of [x, y] polygon vertices")
    normalized: bool = Field(default=True, description="If true, points are 0-1 normalized; if false, pixel coordinates")
    frame_w: Optional[int] = Field(default=None, gt=0, description="Frame width used when ROI was authored")
    frame_h: Optional[int] = Field(default=None, gt=0, description="Frame height used when ROI was authored")


# -------------------------
# Shared / base shapes
# -------------------------

class CameraBaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    site_uuid: uuid.UUID = Field(..., description="Site that owns this camera.")
    device_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Assigned device UUID (Jetson). Required on create; optional on edit if you allow moving devices."
    )

    rtsp_url: str = Field(..., min_length=1, description="RTSP URL for the camera.")
    name: Optional[str] = None
    location: Optional[str] = None
    is_enabled: bool = True
    is_detection_enabled: bool = True
    is_notification_enabled: bool = True
    sample_fps: Optional[float] = Field(default=5.0, ge=0.1)
    decode_backend: Optional[Literal["gstreamer", "opencv"]] = "gstreamer"
    resize: Optional[Tuple[int, int]] = None
    reconnect_base_ms: Optional[int] = Field(default=1000, ge=100)
    reconnect_max_ms: Optional[int] = Field(default=8000, ge=1000)
    poll_interval_ms: Optional[int] = Field(default=500, ge=10)
    request_timeout_s: Optional[float] = Field(default=3.0, ge=0.1)
    detection_path_template: Optional[str] = Field(default="/detection/{camera_uuid}")

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("rtsp_url", "name", "location", "detection_path_template"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        if not self.rtsp_url:
            raise ValueError("rtsp_url must not be empty.")
        return self


class CameraCreateSchema(BaseModel):
    """
    Payload for POST /cameras

    Important:
    - webrtc_url is NOT accepted here (server provisions it)
    - device_uuid is required because we must push config to Jetson
    """
    model_config = ConfigDict(extra="forbid")

    user_id: Optional[int] = Field(default=None, ge=1, description="Optional. Server resolves user from JWT.")

    site_uuid: uuid.UUID
    device_uuid: uuid.UUID
    rtsp_url: str = Field(..., min_length=1)
    device_url: str = Field(..., min_length=1)
    name: Optional[str] = None
    location: Optional[str] = None

    is_enabled: bool = True
    is_detection_enabled: bool = True
    is_notification_enabled: bool = True

    # ROI for detection alerts (optional)
    roi: Optional[ROISchema] = None

    sample_fps: float = Field(default=5.0, ge=0.1)
    decode_backend: Literal["gstreamer", "opencv"] = "gstreamer"
    resize: Optional[Tuple[int, int]] = None
    reconnect_base_ms: int = Field(default=1000, ge=100)
    reconnect_max_ms: int = Field(default=8000, ge=1000)
    poll_interval_ms: int = Field(default=500, ge=10)
    request_timeout_s: float = Field(default=3.0, ge=0.1)
    detection_path_template: Optional[str] = Field(default="/detection/{camera_uuid}")

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("rtsp_url", "name", "location", "detection_path_template"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        if not self.rtsp_url:
            raise ValueError("rtsp_url is required.")
        return self


# -------------------------
# Edit/Patch
# -------------------------

class CameraEditSchema(BaseModel):
    """
    Payload for PATCH /cameras/{camera_uuid}

    Rules:
    - All fields optional: patch semantics
    - Explicitly forbids webrtc_url (immutable/server-managed)
    - If you also want to forbid device_url from client, keep it out entirely.
    """
    model_config = ConfigDict(extra="forbid")

    # Allowed patches
    # Backward compatibility: older clients may still send user_id on PATCH.
    # It is ignored by edit flow.
    user_id: Optional[int] = Field(default=None, ge=1)
    site_uuid: Optional[uuid.UUID] = None
    device_uuid: Optional[uuid.UUID] = None  # allow re-assign device if you want

    rtsp_url: Optional[str] = Field(default=None, min_length=1)
    device_url: Optional[str] = Field(default=None, min_length=1)
    webrtc_url: Optional[str] = Field(default=None, min_length=1)
    name: Optional[str] = None
    location: Optional[str] = None

    is_enabled: Optional[bool] = None
    is_detection_enabled: Optional[bool] = None
    is_notification_enabled: Optional[bool] = None

    # ROI for detection alerts (optional)
    roi: Optional[ROISchema] = None

    sample_fps: Optional[float] = Field(default=None, ge=0.1)
    decode_backend: Optional[Literal["gstreamer", "opencv"]] = None
    resize: Optional[Tuple[int, int]] = None
    reconnect_base_ms: Optional[int] = Field(default=None, ge=100)
    reconnect_max_ms: Optional[int] = Field(default=None, ge=1000)
    poll_interval_ms: Optional[int] = Field(default=None, ge=10)
    request_timeout_s: Optional[float] = Field(default=None, ge=0.1)
    detection_path_template: Optional[str] = None

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("rtsp_url", "name", "location", "detection_path_template"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        return self


# -------------------------
# DB output models
# -------------------------

class CameraSchema(BaseModel):
    """
    Response model for GET /cameras
    Mirrors what you build in list_cameras().
    """
    model_config = ConfigDict(extra="forbid")

    camera_uuid: uuid.UUID
    camera_code: str

    site_uuid: uuid.UUID
    device_uuid: Optional[uuid.UUID] = None

    rtsp_url: str
    webrtc_url: Optional[str] = None

    is_enabled: bool
    is_detection_enabled: bool
    is_notification_enabled: bool

    roi: Optional[Dict[str, Any]] = None

    created_at: datetime
    updated_at: datetime


class CameraWithConfigSchema(CameraSchema):
    """
    Response model for GET /cameras/{camera_uuid} and POST/PATCH responses.
    """
    configuration: Dict[str, Any] = Field(default_factory=dict)
    timezone: Optional[str] = None

class CameraPlaybackSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_uuid: str
    webrtc_url: str
