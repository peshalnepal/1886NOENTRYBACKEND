from __future__ import annotations

from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Tuple, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from application.channels.channel_config import VideoChannelConfig


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


SCHEDULE_TIME_PATTERN = r"^\d{2}:\d{2}(:\d{2})?$"


def _normalize_schedule_fields(model: BaseModel) -> BaseModel:
    for attr in ("timezone", "start_time", "end_time"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            cleaned = value.strip()
            setattr(model, attr, cleaned or None)

    raw_schedule = getattr(model, "schedule", None)
    if raw_schedule is not None:
        normalized_schedule = VideoChannelConfig.normalize_schedule(raw_schedule)
        if raw_schedule and not normalized_schedule:
            raise ValueError("schedule must contain at least one valid day/time window.")
        setattr(model, "schedule", normalized_schedule)
        if normalized_schedule:
            return model

    raw_days = getattr(model, "day_of_week", None)
    if raw_days is not None:
        normalized_days: List[int] = []
        seen_days = set()
        for value in raw_days:
            try:
                day = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("day_of_week values must be integers from 0 to 6.") from exc
            if day < 0 or day > 6:
                raise ValueError("day_of_week values must be between 0 and 6.")
            if day in seen_days:
                continue
            seen_days.add(day)
            normalized_days.append(day)
        setattr(model, "day_of_week", normalized_days)

    day_of_week = getattr(model, "day_of_week", None)
    start_time = getattr(model, "start_time", None)
    end_time = getattr(model, "end_time", None)

    if any(value is not None for value in (day_of_week, start_time, end_time)):
        if not day_of_week:
            raise ValueError("Select at least one day when providing a schedule.")
        if not start_time or not end_time:
            raise ValueError("start_time and end_time are required when providing a schedule.")
        try:
            start_obj = dt_time.fromisoformat(str(start_time))
            end_obj = dt_time.fromisoformat(str(end_time))
        except ValueError as exc:
            raise ValueError("start_time and end_time must use HH:MM or HH:MM:SS format.") from exc
        if start_obj == end_obj:
            raise ValueError("start_time and end_time must be different.")

    return model


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
    timezone: Optional[str] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("rtsp_url", "name", "location", "detection_path_template"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        if not self.rtsp_url:
            raise ValueError("rtsp_url must not be empty.")
        return _normalize_schedule_fields(self)


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
    timezone: Optional[str] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("rtsp_url", "name", "location", "detection_path_template", "timezone"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        if not self.rtsp_url:
            raise ValueError("rtsp_url is required.")
        return _normalize_schedule_fields(self)


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
    timezone: Optional[str] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("rtsp_url", "name", "location", "detection_path_template", "timezone"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        return _normalize_schedule_fields(self)


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
    name: Optional[str] = None
    location: Optional[str] = None

    site_uuid: uuid.UUID
    device_uuid: Optional[uuid.UUID] = None

    rtsp_url: str
    webrtc_url: Optional[str] = None

    is_enabled: bool
    is_detection_enabled: bool
    is_notification_enabled: bool
    use_site_schedule: bool = True

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
