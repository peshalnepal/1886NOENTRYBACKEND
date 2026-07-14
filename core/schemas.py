from __future__ import annotations

from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Optional, Tuple, Literal
import uuid

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)

from application.channels.channel_config import VideoChannelConfig
from core.source_url import is_supported_source_url


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

class CameraCreateSchema(BaseModel):
    """
    Payload for POST /cameras

    Important:
    - webrtc_url is NOT accepted here (server provisions it)
    - device_uuid may be omitted when the site has exactly one linked device;
      the backend will auto-link the camera to that device
    """
    model_config = ConfigDict(extra="forbid")

    user_id: Optional[int] = Field(default=None, ge=1, description="Optional. Server resolves user from JWT.")

    site_uuid: uuid.UUID
    device_uuid: Optional[uuid.UUID] = None
    # The single camera source URL: rtsp/rtsps/webrtc/whep/http/https/rtmp/rtmps/srt.
    source_url: str = Field(..., min_length=1)
    device_url: Optional[str] = Field(default=None, min_length=1)
    name: Optional[str] = None
    location: Optional[str] = None

    is_enabled: bool = True
    is_detection_enabled: bool = True
    is_notification_enabled: bool = True

    notification_trigger_mode: Literal["inherit", "roi_enter", "any_detection"] = Field(
        default="inherit",
        description="Per-camera notification trigger mode. 'inherit' = use site-level Trigger Condition.",
    )
    camera_playback_enabled: Literal["inherit", "always", "never"] = Field(
        default="inherit",
        description="Per-camera clip recording override. 'inherit' = use site default (prerecord list).",
    )

    # ROI for detection alerts (optional)
    roi: Optional[ROISchema] = None

    sample_fps: float = Field(default=5.0, ge=0.1)
    decode_backend: Literal["gstreamer", "opencv"] = "gstreamer"
    resize: Optional[Tuple[int, int]] = None
    reconnect_base_ms: int = Field(default=1000, ge=100)
    reconnect_max_ms: int = Field(default=8000, ge=1000)
    poll_interval_ms: int = Field(default=500, ge=10)
    request_timeout_s: float = Field(default=3.0, ge=0.1)
    detection_path_template: Optional[str] = Field(default="/api/cameras/{camera_uuid}/latest")
    timezone: Optional[str] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None

    @model_validator(mode="after")
    def _strip_blank_strings(self):
        for attr in ("source_url", "device_url", "name", "location", "detection_path_template", "timezone"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        if not self.source_url:
            raise ValueError("source_url is required.")
        if not is_supported_source_url(self.source_url):
            raise ValueError(
                "Unsupported camera source URL. Expected one of: "
                "rtsp/rtsps/webrtc/whep/http/https/rtmp/rtmps/srt."
            )
        self.source_url = self.source_url.strip()
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

    source_url: Optional[str] = Field(default=None, min_length=1)
    device_url: Optional[str] = Field(default=None, min_length=1)
    webrtc_url: Optional[str] = Field(default=None, min_length=1)
    name: Optional[str] = None
    location: Optional[str] = None

    is_enabled: Optional[bool] = None
    is_detection_enabled: Optional[bool] = None
    is_notification_enabled: Optional[bool] = None
    notification_trigger_mode: Optional[Literal["inherit", "roi_enter", "any_detection"]] = Field(
        default=None,
        description="Per-camera notification trigger mode. 'inherit' = use site-level Trigger Condition. Omit on PATCH to leave unchanged.",
    )
    camera_playback_enabled: Optional[Literal["inherit", "always", "never"]] = Field(
        default=None,
        description="Per-camera clip recording override. 'inherit' = use site default (prerecord list). Omit on PATCH to leave unchanged.",
    )

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
        for attr in ("source_url", "name", "location", "detection_path_template", "timezone"):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)
        # Patch semantics: only validate the source when the caller sent one.
        if self.source_url is not None:
            if not is_supported_source_url(self.source_url):
                raise ValueError(
                    "Unsupported camera source URL. Expected one of: "
                    "rtsp/rtsps/webrtc/whep/http/https/rtmp/rtmps/srt."
                )
            self.source_url = self.source_url.strip()
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

    source_url: str
    webrtc_url: Optional[str] = None

    is_enabled: bool
    is_detection_enabled: bool
    is_notification_enabled: bool
    use_site_schedule: bool = True
    notification_trigger_mode: str = "inherit"
    camera_playback_enabled: str = "inherit"

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


# -------------------------
# Site schedule helpers / constants
# -------------------------
SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER = "roi_enter"
SITE_PRERECORD_TRIGGER_MODE_ANY_DETECTION = "any_detection"
SITE_PRERECORD_TRIGGER_MODES = {
    SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER,
    SITE_PRERECORD_TRIGGER_MODE_ANY_DETECTION,
}
SUNDAY_TO_SATURDAY = [6, 0, 1, 2, 3, 4, 5]


# --- Site schemas ---
class SiteCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    site_code: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=255)
    timezone: Optional[str] = Field(default="UTC", max_length=50)
    # Required only when the caller is a platform admin acting cross-tenant
    # (no implicit org). Ignored for normal callers.
    org_id: Optional[int] = None


class SiteUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    site_code: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=255)
    timezone: Optional[str] = Field(default=None, max_length=50)


class SiteOut(BaseModel):
    site_uuid: uuid.UUID
    user_id: int
    name: str
    site_code: Optional[str] = None
    address: Optional[str] = None
    timezone: Optional[str] = "UTC"
    # Caller's effective role on this site: "admin" | "arm_disarm" | "read_only".
    # Org admins and platform admins always see "admin"; plain org members
    # get the role from their site-scoped access grant (or "read_only" as a
    # safe default).
    viewer_role: Optional[str] = None

    class Config:
        from_attributes = True


class LinkDeviceRequest(BaseModel):
    device_uuid: uuid.UUID


class SiteCameraCreate(BaseModel):
    device_uuid: uuid.UUID
    # The single camera source URL: rtsp/rtsps/webrtc/whep/http/https/rtmp/rtmps/srt.
    source_url: str = Field(..., min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    location: Optional[str] = Field(default=None, max_length=255)
    is_enabled: bool = True
    is_detection_enabled: bool = True
    is_notification_enabled: bool = True
    notification_trigger_mode: Literal["inherit", "roi_enter", "any_detection"] = Field(
        default="inherit",
        description="Per-camera notification trigger mode. 'inherit' = use site-level setting.",
    )
    camera_playback_enabled: Literal["inherit", "always", "never"] = Field(
        default="inherit",
        description="Per-camera clip recording override. 'inherit' = use site default (prerecord list).",
    )
    sample_fps: float = Field(default=5.0, ge=0.1)
    timezone: Optional[str] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None

    @model_validator(mode="after")
    def _validate_schedule(self):
        if not self.source_url:
            raise ValueError("source_url is required.")
        if not is_supported_source_url(self.source_url):
            raise ValueError(
                "Unsupported camera source URL. Expected one of: "
                "rtsp/rtsps/webrtc/whep/http/https/rtmp/rtmps/srt."
            )
        self.source_url = self.source_url.strip()
        return _normalize_schedule_fields(self)


class SiteMultiCameraPrerecordRule(BaseModel):
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteMultiCameraPrerecordRuleUpdate(BaseModel):
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteNotificationRule(BaseModel):
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteNotificationRuleUpdate(BaseModel):
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteScheduleRule(BaseModel):
    timezone: str = Field(default="UTC", max_length=50)
    day_of_week: List[int] = Field(default_factory=lambda: list(SUNDAY_TO_SATURDAY))
    start_time: str = Field(default="00:00:00", pattern=SCHEDULE_TIME_PATTERN)
    end_time: str = Field(default="23:59:59", pattern=SCHEDULE_TIME_PATTERN)
    schedule: List[Dict[str, Any]] = Field(default_factory=VideoChannelConfig.default_schedule)


class SiteScheduleRuleUpdate(BaseModel):
    timezone: Optional[str] = Field(default=None, max_length=50)
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def _validate_schedule(self):
        return _normalize_schedule_fields(self)


class SiteSettingsOut(BaseModel):
    site_uuid: uuid.UUID
    schedule: SiteScheduleRule = Field(default_factory=SiteScheduleRule)
    multi_camera_prerecord: SiteMultiCameraPrerecordRule = Field(
        default_factory=SiteMultiCameraPrerecordRule
    )
    notification: SiteNotificationRule = Field(default_factory=SiteNotificationRule)


class SiteSettingsUpdate(BaseModel):
    schedule: Optional[SiteScheduleRuleUpdate] = None
    multi_camera_prerecord: Optional[SiteMultiCameraPrerecordRuleUpdate] = None
    notification: Optional[SiteNotificationRuleUpdate] = None


# --- Device schemas ---
class EdgeCameraListOut(BaseModel):
    device_uuid: uuid.UUID
    device_url: str
    camera_uuids: List[str] = Field(default_factory=list)


class EdgeReconcileOut(BaseModel):
    device_uuid: uuid.UUID
    device_url: str

    to_add: List[str] = Field(default_factory=list)
    to_remove: List[str] = Field(default_factory=list)

    added: List[str] = Field(default_factory=list)
    removed: List[str] = Field(default_factory=list)

    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DeviceCreate(BaseModel):
    device_url: str = Field(..., min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    device_code: Optional[str] = Field(default=None, max_length=64)
    is_enabled: bool = True
    # Required only for platform admins in cross-tenant context (no implicit org).
    org_id: Optional[int] = None


class DeviceUpdate(BaseModel):
    device_url: Optional[str] = Field(default=None, min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    device_code: Optional[str] = Field(default=None, max_length=64)
    is_enabled: Optional[bool] = None


class DeviceOut(BaseModel):
    device_uuid: uuid.UUID
    user_id: int
    device_url: str
    name: Optional[str] = None
    device_code: Optional[str] = None
    is_enabled: bool

    class Config:
        from_attributes = True


# --- Report archive schemas ---
class ReportOut(BaseModel):
    """One archived report PDF (metadata only — download via its endpoint)."""

    id: int
    report_uuid: str
    org_id: int
    report_type: str
    filename: str
    generated_by_email: Optional[str] = None
    site_uuids: List[str] = Field(default_factory=list)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    alert_count: int = 0
    pdf_size: int = 0
    created_at: datetime


# --- Notification schemas ---
class NoteAttributes(BaseModel):
    """Structured, class-aware observations an operator records on an alert.

    Vehicle alerts (car/truck/motorcycle) use ``model`` / ``color`` /
    ``direction``; person alerts use ``gender`` / ``clothing_color`` /
    ``direction``. All optional — the frontend surfaces the relevant subset per
    detected class. ``object_type`` is "vehicle" | "person" | "other".
    """

    object_type: Optional[str] = None
    # Vehicle
    model: Optional[str] = None
    color: Optional[str] = None
    # Person
    gender: Optional[str] = None
    clothing_color: Optional[str] = None
    # Shared
    direction: Optional[str] = None


class NotificationNote(BaseModel):
    text: Optional[str] = None
    action: Optional[str] = None
    attributes: Optional[NoteAttributes] = None
    author_id: Optional[int] = None
    author_name: Optional[str] = None
    created_at: Optional[str] = None


class NotificationOut(BaseModel):
    id: int
    user_id: int
    site_uuid: str
    camera_uuid: Optional[str] = None
    site_name: Optional[str] = None
    camera_name: Optional[str] = None
    device_uuid: Optional[str] = None
    event_type: str
    title: Optional[str] = None
    message: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    image_storage_key: Optional[str] = None
    clip_url: Optional[str] = None
    clip_status: Optional[str] = None
    notes: List[NotificationNote] = Field(default_factory=list)
    detected_at: datetime
    created_at: datetime
    read_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    status: str


class ChartPoint(BaseModel):
    bucket_start: datetime
    count: int


class DetectionsOverTimeOut(BaseModel):
    user_id: int
    site_uuid: Optional[str] = None
    hours: int
    object_class: Optional[str] = None
    roi_only: bool
    bucket_minutes: int
    from_time: datetime = Field(alias="from")
    to: datetime
    total: int
    points: List[ChartPoint]

    class Config:
        populate_by_name = True


class ClearNotificationsRequest(BaseModel):
    site_uuid: Optional[str] = None
    camera_uuid: Optional[str] = None


class DeleteNotificationsRequest(BaseModel):
    site_uuid: Optional[str] = None
    camera_uuid: Optional[str] = None
    notification_ids: Optional[List[int]] = None


# --- Notification email schemas ---
class NotificationEmailCreate(BaseModel):
    email: EmailStr
    # If omitted, email is applied to all sites owned by the user.
    site_uuid: Optional[uuid.UUID] = None


class NotificationEmailOut(BaseModel):
    id: int
    user_id: int
    site_uuid: uuid.UUID
    email: str
    is_enabled: bool


class NotificationEmailCreateResult(BaseModel):
    created: List[NotificationEmailOut]
    created_count: int
    skipped_count: int
    target_site_count: int


# --- User schemas ---
class UserOut(BaseModel):
    id: int
    user_name: str
    user_email: EmailStr


class UserProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str | None = Field(default=None, min_length=1, max_length=255)
    user_email: EmailStr | None = None

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr | None) -> str | None:
        if value is None:
            return None
        return str(value).lower().strip()

    @model_validator(mode="after")
    def validate_has_fields(self):
        if self.user_name is None and self.user_email is None:
            raise ValueError("Provide at least one field to update")
        return self


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: SecretStr
    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 8:
            raise ValueError("New password must be at least 8 characters long")
        return value


class DeleteAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: SecretStr


# --- Detection schemas ---
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
    track_id: Optional[int] = None


class DetectionOut(BaseModel):
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int
    event_type: Optional[str] = None
    reason: Optional[str] = None
    inference_ms: Optional[int] = None
    model_id: Optional[str] = None

    frame_w: Optional[int] = None
    frame_h: Optional[int] = None

    detections: List[DetectionItemOut] = Field(default_factory=list)
    pose: Optional[Any] = None


# --- Clip schemas ---
class ClipOut(BaseModel):
    id: int
    camera_uuid: str
    camera_name: Optional[str] = None
    camera_code: Optional[str] = None
    site_uuid: Optional[str] = None
    site_name: Optional[str] = None
    site_code: Optional[str] = None
    external_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    status: str
    recording_url: Optional[str] = None
    overlay_payload: Optional[Dict[str, Any]] = None
    storage_key: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime


class BulkClipDeleteRequest(BaseModel):
    clip_ids: List[PositiveInt] = Field(..., min_length=1)


class BulkClipDeleteResponse(BaseModel):
    requested: int
    deleted: int
    deleted_ids: List[int] = Field(default_factory=list)


# --- Auth schemas ---
class AuthOrgMembershipOut(BaseModel):
    """One row from the user's org_memberships, used by the frontend
    to decide which Admin / Platform Admin tabs to render."""

    org_id: int
    org_name: str
    org_slug: str
    role: str


class AuthUserOut(BaseModel):
    """Identity payload returned by /auth/me, /auth/login and signup.

    `is_platform_admin` unlocks the Platform Admin section in the UI.
    `organizations` carries every org the user belongs to plus the
    role they hold there, so the frontend can show the Org Admin UI
    whenever any membership has `role == "admin"`.
    """

    id: int
    user_name: str
    user_email: EmailStr
    is_platform_admin: bool = False
    organizations: list[AuthOrgMembershipOut] = []


class AuthTokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserOut


class SignupRequestCode(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str = Field(..., min_length=1, max_length=255)
    user_email: EmailStr
    password: SecretStr

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return value


class SignupVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    signup_token: str = Field(..., min_length=16, max_length=255)
    code: str = Field(..., min_length=6, max_length=10)

    @field_validator("signup_token")
    @classmethod
    def normalize_token(cls, value: str) -> str:
        token = value.strip()
        if not token:
            raise ValueError("signup_token is required")
        return token

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        code = value.strip()
        if not code.isdigit():
            raise ValueError("Verification code must contain digits only")
        return code


class SignupCodeOut(BaseModel):
    message: str
    signup_token: str
    expires_at: datetime
    debug_code: str | None = None


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_email: EmailStr
    password: SecretStr

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower().strip()


# -------------------------
# Custom walls
# -------------------------

# Bounds how many streams a single share link can fan out to a media server that
# has no per-viewer limits of its own.
MAX_WALL_CAMERAS = 24


class WallCameraSchema(BaseModel):
    """A camera as seen by an authenticated wall viewer."""

    model_config = ConfigDict(extra="forbid")

    camera_uuid: uuid.UUID
    name: Optional[str] = None
    location: Optional[str] = None
    site_uuid: uuid.UUID
    site_name: Optional[str] = None
    webrtc_url: Optional[str] = None
    position: int


class WallSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_uuid: uuid.UUID
    name: str
    cameras: List[WallCameraSchema] = Field(default_factory=list)

    # Cameras stored on the wall that this caller may not see (their site grant
    # was revoked, the site was soft-deleted, or the camera was disabled). Lets
    # the UI say "showing 3 of 5" instead of silently dropping them.
    total_camera_count: int = 0

    share_enabled: bool = False
    share_token: Optional[str] = None
    share_expires_at: Optional[datetime] = None

    created_at: datetime
    updated_at: datetime


class WallCreateSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    camera_uuids: List[uuid.UUID] = Field(default_factory=list, max_length=MAX_WALL_CAMERAS)

    @field_validator("camera_uuids")
    @classmethod
    def reject_duplicates(cls, value: List[uuid.UUID]) -> List[uuid.UUID]:
        if len(set(value)) != len(value):
            raise ValueError("camera_uuids contains duplicates")
        return value


class WallEditSchema(BaseModel):
    """PATCH — omitted fields are left unchanged. ``camera_uuids`` replaces the
    whole ordered membership when present."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    camera_uuids: Optional[List[uuid.UUID]] = Field(default=None, max_length=MAX_WALL_CAMERAS)

    @field_validator("camera_uuids")
    @classmethod
    def reject_duplicates(cls, value: Optional[List[uuid.UUID]]) -> Optional[List[uuid.UUID]]:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("camera_uuids contains duplicates")
        return value


class WallShareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expires_at: Optional[datetime] = None


# --- Public (unauthenticated) shapes ---
#
# Deliberately NOT derived from CameraSchema: that model requires `source_url`,
# which can embed RTSP credentials. ROI polygons are omitted too — they reveal
# where the tripwires are.


class PublicWallCameraSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_uuid: uuid.UUID
    name: Optional[str] = None
    location: Optional[str] = None
    site_name: Optional[str] = None
    webrtc_url: Optional[str] = None
    position: int


class PublicWallSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    cameras: List[PublicWallCameraSchema] = Field(default_factory=list)
