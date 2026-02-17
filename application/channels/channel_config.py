from typing import Optional, Literal, Tuple, Any, Dict
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dto import ChannelConfig


class VideoChannelConfig(BaseModel, ChannelConfig):
    """
    One unified config for a Video Channel (create/edit/runtime).

    Rules:
    - CREATE (camera_uuid is None):
        requires rtsp_url, site_uuid, device_uuid
    - EDIT (camera_uuid is provided):
        all fields are optional; only provided ones should be applied.
    - webrtc_url and device_url:
        server-managed (read-only from client perspective). Keep them here so
        you can return them and pass them around internally.
    """
    model_config = ConfigDict(extra="forbid")
    camera_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Camera UUID. Omit on create; include on edit."
    )
    channel_id: Optional[str] = Field(
        default=None,
        description="Runtime channel key. If omitted, use str(camera_uuid)."
    )

    site_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Owning site UUID (required on create)."
    )

    device_uuid: Optional[uuid.UUID] = Field(
        default=None,
        description="Assigned device UUID (required on create)."
    )

    rtsp_url: Optional[str] = Field(
        default=None,
    )

    webrtc_url: Optional[str] = Field(
        default=None,
    )

    device_url: Optional[str] = Field(
        default=None,
        description="Server-managed base URL of assigned Jetson/device.",
        json_schema_extra={"readOnly": True},
    )
    name: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    enabled: Optional[bool] = Field(default=True)
    detection_enabled: Optional[bool] = Field(default=True)
    notification_enabled: Optional[bool] = Field(default=True)
    sample_fps: Optional[float] = Field(default=5.0, ge=0.1)
    decode_backend: Optional[Literal["gstreamer", "opencv"]] = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = Field(default=None, description="(width, height)")
    reconnect_base_ms: Optional[int] = Field(default=1000, ge=100)
    reconnect_max_ms: Optional[int] = Field(default=8000, ge=1000)
    poll_interval_ms: Optional[int] = Field(default=500, ge=10)
    request_timeout_s: Optional[float] = Field(default=10.0, ge=0.1)
    detection_path_template: Optional[str] = Field(
        default="/detection/{camera_uuid}",
        description="Jetson detection endpoint template.",
    )
    emit_format: Optional[Literal["raw", "jpeg"]] = Field(
        default=None,
        description="DEBUG ONLY. Backend should not stream frames in production."
    )
    jpeg_quality: Optional[int] = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_and_normalize(self):
        for attr in (
            "channel_id",
            "rtsp_url",
            "webrtc_url",
            "device_url",
            "name",
            "location",
            "timezone",
            "detection_path_template",
        ):
            v = getattr(self, attr, None)
            if isinstance(v, str) and not v.strip():
                setattr(self, attr, None)

        if self.channel_id is None and self.camera_uuid is not None:
            self.channel_id = str(self.camera_uuid)

        if self.camera_uuid is None:
            if not self.rtsp_url:
                raise ValueError("rtsp_url is required when creating a new camera (camera_uuid is None).")
            if self.site_uuid is None:
                raise ValueError("site_uuid is required when creating a new camera.")
            if self.device_uuid is None:
                raise ValueError("device_uuid is required when creating a new camera.")

        return self

    # -----------------------------
    # Helpers (optional but useful)
    # -----------------------------
    def to_patch_dict(self) -> Dict[str, Any]:
        """
        For edit operations: returns only fields that were actually provided by the client,
        excluding read-only server-managed fields.
        """
        d = self.model_dump(exclude_none=True, exclude_unset=True)
        d.pop("webrtc_url", None)
        d.pop("device_url", None)
        return d
