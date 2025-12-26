from typing import Optional, Literal, Tuple
import uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator

from domain.channel import ChannelConfig


class VideoChannelConfig(BaseModel, ChannelConfig):
    """
    API/DB upsert input config.
    - Allows partial payloads (client can omit channel_id, rtsp_url when updating an existing camera_uuid).
    - Defaults are applied here so POST body stays small.
    """
    model_config = ConfigDict(extra="forbid")

    # identity (client may omit channel_id; server will set it)
    channel_id: Optional[str] = Field(
        default=None,
        description="Optional. If omitted, server uses str(camera_uuid) at runtime."
    )
    camera_uuid: Optional[uuid.UUID] = Field(default=None, description="Existing camera UUID if updating.")
    rtsp_url: Optional[str] = Field(default=None, description="Required when creating a new camera.")

    # enable/sampling
    enabled: bool = Field(default=True)
    sample_fps: float = Field(default=5.0, ge=0.1, description="Frames/sec to publish as RTSPEvent")

    # decoding strategy (Jetson: prefer gstreamer)
    decode_backend: Literal["gstreamer", "opencv"] = Field(default="gstreamer")

    # optional resize before publish/inference to reduce load
    resize: Optional[Tuple[int, int]] = Field(default=None, description="(width, height)")

    # reconnect policy tuning
    reconnect_base_ms: int = Field(default=1000, ge=100)
    reconnect_max_ms: int = Field(default=8000, ge=1000)

    # what your RTSPEvent will carry
    emit_format: Literal["raw", "jpeg"] = Field(default="raw", description="raw => np.ndarray in RTSPEvent.frame")
    jpeg_quality: int = Field(default=80, ge=1, le=100)

    @model_validator(mode="after")
    def _validate_new_camera_requires_rtsp(self):
        # If creating a new camera (no camera_uuid), rtsp_url must be provided.
        if self.camera_uuid is None and not self.rtsp_url:
            raise ValueError("rtsp_url is required when camera_uuid is not provided (new camera).")
        return self

