"""Pipeline events: the vendor-agnostic vocabulary shared by channels, models
and the notification layer."""

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """Base class for every event.

    `arbitrary_types_allowed` lets an event carry an in-memory frame
    (`np.ndarray`); such events must never be JSON-serialized.
    """
    event_type: str = "Event"
    payload: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChannelEvent(Event):
    """Base class for events emitted by a channel (RTSP / web / …)."""
    event_type: str = "ChannelEvent"
    channel_id: Optional[uuid.UUID] = None


EncodedFormat = Literal["raw", "jpeg", "png", "h264", "h265"]


class RTSPEvent(ChannelEvent):
    """One video frame from a camera, carried either raw or encoded.

    In-process (fastest, not serializable): set `frame` to an ndarray with
    `format="raw"` and `frame_shape=(H, W, C)`. This is what the Jetson's
    GStreamer appsink produces.

    Over the network: set `encoded` to the compressed bytes with a matching
    `format` ("jpeg" / "h264" / "h265"); `width`/`height` are optional but
    worth setting.
    """
    event_type: str = "RTSPEvent"

    camera_uuid: str
    ts_ms: int
    seq: int
    format: EncodedFormat = "raw"
    detection_enabled: bool = True
    frame: Any = None
    frame_shape: Optional[Tuple[int, int, int]] = None
    encoded: Optional[bytes] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps_hint: Optional[float] = None
    keyframe: Optional[bool] = None
    codec: Optional[str] = None


class ChannelConnectedEvent(ChannelEvent):
    event_type: str = "ChannelConnectedEvent"
    camera_uuid: str
    source_url: str
    device_url: Optional[str] = None
    webrtc_url: Optional[str] = None

class ChannelDisconnectedEvent(ChannelEvent):
    event_type: str = "ChannelDisconnectedEvent"
    camera_uuid: str
    reason: str
    device_url: Optional[str] = None


class ChannelCreateEvent(Event):
    event_type: Literal["Create_Channel"] = "Create_Channel"
    configs: Dict[str, Any]


class ChannelRemoveEvent(Event):
    event_type: Literal["Remove_Channel"] = "Remove_Channel"
    channel_id: uuid.UUID


class ChannelEditEvent(Event):
    event_type: Literal["Edit_Channel"] = "Edit_Channel"
    channel_id: uuid.UUID
    configs: Dict[str, Any]


class FrameDroppedEvent(ChannelEvent):
    event_type: str = "FrameDroppedEvent"
    camera_uuid: str
    reason: str
    dropped_count: int = 1


VideoChannelEvent = Union[
    ChannelConnectedEvent,
    ChannelDisconnectedEvent,
    ChannelCreateEvent,
    ChannelRemoveEvent,
    ChannelEditEvent,
]

class DetectionBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class BoxNorm(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectionItem(BaseModel):
    cls_name: str
    conf: float
    box: DetectionBox
    box_norm: Optional[BoxNorm] = None


class PoseKeypoint(BaseModel):
    x: float
    y: float
    conf: Optional[float] = None


class SkeletonItem(BaseModel):
    conf: float
    box: DetectionBox
    keypoints: List[PoseKeypoint] = Field(default_factory=list)


class PoseResult(BaseModel):
    """Pose/skeleton output for one frame."""
    format: Literal["xy", "xyn"] = "xy"  # pixel or normalized coordinates
    skeletons: List[SkeletonItem] = Field(default_factory=list)


class SkeletonProducedEvent(ChannelEvent):
    """Pose-only output, for consumers that do not care about detections."""
    event_type: str = "SkeletonProducedEvent"
    camera_uuid: str
    model_id: str
    frame_ts_ms: int
    frame_seq: int
    pose: PoseResult


class DetectionsProducedEvent(ChannelEvent):
    """Emitted after inference runs on one `RTSPEvent` frame."""
    event_type: str = "DetectionsProducedEvent"
    camera_uuid: str
    model_id: str
    frame_ts_ms: int
    frame_seq: int
    detections: List[DetectionItem] = Field(default_factory=list)
    inference_ms: Optional[int] = None
    pose: Optional[PoseResult] = None


class InferenceFailedEvent(ChannelEvent):
    event_type: str = "InferenceFailedEvent"
    camera_uuid: str
    model_id: str
    frame_ts_ms: int
    frame_seq: int
    reason: str


class AlertRaisedEvent(ChannelEvent):
    event_type: str = "AlertRaisedEvent"
    camera_uuid: str
    zone_id: str
    cls_name: str
    conf: float
    box: DetectionBox
    frame_ts_ms: int
    frame_seq: int
    rule_id: Optional[str] = None


class ClipRequestEvent(ChannelEvent):
    """Request the last `window_ms` of footage for a camera."""
    event_type: str = "ClipRequestEvent"
    camera_uuid: str
    request_id: Optional[str] = None
    end_ts_ms: Optional[int] = None
    window_ms: int = 20_000


class ClipReadyEvent(ChannelEvent):
    """Response once a clip exists, as a file path or URL in `clip_ref`."""
    event_type: str = "ClipReadyEvent"
    camera_uuid: str
    request_id: str
    clip_ref: str  # local file path or URL
    start_ts_ms: Optional[int] = None
    end_ts_ms: Optional[int] = None
