# agents/domain/events.py

import logging
from typing import Any, Dict, List, Optional, Tuple, Literal,Union
import uuid
from pydantic import BaseModel, Field,ConfigDict

logger = logging.getLogger(__name__)

# =========================================================
# Base Event System
# =========================================================

class Event(BaseModel):
    """
    Base class for all events in the system.

    Notes:
    - arbitrary_types_allowed=True so we can carry in-memory frames (e.g., np.ndarray).
    - Do NOT try to JSON serialize events that contain raw frames.
    """
    event_type: str = "Event"
    payload: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChannelEvent(Event):
    """
    Base class for events emitted by a channel (RTSP/Web/VAPI/etc).
    """
    event_type: str = "ChannelEvent"
    channel_id: Optional[uuid.UUID] = None


# =========================================================
# RTSP / Video Pipeline Events
# =========================================================

EncodedFormat = Literal["raw", "jpeg", "png", "h264", "h265"]


class RTSPEvent(ChannelEvent):
    """
    Canonical RTSP event carrying an IMAGE/VIDEO FRAME.

    You can carry frames in two ways:

    1) In-process raw frame (fastest, not serializable):
       - frame: np.ndarray (or similar)
       - format="raw"
       - frame_shape=(H, W, C)

    2) Encoded payload (serializable if you base64 it later):
       - encoded: bytes (e.g., JPEG bytes or H264/H265 access unit)
       - format="jpeg" / "h264" / "h265"
       - width/height optional but recommended

    Recommended for Jetson local pipeline:
      - Use raw frames (np.ndarray) from GStreamer appsink, keep everything in-process.
    Recommended if sending frames over network:
      - Use encoded bytes (jpeg/h264) and avoid raw frames.
    """
    event_type: str = "RTSPEvent"

    camera_uuid: str
    ts_ms: int
    seq: int 
    format: EncodedFormat = "raw"
    detection_enabled:bool=True
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
    """
    Event to Add a Channel.
    """
    event_type: Literal["Create_Channel"] = "Create_Channel"
    configs: Dict[str, Any]

class ChannelRemoveEvent(Event):
    """
    Event to Remove a Channel.
    """
    event_type: Literal["Remove_Channel"] = "Remove_Channel"
    channel_id: uuid.UUID
    
class ChannelEditEvent(Event):
    """
    Event to Edit a Channel.
    """
    event_type: Literal["Edit_Channel"] = "Edit_Channel"
    channel_id: uuid.UUID
    configs: Dict[str, Any]
    
class FrameDroppedEvent(ChannelEvent):
    event_type: str = "FrameDroppedEvent"
    camera_uuid: str
    reason: str
    dropped_count: int = 1

VideoChannelEvent=Union[ChannelConnectedEvent,ChannelDisconnectedEvent,ChannelCreateEvent,ChannelRemoveEvent,ChannelEditEvent]
# =========================================================
# Detection + Alerts
# =========================================================

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
    """
    Pose/skeleton output for a frame.
    """
    format: Literal["xy", "xyn"] = "xy"  # pixel coords or normalized coords
    skeletons: List[SkeletonItem] = Field(default_factory=list)


class SkeletonProducedEvent(ChannelEvent):
    """
    Dedicated event for pose/skeleton output.
    Useful if some consumers only care about pose.
    """
    event_type: str = "SkeletonProducedEvent"
    camera_uuid: str
    model_id: str
    frame_ts_ms: int
    frame_seq: int
    pose: PoseResult
    
class DetectionsProducedEvent(ChannelEvent):
    """
    Emitted after inference runs on a specific RTSPEvent frame.
    """
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


# =========================================================
# Clip (Last 20s) Events (optional)
# =========================================================

class ClipRequestEvent(ChannelEvent):
    """
    Request the last N ms clip window for a camera.
    """
    event_type: str = "ClipRequestEvent"
    camera_uuid: str
    request_id: Optional[str] = None
    end_ts_ms: Optional[int] = None
    window_ms: int = 20_000


class ClipReadyEvent(ChannelEvent):
    """
    Response event when a clip is generated (file path or URL).
    """
    event_type: str = "ClipReadyEvent"
    camera_uuid: str
    request_id: str
    clip_ref: str  # local file path or URL
    start_ts_ms: Optional[int] = None
    end_ts_ms: Optional[int] = None
