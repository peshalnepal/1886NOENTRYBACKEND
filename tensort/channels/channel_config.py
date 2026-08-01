# video_channel_config.py
import uuid


class VideoChannelConfig(object):
    """
    Minimal config for RTSP ingest on Jetson (Python 3.6).
    No Pydantic, no external deps.

    Notes:
      - If channel_id is None, it is derived from camera_uuid (if present).
      - If camera_uuid is None (new camera), source_url must be provided.
    """

    __slots__ = (
        "channel_id",
        "camera_uuid",
        "source_url",
        "enabled",
        "detection_enabled",
        "notification_enabled",
        "sample_fps",
        "decode_backend",
        "resize",
        "reconnect_base_ms",
        "reconnect_max_ms",
        "emit_format",
        "jpeg_quality",
        "gst_latency_ms",
        "rtsp_transport",
        "gst_decoder",
    )

    def __init__(
        self,
        channel_id=None,
        camera_uuid=None,
        source_url=None,
        enabled=True,
        detection_enabled=True,
        notification_enabled=True,
        sample_fps=5.0,
        decode_backend="gstreamer",  # "gstreamer" or "opencv"
        resize=None,                # (width, height) or None
        reconnect_base_ms=1000,
        reconnect_max_ms=8000,
        emit_format="raw",          # "raw" or "jpeg"
        jpeg_quality=80,
        gst_latency_ms=5,
        rtsp_transport="tcp",       # "tcp" or "udp"
        gst_decoder="nvv4l2decoder" # Jetson HW decode; fallback happens in code
    ):
        self.channel_id = channel_id
        self.camera_uuid = camera_uuid
        self.source_url = source_url
        self.enabled = bool(enabled)
        self.detection_enabled = bool(detection_enabled)
        self.notification_enabled = bool(notification_enabled)
        self.sample_fps = float(sample_fps)
        self.decode_backend = str(decode_backend)
        self.resize = resize
        self.reconnect_base_ms = int(reconnect_base_ms)
        self.reconnect_max_ms = int(reconnect_max_ms)
        self.emit_format = str(emit_format)
        self.jpeg_quality = int(jpeg_quality)
        self.gst_latency_ms = int(gst_latency_ms)
        self.rtsp_transport = str(rtsp_transport)
        self.gst_decoder = str(gst_decoder)

        self._normalize()
        self.validate()

    def _normalize(self):
        # normalize camera_uuid -> string (optional)
        if self.camera_uuid is not None:
            if isinstance(self.camera_uuid, uuid.UUID):
                self.camera_uuid = str(self.camera_uuid)
            else:
                self.camera_uuid = str(self.camera_uuid)

        # default channel_id from camera_uuid
        if not self.channel_id:
            if self.camera_uuid:
                self.channel_id = str(self.camera_uuid)
            else:
                self.channel_id = "unknown-channel"

        # normalize decode_backend
        if self.decode_backend not in ("gstreamer", "opencv"):
            # keep it simple: fallback to opencv
            self.decode_backend = "opencv"

        # normalize emit_format
        if self.emit_format not in ("raw", "jpeg"):
            self.emit_format = "raw"

        # normalize resize
        if self.resize is not None:
            if (not isinstance(self.resize, (tuple, list))) or len(self.resize) != 2:
                self.resize = None
            else:
                w = int(self.resize[0])
                h = int(self.resize[1])
                if w <= 0 or h <= 0:
                    self.resize = None
                else:
                    self.resize = (w, h)

        # normalize transport
        if self.rtsp_transport not in ("tcp", "udp"):
            self.rtsp_transport = "tcp"

    def validate(self):
        # If creating new camera (no camera_uuid), a source must be provided
        if not self.camera_uuid and not self.source_url:
            raise ValueError("source_url is required when camera_uuid is not provided (new camera).")

        if self.sample_fps <= 0.0:
            raise ValueError("sample_fps must be > 0")

        if self.reconnect_base_ms < 100:
            raise ValueError("reconnect_base_ms must be >= 100")

        if self.reconnect_max_ms < self.reconnect_base_ms:
            raise ValueError("reconnect_max_ms must be >= reconnect_base_ms")

        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be in [1, 100]")
