"""Supported camera source-URL schemes + helpers.

A camera's video source is a single ``source_url`` accepting any of the schemes
in ``ACCEPTED_SOURCE_SCHEMES`` (RTSP, WebRTC/WHEP, HLS/MJPEG over HTTP, RTMP,
SRT). It is the one source of truth — there is no separate ``rtsp_url``.
"""

from typing import Optional

# Scheme prefixes we accept as a camera source. MediaMTX (the WebRTC gateway)
# can ingest all of these as a path ``source``; the Jetson edge decodes RTSP via
# its GStreamer pipeline and falls back to OpenCV for the rest.
ACCEPTED_SOURCE_SCHEMES = (
    "rtsp://",
    "rtsps://",
    "webrtc://",
    "whep://",
    "wheps://",
    "http://",
    "https://",
    "rtmp://",
    "rtmps://",
    "srt://",
)


# Human-facing catalogue of supported source schemes, exposed to the frontend
# (GET /api/cameras/source-schemes) so users know exactly what to enter. This is
# the single source of truth for both validation (above) and the UI hint.
SOURCE_SCHEME_CATALOG = (
    {"scheme": "rtsp", "label": "RTSP", "example": "rtsp://user:pass@192.168.1.10:554/stream"},
    {"scheme": "rtsps", "label": "RTSP over TLS", "example": "rtsps://user:pass@host:322/stream"},
    {"scheme": "webrtc", "label": "WebRTC (WHEP)", "example": "webrtc://host/path"},
    {"scheme": "whep", "label": "WebRTC (WHEP)", "example": "whep://host/stream/whep"},
    {"scheme": "wheps", "label": "WebRTC (WHEP over HTTPS)", "example": "wheps://host/stream/whep"},
    {"scheme": "http", "label": "HLS / MJPEG", "example": "http://host/stream/index.m3u8"},
    {"scheme": "https", "label": "HLS / MJPEG (TLS)", "example": "https://host/stream/index.m3u8"},
    {"scheme": "rtmp", "label": "RTMP", "example": "rtmp://host/app/streamkey"},
    {"scheme": "rtmps", "label": "RTMP over TLS", "example": "rtmps://host/app/streamkey"},
    {"scheme": "srt", "label": "SRT", "example": "srt://host:8890?streamid=..."},
)


def supported_source_schemes() -> dict:
    """Structured description of accepted source schemes for API/UI consumers."""
    return {
        "schemes": [s.rstrip(":/") for s in ACCEPTED_SOURCE_SCHEMES],
        "prefixes": list(ACCEPTED_SOURCE_SCHEMES),
        "items": [dict(item) for item in SOURCE_SCHEME_CATALOG],
    }


def is_supported_source_url(url: Optional[str]) -> bool:
    """True when ``url`` is a non-empty string starting with an accepted scheme."""
    if not url or not isinstance(url, str):
        return False
    lowered = url.strip().lower()
    return any(lowered.startswith(scheme) for scheme in ACCEPTED_SOURCE_SCHEMES)


def is_rtsp_source(url: Optional[str]) -> bool:
    """True when ``url`` is an RTSP/RTSPS source (the only scheme the Jetson
    GStreamer pipeline and MediaMTX ``rtspTransport`` option apply to)."""
    if not url or not isinstance(url, str):
        return False
    lowered = url.strip().lower()
    return lowered.startswith("rtsp://") or lowered.startswith("rtsps://")
