# routes/helpers.py
"""Request-parsing helpers shared by the route blueprints."""

from typing import Any, Dict

from flask import request

try:
    from routes.runtime_ref import get_runtime
except Exception:
    from .runtime_ref import get_runtime


def json_body() -> Dict[str, Any]:
    """Parse the request body as a dict, tolerating malformed/absent JSON."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


# Camera source schemes accepted by the edge. RTSP is decoded by the GStreamer
# pipeline; the others fall back to OpenCV's generic capture (see VideoChannel).
ACCEPTED_SOURCE_SCHEMES = (
    "rtsp://", "rtsps://",
    "webrtc://", "whep://", "wheps://",
    "http://", "https://",
    "rtmp://", "rtmps://",
    "srt://",
)


def require_source(url) -> bool:
    if not url or not isinstance(url, str):
        return False
    lowered = url.strip().lower()
    return any(lowered.startswith(scheme) for scheme in ACCEPTED_SOURCE_SCHEMES)


def runtime_status(include_stats: bool) -> Dict[str, Any]:
    runtime = get_runtime()
    payload = {
        "ok": True,
        "service": "jetson-tensort",
        "pipeline_ready": bool(runtime.pipeline is not None and runtime.loop is not None),
    }
    if include_stats:
        payload["stats"] = runtime.get_stats()

    # Surfaced on both / and /health so the cloud can tell a Jetson that is
    # actively discovering from one where the scanner died, without a second
    # round trip.
    discovery = getattr(runtime, "discovery", None)
    if discovery is not None:
        try:
            payload["discovery"] = discovery.status()
        except Exception:
            payload["discovery"] = {"enabled": True, "running": False, "error": True}
    else:
        payload["discovery"] = {"enabled": False, "running": False}

    return payload
