import os
from urllib.parse import quote


def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


STREAM_GATEWAY_PUBLIC_BASE = _env("STREAM_GATEWAY_PUBLIC_BASE", "http://localhost:8889")

STREAM_GATEWAY_PLAYBACK_MODE = _env("STREAM_GATEWAY_PLAYBACK_MODE", "path")

STREAM_GATEWAY_PATH_PREFIX = _env("STREAM_GATEWAY_PATH_PREFIX", "cam")


def gateway_path(camera_uuid: str) -> str:
    return f"{STREAM_GATEWAY_PATH_PREFIX}-{camera_uuid}"


def build_playback(camera_uuid: str) -> dict:
    """
    Returns a JSON that frontend can use to open WebRTC on the gateway.
    You can change format later without touching camera_routes.
    """
    path = gateway_path(camera_uuid)
    base = STREAM_GATEWAY_PUBLIC_BASE.rstrip("/")

    if STREAM_GATEWAY_PLAYBACK_MODE == "whep":
        url = f"{base}/{quote(path)}/whep"
        return {"type": "webrtc", "mode": "whep", "path": path, "url": url}

    url = f"{base}/{quote(path)}/whep"
    return {"type": "webrtc", "mode": "path", "path": path, "url": url}
