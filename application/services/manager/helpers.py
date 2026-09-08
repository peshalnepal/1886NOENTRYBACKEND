"""Module-level helpers and constants for the manager package."""

from __future__ import annotations

from typing import Any, Dict, Optional

from application.channels.channel_config import VideoChannelConfig


def _edge_health_ready(health: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(health, dict):
        return False
    if health.get("ok") is False:
        return False
    if health.get("pipeline_ready") is False:
        return False
    return True


JETSON_PATCH_KEYS = {
    "source_url",
    "enabled",
    "detection_enabled",
    "notification_enabled",
    "sample_fps",
    "decode_backend",
    "resize",
    "emit_format",
    "jpeg_quality",
    "reconnect_base_ms",
    "reconnect_max_ms",
    "camera_uuid",
    "channel_id",
}


def _camera_config_json(cam: Any) -> Dict[str, Any]:
    """Return a camera's persisted channel configuration as a plain mapping."""
    channel_config = getattr(cam, "channel_configuration", None)
    configuration = getattr(channel_config, "configuration", None)
    return dict(configuration) if isinstance(configuration, dict) else {}


def _only_jetson_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (patch or {}).items() if k in JETSON_PATCH_KEYS and v is not None}


RUNTIME_CONFIG_FORBIDDEN_KEYS = {
    "camera_uuid",
    "channel_id",
    "site_uuid",
    "device_uuid",
    "device_url",
    "source_url",
    "webrtc_url",
    "enabled",
    "detection_enabled",
    "notification_enabled",
    "is_enabled",
    "is_detection_enabled",
    "is_notification_enabled",
    "user_id",
    "roi",
}

RUNTIME_CONFIG_ALLOWED_KEYS = set(VideoChannelConfig.model_fields.keys())


def _coerce_tri_trigger_mode(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ("roi_enter", "any_detection", "inherit"):
        return s
    return "inherit"


def _coerce_tri_playback_mode(v: Any) -> str:
    if v is True:
        return "always"
    if v is False:
        return "never"
    s = str(v or "").strip().lower()
    if s in ("always", "never", "inherit"):
        return s
    return "inherit"


def _runtime_config_overrides(cfg: Dict[str, Any], *, extra_forbidden: Optional[set] = None) -> Dict[str, Any]:
    forbidden = set(RUNTIME_CONFIG_FORBIDDEN_KEYS)
    if extra_forbidden:
        forbidden.update(extra_forbidden)

    out: Dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        if k not in RUNTIME_CONFIG_ALLOWED_KEYS:
            continue
        if k in forbidden:
            continue
        if v is None:
            continue
        if k == "notification_trigger_mode":
            out[k] = _coerce_tri_trigger_mode(v)
        elif k == "camera_playback_enabled":
            out[k] = _coerce_tri_playback_mode(v)
        else:
            out[k] = v
    return out



# Schedule fields are always supplied explicitly from the resolved schedule
# state, so they must never leak in via the free-form runtime overrides.
_SCHEDULE_FORBIDDEN_KEYS = {"schedule", "timezone", "use_site_schedule"}
_CAPTURE_FORBIDDEN_KEYS = {"sample_fps", "decode_backend", "request_timeout_s"}


def build_video_channel_config(
    cam: Any,
    *,
    device_uuid: Any,
    device_url: str,
    schedule_state: Dict[str, Any],
    cfg_json: Optional[Dict[str, Any]] = None,
    default_request_timeout_s: float = 3.0,
    include_capture_fields: bool = True,
) -> "VideoChannelConfig":
    """Build a runtime ``VideoChannelConfig`` from a persisted ``Camera`` row.

    The single place that maps a DB camera + resolved schedule into the
    in-memory channel config, shared by the create/edit/load/schedule-sync paths.

    ``include_capture_fields=False`` lets capture fields (sample_fps etc.) pass
    through ``runtime_overrides`` instead of being set explicitly — used on the
    edit path where they are patched, not recomputed.
    """
    cfg = cfg_json or {}
    extra_forbidden = set(_SCHEDULE_FORBIDDEN_KEYS)
    explicit: Dict[str, Any] = {}

    if include_capture_fields:
        extra_forbidden |= _CAPTURE_FORBIDDEN_KEYS
        explicit.update(
            sample_fps=float(cfg.get("sample_fps", 5.0)),
            decode_backend=str(cfg.get("decode_backend", "gstreamer")),
            request_timeout_s=float(cfg.get("request_timeout_s", default_request_timeout_s)),
        )

    runtime_overrides = _runtime_config_overrides(cfg, extra_forbidden=extra_forbidden)

    return VideoChannelConfig(
        camera_uuid=cam.camera_uuid,
        source_url=cam.source_url,
        webrtc_url=cam.webrtc_url or "",
        site_uuid=cam.site_uuid,
        device_uuid=device_uuid,
        device_url=device_url,
        enabled=bool(getattr(cam, "is_enabled", True)),
        detection_enabled=bool(getattr(cam, "is_detection_enabled", True)),
        notification_enabled=bool(getattr(cam, "is_notification_enabled", True)),
        timezone=schedule_state["timezone"],
        schedule=schedule_state["schedule"],
        use_site_schedule=schedule_state["use_site_schedule"],
        **explicit,
        **runtime_overrides,
    )
