"""Shared camera response builders.

Camera rows reach the HTTP boundary in two shapes: the ORM `Camera` (read
paths) and the manager's `CameraOut` DTO (create/edit paths). Both must
serialize to the same contract, so the mapping lives here rather than in either
router — otherwise one route ends up importing the other's private helper.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from application.services.webrtcgateway import resolve_camera_webrtc_url
from core.schemas import CameraSchema, CameraWithConfigSchema


def camera_webrtc_url(cam: Any) -> Optional[str]:
    """Playback URL for a camera row or DTO."""
    return resolve_camera_webrtc_url(
        camera_code=getattr(cam, "camera_code", None),
        stored_url=getattr(cam, "webrtc_url", None),
    )


def tri_state(cam: Any, attr: str) -> str:
    """Read a tri-state camera column, defaulting to "inherit"."""
    return str(getattr(cam, attr, "inherit") or "inherit")


def camera_out(cam: Any, *, device_uuid: Optional[Any] = None) -> CameraSchema:
    """Build the list response from an ORM `Camera`."""
    return CameraSchema(
        camera_uuid=cam.camera_uuid,
        camera_code=cam.camera_code,
        name=getattr(cam, "name", None),
        location=getattr(cam, "location", None),
        site_uuid=cam.site_uuid,
        device_uuid=device_uuid,
        source_url=cam.source_url,
        webrtc_url=camera_webrtc_url(cam),
        is_enabled=cam.is_enabled,
        is_detection_enabled=cam.is_detection_enabled,
        is_notification_enabled=cam.is_notification_enabled,
        notification_trigger_mode=tri_state(cam, "notification_trigger_mode"),
        camera_playback_enabled=tri_state(cam, "camera_playback_enabled"),
        use_site_schedule=bool(getattr(cam, "use_site_schedule", True)),
        roi=getattr(cam, "roi", None),
        created_at=cam.created_at,
        updated_at=cam.updated_at,
    )


def camera_with_config_out(
    cam: Any,
    *,
    webrtc_url: Optional[str],
    configuration: Optional[dict] = None,
    timezone_name: Optional[str] = None,
    device_uuid: Optional[Any] = None,
) -> CameraWithConfigSchema:
    """Build the detail response from an ORM `Camera` or a `CameraOut` DTO.

    `CameraOut` names the flags `enabled` / `detection_enabled` / …, while the
    ORM and the HTTP contract use the `is_` prefix, so the flags are read
    through both names instead of mapped automatically.
    """
    now = datetime.now(timezone.utc)
    return CameraWithConfigSchema(
        camera_uuid=cam.camera_uuid,
        camera_code=cam.camera_code,
        name=getattr(cam, "name", None),
        location=getattr(cam, "location", None),
        site_uuid=cam.site_uuid,
        device_uuid=device_uuid if device_uuid is not None else getattr(cam, "device_uuid", None),
        source_url=cam.source_url,
        webrtc_url=webrtc_url,
        is_enabled=_flag(cam, "is_enabled", "enabled"),
        is_detection_enabled=_flag(cam, "is_detection_enabled", "detection_enabled"),
        is_notification_enabled=_flag(cam, "is_notification_enabled", "notification_enabled"),
        notification_trigger_mode=tri_state(cam, "notification_trigger_mode"),
        camera_playback_enabled=tri_state(cam, "camera_playback_enabled"),
        use_site_schedule=bool(getattr(cam, "use_site_schedule", True)),
        roi=getattr(cam, "roi", None),
        configuration=configuration if configuration is not None else getattr(cam, "configuration", {}),
        timezone=timezone_name if timezone_name is not None else getattr(cam, "timezone", None),
        created_at=getattr(cam, "created_at", None) or now,
        updated_at=getattr(cam, "updated_at", None) or now,
    )


def _flag(cam: Any, orm_attr: str, dto_attr: str) -> bool:
    """Read a boolean flag under whichever name the object carries."""
    value = getattr(cam, orm_attr, None)
    if value is None:
        value = getattr(cam, dto_attr, None)
    return bool(value)
