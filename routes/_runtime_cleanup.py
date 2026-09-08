"""Best-effort helpers for removing a camera from runtime services."""

import logging
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


async def cleanup_camera_resources(
    manager: Any,
    *,
    camera_uuid: Any,
    camera_code: Optional[str],
    device_urls: Iterable[str],
    pipeline: Optional[Any] = None,
    log_prefix: str,
) -> None:
    """Remove one camera from edge, WebRTC, and an already-loaded pipeline.

    Runtime cleanup must never prevent the database delete.  Callers can use
    this helper from both camera and site deletion flows without changing their
    transaction or timeout boundaries.
    """
    targets = list(dict.fromkeys(url.strip() for url in device_urls if url and url.strip()))
    camera_id = str(camera_uuid)

    for device_url in targets:
        try:
            await manager.edge.delete_camera(
                device_url=device_url,
                camera_uuid=camera_id,
            )
        except Exception as exc:
            logger.warning(
                "%s Edge delete failed cam=%s url=%s: %s",
                log_prefix,
                camera_uuid,
                device_url,
                exc,
            )

    if camera_code:
        try:
            await manager.webrtc.delete_stream(stream_key=str(camera_code))
        except Exception as exc:
            logger.warning(
                "%s WebRTC delete failed cam=%s code=%s: %s",
                log_prefix,
                camera_uuid,
                camera_code,
                exc,
            )

    if pipeline is not None:
        try:
            await pipeline.remove_channel(camera_uuid)
        except Exception as exc:
            logger.warning(
                "%s Pipeline evict failed cam=%s: %s",
                log_prefix,
                camera_uuid,
                exc,
            )
