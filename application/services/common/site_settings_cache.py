"""Shared site trigger-mode resolver.

``ModelPipeline`` and ``NotificationService`` both read
``SiteSettings.config`` to decide whether a site emits all notifications
("any_detection") or only ROI-enter alerts ("roi_enter"), each with its own
TTL cache. The two implementations were identical apart from the fallback
default, which is now a constructor argument.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_MODES = {"roi_enter", "any_detection"}


class SiteTriggerModeResolver:
    """TTL-cached lookup of a site's notification trigger mode."""

    def __init__(self, *, default: str = "roi_enter", ttl_s: float = 30.0) -> None:
        self._default = default if default in _VALID_MODES else "roi_enter"
        self._ttl_s = max(5.0, float(ttl_s))
        self._session_factory = None
        # key -> (expires_at_monotonic, mode)
        self._cache: Dict[str, Tuple[float, str]] = {}

    def set_session_factory(self, session_factory) -> None:
        self._session_factory = session_factory

    def invalidate(self, site_uuid: str) -> None:
        self._cache.pop(str(site_uuid), None)

    async def resolve(self, site_uuid: Optional[str]) -> str:
        if not site_uuid:
            return self._default

        key = str(site_uuid)
        now = time.monotonic()

        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return cached[1]

        sf = self._session_factory
        if sf is None:
            return self._default

        try:
            su = uuid.UUID(key)
        except Exception:
            return self._default

        mode = self._default
        try:
            from application.repositories.site_repository import SiteRepository

            async with sf() as db:
                settings = await SiteRepository().get_site_settings(db, site_uuid=su)
                config = settings.config if settings is not None else None
            mode = self._extract_mode(config)
        except Exception:
            logger.exception("Failed to load site trigger_mode site_uuid=%s", key)
            mode = self._default

        self._cache[key] = (now + self._ttl_s, mode)
        return mode

    def _extract_mode(self, config) -> str:
        if not isinstance(config, dict):
            return self._default

        raw = None
        notification_rule = config.get("notification")
        if isinstance(notification_rule, dict):
            raw = notification_rule.get("trigger_mode")
        if raw is None:
            legacy = config.get("multi_camera_prerecord")
            if isinstance(legacy, dict):
                raw = legacy.get("trigger_mode")

        value = str(raw or "").strip().lower()
        return value if value in _VALID_MODES else self._default
