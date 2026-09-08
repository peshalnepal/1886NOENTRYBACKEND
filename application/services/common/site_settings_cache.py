"""Shared site trigger-mode and arm-state resolvers.

Reads ``SiteSettings.config`` to decide whether a site emits all notifications
("any_detection") or only ROI-enter alerts ("roi_enter"). Shared by
``ModelPipeline`` and ``NotificationService``, which differ only in the fallback
default — hence the constructor argument.
"""

from __future__ import annotations

import datetime as _dt
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
    
class SiteArmStateResolver:
    """TTL-cached lookup of a site's temporary arm/disarm override.

    ``resolve`` returns the *effective* override decision:
      * ``True``  -> site force-armed (notifications/detection allowed),
      * ``False`` -> site force-disarmed,
      * ``None``  -> no active override; the caller follows the schedule.

    The override value and its expiry (``arm_override_until``) are read from the
    ``Site`` row and cached for ``ttl_s``; expiry is evaluated live on every call
    so an override clears exactly at its schedule boundary without waiting for the
    cache to lapse. A short TTL keeps user arm/disarm actions near-immediate; the
    arm route also invalidates the entry for instant effect.
    """

    def __init__(self, *, ttl_s: float = 10.0) -> None:
        self._ttl_s = max(2.0, float(ttl_s))
        self._session_factory = None
        # key -> (expires_at_monotonic, (arm_override, arm_override_until))
        self._cache: Dict[str, Tuple[float, Tuple[Optional[bool], Optional[_dt.datetime]]]] = {}

    def set_session_factory(self, session_factory) -> None:
        self._session_factory = session_factory

    def invalidate(self, site_uuid: str) -> None:
        self._cache.pop(str(site_uuid), None)

    async def resolve(self, site_uuid: Optional[str]) -> Optional[bool]:
        """Input: a site uuid str (or ``None``). Output: the effective override
        decision -- ``True`` (force-armed), ``False`` (force-disarmed) or ``None``
        (no override; follow the schedule).

        Only the raw ``(arm_override, arm_override_until)`` pair is TTL-cached; the
        expiry is re-evaluated on every call so the override clears at its exact
        boundary even if the cache entry has not lapsed. The TTL therefore bounds
        only staleness of *user edits*, which the arm route flushes via
        ``invalidate`` anyway.
        """
        if not site_uuid:
            return None

        key = str(site_uuid)
        now = time.monotonic()

        cached = self._cache.get(key)
        if cached and cached[0] > now:
            override, until = cached[1]
        else:
            override, until = await self._load(key)
            self._cache[key] = (now + self._ttl_s, (override, until))

        from application.channels.channel_config import VideoChannelConfig

        return VideoChannelConfig.effective_arm_override(override, until)

    async def _load(self, key: str) -> Tuple[Optional[bool], Optional[_dt.datetime]]:
        """Read the raw ``(arm_override, arm_override_until)`` for a site from the
        DB. Returns ``(None, None)`` on any failure (no factory, bad uuid, missing
        row, query error) so a lookup problem degrades to "follow the schedule"
        rather than raising into the per-frame notification path."""
        sf = self._session_factory
        if sf is None:
            return (None, None)
        try:
            su = uuid.UUID(key)
        except Exception:
            return (None, None)

        try:
            from core.database_orm import Site
            from sqlalchemy import select

            async with sf() as db:
                row = (
                    await db.execute(
                        select(Site.arm_override, Site.arm_override_until)
                        .where(Site.site_uuid == su)
                        .limit(1)
                    )
                ).first()
            if row is None:
                return (None, None)
            override = None if row[0] is None else bool(row[0])
            return (override, row[1])
        except Exception:
            logger.exception("Failed to load site arm override site_uuid=%s", key)
            return (None, None)
