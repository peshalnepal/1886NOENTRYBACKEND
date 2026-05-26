"""Manager site-schedule loading + runtime sync.

Extracted from the former monolithic application/services/manager.py.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession

from application.channels.channel_config import VideoChannelConfig
from application.channels.channel import VideoChannel
from core.database_orm import Camera

from application.services.manager.helpers import build_video_channel_config
from application.services.manager.controllers._state import ManagerState

logger = logging.getLogger(__name__)


class ScheduleResolver:
    def __init__(self, state: ManagerState):
        self._state = state

    async def _load_site_schedule_state(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cache_key = str(site_uuid)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        site = await self._state.site_repo.get_site(db, site_uuid=site_uuid, raise_if_missing=False)
        site_timezone = getattr(site, "timezone", None) if site is not None else None
        settings_row = await self._state.site_repo.get_site_settings(db, site_uuid=site_uuid)

        config = (
            dict(settings_row.config or {})
            if settings_row is not None and isinstance(getattr(settings_row, "config", None), dict)
            else {}
        )
        state = {
            "schedule": VideoChannelConfig.normalize_schedule(config.get("schedule")),
            "timezone": str(config.get("timezone") or site_timezone or "UTC"),
        }
        if cache is not None:
            cache[cache_key] = state
        return state

    def _get_loaded_channel_configuration(self, cam: Camera) -> Optional[Any]:
        try:
            state = inspect(cam)
        except Exception:
            return None

        if "channel_configuration" in getattr(state, "unloaded", set()):
            return None

        try:
            return getattr(cam, "channel_configuration", None)
        except Exception:
            return None

    async def resolve_runtime_schedule(
        self,
        db: AsyncSession,
        *,
        cam: Camera,
        cfg_json: Optional[Dict[str, Any]] = None,
        cfg_timezone: Optional[str] = None,
        site_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cfg = dict(cfg_json or {})
        use_site_schedule = bool(cfg.get("use_site_schedule", getattr(cam, "use_site_schedule", True)))
        loaded_channel_cfg = self._get_loaded_channel_configuration(cam)

        schedule = VideoChannelConfig.normalize_schedule(cfg.get("schedule"))
        timezone_name = str(
            cfg.get("timezone")
            or cfg_timezone
            or getattr(loaded_channel_cfg, "timezone", None)
            or "UTC"
        )

        if use_site_schedule:
            site_state = await self._load_site_schedule_state(
                db,
                site_uuid=cam.site_uuid,
                cache=site_cache,
            )
            if site_state.get("schedule"):
                schedule = site_state["schedule"]
            timezone_name = str(site_state.get("timezone") or timezone_name or "UTC")

        if not schedule:
            schedule = VideoChannelConfig.default_schedule()

        return {
            "schedule": schedule,
            "timezone": timezone_name or "UTC",
            "use_site_schedule": use_site_schedule,
            "active": VideoChannelConfig.schedule_is_active(
                schedule,
                timezone_name or "UTC",
                now_utc=datetime.now(timezone.utc),
            ),
        }

    async def sync_site_schedule_runtime(
        self,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Rebuild in-memory runtime configs for site cameras that inherit the site schedule.

        Returns the affected device UUIDs so callers can trigger best-effort edge reconcile
        without coupling route code to camera/device lookup details.
        """
        uid = int(user_id)
        active = self._state.pipelines_by_user.get(uid)
        refreshed = 0
        skipped = 0
        device_uuids: Set[uuid.UUID] = set()

        async with self._state.session_factory() as db:
            cams = await self._state.channel_repo.list_cameras(
                db,
                user_id=uid,
                site_uuid=site_uuid,
                include_config=True,
                include_device=True,
            )
            site_schedule_cache: Dict[str, Dict[str, Any]] = {}

            for cam in cams:
                cfg_json = {}
                if getattr(cam, "channel_configuration", None) and getattr(cam.channel_configuration, "configuration", None):
                    cfg_json = cam.channel_configuration.configuration or {}

                use_site_schedule = bool(
                    cfg_json.get("use_site_schedule", getattr(cam, "use_site_schedule", True))
                )
                if not use_site_schedule:
                    skipped += 1
                    continue

                devices = [cam.device] if getattr(cam, "device", None) else []
                if len(devices) == 0:
                    logger.warning(
                        "Skipping site schedule refresh for camera %s because device count=%s",
                        cam.camera_uuid,
                        len(devices),
                    )
                    skipped += 1
                    continue
                if len(devices) > 1:
                    logger.warning(
                        "Camera %s has %s linked devices during site schedule refresh; using most recent device %s",
                        cam.camera_uuid,
                        len(devices),
                        getattr(devices[0], "device_uuid", None),
                    )

                device = devices[0]
                if getattr(device, "device_uuid", None) is not None:
                    device_uuids.add(device.device_uuid)

                if active is None:
                    continue

                schedule_state = await self.resolve_runtime_schedule(
                    db,
                    cam=cam,
                    cfg_json=cfg_json,
                    cfg_timezone=getattr(getattr(cam, "channel_configuration", None), "timezone", None),
                    site_cache=site_schedule_cache,
                )
                vcc = build_video_channel_config(
                    cam,
                    device_uuid=device.device_uuid,
                    device_url=device.device_url,
                    schedule_state=schedule_state,
                    cfg_json=cfg_json,
                    default_request_timeout_s=self._state.default_request_timeout_s,
                )
                await active.edit_channel(VideoChannel(config=vcc))
                refreshed += 1

        return {
            "refreshed": refreshed,
            "skipped": skipped,
            "device_uuids": sorted(device_uuids, key=str),
        }
