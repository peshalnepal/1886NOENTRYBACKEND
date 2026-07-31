"""Manager device reconcile loops + per-device retry logic.

Extracted from the former monolithic application/services/manager.py.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Union

from sqlalchemy.ext.asyncio import AsyncSession

from application.services.edgeinference import EdgeCameraInventoryError
from application.repositories.device_repository import normalize_device_url

from application.services.manager.helpers import (
    _edge_health_ready,
    _only_jetson_config,
)
from application.services.manager.types import EdgeDeviceUnavailableError
from application.services.manager.controllers._state import ManagerState
from application.services.manager.controllers.schedule import ScheduleResolver

logger = logging.getLogger(__name__)


class DeviceReconciler:
    def __init__(self, state: ManagerState, schedule_resolver: ScheduleResolver):
        self._state = state
        self._schedule_resolver = schedule_resolver

    def _as_uuid(self, v: Any, name: str) -> uuid.UUID:
        if isinstance(v, uuid.UUID):
            return v
        try:
            return uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid {name}: {v}") from e

    async def _call_with_timeout(self, coro: asyncio.coroutine, timeout_s: Optional[float] = None) -> Any:
        timeout = timeout_s or self._state.external_timeout_s
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"External service call timed out after {timeout}s") from e

    async def reconcile_devices_best_effort(
        self,
        *,
        user_id: int,
        device_uuids: List[Union[str, uuid.UUID]],
        org_id: Optional[int] = None,
    ) -> None:
        """
        Best-effort targeted reconcile for devices affected by a site schedule change.

        We explicitly allow removal here so cameras that become inactive due to schedule
        changes are removed from edge/WebRTC immediately instead of waiting for the next
        full reconcile cycle.
        """
        uid = int(user_id)
        targets: List[uuid.UUID] = []
        seen_device_keys: Set[str] = set()
        seen_urls: Set[str] = set()

        async with self._state.session_factory() as db:
            for raw_device_uuid in device_uuids or []:
                try:
                    device_uuid = self._as_uuid(raw_device_uuid, "device_uuid")
                except Exception:
                    logger.warning("Skipping invalid device UUID during best-effort reconcile: %r", raw_device_uuid)
                    continue

                device_key = str(device_uuid)
                if device_key in seen_device_keys:
                    continue
                seen_device_keys.add(device_key)

                try:
                    if org_id is not None:
                        dev = await self._state.device_repo.get_device(
                            db, device_uuid=device_uuid, org_id=int(org_id)
                        )
                    else:
                        dev = await self._state.device_repo.get_device(
                            db, device_uuid=device_uuid, user_id=uid
                        )
                    if dev is None:
                        raise ValueError(f"Device not found: {device_key}")
                except Exception:
                    logger.warning(
                        "Skipping missing device during best-effort reconcile device=%s",
                        device_key,
                        exc_info=True,
                    )
                    continue
                device_url = normalize_device_url(getattr(dev, "device_url", None))
                target_key = device_url or device_key
                if target_key in seen_urls:
                    continue
                seen_urls.add(target_key)
                targets.append(device_uuid)

        for device_uuid in targets:
            key = str(device_uuid)
            try:
                await self.reconcile_device_edge_simple(
                    device_uuid=device_uuid,
                    user_id=uid,
                    dry_run=False,
                    delete_unknown=True,
                )
            except Exception:
                logger.warning(
                    "Best-effort reconcile failed after site schedule update device=%s",
                    key,
                    exc_info=True,
                )

    async def _reconcile_device_edge_with_retry(
        self,
        *,
        device_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Wrapper around reconcile_device_edge_simple with exponential backoff retry.
        """
        last_exception = None
        for attempt in range(1, self._state.edge_retry_max_attempts + 1):
            try:
                return await self.reconcile_device_edge_simple(
                    device_uuid=device_uuid,
                    user_id=user_id,
                    dry_run=dry_run,
                    delete_unknown=delete_unknown,
                )
            except Exception as e:
                last_exception = e
                if attempt < self._state.edge_retry_max_attempts:
                    # Exponential backoff: base * (2 ^ (attempt-1))
                    wait_ms = self._state.edge_retry_base_ms * (2 ** (attempt - 1))
                    logger.warning(
                        "Device reconcile attempt %d/%d failed for %s, retrying in %dms: %s",
                        attempt,
                        self._state.edge_retry_max_attempts,
                        device_uuid,
                        wait_ms,
                        str(e),
                    )
                    await asyncio.sleep(wait_ms / 1000.0)
                else:
                    logger.error(
                        "Device reconcile failed after %d attempts for %s: %s",
                        attempt,
                        device_uuid,
                        str(e),
                    )
        raise last_exception or RuntimeError("Device reconcile failed")

    async def reconcile_device_edge_simple(
        self,
        *,
        device_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = True,
    ) -> Dict[str, List[str]]:
        uid = int(user_id) if user_id is not None else None
        async with self._state.session_factory() as db:
            dev = await self._state.device_repo.get_device(db, device_uuid=device_uuid, user_id=uid)
            if dev is None:
                raise ValueError(f"Device not found: {device_uuid}")
            
            raw_url = getattr(dev, "device_url", None)
            if not raw_url:
                raise ValueError(f"Device missing device_url: {device_uuid}")
            
            device_url = normalize_device_url(raw_url)
            if not device_url:
                raise ValueError(f"Device missing device_url: {device_uuid}")

            # 2. Find all logical "peer" devices sharing this exact physical URL.
            # Scoped to the owning org: cameras from another org must never be
            # provisioned onto this box just because the URLs collide.
            peer_devices = await self._state.device_repo.list_devices(
                db,
                device_url=device_url,
                org_id=getattr(dev, "org_id", None),
            )
            reconcile_device_uuids: List[uuid.UUID] = []
            seen_reconcile_devices: Set[str] = set()
            for peer in peer_devices:
                peer_uuid = getattr(peer, "device_uuid", None)
                if peer_uuid is None:
                    continue
                peer_key = str(peer_uuid)
                if peer_key in seen_reconcile_devices:
                    continue
                seen_reconcile_devices.add(peer_key)
                reconcile_device_uuids.append(peer_uuid)

            if not reconcile_device_uuids:
                reconcile_device_uuids = [device_uuid]

            cams = await self._state.channel_repo.list_cameras(
                db,
                device_uuids=reconcile_device_uuids,
                include_config=True,
            )
            
            site_schedule_cache: Dict[str, Dict[str, Any]] = {}
            desired_set: Set[str] = set()
            active_streams: Set[str] = set()
            for cam in cams:
                cfg = (
                    cam.channel_configuration.configuration or {}
                    if getattr(cam, "channel_configuration", None) and getattr(cam.channel_configuration, "configuration", None)
                    else {}
                )
                schedule_state = await self._schedule_resolver.resolve_runtime_schedule(
                    db,
                    cam=cam,
                    cfg_json=cfg,
                    cfg_timezone=getattr(getattr(cam, "channel_configuration", None), "timezone", None),
                    site_cache=site_schedule_cache,
                )
                # Detection is provisioned on the edge only when the camera is
                # detection-enabled AND the site is currently armed. "Armed" is the
                # schedule's active state, optionally overridden by a temporary
                # site arm/disarm that clears at the next schedule boundary. A
                # disarmed / off-schedule site has its cameras torn down here, so
                # the Jetson actually stops detecting.
                if bool(cam.is_detection_enabled) and bool(schedule_state.get("armed", True)):
                    desired_set.add(str(cam.camera_uuid))
                # Keep playback paths provisioned for enabled cameras regardless of
                # alert schedule state so live view remains stable.
                if bool(cam.is_enabled) and getattr(cam, "camera_code", None):
                    active_streams.add(str(cam.camera_code))

        # --- WebRTC stream provisioning (independent of edge device) ---
        # Always provision WHEP streams in MediaMTX so live view works even
        # when the Jetson edge device is unreachable.
        try:
            webrtc_list = await self._state.webrtc.list_webrtc_cameras()
        except Exception as e:
            logger.warning("Cannot reach WebRTC gateway during reconcile: %s", e)
            webrtc_list = []
        webrtc_set = {
            str(c.get("stream_key"))
            for c in webrtc_list
            if isinstance(c, dict) and c.get("stream_key")
        }
        known_streams = {str(c.camera_code) for c in cams if c.camera_code}
        to_add_stream = sorted(active_streams - webrtc_set)
        to_remove_stream = sorted((webrtc_set & known_streams) - active_streams)
        cams_by_code = {str(c.camera_code): c for c in cams if c.camera_code}

        webrtc_added: List[str] = []
        webrtc_errors: List[str] = []
        for cu in to_add_stream:
            cam = cams_by_code.get(cu)
            if cam is None:
                webrtc_errors.append(f"Camera not found in DB during WebRTC reconcile: {cu}")
                continue
            try:
                await self._state.webrtc.ensure_stream(stream_key=cu, source_url=cam.source_url)
                webrtc_added.append(str(cam.camera_uuid))
            except Exception as e:
                logger.warning("WebRTC ensure_stream failed during reconcile for %s: %s", cu, e)
                webrtc_errors.append(f"Failed to provision stream {cu}: {e}")

        webrtc_removed: List[str] = []
        if delete_unknown:
            for cu in to_remove_stream:
                try:
                    await self._state.webrtc.delete_stream(stream_key=cu)
                    removed_cam = cams_by_code.get(cu)
                    removed_uuid = str(removed_cam.camera_uuid) if removed_cam else cu
                    webrtc_removed.append(removed_uuid)
                except Exception as e:
                    logger.warning("WebRTC delete_stream failed during reconcile for %s: %s", cu, e)
                    webrtc_errors.append(f"Failed to remove stream {cu}: {e}")

        # --- Edge device reconcile ---
        # `device_url` stays the normalized value computed above; re-reading the
        # raw column here would send un-normalized URLs to the edge client and
        # desync it from the peer lookup that built `desired_set`.
        edge_warnings: List[str] = []
        try:
            edge_set = await self._call_with_timeout(
                self._state.edge.list_cameras(device_url=device_url),
                timeout_s=self._state.external_timeout_s
            )
        except EdgeCameraInventoryError as e:
            if _edge_health_ready(getattr(e, "health", None)):
                logger.warning(
                    "Edge camera inventory unavailable for %s during reconcile; continuing with add-only sync: %s",
                    device_url,
                    e,
                )
                edge_set = set()
                edge_warnings.append(
                    "Edge camera inventory was unavailable, so sync continued in add-only mode. "
                    "Existing unknown cameras on the edge device were not removed."
                )
            else:
                logger.warning(
                    "Cannot reach edge device %s during reconcile (%s): %s",
                    device_url,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                # Even though edge is unreachable, return WebRTC results so they aren't lost
                raise EdgeDeviceUnavailableError(device_url, e) from e
        except Exception as e:
            logger.warning(
                "Cannot reach edge device %s during reconcile (%s): %s",
                device_url,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise EdgeDeviceUnavailableError(device_url, e) from e
        to_add = sorted(desired_set - edge_set)
        to_remove = sorted(edge_set - desired_set)

        out: Dict[str, List[str]] = {
            "to_add": to_add,
            "to_remove": to_remove,
            "to_add_stream": to_add_stream,
            "to_remove_stream": to_remove_stream,
            "added": list(webrtc_added),
            "removed": list(webrtc_removed),
            "errors": list(webrtc_errors),
            "warnings": edge_warnings,
        }

        if dry_run:
            return out

        cams_by_uuid = {str(c.camera_uuid): c for c in cams}

        for cu in to_add:
            cam = cams_by_uuid.get(cu)
            if cam is None:
                out["errors"].append(f"Camera not found in DB during reconcile: {cu}")
                continue
            cfg = (cam.channel_configuration.configuration or {}) if cam.channel_configuration else {}
            payload = {
                **_only_jetson_config(cfg),
                # Explicit fields last so they always win over whatever is in channel config
                "camera_uuid": cu,
                "source_url": cam.source_url,
                "enabled":True,
                "detection_enabled": True,
                "notification_enabled": bool(cam.is_notification_enabled),
            }
            try:
                await self._call_with_timeout(
                    self._state.edge.upsert_camera(device_url=device_url, payload=payload),
                    timeout_s=self._state.external_timeout_s
                )
                out["added"].append(cu)
            except Exception as e:
                logger.warning("Edge upsert failed during reconcile for camera %s", cu, exc_info=True)
                out["errors"].append(f"Failed to add {cu}: {e}")

        if delete_unknown:
            for cu in to_remove:
                try:
                    await self._call_with_timeout(
                        self._state.edge.delete_camera(device_url=device_url, camera_uuid=cu),
                        timeout_s=self._state.external_timeout_s
                    )
                    out["removed"].append(cu)
                except Exception as e:
                    logger.warning("Edge delete failed during reconcile for camera %s", cu, exc_info=True)
                    out["errors"].append(f"Failed to remove {cu}: {e}")

        return out

    async def reconcile_all_devices_edge(
        self,
        *,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = False,
    ) -> Dict[str, Any]:
        """
        Reconcile every device for the user.
        Useful at startup/after edge reboot so Jetson gets re-hydrated from DB state.
        """
        uid = int(user_id) if user_id is not None else None
        async with self._state.session_factory() as db:
            device_rows = await self._state.device_repo.list_devices(
                db, user_id=uid, only_enabled=True
            )
        device_uuids: List[uuid.UUID] = []
        seen_targets: Set[str] = set()
        for dev in device_rows:
            du = getattr(dev, "device_uuid", None)
            if du is None:
                continue
            device_url = normalize_device_url(getattr(dev, "device_url", None))
            key = device_url or str(du)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            device_uuids.append(du)

        summary: Dict[str, Any] = {
            "user_id": uid,
            "device_count": len(device_uuids),
            "device_row_count": len(device_rows),
            "devices": {},
            "errors": [],
            "warnings": [],
        }

        for du in device_uuids:
            key = str(du)
            try:
                result = await self._reconcile_device_edge_with_retry(
                    device_uuid=du,
                    user_id=uid,
                    dry_run=dry_run,
                    delete_unknown=delete_unknown,
                )
                summary["devices"][key] = result
                for warning in result.get("warnings") or []:
                    summary["warnings"].append("{}: {}".format(key, warning))
            except Exception as e:
                logger.warning("Startup reconcile failed for device=%s", key, exc_info=True)
                summary["errors"].append("{}: {}".format(key, e))

        return summary
