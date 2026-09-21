"""Adoption of edge-discovered cameras into the cloud as real Camera rows.

The Jetson is the source of truth for which cameras physically exist: its
discovery sweep finds Hikvision cameras on the LAN, mints a real UUID for each
and starts decoding immediately. Reconcile must therefore never delete an
edge-discovered camera — it adopts it instead.

A camera with no `Camera` row has no site, no org, no camera_code, and so no
MediaMTX stream, no live view, no notification recipients and no schedule, and
can never enter `desired_set`. Adoption closes that loop by creating the Camera
+ ChannelConfiguration rows through the SAME path a manually-created camera
takes (`Manager.update_pipeline` -> `ChannelController.add_channel`), so an
adopted camera is indistinguishable from a hand-added one afterwards.

Identity, not URL, is the dedupe key: the edge's roster identity
("serial:…" > "mac:…" > "ip:…") is stored in the camera's channel configuration
under `discovery_identity`. That is what makes re-adoption idempotent across
DHCP moves — the same physical camera keeps its identity when its IP changes, so
a second sweep links to the existing row instead of creating a duplicate.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.inventory_repository import InventoryRepository
from core.database_orm import Site, SiteDevice
from domain.events import ChannelCreateEvent
from application.services.manager.controllers._state import ManagerState
from application.services.manager.helpers import _camera_config_json

logger = logging.getLogger(__name__)


def _host_of(url: Optional[str]) -> Optional[str]:
    """Hostname of a stream URL, ignoring scheme, credentials, port and path.

    Mirrors `DiscoveryService._host_of` on the edge so both sides agree on what
    "the same camera" means when only a URL is available to compare.
    """
    if not url or not isinstance(url, str):
        return None
    authority = url.split("://", 1)[-1].split("/", 1)[0]
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    return authority.split(":", 1)[0].strip().lower() or None


def _is_channel_identity(identity: Optional[str]) -> bool:
    """True when this identity names one input of a multi-channel recorder.

    The edge appends "#<channel>" for anything behind an NVR (see
    `identity_of` in tensort/discovery.py), because every channel is reached
    through the recorder's single address and so cannot be told apart by host
    alone. Such an identity is already exact, and must never be widened to a
    host comparison: doing so matches a *different* channel on the same NVR.
    """
    return "#" in str(identity or "")


class CameraAdopter:
    """Turns edge discovery roster entries into cloud Camera rows."""

    def __init__(
        self,
        state: ManagerState,
        update_pipeline: Callable[..., Awaitable[Any]],
    ):
        self._state = state
        # Injected rather than reached for: `update_pipeline` lives on the
        # PipelineController, which already owns pipeline-id resolution and
        # cold-start retry. Injecting it keeps adoption on the exact code path
        # the manual create route uses, with no circular import.
        self._update_pipeline = update_pipeline

    @staticmethod
    def _roster_entries(discovery_report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adoptable roster rows: present, with a usable stream URL.

        Absent cameras are skipped — adopting one would create a Camera row for
        hardware that is no longer on the network, and the missing-camera alert
        already covers that case.
        """
        if not isinstance(discovery_report, dict):
            return []

        out: List[Dict[str, Any]] = []
        for entry in discovery_report.get("roster") or []:
            if not isinstance(entry, dict):
                continue
            if not entry.get("is_present"):
                continue
            if not entry.get("source_url"):
                continue
            out.append(entry)
        return out

    @staticmethod
    def _display_name(entry: Dict[str, Any]) -> str:
        """A human label for the adopted camera, best information first."""
        for key in ("device_name", "model"):
            value = str(entry.get(key) or "").strip()
            if value:
                ip = str(entry.get("ip_address") or "").strip()
                return f"{value} ({ip})" if ip else value

        for key in ("ip_address", "serial_number", "identity"):
            value = str(entry.get(key) or "").strip()
            if value:
                return f"Camera {value}"
        return "Discovered camera"

    async def _sites_for_device(
        self, db: AsyncSession, *, device_uuid: uuid.UUID
    ) -> List[uuid.UUID]:
        """Active site uuids this device is linked to, via `site_devices`."""
        rows = (
            await db.execute(
                select(SiteDevice.site_uuid)
                .join(Site, Site.site_uuid == SiteDevice.site_uuid)
                .where(SiteDevice.device_uuid == device_uuid)
                .where(Site.is_deleted == False)  # noqa: E712 — SQL, not Python
            )
        ).scalars().all()
        return list(dict.fromkeys(rows))

    async def _existing_index(
        self, db: AsyncSession, *, device_uuids: List[uuid.UUID]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Index the cameras already registered for these devices.

        Returns (by_identity, by_host). `by_identity` is the authoritative
        match; `by_host` catches a camera that was added by hand before
        discovery ever ran, so adoption links to it rather than creating a
        second row for the same physical stream.
        """
        cams = await self._state.channel_repo.list_cameras(
            db, device_uuids=device_uuids, include_config=True
        )

        by_identity: Dict[str, Any] = {}
        by_host: Dict[str, Any] = {}

        for cam in cams:
            cfg = _camera_config_json(cam)

            identity = str(cfg.get("discovery_identity") or "").strip()
            if identity:
                by_identity[identity] = cam

            # A row that already carries a channel identity is exactly
            # identified, so it must not also be reachable by host: several NVR
            # channels share one host, and whichever happened to be indexed
            # first would then absorb the others.
            if _is_channel_identity(identity):
                continue

            host = _host_of(getattr(cam, "source_url", None))
            # First writer wins: if two cameras somehow share a host, the
            # identity match above is the one that should decide.
            if host and host not in by_host:
                by_host[host] = cam

        return by_identity, by_host

    async def adopt_discovered_cameras(
        self,
        *,
        device_uuid: uuid.UUID,
        device_url: str,
        user_id: int,
        discovery_report: Optional[Dict[str, Any]],
        peer_device_uuids: Optional[List[uuid.UUID]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Create Camera rows for every present, unregistered discovered camera.

        Returns a summary dict: `adopted`, `linked`, `repointed`, `skipped`,
        `errors`. An already-registered camera is `linked`, and additionally
        `repointed` when the sweep found it at a new address.

        Best-effort by contract: a failure on one camera is recorded and the
        sweep moves on, because a single bad roster entry must never block the
        rest of a site coming online.
        """
        out: Dict[str, Any] = {
            "adopted": [],
            "linked": [],
            "repointed": [],
            "skipped": [],
            "errors": [],
        }

        entries = self._roster_entries(discovery_report)
        if not entries:
            return out

        targets = list(peer_device_uuids or [device_uuid])

        async with self._state.session_factory() as db:
            # The device must belong to exactly one site for adoption to know
            # where to file the camera. Zero sites means the device was never
            # linked; several means the choice is genuinely ambiguous and a
            # human has to make it.
            site_uuids = await self._sites_for_device(db, device_uuid=device_uuid)
            if not site_uuids:
                out["errors"].append(
                    "Device is not linked to any site, so discovered cameras cannot be "
                    "filed. Link the device to a site first."
                )
                return out
            if len(site_uuids) > 1:
                out["errors"].append(
                    "Device is linked to {} sites; discovered cameras were not adopted "
                    "because the target site is ambiguous.".format(len(site_uuids))
                )
                return out

            site_uuid = site_uuids[0]
            by_identity, by_host = await self._existing_index(db, device_uuids=targets)
            # A camera the user removed must stay removed. The existing index
            # is built from live Camera rows, so a deleted camera is invisible
            # to it and would be re-created on every sweep; inventory is what
            # remembers the decision after the row is gone.
            blocked = await InventoryRepository().blocked_identities(
                db, device_uuids=targets
            )

        for entry in entries:
            identity = str(entry.get("identity") or "").strip()
            source_url = str(entry.get("source_url") or "").strip()
            host = _host_of(source_url)

            if identity and identity in blocked and identity not in by_identity:
                out["skipped"].append(
                    {"identity": identity, "source_url": source_url, "reason": "removed_by_user"}
                )
                continue

            existing = by_identity.get(identity) if identity else None
            if existing is None and host and not _is_channel_identity(identity):
                existing = by_host.get(host)

            if existing is not None:
                # Already registered. Backfill the provenance if this row was
                # created by hand, so the next sweep matches on identity
                # instead of falling back to the fragile host comparison, and
                # follow the camera if DHCP moved it to a new address.
                out["linked"].append(str(existing.camera_uuid))
                if (identity or source_url) and not dry_run:
                    try:
                        repointed = await self._sync_existing_camera(
                            camera_uuid=existing.camera_uuid,
                            entry=entry,
                            source_url=source_url,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to sync discovery state for camera %s: %s",
                            existing.camera_uuid, exc, exc_info=True,
                        )
                    else:
                        if repointed:
                            out["repointed"].append(
                                {
                                    "camera_uuid": str(existing.camera_uuid),
                                    "identity": identity,
                                    "source_url": source_url,
                                }
                            )
                continue

            if dry_run:
                out["skipped"].append(
                    {"identity": identity, "source_url": source_url, "reason": "dry_run"}
                )
                continue

            try:
                camera_uuid = await self._create_camera(
                    entry=entry,
                    site_uuid=site_uuid,
                    device_uuid=device_uuid,
                    user_id=user_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to adopt discovered camera %s on device %s",
                    identity or source_url, device_url, exc_info=True,
                )
                out["errors"].append(
                    "Failed to adopt {}: {}".format(identity or source_url, exc)
                )
                continue

            out["adopted"].append(str(camera_uuid))
            # Keep the in-memory index current so two roster rows that resolve
            # to the same host inside one sweep cannot both be adopted.
            if identity:
                by_identity[identity] = _AdoptedRef(camera_uuid)
            # Channels of one NVR all share a host, so indexing them here would
            # make the first adopted channel swallow every later channel in the
            # same sweep -- the first import of an 8-channel recorder would
            # yield a single camera. Their "#<channel>" identities already
            # dedupe them above.
            if host and not _is_channel_identity(identity):
                by_host.setdefault(host, _AdoptedRef(camera_uuid))

            logger.info(
                "Adopted discovered camera %s (%s) into site %s as %s",
                entry.get("ip_address"), identity or "no identity", site_uuid, camera_uuid,
            )

        return out

    async def create_from_inventory(
        self,
        *,
        site_uuid: uuid.UUID,
        device_uuid: uuid.UUID,
        user_id: int,
        entry: Dict[str, Any],
    ) -> uuid.UUID:
        """Create one camera from an inventory row into an explicit site.

        The site is passed in rather than derived, which is the difference from
        `adopt_discovered_cameras`: that path re-resolves the site per sweep
        and cannot act on a camera the user is adding by hand.
        """
        return await self._create_camera(
            entry=entry,
            site_uuid=site_uuid,
            device_uuid=device_uuid,
            user_id=user_id,
        )

    async def _create_camera(
        self,
        *,
        entry: Dict[str, Any],
        site_uuid: uuid.UUID,
        device_uuid: uuid.UUID,
        user_id: int,
    ) -> uuid.UUID:
        """Create one Camera through the normal channel-create path.

        Goes through `update_pipeline` — the same call the manual create route
        makes — rather than writing rows directly, so an adopted camera gets
        exactly what a hand-added one does: camera_code, WHEP stream,
        ChannelConfiguration, pipeline membership and an edge upsert. It also
        inherits that path's pipeline-id resolution and cold-start retry, which
        a bespoke implementation here would have to duplicate and keep in sync.
        """
        # Reuse the UUID the edge already minted. The Jetson is decoding under
        # this ID right now, so keeping it means adoption does not restart the
        # stream or orphan the edge's camera_configs row.
        camera_uuid = entry.get("camera_uuid")
        cam_uuid = uuid.UUID(str(camera_uuid)) if camera_uuid else uuid.uuid4()

        configs: Dict[str, Any] = {
            "camera_uuid": cam_uuid,
            "site_uuid": site_uuid,
            "device_uuid": device_uuid,
            "user_id": int(user_id),
            "source_url": entry.get("source_url"),
            "name": self._display_name(entry),
            "location": entry.get("device_name") or None,
            "enabled": True,
            "detection_enabled": True,
            "notification_enabled": True,
            # Provenance — mirrors what the edge writes into camera_configs so
            # both sides can tell an auto-discovered camera from a pushed one.
            "discovered": True,
            "discovery_identity": entry.get("identity"),
            "discovery_ip": entry.get("ip_address"),
            "discovery_mac": entry.get("mac_address"),
            "discovery_model": entry.get("model"),
            "discovery_serial": entry.get("serial_number"),
            "discovery_name": entry.get("device_name"),
            "discovered_at": datetime.now(timezone.utc).isoformat(),
        }
        configs = {k: v for k, v in configs.items() if v is not None}

        ev = ChannelCreateEvent(
            channel_id=None,
            configs=configs,
            created_at=datetime.now(timezone.utc),
        )

        result = await self._update_pipeline(
            None,
            [ev],
            user_id=int(user_id),
            camera_code_prefix="cam",
        )

        if not result or not result.cameras:
            raise RuntimeError("Channel create returned no camera")
        return result.cameras[0].camera_uuid

    async def _sync_existing_camera(
        self,
        *,
        camera_uuid: uuid.UUID,
        entry: Dict[str, Any],
        source_url: str,
    ) -> bool:
        """Reconcile one already-registered camera with this sweep's findings.

        Two things can be stale on a camera the cloud already knows about:

        1. Its discovery provenance, when the row was created by hand before
           discovery ever ran. Stamping it makes the next sweep match on
           identity instead of the fragile host comparison.
        2. Its `source_url`, when DHCP moved the camera. The edge repoints its
           own pipeline the moment it notices, but that patch is local to the
           Jetson — without the update below the cloud keeps the old address,
           MediaMTX keeps streaming from an IP nothing answers on, and the next
           reconcile pushes the stale URL back down over the edge's correct one.

        Returns True when the camera was repointed, so the caller can report it.
        """
        patch = {
            "discovered": True,
            "discovery_identity": entry.get("identity"),
            "discovery_ip": entry.get("ip_address"),
            "discovery_mac": entry.get("mac_address"),
            "discovery_model": entry.get("model"),
            "discovery_serial": entry.get("serial_number"),
            "discovery_name": entry.get("device_name"),
        }
        patch = {k: v for k, v in patch.items() if v is not None}

        repointed = False
        camera_code: Optional[str] = None

        async with self._state.session_factory() as db:
            full = await self._state.channel_repo.get_camera_full(db, camera_uuid=camera_uuid)
            if not full:
                return False
            cam, chan_cfg, _pid = full

            # Only the host is compared: the edge rebuilds the whole URL from
            # its own configured credentials and stream path every sweep, so a
            # byte-comparison would repoint endlessly on any cloud-side edit to
            # the channel or stream number without the address having moved.
            old_host = _host_of(getattr(cam, "source_url", None))
            new_host = _host_of(source_url)
            if source_url and new_host and new_host != old_host:
                cam.source_url = source_url
                camera_code = str(getattr(cam, "camera_code", "") or "").strip() or None
                repointed = True

            cfg_changed = False
            if chan_cfg is not None and patch:
                merged = dict(chan_cfg.configuration or {})
                if any(merged.get(k) != v for k, v in patch.items()):
                    merged.update(patch)
                    # configuration is a plain JSON column, so an in-place dict
                    # mutation would not be detected; reassigning marks it dirty.
                    chan_cfg.configuration = merged
                    cfg_changed = True

            if not repointed and not cfg_changed:
                return False  # already current — skip the write entirely

            await db.commit()

        if repointed:
            logger.info(
                "Repointed camera %s to %s after discovery reported an address change",
                camera_uuid, source_url,
            )
            # Best-effort: the DB is authoritative and the next reconcile pushes
            # the new URL to the edge regardless, so a gateway hiccup here must
            # not fail the sweep. Without a camera_code there is no MediaMTX
            # path to patch.
            if camera_code:
                try:
                    await self._state.webrtc.update_stream(
                        stream_key=camera_code, source_url=source_url
                    )
                except Exception:
                    logger.warning(
                        "Failed to repoint MediaMTX stream %s for camera %s; "
                        "live view keeps the old address until the next reconcile",
                        camera_code, camera_uuid, exc_info=True,
                    )

        return repointed


class _AdoptedRef:
    """Minimal stand-in for a Camera row, used only to de-dupe within a sweep."""

    __slots__ = ("camera_uuid",)

    def __init__(self, camera_uuid: uuid.UUID):
        self.camera_uuid = camera_uuid
