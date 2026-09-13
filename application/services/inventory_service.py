"""
Camera inventory orchestration.

Owns the decisions inventory exists to make: what a device reported, whether a
camera may be added to a site, and what happens when a user takes one out.
Routes call this rather than the repository so add/remove logic lives in one
place.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import InventoryEntryDTO
from application.repositories.inventory_repository import InventoryRepository
from core.database_orm import CameraInventory, INVENTORY_ADDED

logger = logging.getLogger(__name__)

# Roster key -> DTO field. An explicit allowlist, not a passthrough: a device
# must not be able to set a site, camera code or ROI by including them.
_ROSTER_FIELDS = {
    "ip_address": "ip_address",
    "mac_address": "mac_address",
    "serial_number": "serial_number",
    "model": "model",
    "firmware": "firmware",
    "device_name": "device_name",
    "source_url": "source_url",
    "camera_uuid": "edge_camera_uuid",
    "is_present": "is_present",
    "consecutive_misses": "consecutive_misses",
    "first_seen_at": "first_seen_at",
    "last_seen_at": "last_seen_at",
    "missing_since": "missing_since",
}

_TIMESTAMP_FIELDS = ("first_seen_at", "last_seen_at", "missing_since")


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. A bad timestamp must not drop the camera."""
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def redact_source(source_url: Optional[str]) -> Optional[str]:
    """Strip credentials from a source URL so it is safe to display.

    `rtsp://user:pass@10.0.0.9/s` becomes `rtsp://10.0.0.9/s`. Inventory has to
    store the real URL to connect, but a camera list must never hand out a
    password.
    """
    if not source_url:
        return None
    try:
        parts = urlsplit(str(source_url))
    except ValueError:
        return None
    if not parts.hostname:
        return str(source_url)

    authority = parts.hostname
    if parts.port:
        authority = "%s:%d" % (authority, parts.port)
    return urlunsplit((parts.scheme, authority, parts.path, parts.query, parts.fragment))


def entries_from_roster(report: Any) -> List[InventoryEntryDTO]:
    """Convert an edge discovery report into inventory entries.

    Accepts either the full report or a bare roster list. Rows without an
    identity are skipped: identity is the key, and a row without one cannot be
    matched to anything on a later sweep.
    """
    if isinstance(report, dict):
        rows = report.get("roster") or []
    elif isinstance(report, list):
        rows = report
    else:
        return []

    entries: List[InventoryEntryDTO] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        identity = str(row.get("identity") or "").strip()
        if not identity:
            continue

        values: Dict[str, Any] = {"discovery_identity": identity}
        for roster_key, field in _ROSTER_FIELDS.items():
            value = row.get(roster_key)
            if field in _TIMESTAMP_FIELDS:
                value = parse_timestamp(value)
            if value is not None:
                values[field] = value

        entries.append(InventoryEntryDTO(**values))

    return entries


class InventoryService:
    """Add, remove and refresh decisions for reported cameras."""

    def __init__(self, manager: Optional[Any] = None):
        self._manager = manager
        self._repo = InventoryRepository()

    async def refresh_from_device(
        self, db: AsyncSession, *, device_uuid: uuid.UUID, device_url: str
    ) -> Dict[str, Any]:
        """Ask a device what it can see and store the answer.

        Reads the edge's cached sweep, so it is cheap enough to call on every
        link. A device that cannot be reached leaves stored inventory exactly
        as it was — a camera the cloud cannot currently ask about has not gone
        away.
        """
        if self._manager is None:
            return {"fetched": False, "created": 0, "updated": 0, "marked_offline": 0}

        report = await self._manager.edge.fetch_discovery_report(device_url=device_url)
        if report is None:
            return {"fetched": False, "created": 0, "updated": 0, "marked_offline": 0}

        entries = entries_from_roster(report)
        created, updated = await self._repo.upsert_reported(
            db, device_uuid=device_uuid, entries=entries
        )
        marked_offline = await self._repo.mark_absent(
            db,
            device_uuid=device_uuid,
            present_identities={entry.discovery_identity for entry in entries},
        )
        return {
            "fetched": True,
            "created": created,
            "updated": updated,
            "marked_offline": marked_offline,
        }

    async def add_to_site(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        device_uuid: uuid.UUID,
        discovery_identities: Sequence[str],
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """Create a channel for each identity and file it under the site.

        Per-camera results rather than all-or-nothing: one camera failing to
        start must not hide the ones that worked.
        """
        results: List[Dict[str, Any]] = []

        for identity in discovery_identities:
            row = await self._repo.get(
                db, device_uuid=device_uuid, discovery_identity=identity
            )
            if row is None:
                results.append(self._result(identity, False, "not_in_inventory"))
                continue
            if row.state == INVENTORY_ADDED and row.camera_uuid:
                results.append(
                    self._result(identity, True, "already_added", row.camera_uuid)
                )
                continue
            if not row.source_url:
                results.append(self._result(identity, False, "no_source_url"))
                continue

            try:
                camera_uuid = await self._manager.create_camera_from_inventory(
                    site_uuid=site_uuid,
                    device_uuid=device_uuid,
                    user_id=user_id,
                    entry=self._adoption_entry(row),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to add inventory camera %s to site %s",
                    identity, site_uuid, exc_info=True,
                )
                results.append(self._result(identity, False, "failed", detail=str(exc)))
                continue

            await self._repo.mark_added(
                db, row=row, site_uuid=site_uuid, camera_uuid=camera_uuid
            )
            results.append(self._result(identity, True, "added", camera_uuid))

        return results

    async def remove_camera(
        self, db: AsyncSession, *, camera_uuid: uuid.UUID
    ) -> Optional[CameraInventory]:
        """Record that a camera was taken out of its site.

        Called while the camera row is still readable, before the channel is
        torn down. Returns None for a camera no device reported — a manually
        created camera has no inventory row and needs none.
        """
        rows = await self._repo.rows_for_cameras(db, camera_uuids=[camera_uuid])
        row = rows.get(str(camera_uuid))
        if row is None:
            return None
        return await self._repo.mark_removed(db, row=row)

    async def describe_for_site(
        self, db: AsyncSession, *, site_uuid: uuid.UUID
    ) -> List[Dict[str, Any]]:
        rows = await self._repo.list_for_site(db, site_uuid=site_uuid)
        return [self.describe(row) for row in rows]

    async def describe_for_device(
        self, db: AsyncSession, *, device_uuid: uuid.UUID
    ) -> List[Dict[str, Any]]:
        rows = await self._repo.list_for_device(db, device_uuid=device_uuid)
        return [self.describe(row) for row in rows]

    @staticmethod
    def describe(row: CameraInventory) -> Dict[str, Any]:
        """One inventory row as the API exposes it, without credentials."""
        return {
            "discovery_identity": row.discovery_identity,
            "device_uuid": row.device_uuid,
            "site_uuid": row.site_uuid,
            "camera_uuid": row.camera_uuid,
            "state": row.state,
            "display_name": row.device_name or row.model or row.discovery_identity,
            "ip_address": row.ip_address,
            "mac_address": row.mac_address,
            "serial_number": row.serial_number,
            "model": row.model,
            "firmware": row.firmware,
            "source": redact_source(row.source_url),
            "is_present": bool(row.is_present),
            "first_seen_at": row.first_seen_at,
            "last_seen_at": row.last_seen_at,
            "missing_since": row.missing_since,
        }

    @staticmethod
    def _adoption_entry(row: CameraInventory) -> Dict[str, Any]:
        """The roster-shaped dict the adopter builds a channel from."""
        return {
            "identity": row.discovery_identity,
            "source_url": row.source_url,
            "ip_address": row.ip_address,
            "device_name": row.device_name,
            "model": row.model,
            "serial_number": row.serial_number,
        }

    @staticmethod
    def _result(
        identity: str,
        ok: bool,
        outcome: str,
        camera_uuid: Optional[uuid.UUID] = None,
        detail: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "discovery_identity": identity,
            "ok": ok,
            "outcome": outcome,
            "camera_uuid": camera_uuid,
            "detail": detail,
        }
