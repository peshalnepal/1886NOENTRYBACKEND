"""
Camera inventory persistence.

Owns the `camera_inventory` table only: what each device reports it can see,
and whether the cloud has filed it into a site. Camera rows themselves belong
to ChannelRepository.

Transaction policy: this repository never commits. It only flushes so the
caller (service / route) owns the transaction boundary.
"""

import uuid
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import InventoryEntryDTO
from application.repositories._helpers import as_uuid as _as_uuid
from core.database_orm import (
    CameraInventory,
    INVENTORY_ADDED,
    INVENTORY_AVAILABLE,
    INVENTORY_REMOVED,
    SiteDevice,
)

# Columns a device owns. Everything else on the row is a cloud decision and is
# never overwritten by a report, which is what stops a sweep from undoing a
# removal.
_REPORTED_FIELDS = (
    "ip_address",
    "mac_address",
    "serial_number",
    "model",
    "firmware",
    "device_name",
    "source_url",
    "edge_camera_uuid",
    "is_present",
    "consecutive_misses",
    "first_seen_at",
    "last_seen_at",
    "missing_since",
)


class InventoryRepository:
    """All persistence for the CameraInventory model."""

    async def upsert_reported(
        self,
        db: AsyncSession,
        *,
        device_uuid: uuid.UUID,
        entries: Sequence[InventoryEntryDTO],
    ) -> Tuple[int, int]:
        """Store what a device reported. Returns (created, updated).

        Only device-owned columns are written. A row that already exists keeps
        its `state`, `site_uuid` and `camera_uuid`, so a camera the user
        removed stays removed no matter how often the device re-reports it.
        """
        device = _as_uuid(device_uuid)
        if not entries:
            return (0, 0)

        existing = {
            row.discovery_identity: row
            for row in await self.list_for_device(db, device_uuid=device)
        }

        created = updated = 0
        for entry in entries:
            row = existing.get(entry.discovery_identity)
            if row is None:
                row = CameraInventory(
                    device_uuid=device,
                    discovery_identity=entry.discovery_identity,
                    state=INVENTORY_AVAILABLE,
                )
                db.add(row)
                created += 1
            else:
                updated += 1

            for field in _REPORTED_FIELDS:
                value = getattr(entry, field, None)
                if value is not None:
                    setattr(row, field, value)

        await db.flush()
        return (created, updated)

    async def mark_absent(
        self,
        db: AsyncSession,
        *,
        device_uuid: uuid.UUID,
        present_identities: Set[str],
    ) -> int:
        """Flag rows the latest report did not mention.

        Never deletes: a camera that stopped answering is still a camera, and
        its inventory row is what a later re-add depends on.
        """
        marked = 0
        for row in await self.list_for_device(db, device_uuid=device_uuid):
            if row.discovery_identity not in present_identities and row.is_present:
                row.is_present = False
                marked += 1

        await db.flush()
        return marked

    async def list_for_device(
        self, db: AsyncSession, *, device_uuid: uuid.UUID
    ) -> List[CameraInventory]:
        result = await db.execute(
            select(CameraInventory)
            .where(CameraInventory.device_uuid == _as_uuid(device_uuid))
            .order_by(CameraInventory.discovery_identity)
        )
        return list(result.scalars().all())

    async def list_for_site(
        self, db: AsyncSession, *, site_uuid: uuid.UUID
    ) -> List[CameraInventory]:
        """Inventory from every device linked to this site.

        Scoped through `site_devices` rather than `camera_inventory.site_uuid`,
        so cameras that are merely available — or were removed — still show up
        alongside the ones already added.
        """
        result = await db.execute(
            select(CameraInventory)
            .join(SiteDevice, SiteDevice.device_uuid == CameraInventory.device_uuid)
            .where(SiteDevice.site_uuid == _as_uuid(site_uuid))
            .order_by(CameraInventory.discovery_identity)
        )
        return list(result.scalars().all())

    async def get(
        self, db: AsyncSession, *, device_uuid: uuid.UUID, discovery_identity: str
    ) -> Optional[CameraInventory]:
        result = await db.execute(
            select(CameraInventory).where(
                CameraInventory.device_uuid == _as_uuid(device_uuid),
                CameraInventory.discovery_identity == discovery_identity,
            )
        )
        return result.scalar_one_or_none()

    async def blocked_identities(
        self, db: AsyncSession, *, device_uuids: Sequence[uuid.UUID]
    ) -> Set[str]:
        """Identities adoption must not create a camera for.

        A removed camera is deliberately excluded; an added one already has a
        row. Everything else is fair game.
        """
        clean = [_as_uuid(value) for value in device_uuids if value is not None]
        if not clean:
            return set()

        result = await db.execute(
            select(CameraInventory.discovery_identity).where(
                CameraInventory.device_uuid.in_(clean),
                CameraInventory.state.in_([INVENTORY_REMOVED, INVENTORY_ADDED]),
            )
        )
        return {row for row in result.scalars().all() if row}

    async def mark_added(
        self,
        db: AsyncSession,
        *,
        row: CameraInventory,
        site_uuid: uuid.UUID,
        camera_uuid: uuid.UUID,
    ) -> CameraInventory:
        row.state = INVENTORY_ADDED
        row.site_uuid = _as_uuid(site_uuid)
        row.camera_uuid = _as_uuid(camera_uuid)
        await db.flush()
        return row

    async def mark_removed(
        self, db: AsyncSession, *, row: CameraInventory
    ) -> CameraInventory:
        """Take a camera out of its site without forgetting it exists.

        Clearing `site_uuid` and `camera_uuid` is what makes the row stop
        pointing at a channel that is about to be deleted; `state` is what
        stops the next sweep re-adding it.
        """
        row.state = INVENTORY_REMOVED
        row.site_uuid = None
        row.camera_uuid = None
        await db.flush()
        return row

    async def rows_for_cameras(
        self, db: AsyncSession, *, camera_uuids: Sequence[uuid.UUID]
    ) -> Dict[str, CameraInventory]:
        """Inventory rows for these cameras, keyed by camera_uuid string."""
        clean = [_as_uuid(value) for value in camera_uuids if value is not None]
        if not clean:
            return {}

        result = await db.execute(
            select(CameraInventory).where(CameraInventory.camera_uuid.in_(clean))
        )
        return {str(row.camera_uuid): row for row in result.scalars().all()}
