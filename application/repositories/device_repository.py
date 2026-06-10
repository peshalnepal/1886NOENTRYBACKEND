# application/repositories/device_repository.py
"""
Device-centric persistence.

Owns the `devices` table only. Camera queries live in ChannelRepository,
site/site-device-link queries live in SiteRepository.

Transaction policy: this repository never commits. It only flushes so the
caller (service / route) owns the transaction boundary.
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import DeviceCreateDTO, DeviceUpdateDTO
from core.database_orm import Camera, Device, SiteDevice


def _as_uuid(value: Any) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


class DeviceRepository:
    """All persistence for the Device model."""

    def _filtered_devices_query(
        self,
        *,
        device_uuid: Optional[uuid.UUID] = None,
        device_uuids: Optional[List[uuid.UUID]] = None,
        site_uuid: Optional[uuid.UUID] = None,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        device_url: Optional[str] = None,
        only_enabled: Optional[bool] = None,
        order_by_recent: bool = False,
    ):
        """Build a SELECT(Device) statement shared by get_device/list_devices."""
        stmt = select(Device)

        if site_uuid is not None:
            # Device <-> Site is M:N via site_devices.
            stmt = stmt.join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid).where(
                SiteDevice.site_uuid == _as_uuid(site_uuid)
            )

        if camera_uuid is not None:
            stmt = stmt.join(Camera, Camera.device_uuid == Device.device_uuid).where(
                Camera.camera_uuid == _as_uuid(camera_uuid)
            )

        if device_uuid is not None:
            stmt = stmt.where(Device.device_uuid == _as_uuid(device_uuid))

        if device_uuids is not None:
            clean = [du for du in (_as_uuid(d) for d in device_uuids) if du is not None]
            if not clean:
                return None
            stmt = stmt.where(Device.device_uuid.in_(clean))

        if user_id is not None:
            stmt = stmt.where(Device.user_id == int(user_id))

        if org_id is not None:
            stmt = stmt.where(Device.org_id == int(org_id))

        if device_url:
            stmt = stmt.where(Device.device_url == device_url)

        if only_enabled is True:
            stmt = stmt.where(Device.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(Device.is_enabled.is_(False))

        if order_by_recent:
            if site_uuid is not None:
                stmt = stmt.order_by(
                    Device.is_enabled.desc(),
                    SiteDevice.created_at.desc(),
                    Device.created_at.desc(),
                )
            else:
                stmt = stmt.order_by(Device.created_at.desc())

        return stmt

    async def get_device(
        self,
        db: AsyncSession,
        *,
        device_uuid: Optional[uuid.UUID] = None,
        site_uuid: Optional[uuid.UUID] = None,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        device_url: Optional[str] = None,
        only_enabled: Optional[bool] = None,
    ) -> Optional[Device]:
        """Return the first Device matching any combination of the given filters."""
        stmt = self._filtered_devices_query(
            device_uuid=device_uuid,
            site_uuid=site_uuid,
            user_id=user_id,
            org_id=org_id,
            camera_uuid=camera_uuid,
            device_url=device_url,
            only_enabled=only_enabled,
        )
        if stmt is None:
            return None
        return (await db.execute(stmt)).scalars().first()

    async def list_devices(
        self,
        db: AsyncSession,
        *,
        device_uuids: Optional[List[uuid.UUID]] = None,
        site_uuid: Optional[uuid.UUID] = None,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        device_url: Optional[str] = None,
        only_enabled: Optional[bool] = None,
        order_by_recent: bool = False,
    ) -> List[Device]:
        """Return every Device matching any combination of the given filters."""
        stmt = self._filtered_devices_query(
            device_uuids=device_uuids,
            site_uuid=site_uuid,
            user_id=user_id,
            org_id=org_id,
            camera_uuid=camera_uuid,
            device_url=device_url,
            only_enabled=only_enabled,
            order_by_recent=order_by_recent,
        )
        if stmt is None:
            return []
        return (await db.execute(stmt)).scalars().all()

    async def device_exists(self, db: AsyncSession, *, device_uuid: uuid.UUID) -> bool:
        row = (
            await db.execute(
                select(Device.device_uuid).where(Device.device_uuid == _as_uuid(device_uuid))
            )
        ).scalar_one_or_none()
        return row is not None

    async def ensure_device_exists(self, db: AsyncSession, device_uuid: uuid.UUID) -> None:
        if not await self.device_exists(db, device_uuid=device_uuid):
            raise ValueError(f"Device not found: {device_uuid}")

    async def create_device(self, db: AsyncSession, *, dto: DeviceCreateDTO) -> Device:
        """Insert a new Device from a `DeviceCreateDTO`. Flush only; caller commits."""
        user_id=int(dto.user_id) if dto.user_id is not None else None
        device = Device(
            org_id=int(dto.org_id),
            user_id=user_id,
            created_by=user_id,
            device_url=dto.device_url,
            name=dto.name,
            device_code=dto.device_code,
            is_enabled=bool(dto.is_enabled),
        )
        db.add(device)
        await db.flush()
        return device

    async def update_device(
        self,
        db: AsyncSession,
        *,
        device_uuid: uuid.UUID,
        dto: DeviceUpdateDTO,
    ) -> int:
        """Update the Device columns set on `dto`. Returns affected row count."""
        values: Dict[str, Any] = dto.model_dump(exclude_unset=True)
        if not values:
            return 0
        result = await db.execute(
            update(Device)
            .where(Device.device_uuid == _as_uuid(device_uuid))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def delete_device(self, db: AsyncSession, *, device_uuid: uuid.UUID) -> int:
        """Delete a Device by uuid. Returns affected row count."""
        result = await db.execute(
            delete(Device)
            .where(Device.device_uuid == _as_uuid(device_uuid))
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0
