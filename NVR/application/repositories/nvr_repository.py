"""Persistence for the tower's single NVR row. Never commits; the caller owns the transaction."""

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dtos import NvrDTO
from ...core.database_orm import NvrDevice


class NvrRepository:
    async def get(self, db: AsyncSession) -> Optional[NvrDevice]:
        return (await db.execute(select(NvrDevice).where(NvrDevice.singleton == 1))).scalar_one_or_none()

    async def upsert(self, db: AsyncSession, dto: NvrDTO, now: datetime) -> NvrDevice:
        """Create or update the one NVR. A replaced recorder reuses the row (same id)."""
        row = await self.get(db)
        if row is None:
            row = NvrDevice(singleton=1, created_at=now, first_seen_at=None)
            db.add(row)
        row.host = dto.host
        row.rtsp_port = dto.rtsp_port
        row.http_port = dto.http_port
        row.origin = dto.origin
        row.channels_spec = dto.channels_spec
        row.updated_at = now
        await db.flush()
        return row

    def mark_seen(self, row: NvrDevice, now: datetime) -> dict:
        recovered = row.first_seen_at is not None and not row.is_present
        if row.first_seen_at is None:
            row.first_seen_at = now
        row.is_present = True
        row.consecutive_misses = 0
        row.alerted = False
        row.missing_since = None
        row.last_seen_at = now
        row.updated_at = now
        return {"recovered": recovered}

    def mark_missed(self, row: NvrDevice, miss_threshold: int, now: datetime) -> bool:
        """Age the NVR one sweep. True only on the sweep that crosses the threshold."""
        row.consecutive_misses = int(row.consecutive_misses or 0) + 1
        row.updated_at = now
        if row.missing_since is None:
            row.missing_since = now
        if row.consecutive_misses < miss_threshold:
            return False
        row.is_present = False
        if row.alerted:
            return False
        row.alerted = True
        return True

    async def clear(self, db: AsyncSession) -> bool:
        """Drop the NVR row (NVR_HOST emptied). Cameras keep their rows."""
        row = await self.get(db)
        if row is None:
            return False
        await db.delete(row)
        await db.flush()
        return True
