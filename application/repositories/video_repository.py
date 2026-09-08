"""
VideoRecord persistence.

Owns the `video_record` table (per-camera recording clips).

Transaction policy: never commits, only flushes. Caller owns the transaction.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories._helpers import as_uuid as _as_uuid
from core.database_orm import VideoRecord


class VideoRepository:
    """All persistence for the VideoRecord model."""

    def _conditions(
        self,
        *,
        ids: Optional[List[int]] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        camera_uuids: Optional[List[uuid.UUID]] = None,
        created_before: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> list:
        conds: list = []
        if ids is not None:
            conds.append(VideoRecord.id.in_([int(i) for i in ids]))
        if camera_uuid is not None:
            conds.append(VideoRecord.camera_uuid == _as_uuid(camera_uuid))
        if camera_uuids is not None:
            clean = [c for c in (_as_uuid(c) for c in camera_uuids) if c is not None]
            conds.append(VideoRecord.camera_uuid.in_(clean) if clean else VideoRecord.id.is_(None))
        if created_before is not None:
            conds.append(VideoRecord.created_at < created_before)
        if status is not None:
            conds.append(VideoRecord.status == status)
        return conds

    async def list_video_records(
        self,
        db: AsyncSession,
        *,
        ids: Optional[List[int]] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        camera_uuids: Optional[List[uuid.UUID]] = None,
        created_before: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        order_asc: bool = True,
    ) -> List[VideoRecord]:
        """Return VideoRecord rows matching any combination of filters."""
        conds = self._conditions(
            ids=ids, camera_uuid=camera_uuid, camera_uuids=camera_uuids,
            created_before=created_before, status=status,
        )
        stmt = select(VideoRecord)
        if conds:
            stmt = stmt.where(and_(*conds))
        stmt = stmt.order_by(VideoRecord.id.asc() if order_asc else VideoRecord.id.desc())
        if limit is not None:
            stmt = stmt.limit(int(limit))
        return (await db.execute(stmt)).scalars().all()

    async def list_storage_keys(
        self,
        db: AsyncSession,
        *,
        camera_uuid: Optional[uuid.UUID] = None,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> List[str]:
        """Return non-empty storage_key values for the matching records."""
        conds = self._conditions(camera_uuid=camera_uuid, camera_uuids=camera_uuids)
        stmt = select(VideoRecord.storage_key)
        if conds:
            stmt = stmt.where(and_(*conds))
        rows = (await db.execute(stmt)).scalars().all()
        return [str(k).strip() for k in rows if k and str(k).strip()]

    async def delete_video_records(
        self,
        db: AsyncSession,
        *,
        ids: Optional[List[int]] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        camera_uuids: Optional[List[uuid.UUID]] = None,
        created_before: Optional[datetime] = None,
    ) -> int:
        """Hard-delete VideoRecord rows matching the given filters. Returns count."""
        conds = self._conditions(
            ids=ids, camera_uuid=camera_uuid, camera_uuids=camera_uuids,
            created_before=created_before,
        )
        if not conds:
            raise ValueError("delete_video_records requires at least one filter")
        result = await db.execute(
            delete(VideoRecord)
            .where(and_(*conds))
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0
