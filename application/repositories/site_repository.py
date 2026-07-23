# application/repositories/site_repository.py
"""
Site + SiteSettings + SiteDevice persistence.

Owns the `sites`, `site_settings` and `site_devices` tables. Camera queries
belong to ChannelRepository; Device queries belong to DeviceRepository.

Transaction policy: ordinary methods never commit (caller owns the
transaction). The site-graph deletion helpers are the exception: they take a
session *factory* and manage their own per-batch transactions because they
must stream through very large tables.
"""

import logging
import uuid
from datetime import time as dt_time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from application.channels.channel_config import VideoChannelConfig
from application.dtos import SiteCreateDTO, SiteSettingsUpsertDTO, SiteUpdateDTO
from application.repositories._helpers import as_uuid as _as_uuid, normalize_uuid_list
from core.database_orm import (
    Camera,
    ChannelConfiguration,
    Notification,
    NotificationEmail,
    PipelineCamera,
    Site,
    SiteDevice,
    SiteSettings,
    VideoRecord,
)

logger = logging.getLogger(__name__)


class SiteRepository:
    """All persistence for Site, SiteSettings and the SiteDevice link table."""

    # ------------------------------------------------------------------
    # Site reads
    # ------------------------------------------------------------------
    async def get_site(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        raise_if_missing: bool = True,
    ) -> Optional[Site]:
        """Fetch a single active site.

        Scoping precedence (any combination may be supplied):
          * `org_id`     - restrict to the owning organization.
          * `site_uuids` - a member's allow-list of accessible sites
            (None = no restriction; empty list = no access).
          * `user_id`    - legacy creator/owner filter (internal callers).
        """
        if site_uuids is not None and not site_uuids:
            site = None
        else:
            stmt = select(Site).where(
                Site.site_uuid == _as_uuid(site_uuid), Site.is_deleted == False
            )
            if user_id is not None:
                stmt = stmt.where(Site.user_id == int(user_id))
            if org_id is not None:
                stmt = stmt.where(Site.org_id == int(org_id))
            if site_uuids is not None:
                stmt = stmt.where(Site.site_uuid.in_([_as_uuid(s) for s in site_uuids]))
            site = (await db.execute(stmt)).scalar_one_or_none()
        if site is None and raise_if_missing:
            raise HTTPException(status_code=404, detail="Site not found")
        return site

    async def get_sites(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        allow_unscoped: bool = False,
    ) -> List[Site]:
        if not allow_unscoped and user_id is None and org_id is None and site_uuids is None:
            raise HTTPException(status_code=400, detail="A scoping filter is required")
        if site_uuids is not None and not site_uuids:
            return []
        stmt = select(Site).where(Site.is_deleted == False)
        if user_id is not None:
            stmt = stmt.where(Site.user_id == int(user_id))
        if org_id is not None:
            stmt = stmt.where(Site.org_id == int(org_id))
        if site_uuids is not None:
            stmt = stmt.where(Site.site_uuid.in_([_as_uuid(s) for s in site_uuids]))
        stmt = stmt.order_by(Site.created_at.desc())
        return (await db.execute(stmt)).scalars().all()

    async def list_site_uuids(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
    ) -> List[uuid.UUID]:
        """Return just the site uuids in scope (active sites only)."""
        if site_uuids is not None and not site_uuids:
            return []
        stmt = select(Site.site_uuid).where(Site.is_deleted == False)
        if user_id is not None:
            stmt = stmt.where(Site.user_id == int(user_id))
        if org_id is not None:
            stmt = stmt.where(Site.org_id == int(org_id))
        if site_uuids is not None:
            stmt = stmt.where(Site.site_uuid.in_([_as_uuid(s) for s in site_uuids]))
        rows = (await db.execute(stmt)).scalars().all()
        return list(rows)

    # ------------------------------------------------------------------
    # Site writes
    # ------------------------------------------------------------------
    async def create_site(self, db: AsyncSession, *, dto: SiteCreateDTO) -> Site:
        """Insert a new Site from a `SiteCreateDTO`. Flush only; caller commits."""
        site = Site(
            org_id=int(dto.org_id),
            user_id=int(dto.user_id) if dto.user_id is not None else None,
            created_by=int(dto.created_by) if dto.created_by is not None else None,
            name=dto.name,
            address=dto.address,
            timezone=dto.timezone or "UTC",
            site_code=dto.site_code,
        )
        db.add(site)
        await db.flush()
        return site

    async def update_site(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        dto: SiteUpdateDTO,
    ) -> int:
        """Update the Site columns set on `dto`. Returns affected row count."""
        values = dto.model_dump(exclude_unset=True)
        if not values:
            return 0
        result = await db.execute(
            update(Site)
            .where(Site.site_uuid == _as_uuid(site_uuid))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def soft_delete_site(self, db: AsyncSession, *, site_uuid: uuid.UUID) -> int:
        """Mark a site as deleted without removing rows. Returns affected count."""
        result = await db.execute(
            update(Site)
            .where(Site.site_uuid == _as_uuid(site_uuid))
            .values(is_deleted=True)
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    # ------------------------------------------------------------------
    # SiteSettings
    # ------------------------------------------------------------------
    async def get_site_settings(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: Optional[int] = None,
    ) -> Optional[SiteSettings]:
        stmt = select(SiteSettings).where(SiteSettings.site_uuid == _as_uuid(site_uuid))
        return (await db.execute(stmt)).scalar_one_or_none()

    async def upsert_site_settings(
        self,
        db: AsyncSession,
        *,
        dto: SiteSettingsUpsertDTO,
    ) -> SiteSettings:
        """
        Persist one SiteSettings row per site from a `SiteSettingsUpsertDTO`.

        The real multi-day schedule lives in config["schedule"]; the scalar
        columns store a representative window for backward compatibility.
        """
        user_id = dto.user_id
        site_uuid = dto.site_uuid
        config = dto.config
        day_of_week = dto.day_of_week
        start_time = dto.start_time
        end_time = dto.end_time
        is_enabled = dto.is_enabled

        row = await self.get_site_settings(db, site_uuid=site_uuid, user_id=user_id)

        merged_config = dict(config or {})
        normalized_schedule = VideoChannelConfig.normalize_schedule(merged_config.get("schedule"))
        if not normalized_schedule:
            normalized_schedule = VideoChannelConfig.normalize_schedule(
                [
                    {
                        "day_of_week": day_of_week or [6, 0, 1, 2, 3, 4, 5],
                        "start_time": (start_time or dt_time(0, 0, 0)).strftime("%H:%M:%S"),
                        "end_time": (end_time or dt_time(23, 59, 59)).strftime("%H:%M:%S"),
                        "is_enabled": bool(is_enabled),
                    }
                ]
            ) or VideoChannelConfig.default_schedule()

        merged_config["schedule"] = normalized_schedule

        representative = normalized_schedule[0]
        representative_day = int(representative.get("day_of_week", 6))
        resolved_start = dt_time.fromisoformat(str(representative.get("start_time") or "00:00:00"))
        resolved_end = dt_time.fromisoformat(str(representative.get("end_time") or "23:59:59"))

        if resolved_start >= resolved_end:
            resolved_start = dt_time(0, 0, 0)
            resolved_end = dt_time(23, 59, 59)

        if row is None:
            row = SiteSettings(
                user_id=int(user_id),
                site_uuid=_as_uuid(site_uuid),
                config=merged_config,
                day_of_week=representative_day,
                start_time=resolved_start,
                end_time=resolved_end,
                is_enabled=bool(is_enabled),
            )
            db.add(row)
            await db.flush()
            return row

        row.config = merged_config
        row.day_of_week = representative_day
        row.start_time = resolved_start
        row.end_time = resolved_end
        row.is_enabled = bool(is_enabled)

        await db.flush()
        return row

    # ------------------------------------------------------------------
    # SiteDevice link table
    # ------------------------------------------------------------------
    async def site_device_exists(
        self, db: AsyncSession, *, site_uuid: uuid.UUID, device_uuid: uuid.UUID
    ) -> bool:
        row = (
            await db.execute(
                select(SiteDevice.id).where(
                    SiteDevice.site_uuid == _as_uuid(site_uuid),
                    SiteDevice.device_uuid == _as_uuid(device_uuid),
                )
            )
        ).scalar_one_or_none()
        return row is not None

    async def add_device_to_site(
        self, db: AsyncSession, *, site_uuid: uuid.UUID, device_uuid: uuid.UUID
    ) -> SiteDevice:
        """Link a device to a site (idempotent). Flush only; caller commits."""
        existing = (
            await db.execute(
                select(SiteDevice).where(
                    SiteDevice.site_uuid == _as_uuid(site_uuid),
                    SiteDevice.device_uuid == _as_uuid(device_uuid),
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        row = SiteDevice(site_uuid=_as_uuid(site_uuid), device_uuid=_as_uuid(device_uuid))
        db.add(row)
        await db.flush()
        return row

    async def remove_device_from_site(
        self, db: AsyncSession, *, site_uuid: uuid.UUID, device_uuid: uuid.UUID
    ) -> int:
        result = await db.execute(
            delete(SiteDevice)
            .where(
                SiteDevice.site_uuid == _as_uuid(site_uuid),
                SiteDevice.device_uuid == _as_uuid(device_uuid),
            )
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    # ------------------------------------------------------------------
    # Site-graph deletion
    # ------------------------------------------------------------------
    async def delete_site_graph(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> None:
        """
        Delete a site and every row that hangs off it, in one transaction.

        Use delete_site_graph_batched() for large sites — this variant is kept
        for small sites and tests. Flush only; caller commits.
        """
        normalized = normalize_uuid_list(camera_uuids)
        sid = _as_uuid(site_uuid)

        await db.execute(delete(Notification).where(Notification.site_uuid == sid))
        await db.execute(delete(NotificationEmail).where(NotificationEmail.site_uuid == sid))
        await db.execute(delete(SiteSettings).where(SiteSettings.site_uuid == sid))
        await db.execute(delete(SiteDevice).where(SiteDevice.site_uuid == sid))

        if normalized:
            await db.execute(
                delete(PipelineCamera).where(PipelineCamera.camera_uuid.in_(normalized))
            )
            await db.execute(
                delete(ChannelConfiguration).where(
                    ChannelConfiguration.camera_uuid.in_(normalized)
                )
            )
            await db.execute(
                delete(VideoRecord).where(VideoRecord.camera_uuid.in_(normalized))
            )

        await db.execute(delete(Camera).where(Camera.site_uuid == sid))
        await db.execute(delete(Site).where(Site.site_uuid == sid))
        await db.flush()

    async def delete_site_graph_batched(
        self,
        session_factory,  # callable returning an AsyncSession context manager
        *,
        site_uuid: uuid.UUID,
        camera_uuids: Optional[List[uuid.UUID]] = None,
        batch_size: int = 2000,
        extract_alert_key_fn=None,
        extract_clip_keys_fn=None,
        keep_site_row: bool = False,
    ) -> tuple[dict, list[str], list[str]]:
        """
        Foreground fast-delete of the core site graph (settings, device links,
        camera relationships, cameras, the site row). The heavy tables
        (Notification / VideoRecord) are deliberately left for a background
        sweep so the HTTP response stays fast.

        Returns: (stats_dict, alert_blob_keys, clip_blob_keys)
        """
        normalized = normalize_uuid_list(camera_uuids)
        sid = _as_uuid(site_uuid)

        stats = {"notifications": 0, "videos": 0, "cameras": len(normalized)}
        alert_blob_keys: list[str] = []
        clip_blob_keys: list[str] = []

        # Phase 1: small tables — site settings + device links.
        s1 = await self._fast_delete(session_factory, SiteSettings, SiteSettings.site_uuid == sid)
        s2 = await self._fast_delete(session_factory, SiteDevice, SiteDevice.site_uuid == sid)
        logger.info("[Site Delete] Phase 1: %s site settings, %s site-device links", s1, s2)

        # Phase 2: camera relationship tables.
        if normalized:
            pc = await self._fast_delete(
                session_factory, PipelineCamera, PipelineCamera.camera_uuid.in_(normalized)
            )
            cc = await self._fast_delete(
                session_factory, ChannelConfiguration,
                ChannelConfiguration.camera_uuid.in_(normalized),
            )
            logger.info("[Site Delete] Phase 2: %s pipeline-camera links, %s channel configs", pc, cc)

        # Phase 3: cameras (CASCADE-deletes VideoRecords, SET NULLs Notification.camera_uuid).
        camera_count = await self._fast_delete(session_factory, Camera, Camera.site_uuid == sid)
        logger.info("[Site Delete] Phase 3: %s cameras", camera_count)

        # Phase 4: notification emails.
        ne = await self._fast_delete(
            session_factory, NotificationEmail, NotificationEmail.site_uuid == sid
        )
        logger.info("[Site Delete] Phase 4: %s notification emails", ne)

        # Phase 5: the site row itself (CASCADE-deletes remaining Notifications).
        if not keep_site_row:
            sc = await self._fast_delete(session_factory, Site, Site.site_uuid == sid)
            logger.info("[Site Delete] Phase 5: %s site rows", sc)
        else:
            logger.info("[Site Delete] Phase 5: keeping site row for caller cleanup")

        logger.info("[Site Delete] FOREGROUND COMPLETE")
        return stats, alert_blob_keys, clip_blob_keys

    # ------------------------------------------------------------------
    # Deletion internals
    # ------------------------------------------------------------------
    async def _batch_delete(
        self,
        session_factory,
        *,
        table,
        where_clause,
        batch_size: int = 2000,
        label: str = "records",
        extract_col=None,
        extract_alert_fn=None,
        extract_clip_fn=None,
        alert_keys_out=None,
        clip_keys_out=None,
    ) -> int:
        """
        Delete a large table in batches, each in its own transaction, to avoid
        long locks and big memory spikes. Optionally extracts blob keys.
        """
        total_deleted = 0
        batch_num = 0
        pk = table.id

        while True:
            async with session_factory() as session:
                try:
                    if extract_col is not None:
                        stmt = select(pk, extract_col).where(where_clause).limit(batch_size)
                        rows = (await session.execute(stmt)).all()
                        batch_ids = [r[0] for r in rows]

                        for r in rows:
                            col_val = r[1]
                            if not col_val:
                                continue
                            if extract_alert_fn or extract_clip_fn:
                                if extract_alert_fn and alert_keys_out is not None:
                                    k = extract_alert_fn(col_val)
                                    if k:
                                        alert_keys_out.append(k)
                                if extract_clip_fn and clip_keys_out is not None:
                                    k_list = extract_clip_fn(col_val)
                                    if k_list:
                                        clip_keys_out.extend(k_list)
                            elif isinstance(col_val, str) and clip_keys_out is not None:
                                k = col_val.strip()
                                if k:
                                    clip_keys_out.append(k)
                    else:
                        id_stmt = select(pk).where(where_clause).limit(batch_size)
                        batch_ids = (await session.execute(id_stmt)).scalars().all()

                    if not batch_ids:
                        break

                    result = await session.execute(delete(table).where(pk.in_(batch_ids)))
                    await session.commit()

                    deleted_in_batch = result.rowcount or 0
                    total_deleted += deleted_in_batch
                    batch_num += 1
                    logger.info(
                        "[Batch Delete] %s: batch #%s deleted %s, total=%s",
                        label, batch_num, deleted_in_batch, total_deleted,
                    )
                    if deleted_in_batch == 0:
                        break
                except Exception as e:
                    await session.rollback()
                    logger.error("[Batch Delete] %s: batch #%s failed: %s", label, batch_num, e)
                    raise

        return total_deleted

    async def _fast_delete(self, session_factory, table, where_clause) -> int:
        """Single-transaction delete for smaller tables."""
        async with session_factory() as session:
            result = await session.execute(delete(table).where(where_clause))
            await session.commit()
            return result.rowcount or 0

    # ------------------------------------------------------------------
    # User-account deletion helpers
    # ------------------------------------------------------------------
    async def list_video_record_keys_for_cameras(
        self,
        session_factory,
        *,
        camera_uuids: List[uuid.UUID],
    ) -> List[str]:
        """Return non-blank VideoRecord.storage_key values for the given cameras."""
        normalized = normalize_uuid_list(camera_uuids)
        if not normalized:
            return []
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(VideoRecord.storage_key).where(
                        VideoRecord.camera_uuid.in_(normalized)
                    )
                )
            ).scalars().all()
        return [k.strip() for k in rows if k and k.strip()]

    async def delete_cameras_for_user(
        self,
        session_factory,
        *,
        user_id: int,
        camera_uuids: List[uuid.UUID],
    ) -> None:
        """
        Foreground delete of a user's camera rows and their relationship rows
        (pipeline membership + channel configs). Each in its own transaction.
        """
        normalized = normalize_uuid_list(camera_uuids)
        if not normalized:
            return
        async with session_factory() as session:
            await session.execute(
                delete(PipelineCamera).where(PipelineCamera.camera_uuid.in_(normalized))
            )
            await session.execute(
                delete(ChannelConfiguration).where(
                    ChannelConfiguration.camera_uuid.in_(normalized)
                )
            )
            await session.execute(delete(Camera).where(Camera.user_id == int(user_id)))
            await session.commit()

    async def soft_delete_sites(
        self,
        session_factory,
        *,
        site_uuids: List[uuid.UUID],
    ) -> int:
        """Mark the given sites deleted so they leave queries immediately.

        Site-scoped rather than user-scoped: sites belong to an organization,
        so an account deletion may only touch the ones whose org is actually
        being dissolved.
        """
        normalized = normalize_uuid_list(site_uuids)
        if not normalized:
            return 0
        async with session_factory() as session:
            result = await session.execute(
                update(Site)
                .where(Site.site_uuid.in_(normalized))
                .values(is_deleted=True)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return result.rowcount or 0

    async def delete_sites(
        self,
        session_factory,
        *,
        site_uuids: List[uuid.UUID],
    ) -> int:
        """Hard-delete the given sites. Returns affected count."""
        normalized = normalize_uuid_list(site_uuids)
        if not normalized:
            return 0
        async with session_factory() as session:
            result = await session.execute(
                delete(Site).where(Site.site_uuid.in_(normalized))
            )
            await session.commit()
            return result.rowcount or 0

    async def soft_delete_sites_for_user(
        self,
        session_factory,
        *,
        user_id: int,
    ) -> int:
        """Mark every site owned by a user as deleted. Returns affected count."""
        async with session_factory() as session:
            result = await session.execute(
                update(Site)
                .where(Site.user_id == int(user_id))
                .values(is_deleted=True)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return result.rowcount or 0

    async def delete_sites_for_user(
        self,
        session_factory,
        *,
        user_id: int,
    ) -> int:
        """Hard-delete every site owned by a user. Returns affected count."""
        async with session_factory() as session:
            result = await session.execute(
                delete(Site).where(Site.user_id == int(user_id))
            )
            await session.commit()
            return result.rowcount or 0
