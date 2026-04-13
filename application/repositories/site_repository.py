# application/repositories/site_repository.py

import uuid
from datetime import time as dt_time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from application.channels.channel_config import VideoChannelConfig
from core.database_orm import (
    Camera,
    CameraDevice,
    ChannelConfiguration,
    Device,
    Notification,
    NotificationEmail,
    PipelineCamera,
    Site,
    SiteDevice,
    SiteSettings,
    VideoRecord,
)

class SiteRepository:
    """
    Site-centric queries.

    Your ORM facts:
      - Camera.site_uuid -> FK to Site.site_uuid
      - Camera.devices is M:N via camera_devices
      - ChannelConfiguration is 1:1 via camera_uuid
    """

    async def list_cameras_by_site(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id:int,
        include_config: bool = True,
        include_device: bool = True,

        only_enabled: Optional[bool] = None,

    ) -> List[Camera]:
        """
        Returns all cameras linked to a site_uuid.

        include_config=True  -> eager-load Camera.channel_configuration
        include_device=True  -> eager-load Camera.devices (via camera_devices)
        only_enabled:
          - None  -> all
          - True  -> only Camera.is_enabled == True
          - False -> only Camera.is_enabled == False
        """
        opts = []

        if include_config:
            opts.append(selectinload(Camera.channel_configuration))

        if include_device:
            # loads Camera.devices (Device objects) using the secondary table camera_devices
            opts.append(selectinload(Camera.devices))

        stmt = select(Camera).where(Camera.site_uuid == site_uuid,Camera.user_id==user_id)

        if only_enabled is True:
            stmt = stmt.where(Camera.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(Camera.is_enabled.is_(False))

        if opts:
            stmt = stmt.options(*opts)

        stmt = stmt.order_by(Camera.created_at.asc())

        return (await db.execute(stmt)).scalars().all()
    
    async def list_camera_uuids_by_site(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: int,
        only_enabled: Optional[bool] = None,
    ) -> List[uuid.UUID]:
        stmt = select(Camera.camera_uuid).where(
            Camera.site_uuid == site_uuid,
            Camera.user_id == int(user_id),
        )

        if only_enabled is True:
            stmt = stmt.where(Camera.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(Camera.is_enabled.is_(False))

        stmt = stmt.order_by(Camera.created_at.asc())
        return (await db.execute(stmt)).scalars().all()
    
    async def list_cameras_with_device_details_by_site(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: int,
        only_enabled: Optional[bool] = None,
    ) -> List[dict]:
        stmt = (
            select(Camera, Device)
            .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid, isouter=True)
            .join(Device, Device.device_uuid == CameraDevice.device_uuid, isouter=True)
            .where(
                Camera.site_uuid == site_uuid,
                Camera.user_id == int(user_id),
            )
            .order_by(Camera.created_at.asc())
        )

        if only_enabled is True:
            stmt = stmt.where(Camera.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(Camera.is_enabled.is_(False))

        rows = (await db.execute(stmt)).all()

        out: List[dict] = []
        for cam, dev in rows:
            out.append(
                {
                    "camera_uuid": cam.camera_uuid,
                    "camera_code": cam.camera_code,
                    "site_uuid": cam.site_uuid,
                    "rtsp_url": cam.rtsp_url,
                    "webrtc_url": cam.webrtc_url,
                    "is_enabled": cam.is_enabled,
                    "is_detection_enabled": cam.is_detection_enabled,
                    "is_notification_enabled": cam.is_notification_enabled,
                    "roi": cam.roi,
                    "device_uuid": getattr(dev, "device_uuid", None),
                    "device_url": getattr(dev, "device_url", None),
                }
            )

        return out

    async def get_site(self,db:AsyncSession,*,site_uuid: uuid.UUID,user_id:int=None)->Optional[Site]:
        smt = select(Site).where(Site.site_uuid == site_uuid)
        if user_id is not None:
            smt = smt.where(Site.user_id == int(user_id))
        site = (await db.execute(smt)).scalar_one_or_none()
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        return site
    
    async def get_sites(self, db: AsyncSession, *, user_id: int) -> List[Site]:
        if user_id is None:
            raise HTTPException(status_code=400, detail="user_id is required")

        stmt = (
            select(Site)
            .where(Site.user_id == int(user_id))
            .order_by(Site.created_at.desc())
        )
        return (await db.execute(stmt)).scalars().all()
    
    async def get_site_settings(self,db: AsyncSession,*,site_uuid: uuid.UUID,user_id: int=None) -> Optional[SiteSettings]:
        smt = select(SiteSettings).where(
            
            SiteSettings.site_uuid == site_uuid,
        )
        if user_id:
            smt=smt.where(SiteSettings.user_id == int(user_id),)
        site_settings = (await db.execute(smt)).scalar_one_or_none()
        return site_settings

    async def upsert_site_settings(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
        config: Optional[Dict[str, Any]] = None,
        day_of_week: Optional[List[int]] = None,
        start_time: Optional[dt_time] = None,
        end_time: Optional[dt_time] = None,
        is_enabled: bool = True,
    ) -> SiteSettings:
        """
        Persist one SiteSettings row per site.

        Real multi-day schedule is stored in config["schedule"].
        Scalar columns are stored as representative values for backward compatibility.
        """
        row = await self.get_site_settings(
            db,
            site_uuid=site_uuid,
            user_id=user_id,
        )

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
                site_uuid=site_uuid,
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
    
    async def create_site(self,db: AsyncSession):
        pass

    async def delete_site_graph(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> None:
        """DEPRECATED: Use delete_site_graph_batched() for large sites."""
        normalized_camera_uuids: List[uuid.UUID] = []
        seen = set()
        for value in camera_uuids or []:
            parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
            key = str(parsed)
            if key in seen:
                continue
            seen.add(key)
            normalized_camera_uuids.append(parsed)

        # Delete site-owned rows explicitly instead of depending only on FK
        # cascades. This protects deployed databases that may predate newer
        # ON DELETE rules in the ORM metadata.
        await db.execute(
            delete(Notification).where(Notification.site_uuid == site_uuid)
        )
        await db.execute(
            delete(NotificationEmail).where(NotificationEmail.site_uuid == site_uuid)
        )
        await db.execute(
            delete(SiteSettings).where(SiteSettings.site_uuid == site_uuid)
        )
        await db.execute(
            delete(SiteDevice).where(SiteDevice.site_uuid == site_uuid)
        )

        if normalized_camera_uuids:
            await db.execute(
                delete(PipelineCamera).where(
                    PipelineCamera.camera_uuid.in_(normalized_camera_uuids)
                )
            )
            await db.execute(
                delete(CameraDevice).where(
                    CameraDevice.camera_uuid.in_(normalized_camera_uuids)
                )
            )
            await db.execute(
                delete(ChannelConfiguration).where(
                    ChannelConfiguration.camera_uuid.in_(normalized_camera_uuids)
                )
            )
            await db.execute(
                delete(VideoRecord).where(
                    VideoRecord.camera_uuid.in_(normalized_camera_uuids)
                )
            )

        await db.execute(delete(Camera).where(Camera.site_uuid == site_uuid))
        await db.execute(delete(Site).where(Site.site_uuid == site_uuid))

    async def delete_site_graph_batched(
        self,
        db,  # SessionFactory - returns AsyncSession
        *,
        site_uuid: uuid.UUID,
        camera_uuids: Optional[List[uuid.UUID]] = None,
        batch_size: int = 2000,
        extract_alert_key_fn = None,
        extract_clip_keys_fn = None,
    ) -> tuple[dict, list[str], list[str]]:
        """
        OPTIMIZED deletion with batching to avoid 502/503 errors on large sites.
        Extracts blob keys during deletion to avoid double-scanning.
        
        Note: The heavy tables (Notifications/VideoRecords) are passed to a background
        process from the route to make the HTTP response fast.
        
        Returns: (stats_dict, alert_blob_keys, clip_blob_keys)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        normalized_camera_uuids: List[uuid.UUID] = []
        seen = set()
        for value in camera_uuids or []:
            parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
            key = str(parsed)
            if key in seen:
                continue
            seen.add(key)
            normalized_camera_uuids.append(parsed)

        stats = {"notifications": 0, "videos": 0, "cameras": len(normalized_camera_uuids)}
        alert_blob_keys = []
        clip_blob_keys = []

        # Phase 1: Fast delete site settings and device links (small tables)
        site_settings_count = await self._fast_delete(db, SiteSettings, SiteSettings.site_uuid == site_uuid)
        site_device_count = await self._fast_delete(db, SiteDevice, SiteDevice.site_uuid == site_uuid)
        logger.info(f"[Site Delete] Phase 1: Deleted {site_settings_count} site settings, {site_device_count} site-device links")

        # Phase 2: Fast delete camera relationships
        if normalized_camera_uuids:
            logger.info(f"[Site Delete] Phase 2: Deleting camera relationships")
            pipeline_cam_count = await self._fast_delete(
                db,
                PipelineCamera,
                PipelineCamera.camera_uuid.in_(normalized_camera_uuids)
            )
            camera_device_count = await self._fast_delete(
                db,
                CameraDevice,
                CameraDevice.camera_uuid.in_(normalized_camera_uuids)
            )
            channel_config_count = await self._fast_delete(
                db,
                ChannelConfiguration,
                ChannelConfiguration.camera_uuid.in_(normalized_camera_uuids)
            )
            logger.info(
                f"[Site Delete] Deleted {pipeline_cam_count} pipeline-camera links, "
                f"{camera_device_count} camera-device links, {channel_config_count} channel configs"
            )

        # Phase 3: Fast delete cameras (so they disappear from frontend immediately)
        camera_count = await self._fast_delete(db, Camera, Camera.site_uuid == site_uuid)
        logger.info(f"[Site Delete] Phase 3: Deleted {camera_count} cameras")

        # Phase 4: Delete the site itself (so it disappears from frontend immediately)
        site_count = await self._fast_delete(db, Site, Site.site_uuid == site_uuid)
        logger.info(f"[Site Delete] Phase 4: Deleted {site_count} site rows")

        # Phase 5: Fast delete notification emails
        notification_email_count = await self._fast_delete(db, NotificationEmail, NotificationEmail.site_uuid == site_uuid)
        logger.info(f"[Site Delete] Phase 5: Deleted {notification_email_count} notification emails")

        logger.info(f"[Site Delete] FOREGROUND COMPLETE: Deleted core site graph (Settings, Relationships, Cameras, Site)")
        
        # We NO LONGER delete the massive Notification/VideoRecord tables here because 
        # it blocks the HTTP response for too long (15-20 seconds per 2000 rows).
        # This function handles the foreground fast deletes to make the UI snappy.
        
        return stats, alert_blob_keys, clip_blob_keys

    async def _batch_delete(
        self,
        session_factory,  # callable returning AsyncSession
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
        Delete large tables in batches to avoid locking and memory issues.
        Each batch uses a separate transaction. Optionally extracts blob keys.

        Uses a subquery approach because SQLAlchemy's delete() does not
        support .limit(). The pattern is:
            DELETE FROM table WHERE id IN (SELECT id FROM table WHERE … LIMIT N)
        """
        import logging
        logger = logging.getLogger(__name__)
        total_deleted = 0
        batch_num = 0

        pk = table.id  # assumes every table has an `id` primary key column

        while True:
            async with session_factory() as session:
                try:
                    # Select a batch of IDs (and optionally extract columns)
                    if extract_col is not None:
                        stmt = select(pk, extract_col).where(where_clause).limit(batch_size)
                        rows = (await session.execute(stmt)).all()
                        batch_ids = [r[0] for r in rows]

                        for r in rows:
                            col_val = r[1]
                            if not col_val:
                                continue

                            if extract_alert_fn or extract_clip_fn:
                                # Dictionary payload (Notification)
                                if extract_alert_fn and alert_keys_out is not None:
                                    k = extract_alert_fn(col_val)
                                    if k:
                                        alert_keys_out.append(k)
                                if extract_clip_fn and clip_keys_out is not None:
                                    k_list = extract_clip_fn(col_val)
                                    if k_list:
                                        clip_keys_out.extend(k_list)
                            elif isinstance(col_val, str) and clip_keys_out is not None:
                                # Plain string key (VideoRecord)
                                k = col_val.strip()
                                if k:
                                    clip_keys_out.append(k)
                    else:
                        id_stmt = select(pk).where(where_clause).limit(batch_size)
                        batch_ids = (await session.execute(id_stmt)).scalars().all()

                    if not batch_ids:
                        break

                    delete_stmt = delete(table).where(pk.in_(batch_ids))
                    result = await session.execute(delete_stmt)
                    await session.commit()

                    deleted_in_batch = result.rowcount or 0
                    total_deleted += deleted_in_batch
                    batch_num += 1

                    logger.info(
                        f"[Batch Delete] {label}: batch #{batch_num} deleted {deleted_in_batch}, "
                        f"total={total_deleted}"
                    )

                    if deleted_in_batch == 0:
                        break

                except Exception as e:
                    await session.rollback()
                    logger.error(f"[Batch Delete] {label}: batch #{batch_num} failed: {e}")
                    raise

        return total_deleted

    async def _fast_delete(
        self,
        session_factory,  # callable returning AsyncSession context manager
        table,
        where_clause,
    ) -> int:
        """Single-transaction delete for smaller tables."""
        async with session_factory() as session:
            delete_stmt = delete(table).where(where_clause)
            result = await session.execute(delete_stmt)
            await session.commit()
            return result.rowcount or 0
