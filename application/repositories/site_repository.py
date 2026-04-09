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
