# application/repositories/site_repository.py

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database_orm import Camera, ChannelConfiguration, Device, CameraDevice,Site,SiteSettings

from fastapi import APIRouter, Depends, HTTPException, status

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
        only_enabled: Optional[bool] = None,
    ) -> List[uuid.UUID]:
        """
        Lightweight version: returns only camera_uuid list for the site.
        """
        stmt = select(Camera.camera_uuid).where(Camera.site_uuid == site_uuid)

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
        only_enabled: Optional[bool] = None,
    ) -> List[dict]:
        """
        If you want a ready-to-serialize structure (camera + its single device),
        this returns a list of dicts.

        Assumption (your new ORM rule): one device per camera enforced by uq_camera_one_device.
        """
        stmt = (
            select(Camera, Device)
            .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid, isouter=True)
            .join(Device, Device.device_uuid == CameraDevice.device_uuid, isouter=True)
            .where(Camera.site_uuid == site_uuid)
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
        if user_id:
            smt=smt.where(Site.user_id==int(user_id))
        site = (await db.execute(smt)).scalar_one_or_none()
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        return site
    
    async def get_sites(self,db:AsyncSession,*,user_id:int)->Optional[List[Site]]:
        if not user_id:
            raise HTTPException(status_code=404, detail="No User not found")
        smt = select(Site).where(Site.user_id == user_id).order_by(Site.created_at.desc())
        sites = (await db.execute(smt)).scalars().all()
        if type(sites)!=list:
            
            return list(sites)
        return sites
    
    async def get_site_settings(self,db: AsyncSession,*,site_uuid: uuid.UUID,user_id: int=None) -> Optional[SiteSettings]:
        smt = select(SiteSettings).where(
            
            SiteSettings.site_uuid == site_uuid,
        )
        if user_id:
            smt=smt.where(SiteSettings.user_id == int(user_id),)
        site_settings = (await db.execute(smt)).scalar_one_or_none()

        if not site_settings:
            raise HTTPException(status_code=404, detail="Site setting doesn't exist")
        return site_settings

    async def create_site(self,db: AsyncSession):
        pass