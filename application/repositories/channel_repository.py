# application/repositories/channel_repository.py

import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import logging

from core.database_orm import (
    Camera,
    CameraDevice,
    ChannelConfiguration,
    Device,
    Pipeline,
    PipelineCamera,
    Site,
)

logger = logging.getLogger(__name__)

ChannelConfigLike = Union[Dict[str, Any], Any]


class ChannelRepository:
    """
    Camera + ChannelConfiguration + PipelineCamera consistency.

    ✅ New rule:
      - each camera MUST have exactly 1 device assigned (camera_devices has 1 row per camera_uuid)
      - no "primary device"
    """

    async def upsert_camera_from_channel_config(
        self,
        db: AsyncSession,
        *,
        pipeline_id: uuid.UUID,
        channel_config: ChannelConfigLike,
        user_id: Optional[int] = None,
        cam_uuid: Optional[uuid.UUID] = None,
        camera_code: Optional[str] = None,
        site_uuid: Optional[uuid.UUID] = None,
        webrtc_url: Optional[str] = None,
        rtsp_url: Optional[str] = None,
        device_uuid: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        location: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Tuple[Camera, Dict[str, Any], Optional[str]]:

        await self._ensure_pipeline_exists(db, pipeline_id)

        d = self._to_dict(channel_config)

        raw_cam_uuid = cam_uuid or d.get("camera_uuid") or d.get("camera_id")
        cam_uuid = None
        if raw_cam_uuid:
            cam_uuid = raw_cam_uuid if isinstance(raw_cam_uuid, uuid.UUID) else uuid.UUID(str(raw_cam_uuid))

        if device_uuid is None:
            device_uuid = d.get("device_uuid")
        if device_uuid is not None:
            device_uuid = device_uuid if isinstance(device_uuid, uuid.UUID) else uuid.UUID(str(device_uuid))

        rtsp_url = rtsp_url or d.get("rtsp_url")
        enabled = d.get("enabled", d.get("is_enabled", True))
        detection_enabled = d.get("detection_enabled", d.get("is_detection_enabled", True))
        notification_enabled = d.get("notification_enabled", d.get("is_notification_enabled", True))
        has_roi = "roi" in d
        roi = d.get("roi")

        if not rtsp_url:
            raise ValueError("channel_config.rtsp_url is required")

        cam: Optional[Camera] = None
        if cam_uuid:
            cam = await self._get_camera_by_uuid(db, cam_uuid)

        if cam:
            cam.rtsp_url = rtsp_url
            cam.is_enabled = bool(enabled)
            cam.is_detection_enabled = bool(detection_enabled)
            cam.is_notification_enabled = bool(notification_enabled)
            if has_roi:
                cam.roi = roi

            if webrtc_url is not None:
                if cam.webrtc_url is None:
                    cam.webrtc_url = webrtc_url
                elif cam.webrtc_url != webrtc_url:
                    logger.warning(
                        "Ignoring webrtc_url change for camera=%s (immutable). old=%s new=%s",
                        str(cam.camera_uuid),
                        cam.webrtc_url,
                        webrtc_url,
                    )

            if site_uuid is not None:
                if cam.site_uuid is None:
                    cam.site_uuid = site_uuid
                elif cam.site_uuid != site_uuid:
                    raise ValueError("Changing site_uuid for an existing camera is not supported")

            if camera_code is not None:
                cam.camera_code = camera_code
            if name is not None:
                cam.name = name
            if location is not None:
                cam.location = location

            await db.flush()
            if device_uuid is not None:
                await self._set_camera_device(db, camera_uuid=cam.camera_uuid, device_uuid=device_uuid)
            else:
                if cam.is_enabled and cam.is_detection_enabled:
                    await self._ensure_camera_has_exactly_one_device(db, camera_uuid=cam.camera_uuid)

        else:
            if user_id is None:
                raise ValueError("user_id is required to create a new camera")
            if not camera_code:
                raise ValueError("camera_code is required to create a new camera")
            if site_uuid is None:
                raise ValueError("site_uuid is required to create a new camera")
            if device_uuid is None:
                raise ValueError("device_uuid is required to create a new camera (each camera must have a device).")

            await self._ensure_site_exists(db, site_uuid)
            await self._ensure_device_exists(db, device_uuid)

            cam = Camera(
                user_id=user_id,
                site_uuid=site_uuid,
                camera_code=camera_code,
                rtsp_url=rtsp_url,
                webrtc_url=webrtc_url,
                name=name,
                location=location,
                is_enabled=bool(enabled),
                is_detection_enabled=bool(detection_enabled),
                is_notification_enabled=bool(notification_enabled),
                roi=roi,
            )
            if cam_uuid is not None:
                cam.camera_uuid = cam_uuid

            db.add(cam)
            try:
                await db.flush()
            except IntegrityError as e:
                raise ValueError(
                    f"Camera already exists for user_id={user_id} camera_code={camera_code}"
                ) from e

            await self._set_camera_device(db, camera_uuid=cam.camera_uuid, device_uuid=device_uuid)

        cfg_json = self._build_channel_configuration_json(d)
        tz = self._extract_timezone(d, fallback=timezone)

        await self._upsert_channel_configuration(
            db,
            camera_uuid=cam.camera_uuid,
            configuration=cfg_json,
            timezone=tz,
        )

        await self._set_pipeline_membership(db, camera_uuid=cam.camera_uuid, pipeline_id=pipeline_id)

        return cam, cfg_json, tz

    async def get_camera_full(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> Optional[Tuple[Camera, Optional[ChannelConfiguration], Optional[uuid.UUID]]]:
        cam = (
            await db.execute(
                select(Camera)
                .where(Camera.camera_uuid == camera_uuid)
                .options(
                    selectinload(Camera.channel_configuration),
                    selectinload(Camera.devices),
                )
            )
        ).scalar_one_or_none()

        if cam is None:
            return None

        pipeline_id = (
            await db.execute(select(PipelineCamera.pipeline_id).where(PipelineCamera.camera_uuid == cam.camera_uuid))
        ).scalar_one_or_none()

        return cam, cam.channel_configuration, pipeline_id

    async def get_device(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        required: bool = False,
        relaxed: bool = False,
    ) -> Optional[Device]:
        """
        ✅ With new rule, there should be exactly 1 device.
        """
        devices = await self.list_devices(db, camera_uuid=camera_uuid)

        if len(devices) == 1:
            return devices[0]
        if len(devices) == 0 and not required:
            return None
        if relaxed and devices:
            chosen = devices[0]
            logger.warning(
                "Camera %s has %s linked devices; using most recent device %s for legacy compatibility",
                camera_uuid,
                len(devices),
                getattr(chosen, "device_uuid", None),
            )
            return chosen

        raise ValueError(f"Camera {camera_uuid} must have exactly 1 device, found {len(devices)}")

    async def list_devices(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> List[Device]:
        q = (
            select(Device)
            .join(CameraDevice, CameraDevice.device_uuid == Device.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
            .order_by(CameraDevice.created_at.desc(), CameraDevice.id.desc())
        )
        return (await db.execute(q)).scalars().all()

    async def delete_camera(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> None:
        """
        Deletes a camera and its associated configurations.
        """
        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == camera_uuid))
        await db.execute(delete(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid))
        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
        await db.execute(delete(Camera).where(Camera.camera_uuid == camera_uuid))
        await db.flush()

    # -------------------------
    # Internal helpers (NEW/FIXED)
    # -------------------------

    async def _ensure_device_exists(self, db: AsyncSession, device_uuid: uuid.UUID) -> None:
        exists = (await db.execute(select(Device.device_uuid).where(Device.device_uuid == device_uuid))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Device not found: {device_uuid}")

    async def _ensure_camera_has_exactly_one_device(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> None:
        cnt = (
            await db.execute(
                select(func.count(CameraDevice.id)).where(CameraDevice.camera_uuid == camera_uuid)
            )
        ).scalar_one()
        if int(cnt) != 1:
            raise ValueError(f"Camera {camera_uuid} must have exactly 1 device assigned, found {cnt}")

    async def _set_camera_device(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        device_uuid: uuid.UUID,
    ) -> None:

        clean = device_uuid if isinstance(device_uuid, uuid.UUID) else uuid.UUID(str(device_uuid))
        await self._ensure_device_exists(db, clean)

        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
        db.add(CameraDevice(camera_uuid=camera_uuid, device_uuid=clean))
        await db.flush()

    # (keep your existing helpers below unchanged)
    async def _ensure_pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> None:
        exists = (await db.execute(select(Pipeline.id).where(Pipeline.id == pipeline_id))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

    async def _ensure_site_exists(self, db: AsyncSession, site_uuid: uuid.UUID) -> None:
        exists = (await db.execute(select(Site).where(Site.site_uuid == site_uuid))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Site not found: {site_uuid}")

    async def _get_camera_by_uuid(self, db: AsyncSession, camera_uuid: uuid.UUID) -> Optional[Camera]:
        stmt = (
            select(Camera)
            .where(Camera.camera_uuid == camera_uuid)
            .options(selectinload(Camera.channel_configuration))
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    def _to_dict(self, obj: ChannelConfigLike) -> Dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json", exclude_none=True)
        if isinstance(obj, dict):
            return jsonable_encoder(dict(obj), exclude_none=True)
        return jsonable_encoder({k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}, exclude_none=True)

    def _extract_timezone(self, d: Dict[str, Any], fallback: Optional[str]) -> Optional[str]:
        return d.get("timezone", fallback)

    def _build_channel_configuration_json(self, d: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(d)
        for k in (
            "camera_uuid", "camera_id", "channel_id",
            "rtsp_url", "webrtc_url",
            "device_url",
            "enabled", "detection_enabled", "notification_enabled",
            "site_uuid", "device_uuid", "user_id",
            "camera_code", "name", "location", "timezone",
            "is_enabled", "is_detection_enabled", "is_notification_enabled",
            "roi",
        ):
            cfg.pop(k, None)
        return cfg

    async def _upsert_channel_configuration(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        configuration: Dict[str, Any],
        timezone: Optional[str],
    ) -> ChannelConfiguration:
        configuration = jsonable_encoder(configuration, exclude_none=True)
        row = (
            await db.execute(select(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid))
        ).scalar_one_or_none()

        if row is None:
            row = ChannelConfiguration(camera_uuid=camera_uuid, configuration=configuration, timezone=timezone)
            db.add(row)
            await db.flush()
            return row

        row.configuration = configuration
        if timezone is not None:
            row.timezone = timezone
        await db.flush()
        return row

    async def _set_pipeline_membership(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        pipeline_id: uuid.UUID,
    ) -> None:
        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == camera_uuid))
        db.add(PipelineCamera(pipeline_id=pipeline_id, camera_uuid=camera_uuid))
        await db.flush()
