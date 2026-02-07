# application/repositories/channel_repository.py

import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, select
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

# Accept your Pydantic config types or a dict
ChannelConfigLike = Union[Dict[str, Any], Any]  # Any to avoid importing pydantic in repo


class ChannelRepository:
    """
    Camera + ChannelConfiguration + PipelineCamera consistency.

    This repository is used on Azure (DB is the source of truth).
    We also store:
      - Camera.webrtc_url (immutable once set)
      - Camera.site_uuid (required)
      - CameraDevice links (one primary device for inference; optionally multiple devices)

    NOTE: WebRTC provisioning + Jetson provisioning is handled at the service layer (Manager).
    This repo only persists DB state.
    """

    # -------------------------
    # Public API
    # -------------------------

    async def upsert_camera_from_channel_config(
        self,
        db: AsyncSession,
        *,
        pipeline_id: uuid.UUID,
        channel_config: ChannelConfigLike,
        user_id: Optional[int] = None,
        camera_code: Optional[str] = None,
        site_uuid: Optional[uuid.UUID] = None,
        webrtc_url: Optional[str] = None,
        device_uuids: Optional[List[uuid.UUID]] = None,
        primary_device_uuid: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        location: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Camera:
        """
        One entry-point for BOTH create and update.

        Create:
          - if channel_config.camera_uuid is missing/None -> create camera (UUID generated here)
          - requires user_id, camera_code, and site_uuid

        Update:
          - if channel_config.camera_uuid is present -> update that camera

        Always:
          - upsert ChannelConfiguration from channel_config (minus camera fields)
          - set PipelineCamera membership to pipeline_id (1 camera -> 1 pipeline)
          - set CameraDevice links if device_uuids provided
        """
        await self._ensure_pipeline_exists(db, pipeline_id)

        d = self._to_dict(channel_config)

        cam_uuid = d.get("camera_uuid") or d.get("camera_id")  # support both names
        rtsp_url = d.get("rtsp_url")
        enabled = d.get("enabled", True)
        detection_enabled = d.get("detection_enabled", True)
        notification_enabled = d.get("notification_enabled", True)

        if not rtsp_url:
            raise ValueError("channel_config.rtsp_url is required")

        if cam_uuid:
            cam = await self._get_camera_by_uuid(db, cam_uuid)
            if cam is None:
                raise ValueError(f"Camera not found for camera_uuid={cam_uuid}")

            # update camera from config
            cam.rtsp_url = rtsp_url
            cam.is_enabled = bool(enabled)
            cam.is_detection_enabled = bool(detection_enabled)
            cam.is_notification_enabled = bool(notification_enabled)

            # Immutable: webrtc_url should not change once set
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

            # site_uuid can be set if missing; otherwise treat as stable
            if site_uuid is not None:
                if cam.site_uuid is None:
                    cam.site_uuid = site_uuid
                elif cam.site_uuid != site_uuid:
                    # You can relax this later if you want to support moving cameras between sites.
                    raise ValueError("Changing site_uuid for an existing camera is not supported")

            # optional extra fields (only if provided)
            if camera_code is not None:
                cam.camera_code = camera_code
            if name is not None:
                cam.name = name
            if location is not None:
                cam.location = location

            await db.flush()

        else:
            # create new camera
            if user_id is None:
                raise ValueError("user_id is required to create a new camera")
            if not camera_code:
                raise ValueError("camera_code is required to create a new camera")
            if site_uuid is None:
                raise ValueError("site_uuid is required to create a new camera")

            # ensure site exists (and at least exists in DB)
            await self._ensure_site_exists(db, site_uuid)

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
            )

            # allow deterministic UUID if provided in config dict
            if d.get("camera_uuid"):
                cam.camera_uuid = uuid.UUID(str(d["camera_uuid"]))

            db.add(cam)
            try:
                await db.flush()  # assigns cam.id (int) and cam.camera_uuid (uuid)
            except IntegrityError as e:
                # Usually unique (user_id, camera_code)
                raise ValueError(
                    f"Camera already exists for user_id={user_id} camera_code={camera_code}"
                ) from e

        cfg_json = self._build_channel_configuration_json(d)
        tz = self._extract_timezone(d, fallback=timezone)

        await self._upsert_channel_configuration(
            db,
            camera_uuid=cam.camera_uuid,
            configuration=cfg_json,
            timezone=tz,
        )

        # ensure camera is in exactly this pipeline
        await self._set_pipeline_membership(db, camera_uuid=cam.camera_uuid, pipeline_id=pipeline_id)

        # optional: update camera->device links (for Jetson assignment)
        if device_uuids is not None:
            await self._set_camera_devices(
                db,
                camera_uuid=cam.camera_uuid,
                device_uuids=device_uuids,
                primary_device_uuid=primary_device_uuid,
            )

        return cam

    async def delete_camera(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> None:
        """
        Deletes:
          - PipelineCamera rows
          - CameraDevice rows
          - ChannelConfiguration row
          - Camera row
        """
        cam = await self._get_camera_by_uuid(db, camera_uuid)
        if cam is None:
            return

        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == cam.camera_uuid))
        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == cam.camera_uuid))
        await db.execute(delete(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == cam.camera_uuid))
        await db.execute(delete(Camera).where(Camera.camera_uuid == cam.camera_uuid))
        await db.flush()

    async def get_camera_full(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> Optional[Tuple[Camera, Optional[ChannelConfiguration], Optional[uuid.UUID]]]:
        """
        Returns: (Camera, ChannelConfiguration, pipeline_id)
        """
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

    async def get_primary_device(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> Optional[Device]:
        """
        Returns the primary device assigned to this camera, if any.
        """
        q = (
            select(Device)
            .join(CameraDevice, CameraDevice.device_uuid == Device.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
            .order_by(CameraDevice.is_primary.desc())
        )
        return (await db.execute(q)).scalars().first()

    # -------------------------
    # Internal helpers
    # -------------------------

    async def _ensure_pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> None:
        exists = (await db.execute(select(Pipeline.id).where(Pipeline.id == pipeline_id))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

    async def _ensure_site_exists(self, db: AsyncSession, site_uuid: uuid.UUID) -> None:
        exists = (await db.execute(select(Site.id).where(Site.site_uuid == site_uuid))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Site not found: {site_uuid}")

    async def _get_camera_by_uuid(self, db: AsyncSession, camera_uuid: uuid.UUID) -> Optional[Camera]:
        return (await db.execute(select(Camera).where(Camera.camera_uuid == camera_uuid))).scalar_one_or_none()

    def _to_dict(self, obj: ChannelConfigLike) -> Dict[str, Any]:
        # Pydantic BaseModel
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json", exclude_none=True)
        # dict-like
        if isinstance(obj, dict):
            return jsonable_encoder(dict(obj), exclude_none=True)
        # last resort: attributes
        return jsonable_encoder(
            {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")},
            exclude_none=True,
        )

    def _extract_timezone(self, d: Dict[str, Any], fallback: Optional[str]) -> Optional[str]:
        return d.get("timezone", fallback)

    def _build_channel_configuration_json(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store only the "channel knobs" in ChannelConfiguration.configuration.
        Camera metadata stays in Camera table.
        """
        cfg = dict(d)

        # Remove camera-table fields / identity fields from config JSON
        for k in (
            "camera_uuid",
            "camera_id",
            "channel_id",
            "rtsp_url",
            "webrtc_url",
            "enabled",
            "detection_enabled",
            "notification_enabled",
            "site_uuid",
            "device_uuid",
            "device_uuids",
            "primary_device_uuid",
            "user_id",
            "camera_code",
            "name",
            "location",
            "timezone",
            "is_enabled",
            "is_detection_enabled",
            "is_notification_enabled",
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
        row = (await db.execute(select(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid))).scalar_one_or_none()

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
        """
        Enforce 1 camera -> 1 pipeline:
          delete any existing membership then insert the desired one.
        """
        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == camera_uuid))
        db.add(PipelineCamera(pipeline_id=pipeline_id, camera_uuid=camera_uuid))
        await db.flush()

    async def _set_camera_devices(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        device_uuids: List[uuid.UUID],
        primary_device_uuid: Optional[uuid.UUID],
    ) -> None:
        """
        Replace camera->device links with the provided list.
        """
        clean = [uuid.UUID(str(d)) for d in (device_uuids or [])]
        if not clean:
            await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
            await db.flush()
            return

        # delete existing links
        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))

        # normalize primary
        primary = uuid.UUID(str(primary_device_uuid)) if primary_device_uuid else clean[0]

        for du in clean:
            db.add(CameraDevice(
                camera_uuid=camera_uuid,
                device_uuid=du,
                is_primary=(du == primary),
            ))

        await db.flush()
