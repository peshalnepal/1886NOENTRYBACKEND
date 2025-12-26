# application/repositories/channel_repository.py

import uuid
from typing import Any, Dict, Optional, Tuple, Union

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError
import logging
from core.database_orm import Camera, ChannelConfiguration, Pipeline, PipelineCamera
logger = logging.getLogger(__name__)
# Accept your Pydantic config types or a dict
ChannelConfigLike = Union[Dict[str, Any], Any]  # Any to avoid importing pydantic in repo


class ChannelRepository:
    """
    Camera + ChannelConfiguration + PipelineCamera consistency.

    Input: one ChannelConfig / VideoChannelConfig object (or dict)
    Extras only when not present in config:
      - pipeline_id (required for create/update membership)
      - user_id + camera_code (required for create)
      - timezone/name/location (optional)
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
        name: Optional[str] = None,
        location: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Camera:
        """
        One entry-point for BOTH create and update.

        Create:
          - if channel_config.camera_uuid/camera_uuid is missing/None -> create camera
          - requires user_id and camera_code

        Update:
          - if channel_config.camera_uuid/camera_uuid is present -> update that camera

        Always:
          - upsert ChannelConfiguration from channel_config (minus camera fields)
          - set PipelineCamera membership to pipeline_id (one camera -> one pipeline)
        """
        await self._ensure_pipeline_exists(db, pipeline_id)

        d = self._to_dict(channel_config)

        cam_uuid = d.get("camera_uuid") or d.get("camera_id")  # support both names
        rtsp_url = d.get("rtsp_url")
        enabled = d.get("enabled", True)

        if not rtsp_url:
            raise ValueError("channel_config.rtsp_url is required")

        if cam_uuid:
            cam = await self._get_camera_by_uuid(db, cam_uuid)
            if cam is None:
                raise ValueError(f"Camera not found for camera_uuid={cam_uuid}")
            # update camera from config
            cam.rtsp_url = rtsp_url
            cam.is_enabled = bool(enabled)

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

            cam = Camera(
                user_id=user_id,
                camera_code=camera_code,
                rtsp_url=rtsp_url,
                name=name,
                location=location,
                is_enabled=bool(enabled),
            )
            db.add(cam)
            try:
                await db.flush()  # assigns cam.id (int) and cam.camera_uuid (uuid)
            except IntegrityError as e:
                # Usually unique (user_id, camera_code)
                raise ValueError(f"Camera already exists for user_id={user_id} camera_code={camera_code}") from e

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

        return cam

    async def delete_camera(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> None:
        """
        Deletes:
          - PipelineCamera rows for camera.id
          - ChannelConfiguration row for camera.camera_uuid
          - Camera row
        """
        cam = await self._get_camera_by_uuid(db, camera_uuid)
        if cam is None:
            return

        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == cam.camera_uuid))
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
                .options(selectinload(Camera.channel_configuration))
            )
        ).scalar_one_or_none()
        logger.info(cam)
        if cam is None:
            return None

        pipeline_id = (
            await db.execute(select(PipelineCamera.pipeline_id).where(PipelineCamera.camera_uuid == cam.camera_uuid))
        ).scalar_one_or_none()

        return cam, cam.channel_configuration, pipeline_id


    async def _ensure_pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> None:
        exists = (await db.execute(select(Pipeline.id).where(Pipeline.id == pipeline_id))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

    async def _get_camera_by_uuid(self, db: AsyncSession, camera_uuid: uuid.UUID) -> Optional[Camera]:
        return (await db.execute(select(Camera).where(Camera.camera_uuid == camera_uuid))).scalar_one_or_none()

    def _to_dict(self, obj: ChannelConfigLike) -> Dict[str, Any]:
        # Pydantic BaseModel
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # dict-like
        if isinstance(obj, dict):
            return dict(obj)
        # last resort: attributes
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}

    def _extract_timezone(self, d: Dict[str, Any], fallback: Optional[str]) -> Optional[str]:
        # if someday you add timezone into config, this will auto-pick it up
        return d.get("timezone", fallback)

    def _build_channel_configuration_json(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store only the "channel knobs" in ChannelConfiguration.configuration.
        Camera metadata stays in Camera table.
        """
        cfg = dict(d)

        # Remove camera-table fields / identity fields from config JSON
        for k in (
            "is_enabled",
            "user_id",
            "camera_code",
            "name",
            "timezone",
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
        stmt = select(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid)
        row = (await db.execute(stmt)).scalar_one_or_none()

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
