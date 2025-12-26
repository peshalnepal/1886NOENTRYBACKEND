# agents/application/services/agent_manager.py

import asyncio
import logging
import uuid
from typing import Callable, List, Optional, Union,Tuple,Literal

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from fastapi import Query

from application.builder.pipeline_builder import PipelineBuilder
from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository

from domain.model import ModelPipeline
from application.channels.channel_config import VideoChannelConfig
from application.channels.channel import VideoChannel
from domain.template import Template as DomainTemplate

logger = logging.getLogger(__name__)

ChannelConfigLike = Union[VideoChannelConfig, dict]

class CameraOut(BaseModel):
    camera_uuid: uuid.UUID
    channel_id: str
    rtsp_url: str
    enabled: bool
    sample_fps: float = Field(default=5.0, ge=0.1, description="Frames/sec to publish as RTSPEvent")
    decode_backend: Literal["gstreamer", "opencv"] = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = Field(default=None, description="(width, height)")
    emit_format: Literal["raw", "jpeg"] = Field(default="raw", description="raw => np.ndarray in RTSPEvent.frame")
    jpeg_quality: int = Field(default=80, ge=1, le=100)

class PipelineUpdateResult(BaseModel):
    pipeline_id: uuid.UUID
    active_in_memory: bool
    cameras: List[CameraOut] = Field(default_factory=list)



class Manager:
    """
    Single Manager instance:
      - one active ModelPipeline in memory (per device)
      - DB access via per-call AsyncSession from session_factory
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._lock = asyncio.Lock()

        self._active_id: Optional[uuid.UUID] = None
        self._active_pipeline: Optional[ModelPipeline] = None

        self._repo = PipelineRepository()
        self.channel_repo = ChannelRepository()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_activepipeline(self) -> ModelPipeline:
        """
        Returns the already-active pipeline, or creates a default pipeline if none exists.

        DEADLOCK FIX:
        - This method holds self._lock
        - It must NOT call create_pipeline() because create_pipeline() also acquires self._lock
        - Instead we call a private helper that assumes the lock is already held.
        """
        async with self._lock:
            if self._active_pipeline is not None:
                return self._active_pipeline

            async with self._session_factory() as db:
                # Create an empty pipeline (no channels) and activate it
                return await self._create_pipeline_locked(db, configs=[], pipeline_id=None)

    async def get_pipeline(self, pipeline_id: Union[str, uuid.UUID]) -> Optional[ModelPipeline]:
        pid = self._parse_uuid(pipeline_id)
        if pid is None:
            return None

        async with self._lock:
            if self._active_pipeline is not None and self._active_id == pid:
                return self._active_pipeline

            async with self._session_factory() as db:
                configs = await self._load_channel_configs(db, pid)
                if configs is None:
                    return None

                await self._shutdown_active_locked()
                self._active_pipeline = await self._build_pipeline(db, pid, configs)
                self._active_id = pid
                return self._active_pipeline

    async def create_pipeline(
        self,
        channel_configs: Optional[List[ChannelConfigLike]] = None,
        pipeline_id: Optional[Union[str, uuid.UUID]] = None,
    ) -> ModelPipeline:
        """
        Public create.
        Safe: acquires lock once, does not call another lock-taking public method.
        """
        configs = self._normalize_configs(channel_configs or [])
        pid = self._parse_uuid(pipeline_id) if pipeline_id else None

        async with self._lock:
            async with self._session_factory() as db:
                return await self._create_pipeline_locked(db, configs=configs, pipeline_id=pid)

    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID],
        channel_configs: List[ChannelConfigLike],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
        replace_runtime: bool = False,
    ) -> Optional[PipelineUpdateResult]:
        """
        DB upsert + runtime hot-add/hot-swap channels if active pipeline.
        Safe: acquires lock once.
        """
        pid = self._parse_uuid(pipeline_id)
        if pid is None:
            return None

        configs = self._normalize_configs(channel_configs)

        async with self._lock:
            async with self._session_factory() as db:
                exists = await self._repo.pipeline_exists(db, pid)
                if not exists:
                    return None

                runtime_cfgs: List[VideoChannelConfig] = []
                cameras_out: List[CameraOut] = []

                for cfg in configs:
                    # NOTE: your config uses camera_uuid; make sure VideoChannelConfig actually has it
                    # If it uses camera_id instead, change these references consistently.
                    is_new_camera = getattr(cfg, "camera_uuid", None) is None

                    camera_code = None
                    if is_new_camera:
                        if user_id is None:
                            raise ValueError("user_id is required when creating a new camera.")
                        camera_code = f"{camera_code_prefix}-{uuid.uuid4().hex[:8]}"

                    cam = await self.channel_repo.upsert_camera_from_channel_config(
                        db,
                        pipeline_id=pid,
                        channel_config=cfg,
                        user_id=user_id if is_new_camera else None,
                        camera_code=camera_code if is_new_camera else None,
                    )

                    channel_id = getattr(cfg, "channel_id", None) or cam.camera_code or str(cam.camera_uuid)

                    runtime_channel_id = str(cam.camera_uuid)

                    cfg_rt = VideoChannelConfig(
                        channel_id=runtime_channel_id,
                        camera_uuid=cam.camera_uuid,
                        rtsp_url=cam.rtsp_url,
                        enabled=bool(cam.is_enabled),
                        sample_fps=cfg.sample_fps,
                        decode_backend=cfg.decode_backend,
                        resize=cfg.resize,
                        reconnect_base_ms=cfg.reconnect_base_ms,
                        reconnect_max_ms=cfg.reconnect_max_ms,
                        emit_format=cfg.emit_format,
                        jpeg_quality=cfg.jpeg_quality,
                    )
                    runtime_cfgs.append(cfg_rt)
                    cameras_out.append(
                        CameraOut(
                            camera_uuid=cam.camera_uuid,
                            channel_id=runtime_channel_id,
                            rtsp_url=cam.rtsp_url,
                            enabled=bool(cam.is_enabled),
                            sample_fps=cfg.sample_fps,
                            decode_backend=cfg.decode_backend,
                            resize=cfg.resize,
                            emit_format=cfg.emit_format,
                            jpeg_quality=cfg.jpeg_quality,

                        )
                    )
                await db.commit()

            active_in_memory = (self._active_id == pid and self._active_pipeline is not None)

            if not active_in_memory:
                return PipelineUpdateResult(
                    pipeline_id=pid,
                    active_in_memory=False,
                    cameras=cameras_out,
                )

            active: ModelPipeline = self._active_pipeline

            existing_ids = set(active.list_channel_ids()) if hasattr(active, "list_channel_ids") else set()
            desired_ids = set()

            for cfg in runtime_cfgs:
                ch_id = cfg.channel_id
                desired_ids.add(ch_id)

                if ch_id in existing_ids:
                    await active.remove_channel(ch_id)

                await active.add_channel(VideoChannel(config=cfg))

            if replace_runtime:
                for ch_id in (existing_ids - desired_ids):
                    await active.remove_channel(ch_id)

            await active.start()

            return PipelineUpdateResult(
                pipeline_id=pid,
                active_in_memory=True,
                cameras=cameras_out,
            )


    async def shutdown(self) -> None:
        async with self._lock:
            await self._shutdown_active_locked()
            self._active_pipeline = None
            self._active_id = None

    async def add_or_replace_channels(
        self,
        pipeline_id: Union[str, uuid.UUID],
        channel_configs: List[ChannelConfigLike],
    ) -> Optional[ModelPipeline]:
        """
        NOTE: This method was calling upsert with wrong kwargs.
        I fixed the call signature.
        """
        pid = self._parse_uuid(pipeline_id)
        if pid is None:
            return None

        configs = self._normalize_configs(channel_configs)

        async with self._lock:
            async with self._session_factory() as db:
                if not await self._repo.pipeline_exists(db, pid):
                    return None

                for cfg in configs:
                    await self.channel_repo.upsert_camera_from_channel_config(
                        db,
                        pipeline_id=pid,
                        channel_config=cfg,
                    )

                await db.commit()

                if self._active_id == pid and self._active_pipeline is not None:
                    await self._shutdown_active_locked()
                    self._active_pipeline = await self._build_pipeline(db, pid, configs)
                    self._active_id = pid
                    return self._active_pipeline

                return None

    async def _create_pipeline_locked(
        self,
        db: AsyncSession,
        *,
        configs: List[VideoChannelConfig],
        pipeline_id: Optional[uuid.UUID],
    ) -> ModelPipeline:
        """
        Create/Upsert pipeline and activate it.

        This helper DOES NOT acquire self._lock.
        It's only called by public methods that already hold the lock.
        """
        pipeline = await self._repo.upsert_pipeline(db, pipeline_id=pipeline_id)
        pid = pipeline.id

        if configs:
            for cfg in configs:
                await self.channel_repo.upsert_camera_from_channel_config(
                    db,
                    pipeline_id=pid,
                    channel_config=cfg,
                )

        await db.commit()

        await self._shutdown_active_locked()
        self._active_pipeline = await self._build_pipeline(db, pid, configs)
        self._active_id = pid
        return self._active_pipeline


    def _parse_uuid(self, value: Union[str, uuid.UUID, None]) -> Optional[uuid.UUID]:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(str(value))
        except Exception:
            logger.warning(f"Invalid pipeline UUID: {value}")
            return None

    def _normalize_configs(self, configs: List[ChannelConfigLike]) -> List[VideoChannelConfig]:
        out: List[VideoChannelConfig] = []

        for c in configs:
            if isinstance(c, VideoChannelConfig):
                out.append(c)
                continue

            if isinstance(c, dict):
                d = dict(c)

                cam_uuid = d.get("camera_uuid")
                if cam_uuid:
                    d.setdefault("channel_id", f"ch-{cam_uuid.hex[:8]}")
                else:
                    d.setdefault("channel_id", f"ch-{uuid.uuid4().hex[:8]}")

                out.append(VideoChannelConfig(**d))
                continue

            raise TypeError(f"Invalid channel config type: {type(c)}")

        return out

    async def _load_channel_configs(self, db: AsyncSession, pipeline_id: uuid.UUID):
        orm_pipeline = await self._repo.get_full_pipeline(db, pipeline_id)
        if orm_pipeline is None:
            return None

        domain_configs = []
        for cam in (orm_pipeline.cameras or []):
            cfg = {}
            if cam.channel_configuration and cam.channel_configuration.configuration:
                cfg.update(cam.channel_configuration.configuration)

            cfg["camera_uuid"] = cam.camera_uuid
            cfg["rtsp_url"] = cam.rtsp_url
            cfg["enabled"] = cam.is_enabled
            cfg.setdefault("channel_id", cam.camera_code or str(cam.camera_uuid))

            domain_configs.append(cfg)

        return domain_configs
        
    async def remove_camera_from_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID],
        camera_uuid: Union[str, uuid.UUID],
    ) -> bool:
        """
        Removes a camera from DB + removes runtime channel if pipeline is active.

        - DB: deletes PipelineCamera + ChannelConfiguration + Camera
        - Runtime: active_pipeline.remove_channel(str(camera_uuid))
        """
        pid = self._parse_uuid(pipeline_id)
        cam_id = self._parse_uuid(camera_uuid)
        if pid is None or cam_id is None:
            return False
        

        async with self._lock:
            async with self._session_factory() as db:
                if not await self._repo.pipeline_exists(db, pid):
                    return False

                full = await self.channel_repo.get_camera_full(db, camera_uuid=cam_id)
                logger.info(full)
                if not full:
                    logger.info("Delete is being happening")
                    return False

                cam, _chan_cfg, existing_pid = full
                

                if existing_pid is not None and existing_pid != pid:
                    return False
                logger.info("##################################################")
                logger.info("REMOVED CAMERA FROM DB")
                await self.channel_repo.delete_camera(db, camera_uuid=cam.camera_uuid)
                logger.info("REMOVED ")
                if self._active_id == pid and self._active_pipeline is not None:
                    await self._active_pipeline.remove_channel(str(cam.camera_uuid))
                logger.info("##################################################")
                await db.commit()
                return True

    async def _build_pipeline(self, db: AsyncSession, pipeline_id: uuid.UUID, configs: List[VideoChannelConfig]) -> ModelPipeline:
        builder = PipelineBuilder(db)
        spec = DomainTemplate(id=pipeline_id, configs=configs)
        pipeline = await builder.create(spec)
        logger.info(f"Pipeline built: {pipeline_id} | channels={len(configs)}")
        return pipeline

    async def _shutdown_active_locked(self) -> None:
        if not self._active_pipeline:
            return
        try:
            if hasattr(self._active_pipeline, "shutdown") and callable(getattr(self._active_pipeline, "shutdown")):
                await self._active_pipeline.shutdown()
            elif hasattr(self._active_pipeline, "stop") and callable(getattr(self._active_pipeline, "stop")):
                maybe = self._active_pipeline.stop()
                if asyncio.iscoroutine(maybe):
                    await maybe
        except Exception:
            logger.exception("Failed to shutdown active pipeline")
