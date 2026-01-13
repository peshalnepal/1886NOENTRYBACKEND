# agents/application/services/agent_manager.py

import asyncio
import logging
import uuid
from typing import Callable, List, Optional, Union,Tuple,Literal,Any,Dict

from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from fastapi import Query

from domain.events import (
    VideoChannelEvent,
    ChannelCreateEvent,
    ChannelEditEvent,
    ChannelRemoveEvent,
    VideoChannelEvent
)

from application.builder.pipeline_builder import PipelineBuilder
from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository

from domain.model_pipeline import ModelPipeline
from application.channels.channel_config import VideoChannelConfig
from application.models.yolo_config import YoloModelConfig
from application.channels.channel import VideoChannel
from domain.template import Template as DomainTemplate
logger = logging.getLogger(__name__)

ChannelConfigLike = Union[VideoChannelConfig, dict]
ModelConfigLike = Union[YoloModelConfig, dict]

class CameraOut(BaseModel):
    camera_uuid: uuid.UUID
    channel_id: str
    rtsp_url: str
    enabled: bool
    detection_enabled:bool
    notification_enabled:bool
    sample_fps: float = Field(default=5.0, ge=0.1, description="Frames/sec to publish as RTSPEvent")
    decode_backend: Literal["gstreamer", "opencv"] = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = Field(default=None, description="(width, height)")
    emit_format: Literal["raw", "jpeg"] = Field(default="raw", description="raw => np.ndarray in RTSPEvent.frame")
    jpeg_quality: int = Field(default=80, ge=1, le=100)

class PipelineUpdateResult(BaseModel):
    pipeline_id: uuid.UUID
    active_in_memory: bool
    cameras: List[CameraOut] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list) 


SOFT_PATCH_KEYS = {
    "detection_enabled",
    "notification_enabled",
    "sample_fps",
}

HARD_PATCH_KEYS = {
    "rtsp_url",
    "decode_backend",
    "resize",
    "emit_format",
    "jpeg_quality",
    "reconnect_base_ms",
    "reconnect_max_ms",
}


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
                default_model_cfg = YoloModelConfig()
                return await self._create_pipeline_locked(db,configs=[],model_config=default_model_cfg,pipeline_id=None,)

    async def get_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID],
        model_config: Optional[ModelConfigLike] = None,
    ) -> Optional[ModelPipeline]:
        pid = self._parse_uuid(pipeline_id)
        if pid is None:
            return None

        model_cfg = self._normalize_model_config(model_config)

        async with self._lock:
            if self._active_pipeline is not None and self._active_id == pid:
                return self._active_pipeline

            async with self._session_factory() as db:
                configs = await self._load_channel_configs(db, pid)
                if configs is None:
                    return None

                await self._shutdown_active_locked()
                self._active_pipeline = await self._build_pipeline(db, pid, configs, model_cfg)
                self._active_id = pid
                return self._active_pipeline
            
    def _patch_to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_unset=True, exclude_none=True)
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if v is not None}
        return {}
    
    async def _load_existing_camera_config(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        camera_uuid: uuid.UUID,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Returns:
          - base config dict suitable for VideoChannelConfig(**base)
          - cam_db ORM object
        """
        full = await self.channel_repo.get_camera_full(db, camera_uuid=camera_uuid)
        if not full:
            raise ValueError(f"Camera not found: {camera_uuid}")

        cam_db, chan_cfg_db, existing_pid = full
        if existing_pid is not None and existing_pid != pid:
            raise ValueError("Camera does not belong to provided pipeline_id")

        base: Dict[str, Any] = {}

        # stored tuning knobs
        if chan_cfg_db and getattr(chan_cfg_db, "configuration", None):
            base.update(chan_cfg_db.configuration)

        # DB “truth”
        base["camera_uuid"] = cam_db.camera_uuid
        base["channel_id"] = str(cam_db.camera_uuid)  # runtime key
        base["rtsp_url"] = cam_db.rtsp_url
        base["enabled"] = bool(cam_db.is_enabled)
        base["detection_enabled"] = bool(getattr(cam_db, "is_detection_enabled", True))
        base["notification_enabled"] = bool(getattr(cam_db, "is_notification_enabled", True))

        return base, cam_db
    
    async def create_pipeline(
        self,
        channel_configs: Optional[List[ChannelConfigLike]] = None,
        model_config: Optional[ModelConfigLike] = None,
        pipeline_id: Optional[Union[str, uuid.UUID]] = None,
    ) -> ModelPipeline:
        configs = self._normalize_configs(channel_configs or [])
        pid = self._parse_uuid(pipeline_id) if pipeline_id else None
        model_cfg = self._normalize_model_config(model_config)

        async with self._lock:
            async with self._session_factory() as db:
                return await self._create_pipeline_locked(
                    db,
                    configs=configs,
                    model_config=model_cfg,
                    pipeline_id=pid,
                )
                
    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID],
        channel_events: List[VideoChannelEvent],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
    ) -> Optional["PipelineUpdateResult"]:

        pid = self._parse_uuid(pipeline_id)
        if pid is None:
            return None

        async with self._lock:
            async with self._session_factory() as db:
                if not await self._repo.pipeline_exists(db, pid):
                    return None

                runtime_ops: List[Tuple[str, Any]] = []
                cameras_out: List[CameraOut] = []
                events_out: List[Dict[str, Any]] = []

                for ev in (channel_events or []):
                    et = getattr(ev, "event_type", None)

                    if et == "Create_Channel" or isinstance(ev, ChannelCreateEvent):
                        ops, cams, evs = await self.add_channel(
                            db, pid=pid, ev=ev, user_id=user_id, camera_code_prefix=camera_code_prefix
                        )
                    elif et == "Edit_Channel" or isinstance(ev, ChannelEditEvent):
                        ops, cams, evs = await self.edit_channel(db, pid=pid, ev=ev)
                    elif et == "Remove_Channel" or isinstance(ev, ChannelRemoveEvent):
                        ops, cams, evs = await self.remove_channel(db, pid=pid, ev=ev)
                    else:
                        continue

                    runtime_ops.extend(ops)
                    cameras_out.extend(cams)
                    events_out.extend(evs)

                await db.commit()

            active_in_memory = (self._active_id == pid and self._active_pipeline is not None)
            if not active_in_memory:
                return PipelineUpdateResult(
                    pipeline_id=pid,
                    active_in_memory=False,
                    cameras=cameras_out,
                    events=events_out,
                )

            await self._apply_runtime_ops(self._active_pipeline, runtime_ops)

            return PipelineUpdateResult(
                pipeline_id=pid,
                active_in_memory=True,
                cameras=cameras_out,
                events=events_out,
            )


    async def add_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: Optional[int],
        camera_code_prefix: str,
    ) -> Tuple[List[Tuple[str, Any]], List["CameraOut"], List[Dict[str, Any]]]:
        """
        Handle Create_Channel:
          - upsert camera+config in DB
          - returns runtime_ops + CameraOut + events_out
        """
        patch = self._patch_to_dict(getattr(ev, "configs", None))

        cfg_in = VideoChannelConfig(**patch)
        if user_id is None:
            raise ValueError("user_id is required when creating a new camera.")

        camera_code = f"{camera_code_prefix}-{uuid.uuid4().hex[:8]}"

        cam = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config=cfg_in,
            user_id=user_id,
            camera_code=camera_code,
        )
        data = cfg_in.model_dump(exclude_unset=True, exclude_none=True)

        data.update({
            "camera_uuid": cam.camera_uuid,
            "channel_id": str(cam.camera_uuid),   # runtime key
            "rtsp_url": cam.rtsp_url,
            "enabled": bool(cam.is_enabled),
            "detection_enabled": bool(cam.is_detection_enabled),
            "notification_enabled": bool(cam.is_notification_enabled),
        })

        runtime_cfg = VideoChannelConfig(**data)


        runtime_ops: List[Tuple[str, Any]] = []
        if runtime_cfg.enabled:
            runtime_ops.append(("add", runtime_cfg))
        else:
            runtime_ops.append(("remove", str(cam.camera_uuid)))

        cameras_out = [
            CameraOut(
                camera_uuid=cam.camera_uuid,
                channel_id=str(cam.camera_uuid),
                rtsp_url=cam.rtsp_url,
                enabled=bool(cam.is_enabled),
                detection_enabled=bool(cam.is_detection_enabled),
                notification_enabled=bool(cam.is_notification_enabled),
                sample_fps=runtime_cfg.sample_fps,
                decode_backend=runtime_cfg.decode_backend,
                resize=runtime_cfg.resize,
                emit_format=runtime_cfg.emit_format,
                jpeg_quality=runtime_cfg.jpeg_quality,
            )
        ]

        events_out = [
            {
                "event_type": "Create_Channel",
                "camera_uuid": str(cam.camera_uuid),
                "configs": runtime_cfg.model_dump(),
            }
        ]

        return runtime_ops, cameras_out, events_out

    async def edit_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
    ) -> Tuple[List[Tuple[str, Any]], List["CameraOut"], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None)
        if cam_uuid is None:
            raise ValueError("Edit_Channel missing channel_id (camera_uuid).")

        base, _cam_db = await self._load_existing_camera_config(db, pid=pid, camera_uuid=cam_uuid)

        patch = self._patch_to_dict(getattr(ev, "configs", None))
        base.update(patch)

        cfg_merged = VideoChannelConfig(**base)

        cam2 = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config=cfg_merged,
        )

        # full runtime cfg (used if we must restart channel)
        data = cfg_merged.model_dump(exclude_unset=True, exclude_none=True)
        data.update({
            "camera_uuid": cam2.camera_uuid,
            "channel_id": str(cam2.camera_uuid),
            "rtsp_url": cam2.rtsp_url,
            "enabled": bool(cam2.is_enabled),
            "detection_enabled": bool(cam2.is_detection_enabled),
            "notification_enabled": bool(cam2.is_notification_enabled),
        })
        runtime_cfg = VideoChannelConfig(**data)

        runtime_ops: List[Tuple[str, Any]] = []

        # If camera disabled -> remove runtime channel
        if not runtime_cfg.enabled:
            runtime_ops.append(("remove", str(cam2.camera_uuid)))
        else:
            # Build patch payload for runtime patching
            patch_runtime = {k: v for k, v in patch.items() if v is not None}

            # Detect "hard" changes that require restart (decoder changes)
            must_restart = any(k in patch_runtime for k in HARD_PATCH_KEYS)

            runtime_ops.append((
                "patch",
                {
                    "camera_uuid": str(cam2.camera_uuid),
                    "patch": patch_runtime,
                    "must_restart": must_restart,
                    "fallback_cfg": runtime_cfg,  # used if restart needed or patch fails
                }
            ))

        cameras_out = [
            CameraOut(
                camera_uuid=cam2.camera_uuid,
                channel_id=str(cam2.camera_uuid),
                rtsp_url=cam2.rtsp_url,
                enabled=bool(cam2.is_enabled),
                detection_enabled=bool(cam2.is_detection_enabled),
                notification_enabled=bool(cam2.is_notification_enabled),
                sample_fps=runtime_cfg.sample_fps,
                decode_backend=runtime_cfg.decode_backend,
                resize=runtime_cfg.resize,
                emit_format=runtime_cfg.emit_format,
                jpeg_quality=runtime_cfg.jpeg_quality,
            )
        ]

        events_out = [{
            "event_type": "Edit_Channel",
            "camera_uuid": str(cam2.camera_uuid),
            "configs": runtime_cfg.model_dump(),
            "patch": patch,  # optional debug
        }]

        return runtime_ops, cameras_out, events_out

    async def remove_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
    ) -> Tuple[List[Tuple[str, Any]], List["CameraOut"], List[Dict[str, Any]]]:
        """
        Handle Remove_Channel:
          - delete camera+config if exists (idempotent)
          - runtime: remove channel id
        """
        cam_uuid = getattr(ev, "channel_id", None)
        if cam_uuid is None:
            raise ValueError("Remove_Channel missing channel_id (camera_uuid).")

        full = await self.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if full:
            cam_db, _cfg, existing_pid = full
            if existing_pid is not None and existing_pid != pid:
                raise ValueError("Camera does not belong to provided pipeline_id")
            await self.channel_repo.delete_camera(db, camera_uuid=cam_db.camera_uuid)

        runtime_ops = [("remove", str(cam_uuid))]
        cameras_out: List[CameraOut] = []
        events_out = [{"event_type": "Remove_Channel", "camera_uuid": str(cam_uuid)}]
        return runtime_ops, cameras_out, events_out

    # ------------------------------------------------------------
    # Optional: runtime apply helper (keeps update_pipeline clean)
    # ------------------------------------------------------------
    async def _apply_runtime_ops(self, active: ModelPipeline, runtime_ops: List[Tuple[str, Any]]) -> None:
        existing = set(active.list_channel_ids()) if hasattr(active, "list_channel_ids") else set()

        # 1) remove first
        for op, payload in runtime_ops:
            if op == "remove":
                await active.remove_channel(str(payload))
                existing.discard(str(payload))

        # 2) add + patch
        for op, payload in runtime_ops:
            if op == "add":
                cfg = payload
                key = str(getattr(cfg, "camera_uuid", None) or getattr(cfg, "channel_id", ""))
                await active.add_channel(VideoChannel(config=cfg))
                existing.add(key)

            elif op == "patch":
                cam_uuid = payload["camera_uuid"]
                patch = payload.get("patch") or {}
                must_restart = bool(payload.get("must_restart", False))
                fallback_cfg = payload.get("fallback_cfg")

                # If not present in runtime, treat as add
                if cam_uuid not in existing:
                    if fallback_cfg is not None:
                        await active.add_channel(VideoChannel(config=fallback_cfg))
                        existing.add(cam_uuid)
                    continue

                # If change requires decoder restart, do a channel-only restart (not pipeline rebuild)
                if must_restart:
                    if fallback_cfg is not None:
                        await active.remove_channel(cam_uuid)
                        await active.add_channel(VideoChannel(config=fallback_cfg))
                    continue

                # Soft patch in place
                ok = False
                if hasattr(active, "patch_channel_config"):
                    ok = await active.patch_channel_config(cam_uuid, patch)

                # If patch failed for any reason, fallback to channel restart
                if not ok and fallback_cfg is not None:
                    await active.remove_channel(cam_uuid)
                    await active.add_channel(VideoChannel(config=fallback_cfg))

        await active.start()

    async def shutdown(self) -> None:
        async with self._lock:
            await self._shutdown_active_locked()
            self._active_pipeline = None
            self._active_id = None


    async def _create_pipeline_locked(
        self,
        db: AsyncSession,
        *,
        configs: List[VideoChannelConfig],
        model_config: YoloModelConfig,
        pipeline_id: Optional[uuid.UUID],
    ) -> ModelPipeline:
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
        self._active_pipeline = await self._build_pipeline(db, pid, configs, model_config)
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
                    d.setdefault("channel_id", str(cam_uuid))
                else:
                    d.setdefault("channel_id", str(uuid.uuid4()))

                out.append(VideoChannelConfig(**d))
                continue

            raise TypeError(f"Invalid channel config type: {type(c)}")

        return out
    
    def _normalize_model_config(self, cfg: Optional[ModelConfigLike]) -> YoloModelConfig:
        """
        Accepts:
        - None -> default YoloModelConfig()
        - YoloModelConfig -> returns as-is
        - dict -> YoloModelConfig(**dict)
        """
        if cfg is None:
            return YoloModelConfig()

        if isinstance(cfg, YoloModelConfig):
            return cfg

        if isinstance(cfg, dict):
            return YoloModelConfig(**cfg)

        raise TypeError(f"Invalid model config type: {type(cfg)}")
    
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
            cfg["detection_enabled"]=bool(cam.is_detection_enabled)
            cfg["notification_enabled"]=bool(cam.is_notification_enabled)
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

    async def _build_pipeline(
        self,
        db: AsyncSession,
        pipeline_id: uuid.UUID,
        configs: List[VideoChannelConfig],
        modelconfig: YoloModelConfig,
    ) -> ModelPipeline:
        builder = PipelineBuilder(db)
        spec = DomainTemplate(id=pipeline_id, configs=configs, model_config=modelconfig)
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
