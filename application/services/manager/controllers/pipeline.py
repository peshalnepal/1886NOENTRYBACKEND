"""Manager pipeline lifecycle + background tasks."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Union

from application.channels.channel import VideoChannel
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent, VideoChannelEvent
from application.services.pipeline import ModelPipeline

from application.services.manager.helpers import _camera_config_json, build_video_channel_config
from application.services.manager.types import CameraOut, PipelineUpdateResult
from application.repositories._helpers import require_uuid
from application.services.manager.controllers._state import ManagerState
from application.services.manager.controllers.schedule import ScheduleResolver
from application.services.manager.controllers.channel import ChannelController

logger = logging.getLogger(__name__)


class PipelineController:
    def __init__(self, state: ManagerState, schedule_resolver: ScheduleResolver, channel_ctrl: ChannelController):
        self._state = state
        self._schedule_resolver = schedule_resolver
        self._channel_ctrl = channel_ctrl

    def wire_pipeline(self, mp: ModelPipeline, notification_service: Optional[Any]) -> None:
        """Give a freshly-built pipeline its session factory and notification sink."""
        mp.set_session_factory(self._state.session_factory)
        if notification_service is not None:
            mp.set_notification_service(notification_service)

    def _event_includes_roi_patch(self, ev: VideoChannelEvent) -> bool:
        configs = getattr(ev, "configs", None)
        if isinstance(configs, dict):
            return "roi" in configs

        fields_set = getattr(configs, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(configs, "__fields_set__", None)
        if fields_set is not None:
            return "roi" in fields_set

        return False

    def _clear_loaded_pipeline(self, uid: int) -> None:
        """Forget a pipeline whose database row is no longer available."""
        self._state.pipelines_by_user.pop(uid, None)
        self._state.pipeline_id_by_user.pop(uid, None)

    @staticmethod
    def _event_type(ev: VideoChannelEvent) -> str:
        event_type = getattr(ev, "event_type", None)
        if event_type is None and isinstance(ev, dict):
            event_type = ev.get("event_type")
        return str(event_type or "").lower()

    def _invalidate_camera_roi_state(self, camera_uuid: uuid.UUID, notification_service: Optional[Any]) -> None:
        cam = str(camera_uuid)

        svc = notification_service
        invalidate = getattr(svc, "invalidate_camera_roi_state", None) if svc is not None else None
        if callable(invalidate):
            try:
                invalidate(cam)
            except Exception:
                logger.exception("Failed invalidating notification ROI state camera=%s", cam)

        for mp in list(self._state.pipelines_by_user.values()):
            if mp is None:
                continue
            invalidate_mp = getattr(mp, "invalidate_camera_roi_state", None)
            if not callable(invalidate_mp):
                continue
            try:
                invalidate_mp(cam)
            except Exception:
                logger.exception("Failed invalidating pipeline ROI state camera=%s", cam)

    async def _create_pipeline_unlocked(self, uid: int, notification_service: Optional[Any]) -> ModelPipeline:
        async with self._state.session_factory() as db:
            pipeline_name=f"{str(uid)}_pipeline" if uid else "default"
            pipeline_row = await self._state.pipeline_repo.upsert_pipeline(
                db,
                user_id=uid,
                pipeline_id=None,
                name=pipeline_name,
                is_active=True,
            )
            pid = pipeline_row.id

            full_pl = await self._state.pipeline_repo.get_full_pipeline(db, pid)

            from core.env import env_bool
            notify_on_confirmed = env_bool("NOTIFY_ON_CONFIRMED", False)
            mp = ModelPipeline(
                pipeline_id=pid,
                notify_on_confirmed=notify_on_confirmed,
                task_spawner=lambda coro, name: self._state.spawn_bg(coro, name=name),
            )
            self.wire_pipeline(mp, notification_service)

            if full_pl and getattr(full_pl, "cameras", None):
                site_schedule_cache: Dict[str, Dict[str, Any]] = {}
                for cam in full_pl.cameras:
                    enabled = bool(getattr(cam, "is_enabled", True))
                    det_enabled = bool(getattr(cam, "is_detection_enabled", True))
                    devices = [cam.device] if getattr(cam, "device", None) else []
                    if len(devices) == 0:
                        site_devices = await self._state.device_repo.list_devices(
                            db, site_uuid=cam.site_uuid, user_id=uid
                        )
                        repair_device = None
                        if site_devices:
                            enabled_devices = [d for d in site_devices if getattr(d, "is_enabled", True)]
                            candidates = enabled_devices or site_devices
                            if len(candidates) == 1:
                                repair_device = candidates[0]
                        if repair_device is None:
                            logger.warning(
                                "Skipping camera %s (enabled=%s detection=%s) because device count=%s and no unique site device could be inferred",
                                cam.camera_uuid, enabled, det_enabled, len(devices)
                            )
                            continue
                        await self._state.channel_repo.set_camera_device(
                            db,
                            camera_uuid=cam.camera_uuid,
                            device_uuid=repair_device.device_uuid,
                        )
                        devices = [repair_device]
                        logger.info(
                            "Auto-linked missing camera device camera=%s site=%s device=%s during pipeline load",
                            cam.camera_uuid,
                            cam.site_uuid,
                            repair_device.device_uuid,
                        )
                    if len(devices) > 1:
                        logger.warning(
                            "Camera %s has %s linked devices; using first loaded device %s for runtime compatibility",
                            cam.camera_uuid,
                            len(devices),
                            getattr(devices[0], "device_uuid", None),
                        )

                    device = devices[0]
                    d_url = getattr(device, "device_url", None)
                    d_uuid = getattr(device, "device_uuid", None)
                    if not d_url or not d_uuid:
                        raise ValueError(f"Camera {cam.camera_uuid} has invalid device assignment.")

                    cfg_json = _camera_config_json(cam)
                    schedule_state = await self._schedule_resolver.resolve_runtime_schedule(
                        db,
                        cam=cam,
                        cfg_json=cfg_json,
                        cfg_timezone=getattr(getattr(cam, "channel_configuration", None), "timezone", None),
                        site_cache=site_schedule_cache,
                    )

                    vcc = build_video_channel_config(
                        cam,
                        device_uuid=d_uuid,
                        device_url=d_url,
                        schedule_state=schedule_state,
                        cfg_json=cfg_json,
                        default_request_timeout_s=self._state.default_request_timeout_s,
                    )
                    await mp.add_channel(VideoChannel(config=vcc))

            await db.commit()
        self._state.pipelines_by_user[uid] = mp
        self._state.pipeline_id_by_user[uid] = pid
        return mp

    async def create_pipeline(self, user_id: int | None, notification_service: Optional[Any]) -> ModelPipeline:
        """Load (or create) the user's pipeline row and build its ModelPipeline."""
        uid = int(user_id or 1)

        user_lock = self._state.get_user_lock(uid)
        async with user_lock:
            return await self._create_pipeline_unlocked(uid, notification_service)

    async def get_activepipeline(self, user_id: int | None, notification_service: Optional[Any], default_user_id: int) -> ModelPipeline:
        uid = int(user_id or default_user_id)
        mp = self._state.pipelines_by_user.get(uid)
        if mp is None:
            user_lock = self._state.get_user_lock(uid)
            async with user_lock:
                mp = self._state.pipelines_by_user.get(uid)
                if mp is None:
                    mp = await self._create_pipeline_unlocked(uid, notification_service)
        await mp.start()
        return mp

    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID, None],
        channel_events: List[VideoChannelEvent],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
        notification_service: Optional[Any] = None,
        default_user_id: int = 1,
    ) -> Optional[PipelineUpdateResult]:

        uid = int(user_id or default_user_id)

        # Resolved OUTSIDE any manager lock: get_activepipeline takes the
        # per-user lock itself, so holding one here would deadlock.
        model_pipeline = await self.get_activepipeline(uid, notification_service, default_user_id)

        model_pipeline_pid = getattr(model_pipeline, "pipeline_id", None) or self._state.pipeline_id_by_user.get(uid)
        if not model_pipeline_pid:
            user_lock = self._state.get_user_lock(uid)
            async with user_lock:
                self._clear_loaded_pipeline(uid)
            model_pipeline = await self.get_activepipeline(uid, notification_service, default_user_id)
            model_pipeline_pid = getattr(model_pipeline, "pipeline_id", None)

        pid = require_uuid(model_pipeline_pid, "pipeline_id")

        if pipeline_id is not None:
            try:
                supplied_pid = require_uuid(pipeline_id, "pipeline_id")
                if supplied_pid != pid:
                    logger.warning(
                        "update_pipeline called with pipeline_id=%s but active pipeline_id=%s user=%s; using active",
                        supplied_pid, pid, uid
                    )
            except Exception:
                logger.warning("update_pipeline got non-uuid pipeline_id=%s; using active", pipeline_id)

        for attempt in (1, 2):
            async with self._state.session_factory() as db:
                exists = await self._state.pipeline_repo.pipeline_exists(db, pid)
                if not exists:
                    logger.warning(
                        "Active pipeline %s missing in DB for user %s (attempt %s).",
                        pid, uid, attempt
                    )
                    user_lock = self._state.get_user_lock(uid)
                    async with user_lock:
                        self._clear_loaded_pipeline(uid)

                    if attempt == 1:
                        model_pipeline = await self.get_activepipeline(uid, notification_service, default_user_id)
                        model_pipeline_pid = getattr(model_pipeline, "pipeline_id", None) or self._state.pipeline_id_by_user.get(uid)
                        if not model_pipeline_pid:
                            return None
                        pid = require_uuid(model_pipeline_pid, "pipeline_id")
                        continue
                    return None

                cameras_out: List[CameraOut] = []
                events_out: List[Dict[str, Any]] = []
                roi_reset_camera_ids: Set[uuid.UUID] = set()

                for ev in (channel_events or []):
                    et_norm = self._event_type(ev)

                    if et_norm == "create_channel" or isinstance(ev, ChannelCreateEvent):
                        cams, evs = await self._channel_ctrl.add_channel(
                            db, pid=pid, ev=ev, user_id=uid,
                            camera_code_prefix=camera_code_prefix, model_pipeline=model_pipeline
                        )
                    elif et_norm == "edit_channel" or isinstance(ev, ChannelEditEvent):
                        if self._event_includes_roi_patch(ev):
                            cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
                            if cam_uuid is not None:
                                roi_reset_camera_ids.add(require_uuid(cam_uuid, "camera_uuid"))
                        cams, evs = await self._channel_ctrl.edit_channel(db, pid=pid, ev=ev, user_id=uid, model_pipeline=model_pipeline)
                    elif et_norm == "remove_channel" or isinstance(ev, ChannelRemoveEvent):
                        cams, evs = await self._channel_ctrl.remove_channel(db, pid=pid, ev=ev, user_id=uid, model_pipeline=model_pipeline)
                    else:
                        logger.warning("Event type not matched: %s", et_norm)
                        continue

                    cameras_out.extend(cams)
                    events_out.extend(evs)

                await db.commit()
                for cam_uuid in roi_reset_camera_ids:
                    self._invalidate_camera_roi_state(cam_uuid, notification_service)

                return PipelineUpdateResult(
                    pipeline_id=pid,
                    active_in_memory=True,
                    cameras=cameras_out,
                    events=events_out,
                )

        return None
