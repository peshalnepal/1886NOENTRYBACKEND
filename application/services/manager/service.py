"""The `Manager` facade: one object the app holds, delegating to the
controllers under `application/services/manager/controllers/`.

The DB is the source of truth; the WebRTC gateway supplies playback URLs and
the Jetson edge devices supply detections.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.channel_repository import ChannelRepository
from application.repositories.device_repository import DeviceRepository
from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.site_repository import SiteRepository
from application.repositories.user_repository import UserRepository
from application.services.edgeinference import EdgeInferenceClient
from application.services.manager.controllers import (
    CameraAdopter,
    ChannelController,
    CleanupController,
    DeviceReconciler,
    ManagerState,
    PipelineController,
    ScheduleResolver,
)
from application.services.manager.types import PipelineUpdateResult
from application.services.pipeline import ModelPipeline
from application.services.webrtcgateway import WebRTCGatewayClient
from core.env import env_float, env_int
from domain.events import VideoChannelEvent

logger = logging.getLogger(__name__)


class Manager:
    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._bg_tasks: set[asyncio.Task] = set()

        def _spawn_bg(coro, *, name: str) -> asyncio.Task:
            task = asyncio.create_task(coro, name=name)
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            return task

        self._state = ManagerState(
            session_factory=session_factory,
            webrtc=WebRTCGatewayClient(),
            edge=EdgeInferenceClient(),
            channel_repo=ChannelRepository(),
            pipeline_repo=PipelineRepository(),
            user_repo=UserRepository(),
            device_repo=DeviceRepository(),
            site_repo=SiteRepository(),
            bg_tasks=self._bg_tasks,
            spawn_bg=_spawn_bg,
            pipelines_by_user={},
            pipeline_id_by_user={},
            locks_by_user={},
            default_request_timeout_s=env_float("REQUEST_TIMEOUT_S", 3.0),
            external_timeout_s=env_float("EXTERNAL_SERVICE_TIMEOUT_S", 10.0),
            edge_retry_max_attempts=env_int("EDGE_RETRY_MAX_ATTEMPTS", 3, minimum=1),
            edge_retry_base_ms=env_int("EDGE_RETRY_BASE_MS", 500, minimum=100),
        )

        self._notification_service: Optional[Any] = None
        self._default_user_id = env_int("DEFAULT_USER_ID", 1)

        self._schedule_ctrl = ScheduleResolver(self._state)
        self._channel_ctrl = ChannelController(self._state, self._schedule_ctrl)
        self._pipeline_ctrl = PipelineController(
            self._state, self._schedule_ctrl, self._channel_ctrl
        )
        self._adopt_ctrl = CameraAdopter(self._state, self.update_pipeline)
        self._reconcile_ctrl = DeviceReconciler(
            self._state, self._schedule_ctrl, self._adopt_ctrl
        )
        self._cleanup_ctrl = CleanupController(self._state)

    # --- Collaborators callers reach for directly ---

    @property
    def edge(self) -> EdgeInferenceClient:
        """The edge (Jetson) client, for callers doing their own teardown."""
        return self._state.edge

    @property
    def webrtc(self) -> WebRTCGatewayClient:
        """The MediaMTX gateway client, for callers doing their own teardown."""
        return self._state.webrtc

    @property
    def notification_service(self) -> Optional[Any]:
        return self._notification_service

    def set_notification_service(self, notification_service: Optional[Any]) -> None:
        self._notification_service = notification_service
        for mp in list(self._state.pipelines_by_user.values()):
            self._pipeline_ctrl.wire_pipeline(mp, notification_service)

    async def shutdown(self) -> None:
        # Locks are cleared LAST. `get_user_lock` mints a lock on demand, so
        # clearing this map first would let a concurrent `get_activepipeline`
        # take a brand-new lock, observe an empty `pipelines_by_user`, and build
        # a fresh pipeline while we are tearing the old ones down.
        pipelines = list(self._state.pipelines_by_user.values())
        self._state.pipelines_by_user.clear()
        self._state.pipeline_id_by_user.clear()

        for mp in pipelines:
            if mp is None:
                continue
            try:
                await mp.shutdown()
            except Exception:
                logger.exception(
                    "Pipeline shutdown failed pipeline_id=%s",
                    getattr(mp, "pipeline_id", None),
                )
        
        pending = list(self._state.bg_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        await self._state.webrtc.close()
        await self._state.edge.close()
        self._state.locks_by_user.clear()

    async def start_background_pipelines(self) -> Dict[str, Any]:
        """Load every user's pipeline at boot so detections flow without a
        first request to prime them."""
        async with self._state.session_factory() as db:
            raw_user_ids = await self._state.user_repo.list_user_ids(db)
        user_ids = sorted({int(uid) for uid in raw_user_ids if uid is not None})

        started: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for uid in user_ids:
            try:
                mp = await self.get_activepipeline(user_id=uid)
                started.append(
                    {
                        "user_id": uid,
                        "pipeline_id": str(getattr(mp, "pipeline_id", "")),
                        "channel_count": len(mp.list_channel_ids()),
                    }
                )
            except Exception as exc:
                logger.exception("Failed to start background pipeline user=%s", uid)
                errors.append({"user_id": uid, "error": str(exc)})

        return {
            "user_ids": user_ids,
            "started": started,
            "errors": errors,
            "started_count": len(started),
            "error_count": len(errors),
        }

    # --- Pipeline delegation ---


    async def create_pipeline(self, user_id: int | None = None) -> ModelPipeline:
        return await self._pipeline_ctrl.create_pipeline(user_id, self._notification_service)

    async def get_activepipeline(self, user_id: int | None = None) -> ModelPipeline:
        return await self._pipeline_ctrl.get_activepipeline(user_id, self._notification_service, self._default_user_id)

    def get_loaded_pipeline(self, user_id: int | None = None) -> Optional[ModelPipeline]:
        uid = int(user_id or self._default_user_id)
        return self._state.pipelines_by_user.get(uid)

    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID, None],
        channel_events: List[VideoChannelEvent],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
    ) -> Optional[PipelineUpdateResult]:
        return await self._pipeline_ctrl.update_pipeline(
            pipeline_id=pipeline_id,
            channel_events=channel_events,
            user_id=user_id,
            camera_code_prefix=camera_code_prefix,
            notification_service=self._notification_service,
            default_user_id=self._default_user_id,
        )

    # --- Schedule delegation ---
    
    async def sync_site_schedule_runtime(self, *, user_id: int, site_uuid: uuid.UUID) -> Dict[str, Any]:
        return await self._schedule_ctrl.sync_site_schedule_runtime(user_id=user_id, site_uuid=site_uuid)

    # --- Reconcile delegation ---
    
    async def reconcile_devices_best_effort(self, *, user_id: int, device_uuids: List[Union[str, uuid.UUID]], org_id: Optional[int] = None) -> None:
        return await self._reconcile_ctrl.reconcile_devices_best_effort(user_id=user_id, device_uuids=device_uuids, org_id=org_id)

    async def reconcile_device_edge_simple(self, *, device_uuid: uuid.UUID, user_id: Optional[int] = None, dry_run: bool = False, delete_unknown: bool = True) -> Dict[str, Any]:
        return await self._reconcile_ctrl.reconcile_device_edge_simple(device_uuid=device_uuid, user_id=user_id, dry_run=dry_run, delete_unknown=delete_unknown)

    async def reconcile_all_devices_edge(self, *, user_id: Optional[int] = None, dry_run: bool = False, delete_unknown: bool = False) -> Dict[str, Any]:
        return await self._reconcile_ctrl.reconcile_all_devices_edge(user_id=user_id, dry_run=dry_run, delete_unknown=delete_unknown)

    # --- Discovery adoption delegation ---

    async def adopt_discovered_cameras(
        self,
        *,
        device_uuid: uuid.UUID,
        device_url: str,
        user_id: int,
        discovery_report: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Register the edge's discovered cameras as real cloud cameras.

        With no `discovery_report` supplied, the edge is asked to sweep first,
        so this can be called standalone (e.g. right after a device is linked
        to a site) and not only from inside a reconcile.
        """
        if discovery_report is None:
            discovery_report = await self._state.edge.sync_discovery(device_url=device_url)

        return await self._adopt_ctrl.adopt_discovered_cameras(
            device_uuid=device_uuid,
            device_url=device_url,
            user_id=user_id,
            discovery_report=discovery_report,
            dry_run=dry_run,
        )

    # --- Cleanup delegation ---
    
    async def cleanup_user_resources(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> Dict[str, Any]:
        return await self._cleanup_ctrl.cleanup_user_resources(
            db, user_id=user_id, camera_uuids=camera_uuids
        )

    async def cleanup_device_resources(self, db: AsyncSession, *, device_uuid: uuid.UUID, active: Optional[ModelPipeline]) -> None:
        return await self._cleanup_ctrl.cleanup_device_resources(db, device_uuid=device_uuid, active=active)

    async def cleanup_site_resources(self, db: AsyncSession, *, user_id: int, site_uuid: uuid.UUID, active: Optional[ModelPipeline]) -> None:
        return await self._cleanup_ctrl.cleanup_site_resources(db, user_id=user_id, site_uuid=site_uuid, active=active)
