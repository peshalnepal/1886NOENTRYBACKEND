"""Composed Manager class.

Behavior matches the former monolithic application/services/manager.py.
Delegates to the controllers under application/services/manager/controllers/.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.channel_repository import ChannelRepository
from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.device_repository import DeviceRepository
from application.repositories.site_repository import SiteRepository
from application.repositories.user_repository import UserRepository
from application.services.edgeinference import EdgeInferenceClient
from application.services.webrtcgateway import WebRTCGatewayClient
from domain.events import VideoChannelEvent
from application.services.pipeline import ModelPipeline

from application.services.manager.types import PipelineUpdateResult
from application.services.manager.controllers import (
    ManagerState,
    ScheduleResolver,
    ChannelController,
    PipelineController,
    DeviceReconciler,
    CleanupController,
)

logger = logging.getLogger(__name__)


class Manager:
    """
    Azure Manager:
      - DB is source of truth
      - WebRTC gateway provides playback URL
      - Jetson device provides detections
    """

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
            default_request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "3.0")),
            external_timeout_s=float(os.getenv("EXTERNAL_SERVICE_TIMEOUT_S", "10.0")),
            edge_retry_max_attempts=max(1, int(os.getenv("EDGE_RETRY_MAX_ATTEMPTS", "3"))),
            edge_retry_base_ms=max(100, int(os.getenv("EDGE_RETRY_BASE_MS", "500"))),
        )
        
        self._notification_service: Optional[Any] = None
        self._default_user_id = int(os.getenv("DEFAULT_USER_ID", "1"))

        # Controllers
        self._schedule_ctrl = ScheduleResolver(self._state)
        self._channel_ctrl = ChannelController(self._state, self._schedule_ctrl)
        self._pipeline_ctrl = PipelineController(self._state, self._schedule_ctrl, self._channel_ctrl)
        self._reconcile_ctrl = DeviceReconciler(self._state, self._schedule_ctrl)
        self._cleanup_ctrl = CleanupController(self._state)

    def set_notification_service(self, notification_service: Optional[Any]) -> None:
        self._notification_service = notification_service
        for mp in list(self._state.pipelines_by_user.values()):
            self._pipeline_ctrl._wire_pipeline(mp, notification_service)

    async def shutdown(self) -> None:
        all_locks = list(self._state.locks_by_user.values())
        self._state.locks_by_user.clear()
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

    async def start_background_pipelines(self) -> Dict[str, Any]:
        async with self._state.session_factory() as db:
            raw_user_ids = await self._state.user_repo.get_exisiting_users_id(db)
            user_ids = sorted(
                {
                    int(uid)
                    for uid in raw_user_ids
                    if uid is not None
                }
            )

        summary: Dict[str, Any] = {
            "user_ids": user_ids,
            "started": [],
            "errors": [],
        }

        for uid in user_ids:
            try:
                mp = await self.get_activepipeline(user_id=uid)
                summary["started"].append(
                    {
                        "user_id": uid,
                        "pipeline_id": str(getattr(mp, "pipeline_id", "")),
                        "channel_count": len(mp.list_channel_ids()),
                    }
                )
            except Exception as exc:
                logger.exception("Failed to start background pipeline user=%s", uid)
                summary["errors"].append(
                    {
                        "user_id": uid,
                        "error": str(exc),
                    }
                )

        summary["started_count"] = len(summary["started"])
        summary["error_count"] = len(summary["errors"])
        return summary

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
    
    async def reconcile_devices_best_effort(self, *, user_id: int, device_uuids: List[Union[str, uuid.UUID]]) -> None:
        return await self._reconcile_ctrl.reconcile_devices_best_effort(user_id=user_id, device_uuids=device_uuids)

    async def reconcile_device_edge_simple(self, *, device_uuid: uuid.UUID, user_id: Optional[int] = None, dry_run: bool = False, delete_unknown: bool = True) -> Dict[str, List[str]]:
        return await self._reconcile_ctrl.reconcile_device_edge_simple(device_uuid=device_uuid, user_id=user_id, dry_run=dry_run, delete_unknown=delete_unknown)

    async def reconcile_all_devices_edge(self, *, user_id: Optional[int] = None, dry_run: bool = False, delete_unknown: bool = False) -> Dict[str, Any]:
        return await self._reconcile_ctrl.reconcile_all_devices_edge(user_id=user_id, dry_run=dry_run, delete_unknown=delete_unknown)

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
