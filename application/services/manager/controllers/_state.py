"""Shared state object for manager controllers."""

from dataclasses import dataclass
import asyncio
import uuid
from typing import Callable, Dict, Any, Optional, Coroutine

from sqlalchemy.ext.asyncio import AsyncSession
from application.services.pipeline import ModelPipeline
from application.services.webrtcgateway import WebRTCGatewayClient
from application.services.edgeinference import EdgeInferenceClient
from application.repositories.channel_repository import ChannelRepository
from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.device_repository import DeviceRepository
from application.repositories.site_repository import SiteRepository
from application.repositories.user_repository import UserRepository

@dataclass
class ManagerState:
    session_factory: Callable[[], AsyncSession]
    webrtc: WebRTCGatewayClient
    edge: EdgeInferenceClient
    channel_repo: ChannelRepository
    pipeline_repo: PipelineRepository
    device_repo: DeviceRepository
    site_repo: SiteRepository
    user_repo:UserRepository
    bg_tasks: set[asyncio.Task]
    spawn_bg: Callable[[Coroutine[Any, Any, Any], str], asyncio.Task]
    
    pipelines_by_user: Dict[int, ModelPipeline]
    pipeline_id_by_user: Dict[int, uuid.UUID]
    locks_by_user: Dict[int, asyncio.Lock]
    
    default_request_timeout_s: float
    external_timeout_s: float
    edge_retry_max_attempts: int
    edge_retry_base_ms: int
    
    def get_user_lock(self, uid: int) -> asyncio.Lock:
        if uid not in self.locks_by_user:
            self.locks_by_user[uid] = asyncio.Lock()
        return self.locks_by_user[uid]
