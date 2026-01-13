# agents/domain/ai_models.py

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

from domain.events import ChannelEvent, RTSPEvent
from dto import VisionModelConfig


@runtime_checkable
class VisionModel(Protocol):
    """
    Domain port: anything that can turn an RTSPEvent into a domain ChannelEvent.
    (Usually DetectionsProducedEvent or InferenceFailedEvent)
    """
    cfg: VisionModelConfig

    async def infer(self, rtsp_ev: RTSPEvent) -> ChannelEvent: ...

    async def warmup(self) -> None: ...
    async def shutdown(self) -> None: ...
