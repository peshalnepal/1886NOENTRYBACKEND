from __future__ import annotations

from typing import Protocol, runtime_checkable

from application.models.vision_config import VisionModelConfig
from domain.events import ChannelEvent, RTSPEvent


@runtime_checkable
class VisionModel(Protocol):
    """Domain port: turns an `RTSPEvent` into a `ChannelEvent` — usually a
    `DetectionsProducedEvent` or an `InferenceFailedEvent`."""

    cfg: VisionModelConfig

    async def infer(self, rtsp_ev: RTSPEvent) -> ChannelEvent: ...

    async def warmup(self) -> None: ...

    async def shutdown(self) -> None: ...
