# agents/domain/model_pipeline.py

"""
Azure-side pipeline.

The system is migrating to an Edge/Cloud split:

- Jetson device: RTSP ingest + TensorRT inference + detections
- Azure backend: configuration + orchestration + WebRTC playback URL management

Therefore, this ModelPipeline is intentionally **not** an RTSP ingest pipeline.
It is a light runtime registry that can:
- keep track of active channels (by camera_uuid)
- optionally keep a latest-detection store (when Jetson pushes detections to Azure)

Frame streaming methods intentionally raise, because the frontend must use WebRTC.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple
from uuid import UUID

from domain.channel import Channel
from domain.events import DetectionItem, PoseResult, RTSPEvent

logger = logging.getLogger(__name__)
CameraKey = str


@dataclass(frozen=True)
class ObjDetectResponse:
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int

    site_uuid: Optional[str] = None
    device_uuid: Optional[str] = None

    detections: Tuple[DetectionItem, ...] = ()
    pose: Optional[PoseResult] = None
    inference_ms: Optional[int] = None


class ObjDetectStorePort(Protocol):
    async def put(self, resp: ObjDetectResponse) -> None: ...
    async def get_latest(self, camera_uuid: str) -> Optional[ObjDetectResponse]: ...
    async def wait_new(self, camera_uuid: str, *, after_seq: int, timeout_ms: int) -> Optional[ObjDetectResponse]: ...


class InMemoryObjDetectStore:
    """Latest-only per camera_uuid with an asyncio.Event per camera."""

    def __init__(self):
        self._latest: Dict[CameraKey, ObjDetectResponse] = {}
        self._signal: Dict[CameraKey, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def _evt(self, camera_uuid: str) -> asyncio.Event:
        async with self._lock:
            e = self._signal.get(camera_uuid)
            if e is None:
                e = asyncio.Event()
                self._signal[camera_uuid] = e
            return e

    async def put(self, resp: ObjDetectResponse) -> None:
        async with self._lock:
            self._latest[str(resp.camera_uuid)] = resp
        (await self._evt(str(resp.camera_uuid))).set()

    async def get_latest(self, camera_uuid: str) -> Optional[ObjDetectResponse]:
        async with self._lock:
            return self._latest.get(str(camera_uuid))

    async def wait_new(self, camera_uuid: str, *, after_seq: int, timeout_ms: int) -> Optional[ObjDetectResponse]:
        evt = await self._evt(str(camera_uuid))
        timeout_s = max(0.0, timeout_ms / 1000.0)

        while True:
            try:
                await asyncio.wait_for(evt.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return None

            async with self._lock:
                resp = self._latest.get(str(camera_uuid))
                evt.clear()

            if resp is not None and resp.frame_seq > after_seq:
                return resp


class ModelPipeline:
    """Azure-side lightweight runtime registry."""

    def __init__(
        self,
        pipeline_id: UUID,
        channels: Optional[List[Channel]] = None,
        *,
        detect_store: Optional[ObjDetectStorePort] = None,
    ):
        self.pipeline_id = pipeline_id
        self.detect_store = detect_store or InMemoryObjDetectStore()

        self._channels: Dict[str, Channel] = {}
        self._lock = asyncio.Lock()
        self._started = False

        for ch in (channels or []):
            self._channels[self._channel_key(ch)] = ch

    def _channel_key(self, ch: Channel) -> str:
        cfg = getattr(ch, "config", None)
        cid = getattr(cfg, "camera_uuid", None)
        return str(cid) if cid else f"ch-{id(ch)}"

    async def start(self) -> None:
        async with self._lock:
            self._started = True

    async def shutdown(self) -> None:
        async with self._lock:
            self._started = False
            self._channels.clear()

    async def add_channel(self, ch: Channel) -> None:
        async with self._lock:
            self._channels[self._channel_key(ch)] = ch

    async def edit_channel(self, ch: Channel) -> None:
        async with self._lock:
            self._channels[self._channel_key(ch)] = ch

    async def remove_channel(self, camera_uuid: str) -> bool:
        async with self._lock:
            self._channels.pop(str(camera_uuid), None)
        return True

    def list_channel_ids(self) -> List[str]:
        return list(self._channels.keys())

    async def patch_channel_config(self, camera_uuid: str, patch: dict) -> bool:
        key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.get(key)
            if ch is None:
                return False

            cfg = getattr(ch, "config", None)
            if cfg is None:
                return False

            if hasattr(cfg, "model_dump"):
                data = cfg.model_dump()
            elif hasattr(cfg, "__slots__"):
                data = {k: getattr(cfg, k) for k in cfg.__slots__ if hasattr(cfg, k)}
            else:
                try:
                    data = dict(cfg)
                except Exception:
                    return False

            for k, v in (patch or {}).items():
                if v is not None:
                    data[k] = v

            try:
                new_cfg = cfg.__class__(**data)
                setattr(ch, "config", new_cfg)
                return True
            except Exception:
                logger.exception("Failed to patch config for %s", key)
                return False

    # ---- detections (metadata only) ----
    async def put_detection(self, resp: ObjDetectResponse) -> None:
        await self.detect_store.put(resp)

    async def get_latest_detection(self, camera_uuid: str) -> Optional[ObjDetectResponse]:
        return await self.detect_store.get_latest(str(camera_uuid))

    # ---- legacy frame API (disabled) ----
    async def stream(self, camera_uuid: str):
        raise RuntimeError("Frame streaming is disabled. Use WebRTC gateway for video playback.")

    async def stream_with_detections(self, camera_uuid: str, *args, **kwargs):
        raise RuntimeError("Frame streaming is disabled. Use WebRTC gateway for video playback.")


__all__ = [
    "ModelPipeline",
    "ObjDetectResponse",
    "ObjDetectStorePort",
    "InMemoryObjDetectStore",
    "RTSPEvent",
]
