# agents/domain/model.py

import asyncio
import logging
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)
from uuid import UUID

from domain.channel import Channel
from domain.events import (
    ChannelEvent,
    RTSPEvent,
    DetectionItem,
    DetectionsProducedEvent,
)

logger = logging.getLogger(__name__)

CameraKey = str  # camera_uuid only

# -------------------------
# ObjDetectResponse (store)
# -------------------------

@dataclass(frozen=True)
class ObjDetectResponse:
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int
    frame: Any = None
    encoded: Optional[bytes] = None
    detections: Tuple[DetectionItem, ...] = ()
    inference_ms: Optional[int] = None


class ObjDetectStorePort(Protocol):
    async def put(self, resp: ObjDetectResponse) -> None: ...
    async def get_latest(self, camera_uuid: str) -> Optional[ObjDetectResponse]: ...


class InMemoryObjDetectStore:
    def __init__(self):
        self._latest: Dict[CameraKey, ObjDetectResponse] = {}
        self._lock = asyncio.Lock()

    async def put(self, resp: ObjDetectResponse) -> None:
        async with self._lock:
            self._latest[resp.camera_uuid] = resp

    async def get_latest(self, camera_uuid: str) -> Optional[ObjDetectResponse]:
        async with self._lock:
            return self._latest.get(camera_uuid)


# -------------------------
# Stream Buffer (latest-wins per camera_uuid)
# -------------------------

class StreamBufferPort(Protocol):
    async def put(self, ev: RTSPEvent) -> None: ...
    async def get(self, camera_uuid: str) -> RTSPEvent: ...
    async def peek_latest(self, camera_uuid: str) -> Optional[RTSPEvent]: ...


class LatestStreamBuffer:
    """
    Latest frame per camera_uuid.
    get(camera_uuid) blocks until a *new* frame arrives for that camera_uuid.
    """
    def __init__(self):
        self._latest: Dict[CameraKey, RTSPEvent] = {}
        self._signal: Dict[CameraKey, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def _evt(self, camera_uuid: str) -> asyncio.Event:
        async with self._lock:
            e = self._signal.get(camera_uuid)
            if e is None:
                e = asyncio.Event()
                self._signal[camera_uuid] = e
            return e

    async def put(self, ev: RTSPEvent) -> None:
        camera_uuid = ev.camera_uuid
        async with self._lock:
            self._latest[camera_uuid] = ev
        evt = await self._evt(camera_uuid)
        evt.set()

    async def get(self, camera_uuid: str) -> RTSPEvent:
        evt = await self._evt(camera_uuid)
        while True:
            await evt.wait()
            async with self._lock:
                ev = self._latest.get(camera_uuid)
                evt.clear()
            if ev is not None:
                return ev

    async def peek_latest(self, camera_uuid: str) -> Optional[RTSPEvent]:
        async with self._lock:
            return self._latest.get(camera_uuid)


# -------------------------
# Inference Buffer (coalescing, no backlog)
# -------------------------

class InferenceBufferPort(Protocol):
    async def put(self, ev: RTSPEvent) -> None: ...
    async def get(self) -> RTSPEvent: ...


class CoalescingInferenceBuffer:
    """
    Latest-per-camera buffer + queue of camera_uuids.
    Prevents backlog when you push every frame.
    """
    def __init__(self, max_pending_keys: int = 1000):
        self._latest: Dict[CameraKey, RTSPEvent] = {}
        self._pending: asyncio.Queue[CameraKey] = asyncio.Queue(maxsize=max_pending_keys)
        self._in_queue: set[CameraKey] = set()
        self._lock = asyncio.Lock()

    async def put(self, ev: RTSPEvent) -> None:
        key = ev.camera_uuid
        async with self._lock:
            self._latest[key] = ev
            if key in self._in_queue:
                return
            try:
                self._pending.put_nowait(key)
                self._in_queue.add(key)
            except asyncio.QueueFull:
                pass

    async def get(self) -> RTSPEvent:
        while True:
            key = await self._pending.get()
            async with self._lock:
                self._in_queue.discard(key)
                ev = self._latest.get(key)
            if ev is not None:
                return ev


# -------------------------
# Model Pipeline
# -------------------------

_DONE = object()
PostProcessor = Callable[[ChannelEvent], Sequence[ChannelEvent]]

class ModelPipeline:
    """
    Runtime pipeline that:
      - starts long-running pumps once (channels + inference)
      - allows dynamic add/remove of channels
      - keeps yielding events "going forward" via handle_event()

    Works with your VideoChannel where channel_id is inside channel.config
    and stream() signature is stream(self, event: ChannelEvent).
    """

    def __init__(
        self,
        pipeline_id: UUID,
        channels: Optional[List[Channel]] = None,
        model: Optional[Any] = None,
        inference_buffer: Optional["InferenceBufferPort"] = None,
        stream_buffer: Optional["StreamBufferPort"] = None,
        detect_store: Optional["ObjDetectStorePort"] = None,
        postprocessors: Optional[List[PostProcessor]] = None,
        out_queue_maxsize: int = 500,
    ):
        self.pipeline_id = pipeline_id
        self.model = model

        # default buffers if not injected
        self.inference_buffer = inference_buffer or CoalescingInferenceBuffer()
        self.stream_buffer = stream_buffer or LatestStreamBuffer()
        self.detect_store = detect_store or InMemoryObjDetectStore()

        self.postprocessors = postprocessors or []

        self._out_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=out_queue_maxsize)

        # dynamic channel registry
        self._channels: Dict[str, Channel] = {}
        self._channel_tasks: Dict[str, asyncio.Task] = {}

        # lifecycle
        self._lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._inference_task: Optional[asyncio.Task] = None

        for ch in (channels or []):
            key = self._channel_key(ch)
            self._channels[key] = ch

    # -------------------------
    # Channel management
    # -------------------------

    def _channel_key(self, ch: Channel) -> str:
        """
        Your VideoChannel does NOT have .channel_id.
        It exists at ch.config.channel_id.
        """
        cfg = getattr(ch, "config", None)
        cid = getattr(cfg, "camera_uuid", None)
        if cid:
            return str(cid)
        # fallback (should not happen in your system)
        return f"ch-{id(ch)}"

    async def add_channel(self, ch: Channel) -> None:
        """
        Add a channel at runtime. If pipeline already started, it begins pumping immediately.
        """
        key = self._channel_key(ch)
        async with self._lock:
            self._channels[key] = ch
            if self._started and not self._closing:
                t = self._channel_tasks.get(key)
                if t is None or t.done():
                    self._start_channel_task(key, ch)

    async def add_channels(self, channels: List[Channel]) -> None:
        for ch in channels:
            await self.add_channel(ch)

    async def remove_channel(self, camera_uuid: str) -> bool:
        """
        Stop pumping a channel and remove it.
        (channel_id here is the config.channel_id)
        """
        async with self._lock:
            t = self._channel_tasks.pop(camera_uuid, None)
            self._channels.pop(camera_uuid, None)

        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                return False
        return True

    def list_channel_ids(self) -> List[str]:
        return list(self._channels.keys())

    # -------------------------
    # Lifecycle
    # -------------------------

    async def start(self) -> None:
        """
        Start inference loop + start pumping all currently registered channels.
        Safe to call multiple times.
        """
        async with self._lock:
            if self._started:
                return
            self._closing = False
            self._started = True

            self._inference_task = asyncio.create_task(self._pump_inference(), name="pump_inference")

            for key, ch in list(self._channels.items()):
                self._start_channel_task(key, ch)

    async def shutdown(self) -> None:
        """
        Stop all pumps. After shutdown, handle_event() will eventually terminate.
        """
        async with self._lock:
            if not self._started:
                return
            self._closing = True

            for t in list(self._channel_tasks.values()):
                if t and not t.done():
                    t.cancel()

            if self._inference_task and not self._inference_task.done():
                self._inference_task.cancel()

        # await outside lock
        for key, t in list(self._channel_tasks.items()):
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Channel task failed during shutdown: %s", key)

        if self._inference_task:
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Inference task failed during shutdown")

        async with self._lock:
            self._channel_tasks.clear()
            self._started = False

        # signal consumers
        await self._out_q.put(_DONE)

    # -------------------------
    # Event APIs
    # -------------------------

    async def handle_event(self) -> AsyncGenerator[ChannelEvent, None]:
        """
        Yields non-frame events + detection events continuously.
        IMPORTANT: this does NOT start/cancel pumps every call.
        """
        if not self._started:
            await self.start()

        while True:
            item = await self._out_q.get()
            if item is _DONE:
                break
            yield item
            self._out_q.task_done()

    async def stream(self, camera_uuid: str) -> AsyncGenerator[RTSPEvent, None]:
        """
        Per-camera stream of latest RTSPEvent.
        """
        if not self._started:
            await self.start()

        while True:
            ev = await self.stream_buffer.get(camera_uuid)
            yield ev

    async def get_latest_detection(self, camera_uuid: str) -> Optional["ObjDetectResponse"]:
        return await self.detect_store.get_latest(camera_uuid)

    # -------------------------
    # Internals: channel pumps
    # -------------------------

    def _start_channel_task(self, key: str, ch: Channel) -> None:
        self._channel_tasks[key] = asyncio.create_task(
            self._pump_single_channel(key, ch),
            name=f"pump_channel:{key}",
        )

    async def _pump_single_channel(self, key: str, ch: Channel) -> None:
        """
        Consume a single channel forever and route RTSPEvents to buffers.
        Your VideoChannel signature is stream(self, event: ChannelEvent),
        so we call: ch.stream(None)
        """
        try:
            async for ev in ch.stream(None):  # <-- IMPORTANT for your VideoChannel
                if self._closing:
                    break

                if isinstance(ev, RTSPEvent):
                    await self.stream_buffer.put(ev)
                    await self.inference_buffer.put(ev)
                else:
                    await self._out_q.put(ev)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Channel pump failed (%s): %s", key, e, exc_info=True)


    async def _pump_inference(self) -> None:
        """
        Single-model inference loop (YOLO) reading from inference_buffer.
        """
        try:
            while not self._closing:
                rtsp_ev = await self.inference_buffer.get()

                if not self.model:
                    continue

                det_event = await self.model.infer(rtsp_ev)

                if isinstance(det_event, DetectionsProducedEvent):
                    resp = ObjDetectResponse(
                        camera_uuid=det_event.camera_uuid,
                        frame_ts_ms=det_event.frame_ts_ms,
                        frame_seq=det_event.frame_seq,
                        frame=getattr(rtsp_ev, "frame", None),
                        encoded=getattr(rtsp_ev, "encoded", None),
                        detections=tuple(det_event.detections),
                        inference_ms=getattr(det_event, "inference_ms", None),
                    )
                    await self.detect_store.put(resp)

                await self._out_q.put(det_event)

                for pp in self.postprocessors:
                    for out_ev in pp(det_event):
                        await self._out_q.put(out_ev)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Inference pump failed: %s", e, exc_info=True)