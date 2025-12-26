# agents/application/channels/video_channel.py

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Deque, Optional, Tuple

import cv2

from application.channels.channel_config import VideoChannelConfig
from domain.channel import Channel
from domain.events import (
    ChannelEvent,
    RTSPEvent,
    ChannelConnectedEvent,
    ChannelDisconnectedEvent,
    FrameDroppedEvent,
)
from domain.model import ModelPipeline
logger = logging.getLogger(__name__)

_DONE_SENTINEL = object()


class EventBufferPort:
    async def put(self, event: RTSPEvent) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class _FramePacket:
    ts_ms: int
    frame: Any  # numpy ndarray (BGR)


class VideoChannel(Channel):
    """
    RTSP ingest channel.

    NEW behavior:
      - capture loop fills a per-channel ring buffer + latest pointer
      - emit loop samples at sample_fps, pulls latest from ring, builds RTSPEvent,
        and pushes into the shared event buffer for YOLO
      - stream() yields ChannelEvents (connected/disconnected/dropped + optional RTSPEvent)
    """

    # Helps orchestrator avoid double-publishing RTSPEvent into the shared buffer.
    publishes_to_event_buffer: bool = True

    def __init__(self, config: VideoChannelConfig):
        self.config = config

        # observer stream queue (yields outward)
        self._out_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=50)

        # tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._emit_task: Optional[asyncio.Task] = None

        # sequencing
        self._seq: int = 0
        self._last_emitted_frame_ts_ms: int = -1

        # ring buffer (for clip writer + "latest frame" sampling)
        self._frame_lock = asyncio.Lock()
        self._latest_frame: Optional[_FramePacket] = None
        self._ring: Deque[_FramePacket] = deque(maxlen=self._ring_maxlen())
        # connection coordination (emit loop waits for connection)
        self._connected_evt = asyncio.Event()

        logger.info(
            "VideoChannel initialized channel_id=%s camera_uuid=%s backend=%s sample_fps=%.2f ring_maxlen=%d",
            self.config.channel_id,
            str(self.config.camera_uuid),
            self.config.decode_backend,
            self.config.sample_fps,
            self._ring.maxlen or 0,
        )

    # ---------------------------
    # Ring sizing
    # ---------------------------
    def _ring_maxlen(self) -> int:
        """
        Prefer an explicit config if available, otherwise fallback to seconds * fps_hint.
        This is for storing the last ~N seconds of frames for clip writing.
        """
        max_frames = getattr(self.config, "ring_buffer_max_frames", None)
        if max_frames is not None:
            return max(1, int(max_frames))

        seconds = int(getattr(self.config, "ring_buffer_seconds", 20))
        fps_hint = float(getattr(self.config, "capture_fps_hint", 30.0))
        return max(1, int(seconds * fps_hint))

    # ---------------------------
    # RTSP open helpers
    # ---------------------------
    def _build_gst_pipeline(self, rtsp_url: str) -> str:
        return (
            f"rtspsrc location={rtsp_url} latency=200 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false"
        )

    def _open_capture(self) -> cv2.VideoCapture:
        # if self.config.decode_backend == "gstreamer":
        #     gst = self._build_gst_pipeline(self.config.rtsp_url)
        #     cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        # else:
        cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
            

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    # ---------------------------
    # Frame transforms for inference event
    # ---------------------------
    def _maybe_resize(self, frame):
        if not self.config.resize:
            return frame
        w, h = self.config.resize
        return cv2.resize(frame, (w, h))

    def _encode_if_needed(self, frame):
        if self.config.emit_format == "jpeg":
            ok, enc = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)],
            )
            if not ok:
                return None, None
            return None, enc.tobytes()
        return frame, None
    
    async def _put_latest_event(self, ev: ChannelEvent) -> None:
        """
        latest-wins: keep only most recent RTSPEvent in frame queue
        """
        if self._out_q.full():
            try:
                _ = self._out_q.get_nowait()
            except Exception:
                pass
        try:
            self._out_q.put_nowait(ev)
        except Exception:
            # if still fails, silently drop
            pass

    # ---------------------------
    # Capture loop: fills ring buffer + latest pointer
    # ---------------------------
    async def _capture_loop(self) -> None:
        backoff_ms = int(self.config.reconnect_base_ms)
        while True:
            cap = None
            try:
                cap = await asyncio.to_thread(self._open_capture)
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open RTSP stream")

                # Connected
                self._connected_evt.set()
                
                await self._put_latest_event(
                    ChannelConnectedEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=str(self.config.camera_uuid),
                        rtsp_url=self.config.rtsp_url,
                    )
                )

                backoff_ms = int(self.config.reconnect_base_ms)
                while True:
                    ok, frame = await asyncio.to_thread(cap.read)
                    if not ok or frame is None:
                        raise RuntimeError("Frame read failed")

                    ts_ms = int(time.time() * 1000)

                    self._seq += 1

                    # build inference payload from latest frame
                    frame_for_ev = self._maybe_resize(frame)
                    raw_frame, enc_bytes = self._encode_if_needed(frame_for_ev)

                    if self.config.emit_format == "jpeg" and enc_bytes is None:
                        await self._safe_emit_drop("jpeg_encode_failed")
                        continue

                    h = int(frame_for_ev.shape[0]) if hasattr(frame_for_ev, "shape") else None
                    w = int(frame_for_ev.shape[1]) if hasattr(frame_for_ev, "shape") else None
                    shape = getattr(frame_for_ev, "shape", None)
                    ev = RTSPEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=str(self.config.camera_uuid),
                        ts_ms=ts_ms,
                        seq=self._seq,
                        format=self.config.emit_format,
                        frame=raw_frame,
                        frame_shape=shape,
                        encoded=enc_bytes,
                        width=w,
                        height=h,
                        fps_hint=float(self.config.sample_fps),
                    )
                    try:
                        await self._put_latest_event(ev)
                    except:
                        self._safe_emit_drop("ubable to store the values")

            except asyncio.CancelledError:
                raise
            
            except Exception as e:
                # Disconnected
                self._connected_evt.clear()
                logger.warning(
                    "VideoChannel disconnected channel_id=%s camera_uuid=%s reason=%s",
                    self.config.channel_id,
                    str(self.config.camera_uuid),
                    str(e),
                )
                try:
                    await self._put_latest_event(
                        ChannelDisconnectedEvent(
                            channel_id=self.config.channel_id,
                            camera_uuid=str(self.config.camera_uuid),
                            reason=str(e),
                        )
                    )
                except Exception:
                    pass

                # reconnect with bounded backoff
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, int(self.config.reconnect_max_ms))

            finally:
                try:
                    if cap is not None:
                        await asyncio.to_thread(cap.release)
                except Exception:
                    pass
                
    async def _safe_emit_drop(self, reason: str, dropped_count: int = 1) -> None:
        try:
            await self._put_latest_event(
                FrameDroppedEvent(
                    channel_id=self.config.channel_id,
                    camera_uuid=str(self.config.camera_uuid),
                    reason=reason,
                    dropped_count=dropped_count,
                )
            )
        except Exception:
            pass

    # ---------------------------
    # Channel API
    # ---------------------------
    async def stream(self, event: ChannelEvent) -> AsyncGenerator[ChannelEvent, None]:
        """
        Starts capture + emit on first call and yields:
          - ChannelConnectedEvent / ChannelDisconnectedEvent
          - RTSPEvent (built from latest frame in ring buffer)
          - FrameDroppedEvent
        """
        if not self.config.enabled:
            logger.info("VideoChannel disabled channel_id=%s", self.config.channel_id)
            return

        # start tasks once
        if self._capture_task is None or self._capture_task.done():
            self._capture_task = asyncio.create_task(self._capture_loop())

        try:
            while True:
                item = await self._out_q.get()
                if item is _DONE_SENTINEL:
                    break
                try:
                    yield item
                finally:
                    self._out_q.task_done()
        finally:
            # stop tasks if consumer stops
            if self._capture_task and not self._capture_task.done():
                self._capture_task.cancel()
                try:
                    await self._capture_task
                except asyncio.CancelledError:
                    pass

    # Optional: expose ring buffer to clip-writer (read-only snapshot)
    async def snapshot_ring(self) -> Tuple[_FramePacket, ...]:
        async with self._frame_lock:
            return tuple(self._ring)
