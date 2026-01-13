# agents/application/channels/video_channel.py
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;2000000"

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Deque, Optional, Tuple

import cv2
import threading

from application.channels.channel_config import VideoChannelConfig
from domain.channel import Channel
from domain.events import (
    ChannelEvent,
    RTSPEvent,
    ChannelConnectedEvent,
    ChannelDisconnectedEvent,
    FrameDroppedEvent,
)
from domain.model_pipeline import ModelPipeline
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
        self._out_q: asyncio.Queue[Any] = asyncio.Queue(maxsize=2)

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
        self._stop_evt = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._worker_thread: Optional[threading.Thread] = None
        self._stop_thread_evt = threading.Event()
        
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

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
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
    
    def _put_latest_event(self, ev: ChannelEvent) -> None:
        """
        latest-wins: keep only most recent RTSPEvent in frame queue
        """
        if self._out_q.full():
            try:
                _ = self._out_q.get_nowait()
                # Optional: don't use task_done at all unless you call queue.join()
                # self._out_q.task_done()
            except Exception:
                pass

        try:
            self._out_q.put_nowait(ev)
        except Exception:
            pass
        
    def _push_from_thread(self, ev: Any) -> None:
        """
        Called from capture thread.
        Schedules the actual queue put onto the asyncio loop thread.
        """
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._put_latest_event, ev)

    def _start_worker_thread(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_thread_evt.clear()

        self._worker_thread = threading.Thread(
            target=self._capture_worker,
            name=f"VideoChannel-{self.config.channel_id}",
            daemon=True,  # ensures app can exit even if a thread is stuck
        )
        self._worker_thread.start()

   
    async def stop(self) -> None:
        self._stop_evt.set()
        self._stop_thread_evt.set()

        # unblock stream consumer
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._put_latest_event, _DONE_SENTINEL)

        # join thread without blocking event loop
        t = self._worker_thread
        if t and t.is_alive():
            await asyncio.to_thread(t.join, 1.0)
            
            
    def _capture_worker(self) -> None:
        backoff_ms = int(self.config.reconnect_base_ms)

        while not self._stop_thread_evt.is_set():
            cap = None
            try:
                cap = self._open_capture()
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open RTSP stream")

                self._connected_evt.set()
                self._push_from_thread(
                    ChannelConnectedEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=str(self.config.camera_uuid),
                        rtsp_url=self.config.rtsp_url,
                    )
                )

                backoff_ms = int(self.config.reconnect_base_ms)

                sample_fps = float(self.config.sample_fps or 0.0)
                emit_interval_ms = int(1000 / sample_fps) if sample_fps > 0 else 0
                last_emit_ms = 0

                while not self._stop_thread_evt.is_set():
                    ok = cap.grab()
                    if not ok:
                        raise RuntimeError("Frame grab failed")

                    ts_ms = int(time.time() * 1000)

                    if emit_interval_ms and (ts_ms - last_emit_ms) < emit_interval_ms:
                        continue
                    last_emit_ms = ts_ms

                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError("Frame retrieve failed")
                    if self.config.resize:
                        frame_for_ev = self._maybe_resize(frame)   # allocates new
                    else:
                        frame_for_ev = frame.copy()                # safe buffer

                    self._seq += 1
                    shape = getattr(frame_for_ev, "shape", None)
                    h = int(shape[0]) if shape is not None else None
                    w = int(shape[1]) if shape is not None else None
                    frame_field, encoded = self._encode_if_needed(frame_for_ev)

                    ev = RTSPEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=str(self.config.camera_uuid),
                        ts_ms=ts_ms,
                        detection_enabled=getattr(self.config, "detection_enabled", True),
                        seq=self._seq,
                        format=self.config.emit_format,
                        frame=frame_field,
                        frame_shape=shape,
                        encoded=encoded,
                        width=w,
                        height=h,
                        fps_hint=float(self.config.sample_fps),
                    )
                    self._push_from_thread(ev)

            except Exception as e:
                self._connected_evt.clear()
                self._push_from_thread(
                    ChannelDisconnectedEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=str(self.config.camera_uuid),
                        reason=str(e),
                    )
                )

                # bounded backoff
                time.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, int(self.config.reconnect_max_ms))

            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
     
    async def _safe_emit_drop(self, reason: str, dropped_count: int = 1) -> None:
        try:
            self._put_latest_event(
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
        if not self.config.enabled:
            logger.info("VideoChannel disabled channel_id=%s", self.config.channel_id)
            return

        # Capture the running loop ONCE (must be done inside async context)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        self._start_worker_thread()

        try:
            while True:
                item = await self._out_q.get()
                if item is _DONE_SENTINEL:
                    break
                try:
                    yield item
                finally:
                    pass
        finally:
            pass


    # Optional: expose ring buffer to clip-writer (read-only snapshot)
    async def snapshot_ring(self) -> Tuple[_FramePacket, ...]:
        async with self._frame_lock:
            return tuple(self._ring)

