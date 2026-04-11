# video_channel.py
import os
from typing import AsyncGenerator

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;2000000"
)

import asyncio
import logging
import time
import threading

import cv2

try:
    # Script mode (python main.py from Backend/tensort)
    from channels.channel_config import VideoChannelConfig
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from .channel_config import VideoChannelConfig

logger = logging.getLogger(__name__)

_DONE = object()



class ChannelEvent(object):
    __slots__ = ("type", "channel_id", "camera_uuid", "ts_ms")

    def __init__(self, event_type, channel_id, camera_uuid, ts_ms):
        self.type = event_type
        self.channel_id = channel_id
        self.camera_uuid = camera_uuid
        self.ts_ms = ts_ms


class ChannelConnectedEvent(ChannelEvent):
    __slots__ = ("rtsp_url",)

    def __init__(self, channel_id, camera_uuid, rtsp_url, ts_ms):
        ChannelEvent.__init__(self, "connected", channel_id, camera_uuid, ts_ms)
        self.rtsp_url = rtsp_url


class ChannelDisconnectedEvent(ChannelEvent):
    __slots__ = ("reason",)

    def __init__(self, channel_id, camera_uuid, reason, ts_ms):
        ChannelEvent.__init__(self, "disconnected", channel_id, camera_uuid, ts_ms)
        self.reason = reason


class FrameDroppedEvent(ChannelEvent):
    __slots__ = ("reason", "dropped_count")

    def __init__(self, channel_id, camera_uuid, reason, dropped_count, ts_ms):
        ChannelEvent.__init__(self, "dropped", channel_id, camera_uuid, ts_ms)
        self.reason = reason
        self.dropped_count = dropped_count


class RTSPEvent(ChannelEvent):
    __slots__ = (
        "seq",
        "format",
        "frame",
        "width",
        "height",
        "fps_hint",
        "detection_enabled",
    )

    def __init__(
        self,
        channel_id,
        camera_uuid,
        ts_ms,
        seq,
        fmt,
        frame,
        width,
        height,
        fps_hint,
        detection_enabled=True,
    ):
        ChannelEvent.__init__(self, "rtsp_frame", channel_id, camera_uuid, ts_ms)
        self.seq = seq
        self.format = fmt
        self.frame = frame
        self.width = width
        self.height = height
        self.fps_hint = fps_hint
        self.detection_enabled = detection_enabled




class VideoChannel():
    """
    RTSP ingest for Jetson inference only.

    - Reads RTSP frames (GStreamer preferred on Jetson; OpenCV fallback)
    - Emits RTSPEvent for your inference pipeline
    - Frontend video streaming should come from your Azure Media Server (WebRTC/HLS)
    """

    def __init__(self, config):
        if not isinstance(config, VideoChannelConfig):
            raise TypeError("config must be a VideoChannelConfig")

        self.config = config

        self._seq = 0
        self._loop = None
        self._thread = None
        self._stop_thread_evt = threading.Event()
        try:
            out_q_max = max(1, int(os.getenv("CHANNEL_OUT_Q_MAX", "1")))
        except Exception:
            out_q_max = 4
        self._out_q = asyncio.Queue(maxsize=out_q_max)
        self._event_queue = None
        self._cap = None
        self._stopping = False
        self._cap_lock = threading.Lock()
        
    def _build_gst_pipeline(self, rtsp_url, decoder):
        lat = int(self.config.gst_latency_ms)
        proto = self.config.rtsp_transport

        # Good practice: add a leaky queue so slow consumers don't blow up memory
        q = "queue max-size-buffers=1 leaky=downstream ! "

        if decoder == "nvv4l2decoder":
            # Jetson HW decode (NVMM) -> nvvidconv to CPU BGRx -> videoconvert to BGR for OpenCV
            return (
                "rtspsrc location={url} latency={lat} protocols={proto} "
                "drop-on-latency=true do-retransmission=false ! "
                + q +
                "rtph264depay ! "
                "h264parse config-interval=1 ! "
                "video/x-h264,stream-format=byte-stream,alignment=au ! "
                "nvv4l2decoder enable-max-performance=1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            ).format(url=rtsp_url, lat=lat, proto=proto)

        # CPU fallback decoder (avdec_h264)
        return (
            "rtspsrc location={url} latency={lat} protocols={proto} "
            "drop-on-latency=true do-retransmission=false ! "
            + q +
            "rtph264depay ! h264parse config-interval=1 ! "
            "{dec} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true sync=false max-buffers=1"
        ).format(url=rtsp_url, lat=lat, proto=proto, dec=decoder)

    def _open_capture(self):
        if self.config.decode_backend == "gstreamer":
            for dec in (self.config.gst_decoder, "avdec_h264"):
                gst = self._build_gst_pipeline(self.config.rtsp_url, dec)
                cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cap is not None and cap.isOpened():
                    logger.info("[%s] opened with gstreamer decoder=%s", self.config.camera_uuid, dec)
                    return cap

            logger.warning("[%s] gstreamer failed, falling back to OpenCV default RTSP capture", self.config.camera_uuid)
            cap = cv2.VideoCapture(self.config.rtsp_url)
            return cap

        try:
            cap = cv2.VideoCapture(self.config.rtsp_url, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(self.config.rtsp_url)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        return cap

    def _maybe_resize(self, frame):
        if self.config.resize is None:
            return frame
        target_w, target_h = self.config.resize
        fh, fw = frame.shape[:2]
        # Maintain aspect ratio: scale to fit within target dimensions
        scale = min(target_w / max(fw, 1), target_h / max(fh, 1))
        if scale >= 1.0:
            return frame  # don't upscale
        new_w = int(round(fw * scale))
        new_h = int(round(fh * scale))
        return cv2.resize(frame, (new_w, new_h))
    
    def _put_latest(self, ev):
        # if we're stopping, do not enqueue any new RTSPEvents
        if getattr(self, "_stopping", False) and ev is not _DONE:
            return

        # make _DONE sticky: clear queue and insert only _DONE
        if ev is _DONE:
            try:
                while True:
                    self._out_q.get_nowait()
            except Exception:
                pass
            try:
                self._out_q.put_nowait(_DONE)
            except Exception:
                pass
            return

        # normal leaky behavior for frames
        if self._out_q.full():
            try:
                self._out_q.get_nowait()
            except Exception:
                pass
        try:
            self._out_q.put_nowait(ev)
        except Exception:
            pass

    def _handle_in_loop(self, ev):
        self._put_latest(ev)
        if self._event_queue is not None and isinstance(ev, RTSPEvent):
            try:
                self._event_queue.put_nowait(ev)
            except Exception:
                pass

    def _push_from_thread(self, ev):
        if self._loop is None:
            return
        if self._stop_thread_evt.is_set():
            return
        self._loop.call_soon_threadsafe(self._handle_in_loop, ev)

    def _worker(self):
        backoff_ms = int(self.config.reconnect_base_ms)

        while not self._stop_thread_evt.is_set():
            cap = None
            try:
                cap = self._open_capture()
                
                with self._cap_lock:
                    self._cap = cap

                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open RTSP stream")

                ts_ms = int(time.time() * 1000)
                self._push_from_thread(
                    ChannelConnectedEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=self.config.camera_uuid,
                        rtsp_url=self.config.rtsp_url,
                        ts_ms=ts_ms,
                    )
                )

                backoff_ms = int(self.config.reconnect_base_ms)

                sample_fps = float(self.config.sample_fps)
                emit_interval_ms = int(1000.0 / sample_fps) if sample_fps > 0 else 0
                last_emit_ms = 0

                while not self._stop_thread_evt.is_set():
                    grabbed = False
                    try:
                        grabbed = cap.grab()
                    except Exception:
                        grabbed = False

                    if not grabbed:
                        raise RuntimeError("Frame grab failed")

                    ts_ms = int(time.time() * 1000)

                    if emit_interval_ms and (ts_ms - last_emit_ms) < emit_interval_ms:
                        continue

                    last_emit_ms = ts_ms

                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError("Frame retrieve failed")

                    frame = self._maybe_resize(frame)
                    self._seq += 1
                    h, w = frame.shape[:2]


                    ev = RTSPEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=self.config.camera_uuid,
                        ts_ms=ts_ms,
                        seq=self._seq,
                        fmt=self.config.emit_format,
                        frame=frame,
                        width=int(w),
                        height=int(h),
                        fps_hint=float(self.config.sample_fps),
                        detection_enabled=bool(self.config.detection_enabled),
                    )
                    self._push_from_thread(ev)

            except Exception as e:
                ts_ms = int(time.time() * 1000)
                self._push_from_thread(
                    ChannelDisconnectedEvent(
                        channel_id=self.config.channel_id,
                        camera_uuid=self.config.camera_uuid,
                        reason=str(e),
                        ts_ms=ts_ms,
                    )
                )

                if self._stop_thread_evt.wait(backoff_ms / 1000.0):
                    break
                backoff_ms = min(backoff_ms * 2, int(self.config.reconnect_max_ms))

            finally:
                try:
                    with self._cap_lock:
                        if self._cap is not None:
                            self._cap.release()
                        self._cap = None
                except Exception:
                    pass

    async def stream(self, event_queue=None) -> AsyncGenerator[ChannelEvent, None]:
        """
        Async generator of ChannelEvent.

        If you pass event_queue (recommended: asyncio.Queue),
        RTSPEvent is also pushed there best-effort for your inference pipeline.
        """
        if not self.config.enabled:
            return

        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        self._event_queue = event_queue

        if self._thread is None or not self._thread.is_alive():
            self._stop_thread_evt.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="VideoChannel-{}".format(self.config.channel_id),
                daemon=True,
            )
            self._thread.start()

        while True:
            item = await self._out_q.get()
            if item is _DONE:
                break
            yield item

    async def stop(self):
        # idempotent
        if getattr(self, "_stopping", False):
            return
        self._stopping = True

        cam = getattr(self.config, "camera_uuid", None) or getattr(self.config, "channel_id", "unknown")
        logger.info(f"[{cam}] Stopping channel thread...")
        self._stop_thread_evt.set()

        # Signal stream() to end cleanly (unblocks async consumers)
        if self._loop:
            self._loop.call_soon_threadsafe(self._put_latest, _DONE)

        # DO NOT call self._cap.release() here (race -> segfault).
        # Worker thread will exit and release in its finally.

        t = self._thread
        if t and t.is_alive():
            # Never block the asyncio loop with a thread join. A stalled RTSP read
            # can take several seconds to unwind; if we join inline, every Flask
            # request that calls into the pipeline (latest, detection, patch) can
            # time out waiting for the loop.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, t.join, 6.0)
            if t.is_alive():
                logger.warning("[%s] Channel thread did not stop within 6s", cam)

        if t and not t.is_alive():
            self._thread = None
