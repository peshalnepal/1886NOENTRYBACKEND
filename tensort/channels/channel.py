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



class RTSPEvent(object):
    """One decoded frame on its way to the inference pipeline.

    The only event type the channel emits. Connect/disconnect used to be
    events too, but nothing on the edge consumed them — the cloud infers link
    state from the detection flow — so they are plain log lines now.
    """

    __slots__ = (
        "channel_id",
        "camera_uuid",
        "ts_ms",
        "seq",
        "frame",
        "detection_enabled",
    )

    def __init__(
        self,
        channel_id,
        camera_uuid,
        ts_ms,
        seq,
        frame,
        detection_enabled=True,
    ):
        self.channel_id = channel_id
        self.camera_uuid = camera_uuid
        self.ts_ms = ts_ms
        self.seq = seq
        self.frame = frame
        self.detection_enabled = detection_enabled


class VideoChannel():
    """
    Multi-scheme video ingest for Jetson inference.

    Accepts any source_url scheme: rtsp/rtsps, srt, rtmp/rtmps, http/https
    (HLS / MJPEG / progressive) and webrtc/whep. On Jetson it prefers
    hardware-accelerated GStreamer pipelines (nvv4l2decoder + nvvidconv) and
    falls back to CPU GStreamer, then to OpenCV/FFmpeg generic capture.

    - Emits RTSPEvent for your inference pipeline (name kept for compatibility)
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
        self._cap = None
        self._stopping = False
        self._cap_lock = threading.Lock()
        
    @staticmethod
    def _url_scheme(url):
        url = str(url or "").strip().lower()
        return url.split("://", 1)[0] if "://" in url else ""

    def _gst_appsink_tail(self, hw):
        """Trailing convert+resize+appsink segment shared by every pipeline.

        ``hw=True`` uses Jetson's nvvidconv (handles NVMM output from
        nvv4l2decoder); ``hw=False`` uses CPU videoscale/videoconvert. Both end
        in BGR frames for the OpenCV appsink. Resize matches the historical RTSP
        behaviour (fixed WxH; aspect handled later by the TRT letterbox).
        """
        resize = self.config.resize
        if hw:
            if resize is not None:
                w, h = resize
                conv = (
                    "nvvidconv interpolation-method=1 ! "
                    "video/x-raw,width={w},height={h},format=BGRx ! "
                ).format(w=int(w), h=int(h))
            else:
                conv = "nvvidconv ! video/x-raw,format=BGRx ! "
        else:
            if resize is not None:
                w, h = resize
                conv = "videoscale ! video/x-raw,width={w},height={h} ! ".format(
                    w=int(w), h=int(h)
                )
            else:
                conv = ""
        return conv + "videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1"

    def _build_rtsp_pipeline(self, url, decoder):
        """RTSP fast path: depay/parse H.264 + (HW or CPU) decode."""
        lat = int(self.config.gst_latency_ms)
        proto = self.config.rtsp_transport
        q = "queue max-size-buffers=1 leaky=downstream ! "

        if decoder == "nvv4l2decoder":
            return (
                "rtspsrc location={url} latency={lat} protocols={proto} "
                "drop-on-latency=true do-retransmission=false ! "
                + q +
                "rtph264depay ! "
                "h264parse config-interval=1 ! "
                "video/x-h264,stream-format=byte-stream,alignment=au ! "
                "nvv4l2decoder enable-max-performance=1 ! "
                + self._gst_appsink_tail(hw=True)
            ).format(url=url, lat=lat, proto=proto)
        # CPU fallback decoder (avdec_h264)
        return (
            "rtspsrc location={url} latency={lat} protocols={proto} "
            "drop-on-latency=true do-retransmission=false ! "
            + q +
            "rtph264depay ! h264parse config-interval=1 ! "
            "{dec} ! "
            + self._gst_appsink_tail(hw=False)
        ).format(url=url, lat=lat, proto=proto, dec=decoder)

    def _build_uridecodebin_pipeline(self, url, hw):
        """Generic pipeline for srt/rtmp/http(HLS/MJPEG/progressive) and rtsp.

        ``uridecodebin`` auto-plugs the right source + demux + decoder for the
        URI (and on Jetson picks nvv4l2decoder for H.264/H.265, emitting NVMM
        that the hw tail's nvvidconv consumes). The quoted uri keeps query
        strings (``?...``) from breaking gst-launch parsing.
        """
        q = "queue max-size-buffers=1 leaky=downstream ! "
        return (
            'uridecodebin uri="{url}" ! '.format(url=url)
            + q
            + self._gst_appsink_tail(hw=hw)
        )

    def _build_whep_pipeline(self, url, hw):
        """Best-effort WebRTC/WHEP ingest via the gst webrtchttp ``whepsrc``.

        Requires the GStreamer ``webrtchttp`` plugin (gst-plugins-rs). The
        whep/webrtc scheme is normalised to the https signalling endpoint.
        Falls back to other candidates / OpenCV if the plugin is unavailable.
        """
        endpoint = url
        # wheps:// is WHEP over HTTPS, whep://webrtc:// are plain; map both to the
        # http(s) signalling endpoint whepsrc expects.
        for prefix, target in (("wheps://", "https://"), ("whep://", "https://"), ("webrtc://", "https://")):
            if endpoint.lower().startswith(prefix):
                endpoint = target + endpoint[len(prefix):]
                break
        q = "queue max-size-buffers=1 leaky=downstream ! "
        return (
            'whepsrc whep-endpoint="{ep}" ! '.format(ep=endpoint)
            + q
            + "decodebin ! "
            + self._gst_appsink_tail(hw=hw)
        )

    def _gst_candidates(self):
        """Ordered (name, pipeline) GStreamer attempts for the source scheme."""
        url = self.config.source_url or ""
        scheme = self._url_scheme(url)
        cands = []
        if scheme in ("rtsp", "rtsps"):
            cands.append(("rtsp-nvv4l2decoder", self._build_rtsp_pipeline(url, self.config.gst_decoder)))
            cands.append(("rtsp-avdec_h264", self._build_rtsp_pipeline(url, "avdec_h264")))
            # Generic fallback also covers H.265 / non-H264 RTSP cameras.
            cands.append(("rtsp-uridecodebin-hw", self._build_uridecodebin_pipeline(url, hw=True)))
            cands.append(("rtsp-uridecodebin-cpu", self._build_uridecodebin_pipeline(url, hw=False)))
        elif scheme in ("whep", "wheps", "webrtc"):
            cands.append(("whep-hw", self._build_whep_pipeline(url, hw=True)))
            cands.append(("whep-cpu", self._build_whep_pipeline(url, hw=False)))
        else:
            # srt / rtmp / rtmps / http / https (HLS, MJPEG, progressive) / other
            cands.append(("uridecodebin-hw", self._build_uridecodebin_pipeline(url, hw=True)))
            cands.append(("uridecodebin-cpu", self._build_uridecodebin_pipeline(url, hw=False)))
        return cands

    def _open_capture(self):
        url = self.config.source_url or ""
        scheme = self._url_scheme(url) or "?"

        if self.config.decode_backend == "gstreamer":
            for name, gst in self._gst_candidates():
                cap = None
                try:
                    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                except Exception:
                    cap = None
                if cap is not None and cap.isOpened():
                    logger.info(
                        "[%s] opened %s source via gstreamer pipeline=%s",
                        self.config.camera_uuid, scheme, name,
                    )
                    return cap
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

            logger.warning(
                "[%s] all gstreamer pipelines failed for %s source; falling back to OpenCV/FFmpeg",
                self.config.camera_uuid, scheme,
            )

        # OpenCV/FFmpeg generic capture (decode_backend="opencv" or gst fallback).
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(url)

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

    def _push_from_thread(self, ev):
        if self._loop is None:
            return
        if self._stop_thread_evt.is_set():
            return
        self._loop.call_soon_threadsafe(self._put_latest, ev)

    def _worker(self):
        backoff_ms = int(self.config.reconnect_base_ms)

        while not self._stop_thread_evt.is_set():
            cap = None
            try:
                cap = self._open_capture()
                
                with self._cap_lock:
                    self._cap = cap

                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open video source")

                logger.info("[%s] connected", self.config.camera_uuid)

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

                    self._push_from_thread(
                        RTSPEvent(
                            channel_id=self.config.channel_id,
                            camera_uuid=self.config.camera_uuid,
                            ts_ms=ts_ms,
                            seq=self._seq,
                            frame=frame,
                            detection_enabled=bool(self.config.detection_enabled),
                        )
                    )

            except Exception as e:
                logger.warning(
                    "[%s] disconnected: %s (retry in %dms)",
                    self.config.camera_uuid, e, backoff_ms,
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

    async def stream(self) -> AsyncGenerator[RTSPEvent, None]:
        """Async generator of decoded frames, newest-first under backpressure."""
        if not self.config.enabled:
            return

        if self._loop is None:
            self._loop = asyncio.get_event_loop()

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
