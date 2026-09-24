# video_channel.py
import os
from typing import AsyncGenerator

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|max_delay;2000000"
)

import asyncio
import logging
import random
import time
import threading
from fractions import Fraction

import cv2

from .channel_config import VideoChannelConfig
from .gstreamer_capture import GstCapture, native_gstreamer_available

logger = logging.getLogger(__name__)

_DONE = object()


class _ConnectGate(object):
    """Serialises how fast capture sessions may be opened, process-wide.

    Every camera behind one NVR shares that NVR's uplink and its pool of RTSP
    sessions. When all of them open at once — at boot, or one second after a
    common failure — they race each other, the NVR sheds sessions, and every
    channel retries in lockstep. The result is a permanent storm of "Internal
    data stream error" in which no camera ever establishes a stable stream.

    This gate spaces connection attempts out. It is deliberately global rather
    than per-camera: the contended resource is the NVR, not the channel.
    """

    def __init__(self, min_interval_s):
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait_turn(self, stop_evt=None):
        """Block until this caller may open a capture.

        Returns False if `stop_evt` fired while waiting, so a channel being
        torn down abandons its turn instead of connecting on the way out.
        """
        if self._min_interval_s <= 0:
            return True

        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_allowed_at)
            self._next_allowed_at = start_at + self._min_interval_s
        delay = start_at - time.monotonic()
        if delay <= 0:
            return True
        if stop_evt is not None:
            if stop_evt.wait(delay):
                # Give the slot back. A channel torn down while queued used to
                # keep its reservation, so under add/replace churn every
                # surviving camera was pushed further back by cameras that never
                # connected at all — the queue drained slower than it filled and
                # later cameras never got a turn.
                with self._lock:
                    if self._next_allowed_at >= start_at + self._min_interval_s:
                        self._next_allowed_at = max(
                            time.monotonic(),
                            self._next_allowed_at - self._min_interval_s,
                        )
                return False
            return True
        time.sleep(delay)
        return True


try:
    _STAGGER_S = float(os.getenv("CAMERA_START_STAGGER_S", "1.5"))
except Exception:
    _STAGGER_S = 1.5

CONNECT_GATE = _ConnectGate(_STAGGER_S)


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
        # Coalesce before scheduling on asyncio: Queue(1) alone cannot bound
        # call_soon_threadsafe's pending callbacks when the loop is busy.
        self._handoff_lock = threading.Lock()
        self._pending_frame = None
        self._handoff_scheduled = False
        self._capture_backend = None
        self._gstreamer_failures = {}
        self._connected = False
        self._handoff_dropped = 0
        self._frames_emitted = 0
        self._last_frame_monotonic = None
        
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
        fps = Fraction(str(self.config.sample_fps)).limit_denominator(1000)
        rate = "videorate drop-only=true ! video/x-raw,framerate={}/{} ! ".format(fps.numerator, fps.denominator)
        # Limit CPU color conversion to the requested inference rate. Hardware
        # scaling remains before videorate to bring NVMM into system memory.
        prefix = conv + rate if hw else rate + conv
        return prefix + "videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=1"

    @staticmethod
    def _gst_quote(value):
        # Escape $ to prevent gst_parse_launch variable expansion
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$') + '"'

    def _build_rtsp_pipeline(self, url, decoder, codec="h264"):
        """RTSP fast path: depay/parse + (HW or CPU) decode."""
        lat = int(self.config.gst_latency_ms)
        proto = self.config.rtsp_transport
        # Never drop compressed RTP packets before depay/decode: doing so
        # corrupts reference frames until the camera sends another keyframe.
        url = self._gst_quote(url)

        depay = "rtph265depay ! h265parse config-interval=1 ! video/x-h265" if codec == "h265" else "rtph264depay ! h264parse config-interval=1 ! video/x-h264"

        # `drop-on-latency` throws away buffers that outlive the jitter buffer.
        # That is reasonable on a LAN with a tiny buffer, but over the internet
        # it defeats the larger buffer entirely: the late packets the buffer
        # exists to absorb get dropped anyway, reference frames break, and the
        # stream collapses. Only drop when the buffer is small enough that
        # holding late data would add real latency.
        drop = "true" if lat <= 50 else "false"
        # Retransmission recovers the occasional lost packet on a WAN link,
        # which is exactly the case a large buffer is already tolerating.
        rtx = "false" if lat <= 50 else "true"

        if decoder == "nvv4l2decoder":
            return (
                "rtspsrc location={url} latency={lat} protocols={proto} "
                "drop-on-latency={drop} do-retransmission={rtx} ! "
                "{depay},stream-format=byte-stream,alignment=au ! "
                "nvv4l2decoder enable-max-performance=1 ! "
                + self._gst_appsink_tail(hw=True)
            ).format(url=url, lat=lat, proto=proto, depay=depay, drop=drop, rtx=rtx)
        # CPU fallback decoder (avdec_h264 / avdec_h265)
        return (
            "rtspsrc location={url} latency={lat} protocols={proto} "
            "drop-on-latency={drop} do-retransmission={rtx} ! "
            "{depay} ! "
            "{dec} ! "
            + self._gst_appsink_tail(hw=False)
        ).format(url=url, lat=lat, proto=proto, dec=decoder, depay=depay, drop=drop, rtx=rtx)

    def _build_uridecodebin_pipeline(self, url, hw):
        """Generic pipeline for srt/rtmp/http(HLS/MJPEG/progressive) and rtsp.

        ``uridecodebin`` auto-plugs the right source + demux + decoder for the
        URI (and on Jetson picks nvv4l2decoder for H.264/H.265, emitting NVMM
        that the hw tail's nvvidconv consumes). The quoted uri keeps query
        strings (``?...``) from breaking gst-launch parsing.

        ``source::latency`` forwards the configured jitter buffer to the
        rtspsrc that uridecodebin builds internally. Without it rtspsrc keeps
        its own 2000ms default and GST_LATENCY_MS silently does nothing on this
        path — the opposite of the intent for a LAN deployment, and unrelated to
        the tuned value for a WAN one. Ignored by non-RTSP sources.

        ``source::protocols`` matters even more. uridecodebin's internal rtspsrc
        defaults to trying UDP first, and an NVR reached across the internet
        through NAT then delivers a few dozen RTP packets and stalls: the RTP
        arrives from the router's rewritten address, no keyframe ever completes,
        decodebin never plugs a decoder, and the stream looks "connected but
        dead". TCP interleaves RTP on the RTSP connection that is already open
        through the port forward, so it is the only reliable transport here.
        """
        q = "queue max-size-buffers=1 leaky=downstream ! "
        # Only rtspsrc has these properties. Setting source::* for an
        # http/srt/rtmp uri makes gst_parse_launch reject the whole pipeline.
        props = "" if hw else " force-sw-decoders=true"
        if self._url_scheme(url) in ("rtsp", "rtsps"):
            props += " source::latency={} source::protocols={}".format(
                int(self.config.gst_latency_ms), self.config.rtsp_transport,
            )
        return (
            'uridecodebin uri={url}{props} ! '.format(
                url=self._gst_quote(url), props=props,
            )
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
        return (
            'whepsrc whep-endpoint={ep} ! '.format(ep=self._gst_quote(endpoint))
            + "decodebin ! "
            + self._gst_appsink_tail(hw=hw)
        )

    def _gst_candidates(self):
        """Ordered (name, pipeline) GStreamer attempts for the source scheme."""
        url = self.config.source_url or ""
        scheme = self._url_scheme(url)
        cands = []
        if scheme in ("rtsp", "rtsps"):
            cands.append(("rtsp-uridecodebin-hw", self._build_uridecodebin_pipeline(url, hw=True)))
            cands.append(("rtsp-nvv4l2decoder-h265", self._build_rtsp_pipeline(url, self.config.gst_decoder, codec="h265")))
            cands.append(("rtsp-nvv4l2decoder-h264", self._build_rtsp_pipeline(url, self.config.gst_decoder, codec="h264")))
            cands.append(("rtsp-avdec_h265", self._build_rtsp_pipeline(url, "avdec_h265", codec="h265")))
            cands.append(("rtsp-avdec_h264", self._build_rtsp_pipeline(url, "avdec_h264", codec="h264")))
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
        self._capture_backend = None
        self._gstreamer_failures = {}

        if self.config.decode_backend == "gstreamer" and not native_gstreamer_available():
            self._gstreamer_failures["bindings"] = "GStreamer GI bindings unavailable"
            logger.warning(
                "[%s] native GStreamer bindings unavailable; install python3-gi and "
                "gir1.2-gst-plugins-base-1.0 in the system environment; using FFmpeg for %s",
                self.config.camera_uuid, scheme,
            )
        elif self.config.decode_backend == "gstreamer":
            try:
                probe_budget_s = float(os.getenv("CAPTURE_PROBE_S", "12"))
            except Exception:
                probe_budget_s = 12.0
            is_first = True
            for name, gst in self._gst_candidates():
                # Check between candidates, not just inside the frame probe. A
                # channel torn down mid-open would otherwise keep walking the
                # whole ladder, holding the worker thread for the sum of every
                # candidate's probe budget while stop() waits on it.
                if self._stop_thread_evt.is_set():
                    return None
                cap = None
                try:
                    cap = GstCapture(gst, stop_event=self._stop_thread_evt)
                except Exception as error:
                    self._gstreamer_failures[name] = str(error)
                    logger.debug("[%s] pipeline=%s creation failed: %s",
                                 self.config.camera_uuid, name, error)
                    cap = None
                # Fallbacks also need time to fill the jitter buffer and decode.
                fallback_s = max(3.0, self.config.gst_latency_ms / 1000.0 + 2.0)
                probe_s = None if is_first else min(fallback_s, probe_budget_s)
                is_first = False
                if cap is not None and cap.isOpened() and self._probe_first_frame(cap, probe_s):
                    self._capture_backend = name
                    logger.info(
                        "[%s] opened %s source via native GStreamer appsink pipeline=%s",
                        self.config.camera_uuid, scheme, name,
                    )
                    return cap
                if cap is not None:
                    self._gstreamer_failures[name] = cap.last_error or "frame probe timed out"
                    logger.debug(
                        "[%s] pipeline=%s did not deliver a frame (%s); trying next",
                        self.config.camera_uuid, name, cap.last_error or "probe timed out",
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass

            logger.warning(
                "[%s] all gstreamer pipelines failed for %s source; falling back to OpenCV/FFmpeg: %s",
                self.config.camera_uuid, scheme, self._gstreamer_failures,
            )

        # OpenCV/FFmpeg generic capture (decode_backend="opencv" or gst fallback).
        if self._stop_thread_evt.is_set():
            return None

        try:
            cap = self._capture_with_timeout(url, cv2.CAP_FFMPEG)
        except Exception:
            cap = cv2.VideoCapture(url)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        # Say so at INFO. This path means CPU decode, which an Orin Nano cannot
        # sustain across a full camera set, so a silent demotion here looks like
        # "cameras connected but no detections" with nothing in the log to
        # explain it.
        self._capture_backend = "opencv-ffmpeg"
        logger.warning(
            "[%s] using CPU OpenCV/FFmpeg decode for %s — hardware decode was "
            "not available for this source; expect low throughput",
            self.config.camera_uuid, scheme,
        )
        return cap

    def _probe_first_frame(self, cap, budget_s=None):
        """True when `cap` delivers one real frame within the probe window.

        `budget_s` of None means "use CAPTURE_PROBE_S"; callers pass a shorter
        window for the codec-guessing candidates.

        A remote NVR reached over the internet does not hand over a frame
        immediately: rtspsrc must fill its jitter buffer and the decoder must
        wait for the first keyframe, which on a WAN link is routinely a few
        seconds after PAUSED. So this waits rather than testing once — a single
        immediate read would reject every working WAN pipeline.

        The frame is discarded. Losing one frame at connect time is irrelevant
        at the sample rates this service runs at, and it is the only way to tell
        a negotiated-but-dead pipeline from a live one.
        """
        if budget_s is None:
            try:
                budget_s = float(os.getenv("CAPTURE_PROBE_S", "12"))
            except Exception:
                budget_s = 12.0
        deadline = time.monotonic() + max(1.0, float(budget_s))

        while time.monotonic() < deadline:
            if self._stop_thread_evt.is_set():
                return False
            frame = None
            try:
                with self._cap_lock:
                    if isinstance(cap, GstCapture):
                        ok = cap.grab(timeout_s=max(0.0, deadline - time.monotonic()))
                    else:
                        ok = cap.grab()
                    if ok:
                        ok, frame = cap.retrieve()
            except Exception:
                return False
            if ok and frame is not None and getattr(frame, "size", 1) > 0:
                self._last_frame_monotonic = time.monotonic()
                return True
            if self._stop_thread_evt.wait(0.1):
                return False
        return False

    @staticmethod
    def _capture_with_timeout(source, backend):
        # Open-only properties must be passed at construction, not cap.set().
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC") and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            return cv2.VideoCapture(source, backend, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
            ])
        return cv2.VideoCapture(source, backend)

    def status(self):
        return {"connected": self._connected, "backend": self._capture_backend,
                "gstreamer_failures": dict(self._gstreamer_failures),
                "capture_interface": ("native-gstreamer" if isinstance(self._cap, GstCapture) else
                                      "opencv-ffmpeg" if self._capture_backend == "opencv-ffmpeg" else None),
                "last_frame_age_ms": (None if self._last_frame_monotonic is None else
                                      int((time.monotonic() - self._last_frame_monotonic) * 1000)),
                "frames_emitted": self._frames_emitted, "handoff_dropped": self._handoff_dropped,
                "sample_fps": self.config.sample_fps}

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
        with self._handoff_lock:
            if self._pending_frame is not None:
                self._handoff_dropped += 1
            self._pending_frame = ev
            if self._handoff_scheduled:
                return
            self._handoff_scheduled = True
        try:
            self._loop.call_soon_threadsafe(self._drain_handoff)
        except RuntimeError:
            with self._handoff_lock:
                self._pending_frame = None
                self._handoff_scheduled = False

    def _drain_handoff(self):
        with self._handoff_lock:
            ev = self._pending_frame
            self._pending_frame = None
            self._handoff_scheduled = False
        if ev is not None:
            self._put_latest(ev)

    def _worker(self):
        backoff_ms = int(self.config.reconnect_base_ms)

        while not self._stop_thread_evt.is_set():
            cap = None
            try:
                if not CONNECT_GATE.wait_turn(self._stop_thread_evt):
                    break

                cap = self._open_capture()

                with self._cap_lock:
                    self._cap = cap

                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open video source")

                logger.info("[%s] connected", self.config.camera_uuid)
                self._connected = True

                sample_fps = float(self.config.sample_fps)
                emit_interval_s = 1.0 / sample_fps
                next_emit = 0.0
                try:
                    max_grab_misses = int(os.getenv("GRAB_MISS_TOLERANCE", "15"))
                except Exception:
                    max_grab_misses = 15
                grab_misses = 0

                while not self._stop_thread_evt.is_set():
                    grabbed = False
                    try:
                        with self._cap_lock:
                            if self._cap is None:
                                break
                            grabbed = self._cap.grab()
                    except Exception:
                        grabbed = False

                    if not grabbed:
                        grab_misses += 1
                        if grab_misses >= max_grab_misses:
                            raise RuntimeError(
                                "Frame grab failed {} times consecutively".format(grab_misses)
                            )
                        if self._stop_thread_evt.wait(0.1):
                            break
                        continue

                    grab_misses = 0
                    ts_ms = int(time.time() * 1000)

                    now = time.monotonic()
                    if self._capture_backend == "opencv-ffmpeg" and now < next_emit:
                        continue

                    # Keep the phase despite scheduler jitter, but never catch
                    # up by emitting a burst after a blocked read.
                    next_emit = max(next_emit + emit_interval_s, now + emit_interval_s * 0.5)

                    with self._cap_lock:
                        if self._cap is None:
                            break
                        ok, frame = self._cap.retrieve()
                    if not ok or frame is None:
                        raise RuntimeError("Frame retrieve failed")

                    # Opening a handle is not proof of a healthy stream. Reset
                    # retry delay only when a frame has actually been decoded.
                    backoff_ms = int(self.config.reconnect_base_ms)
                    self._last_frame_monotonic = time.monotonic()

                    frame = self._maybe_resize(frame)
                    self._seq += 1
                    self._frames_emitted += 1

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
                self._connected = False
                sleep_s = (backoff_ms / 1000.0) * (0.75 + random.random() * 0.5)
                logger.warning(
                    "[%s] disconnected: %s (retry in %.1fs)",
                    self.config.camera_uuid, e, sleep_s,
                )

                # Close the failed NVR session before backing off, not after
                # the delay. The capture thread remains the sole owner.
                with self._cap_lock:
                    failed_cap, self._cap = self._cap, None
                if failed_cap is not None:
                    try:
                        failed_cap.release()
                    except Exception:
                        pass

                if self._stop_thread_evt.wait(sleep_s):
                    break
                backoff_ms = min(backoff_ms * 2, int(self.config.reconnect_max_ms))

            finally:
                # Detach the handle under the lock BEFORE releasing it, so any
                # other code path that takes the lock sees None rather than a
                # capture that is midway through being torn down.
                try:
                    with self._cap_lock:
                        doomed = self._cap
                        self._cap = None
                    if doomed is not None:
                        doomed.release()
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
        # Repeated calls still join a thread whose earlier stop timed out.
        already_stopping = self._stopping
        self._stopping = True

        cam = getattr(self.config, "camera_uuid", None) or getattr(self.config, "channel_id", "unknown")
        if not already_stopping:
            logger.info(f"[{cam}] Stopping channel thread...")
        self._stop_thread_evt.set()
        self._connected = False
        with self._handoff_lock:
            self._pending_frame = None
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
                logger.warning(
                    "[%s] capture thread still unwinding after 6s; it will exit on "
                    "its own and emit nothing in the meantime", cam,
                )
                return

        if t and not t.is_alive():
            self._thread = None
