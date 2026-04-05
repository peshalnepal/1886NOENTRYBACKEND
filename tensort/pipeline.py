# simple_model_pipeline.py  (Python 3.6)
import asyncio
import logging
import re
import threading
import queue
import os
import time
from typing import Any, Dict, Optional, List

try:
    # Script mode (python main.py from Backend/tensort)
    from channels.channel_config import VideoChannelConfig
    from channels.channel import VideoChannel, RTSPEvent
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from .channels.channel_config import VideoChannelConfig
    from .channels.channel import VideoChannel, RTSPEvent

logger = logging.getLogger(__name__)
_DONE = object()


def _encode_jpeg_bytes(frame_bgr, *, max_edge: int = 960, jpeg_quality: int = 75) -> Optional[bytes]:
    if frame_bgr is None:
        return None

    try:
        import cv2

        height, width = frame_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return None

        scale = min(float(max_edge) / float(max(height, width)), 1.0)
        output = frame_bgr
        if scale < 1.0:
            output = cv2.resize(
                frame_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            return None

        return encoded.tobytes()
    except Exception:
        return None


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = int(default)
    return max(minimum, v)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except Exception:
        value = float(default)
    return max(float(minimum), float(value))


class Broadcaster:
    def __init__(self):
        self._subscribers = set()
        self._lock = asyncio.Lock()

    async def subscribe(self):
        q = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q):
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def broadcast(self, msg):
        async with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

class CoalescingBuffer(object):
    def __init__(self, max_pending_keys=1000):
        self._latest = {}  # camera_uuid -> RTSPEvent
        self._pending = asyncio.Queue(maxsize=max_pending_keys)
        self._in_queue = set()
        self._lock = asyncio.Lock()

    async def put(self, ev):
        key = str(ev.camera_uuid)
        async with self._lock:
            self._latest[key] = ev
            if key in self._in_queue:
                return
            try:
                self._pending.put_nowait(key)
                self._in_queue.add(key)
            except asyncio.QueueFull:
                pass

    async def get(self):
        while True:
            key = await self._pending.get()
            async with self._lock:
                if key in self._in_queue:
                    self._in_queue.remove(key)
                ev = self._latest.get(key)
            if ev is not None:
                return ev


class InferenceWorker(object):
    """
    Dedicated inference thread that OWNS:
      - PyCUDA context (via trt_infer import)
      - TRT engines
      - CUDA streams

    You must not call TRT from other threads.
    """
    def __init__(self, loop, max_q=2):
        self._loop = loop
        self._q = queue.Queue(maxsize=max_q)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trt-infer-worker", daemon=True)

        self._infer = None
        self._thread.start()

    def submit(self, bgr, meta, fut):
        """
        bgr: numpy ndarray (BGR)
        meta: dict
        fut: asyncio.Future (created in event loop thread)
        """
        try:
            self._q.put_nowait((bgr, meta, fut))
            return True
        except queue.Full:
            # drop if overloaded: set a failure result (non-blocking)
            def _set():
                if not fut.done():
                    fut.set_result({
                        "type": "InferenceFailedEvent",
                        "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                        "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
                        "frame_seq": int(meta.get("frame_seq", 0)),
                        "reason": "Inference queue full (dropped)",
                    })
            self._loop.call_soon_threadsafe(_set)
            return False

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass

    def join(self, timeout=2.0):
        try:
            self._thread.join(timeout)
        except Exception:
            pass

    def _run(self):
        """
        Runs inside the inference thread.
        Import trt_infer here so PyCUDA context is created in this thread.
        """
        from trt_infer import build_default
        try:
            self._infer = build_default()
            # logger.info("TRTInfer initialized inside inference thread.")
        except Exception as e:
            logger.exception("Failed to init TRT infer in worker thread: %s", e)
            self._infer = None

        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except Exception:
                continue

            if item is None:
                continue

            bgr, meta, fut = item

            if self._infer is None:
                res = {
                    "type": "InferenceFailedEvent",
                    "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                    "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
                    "frame_seq": int(meta.get("frame_seq", 0)),
                    "reason": "TRT inference not initialized",
                }
                self._loop.call_soon_threadsafe(self._safe_set_result, fut, res)
                continue

            try:
                res = self._infer.infer_multitask(bgr, meta)
            except Exception as e:
                res = {
                    "type": "InferenceFailedEvent",
                    "camera_uuid": str(meta.get("camera_uuid", "unknown")),
                    "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
                    "frame_seq": int(meta.get("frame_seq", 0)),
                    "reason": "{}: {}".format(type(e).__name__, e),
                }

            self._loop.call_soon_threadsafe(self._safe_set_result, fut, res)

    def _safe_set_result(self, fut, res):
        try:
            if not fut.done():
                fut.set_result(res)
        except Exception:
            pass


class SimpleInferencePipeline(object):
    def __init__(self, out_queue_max=None, infer_q_max=None):
        if out_queue_max is None:
            out_queue_max = _env_int("PIPELINE_OUT_QUEUE_MAX", 500, minimum=10)
        if infer_q_max is None:
            infer_q_max = _env_int("INFER_QUEUE_MAX", 8, minimum=1)

        self._channels = {}
        self._channel_tasks = {}
        self._closing = False
        self._started = False

        self._buffer = CoalescingBuffer(max_pending_keys=_env_int("PENDING_KEY_MAX", 1000, minimum=100))
        self._out_q = asyncio.Queue(maxsize=out_queue_max)

        self._latest = {}
        self._latest_snapshots = {}
        self._latest_snapshot_ts_ms = {}
        self._latest_lock = asyncio.Lock()
        self._snapshot_enabled = _env_bool("ENABLE_SNAPSHOT_CACHE", True)
        self._snapshot_min_interval_ms = _env_int("SNAPSHOT_MIN_INTERVAL_MS", 1000, minimum=0)
        self._snapshot_max_edge = _env_int("SNAPSHOT_MAX_EDGE", 960, minimum=64)
        self._snapshot_jpeg_quality = _env_int("SNAPSHOT_JPEG_QUALITY", 75, minimum=1)
        self._snapshot_on_detection_only = _env_bool("SNAPSHOT_ON_DETECTION_ONLY", False)
        self._emit_empty_detections = _env_bool("EMIT_EMPTY_DETECTIONS", False)
        self._infer_result_timeout_s = _env_float("INFER_RESULT_TIMEOUT_S", 10.0, minimum=0.0)
        self._infer_error_log_interval_s = _env_float("INFER_ERROR_LOG_INTERVAL_S", 10.0, minimum=0.0)
        self._last_infer_error_sig = {}
        self._last_infer_error_ts = {}

        self._lock = asyncio.Lock()
        self._inference_task = None
        self._infer_worker = None  # created on start()
        self._infer_q_max = int(infer_q_max)
        self._log_every_n = _env_int("PIPELINE_LOG_EVERY_N_FRAMES", 0, minimum=0)
        self._stats = {
            "frames_in": 0,
            "infer_ok": 0,
            "infer_fail": 0,
            "infer_dropped": 0,
            "detections_total": 0,
            "alerts_attempted": 0,
        }
        self.broadcaster = Broadcaster()

    def _should_log_infer_failure(self, camera_uuid, reason):
        camera_key = str(camera_uuid)
        normalized_reason = str(reason or "").strip()
        normalized_reason = re.sub(r"\s+\(after\s+\d+\s+ms\)\s*$", "", normalized_reason)
        sig = "{}|{}".format(camera_key, normalized_reason)
        now_s = float(time.monotonic())
        last_sig = self._last_infer_error_sig.get(camera_key)
        last_ts = float(self._last_infer_error_ts.get(camera_key, 0.0) or 0.0)
        if sig != last_sig:
            self._last_infer_error_sig[camera_key] = sig
            self._last_infer_error_ts[camera_key] = now_s
            return True
        if self._infer_error_log_interval_s <= 0.0:
            self._last_infer_error_ts[camera_key] = now_s
            return True
        if (now_s - last_ts) >= self._infer_error_log_interval_s:
            self._last_infer_error_ts[camera_key] = now_s
            return True
        return False

    async def add_channel(self, cfg):
        camera_key = str(cfg.camera_uuid)
        new_channel = VideoChannel(cfg)

        async with self._lock:
            old_ch = self._channels.pop(camera_key, None)
            old_task = self._channel_tasks.pop(camera_key, None)

            self._channels[camera_key] = new_channel
            should_start = self._started and (not self._closing)

        if old_task is not None and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except Exception:
                pass

        if old_ch is not None:
            try:
                await old_ch.stop()
            except Exception:
                pass

        if should_start:
            # Record the task inside the lock so that a concurrent remove_channel
            # that runs between the lock release above and here sees the task and
            # can cancel it instead of letting it run orphaned.
            async with self._lock:
                if camera_key in self._channels and not self._closing:
                    self._start_channel_task(camera_key)

    async def remove_channel(self, camera_uuid):
        camera_key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.pop(camera_key, None)
            t = self._channel_tasks.pop(camera_key, None)

        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except Exception:
                pass

        if ch is not None:
            try:
                await ch.stop()
            except Exception:
                pass

        async with self._latest_lock:
            if camera_key in self._latest:
                del self._latest[camera_key]
            if camera_key in self._latest_snapshots:
                del self._latest_snapshots[camera_key]
            if camera_key in self._latest_snapshot_ts_ms:
                del self._latest_snapshot_ts_ms[camera_key]
        self._last_infer_error_sig.pop(camera_key, None)
        self._last_infer_error_ts.pop(camera_key, None)

    def list_channels(self):
        return list(self._channels.keys())

    async def start(self):
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._closing = False

            loop = asyncio.get_event_loop()
            self._infer_worker = InferenceWorker(loop=loop, max_q=self._infer_q_max)

            self._inference_task = asyncio.ensure_future(self._pump_inference())

            for camera_key in list(self._channels.keys()):
                self._start_channel_task(camera_key)

    async def shutdown(self):
        async with self._lock:
            if not self._started:
                return
            self._closing = True

            for t in list(self._channel_tasks.values()):
                if t is not None and not t.done():
                    t.cancel()

            if self._inference_task is not None and not self._inference_task.done():
                self._inference_task.cancel()

        for t in list(self._channel_tasks.values()):
            try:
                await t
            except Exception:
                pass

        for ch in list(self._channels.values()):
            try:
                await ch.stop()
            except Exception:
                pass

        if self._inference_task is not None:
            try:
                await self._inference_task
            except Exception:
                pass

        if self._infer_worker is not None:
            self._infer_worker.stop()
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._infer_worker.join, 2.0)
            except Exception:
                pass

        async with self._lock:
            self._channel_tasks = {}
            self._channels = {}
            self._started = False
        self._last_infer_error_sig = {}
        self._last_infer_error_ts = {}

        await self._put_out(_DONE)

    async def _put_out(self, ev):
        if self._out_q.full():
            try:
                _ = self._out_q.get_nowait()
            except Exception:
                pass
        try:
            self._out_q.put_nowait(ev)
        except Exception:
            pass

    def _start_channel_task(self, camera_key):
        ch = self._channels.get(camera_key)
        if ch is None:
            return
        self._channel_tasks[camera_key] = asyncio.ensure_future(self._pump_channel(camera_key, ch))

    async def _pump_channel(self, camera_key, ch):
        try:
            # logger.info("[Jetson] Starting pump for camera %s", camera_key)
            async for ev in ch.stream(event_queue=None):
                if self._closing:
                    break

                if isinstance(ev, RTSPEvent):
                    if getattr(ev, "detection_enabled", True):
                        await self._buffer.put(ev)
                else:
                    await self._put_out(ev)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._put_out({"type": "ChannelPumpFailed", "camera_uuid": camera_key, "reason": str(e)})
        finally:
            try:
                await ch.stop()
            except Exception:
                pass
            
    async def _pump_inference(self):
        """
        Sends frames to the TRT inference thread.
        Stores latest result per camera.
        Emits only failures or positive detections by default.
        No external alert webhook push from Jetson.
        """
        loop = asyncio.get_event_loop()

        try:
            while not self._closing:
                rtsp_ev = await self._buffer.get()
                self._stats["frames_in"] += 1

                if self._log_every_n and (int(getattr(rtsp_ev, "seq", 0)) % self._log_every_n == 0):
                    logger.debug(
                        "[Jetson] Received frame camera=%s seq=%s",
                        rtsp_ev.camera_uuid,
                        rtsp_ev.seq,
                    )

                bgr = getattr(rtsp_ev, "frame", None)

                if bgr is None:
                    self._stats["infer_fail"] += 1
                    result = {
                        "type": "InferenceFailedEvent",
                        "camera_uuid": str(rtsp_ev.camera_uuid),
                        "frame_ts_ms": int(rtsp_ev.ts_ms),
                        "frame_seq": int(rtsp_ev.seq),
                        "reason": "No frame data (emit_format=raw required)",
                    }

                    async with self._latest_lock:
                        self._latest[str(rtsp_ev.camera_uuid)] = result

                    await self.broadcaster.broadcast(result)
                    await self._put_out(result)
                    continue

                meta = {
                    "camera_uuid": str(rtsp_ev.camera_uuid),
                    "channel_id": getattr(rtsp_ev, "channel_id", None),
                    "frame_ts_ms": int(rtsp_ev.ts_ms),
                    "frame_seq": int(rtsp_ev.seq),
                }

                fut = loop.create_future()

                if self._infer_worker is None:
                    result = {
                        "type": "InferenceFailedEvent",
                        "camera_uuid": meta["camera_uuid"],
                        "frame_ts_ms": meta["frame_ts_ms"],
                        "frame_seq": meta["frame_seq"],
                        "reason": "Inference worker not running",
                    }
                else:
                    ok = self._infer_worker.submit(bgr, meta, fut)

                    if not ok:
                        result = await fut
                    else:
                        try:
                            if self._infer_result_timeout_s > 0.0:
                                result = await asyncio.wait_for(
                                    fut,
                                    timeout=self._infer_result_timeout_s,
                                )
                            else:
                                result = await fut
                        except asyncio.TimeoutError:
                            result = {
                                "type": "InferenceFailedEvent",
                                "camera_uuid": meta["camera_uuid"],
                                "frame_ts_ms": meta["frame_ts_ms"],
                                "frame_seq": meta["frame_seq"],
                                "reason": "Inference timed out",
                            }

                has_detections = False

                if isinstance(result, dict) and result.get("type") == "InferenceFailedEvent":
                    self._stats["infer_fail"] += 1
                    reason = str(result.get("reason", "") or "")

                    if "queue full" in reason.lower():
                        self._stats["infer_dropped"] += 1

                    if self._should_log_infer_failure(rtsp_ev.camera_uuid, reason):
                        logger.warning(
                            "[Jetson] Inference failed camera=%s seq=%s reason=%s",
                            rtsp_ev.camera_uuid,
                            rtsp_ev.seq,
                            reason or "unknown error",
                        )
                else:
                    self._stats["infer_ok"] += 1

                    detections = []
                    if isinstance(result, dict):
                        detections = result.get("detections", []) or []

                    self._stats["detections_total"] += len(detections)
                    has_detections = bool(detections)

                    # only cache snapshot when needed
                    if self._snapshot_enabled and ((not self._snapshot_on_detection_only) or has_detections):
                        await self._cache_snapshot(
                            camera_uuid=rtsp_ev.camera_uuid,
                            frame_bgr=bgr,
                            ts_ms=int(rtsp_ev.ts_ms),
                        )

                async with self._latest_lock:
                    self._latest[str(rtsp_ev.camera_uuid)] = result

                should_emit = True
                if isinstance(result, dict) and result.get("type") != "InferenceFailedEvent":
                    should_emit = self._emit_empty_detections or has_detections

                if should_emit:
                    await self.broadcaster.broadcast(result)
                    await self._put_out(result)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._put_out({"type": "InferencePumpFailed", "reason": str(e)})
    async def events(self):
        if not self._started:
            await self.start()

        while True:
            item = await self._out_q.get()
            if item is _DONE:
                break
            yield item

    async def get_latest(self, camera_uuid):
        async with self._latest_lock:
            return self._latest.get(str(camera_uuid))

    async def get_latest_snapshot(self, camera_uuid):
        async with self._latest_lock:
            return self._latest_snapshots.get(str(camera_uuid))

    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            channel_count = len(self._channels)
        async with self._latest_lock:
            latest_count = len(self._latest)
            snapshot_count = len(self._latest_snapshots)
        stats = dict(self._stats)
        stats.update({
            "channel_count": int(channel_count),
            "latest_cache_size": int(latest_count),
            "snapshot_cache_size": int(snapshot_count),
            "infer_queue_max": int(self._infer_q_max),
            "out_queue_max": int(self._out_q.maxsize),
        })
        return stats

    async def _cache_snapshot(self, *, camera_uuid: str, frame_bgr, ts_ms: int) -> None:
        camera_key = str(camera_uuid)
        snapshot_ts_ms = int(ts_ms)

        async with self._latest_lock:
            last_ts = int(self._latest_snapshot_ts_ms.get(camera_key, 0) or 0)
        if self._snapshot_min_interval_ms and last_ts and (snapshot_ts_ms - last_ts) < self._snapshot_min_interval_ms:
            return

        encoded = _encode_jpeg_bytes(
            frame_bgr,
            max_edge=self._snapshot_max_edge,
            jpeg_quality=self._snapshot_jpeg_quality,
        )
        if not encoded:
            return

        async with self._latest_lock:
            last_ts = int(self._latest_snapshot_ts_ms.get(camera_key, 0) or 0)
            if self._snapshot_min_interval_ms and last_ts and (snapshot_ts_ms - last_ts) < self._snapshot_min_interval_ms:
                return
            self._latest_snapshots[camera_key] = encoded
            self._latest_snapshot_ts_ms[camera_key] = snapshot_ts_ms
