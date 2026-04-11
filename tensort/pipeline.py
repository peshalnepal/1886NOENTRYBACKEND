# pipeline.py  (Python 3.6)
#
# Fixes applied vs original:
#  1. InferenceWorkerPool  — N parallel inference threads instead of 1.
#     Each camera is pinned to a worker (round-robin at submit time) so
#     camera N never waits for camera N-1's frame to finish.
#  2. _pump_inference  — fire-and-forget dispatch; futures are resolved via
#     done-callbacks so the dispatch coroutine never awaits a single future.
#     All in-flight futures are tracked in a dict keyed by (camera_uuid, seq).
#  3. CoalescingBuffer  — CPython dict assignment is atomic under the GIL;
#     the asyncio.Lock now only guards _in_queue membership, not the hot path.
#  4. _cache_snapshot  — JPEG encode offloaded to executor so the event loop
#     is never stalled by cv2.imencode.
#  5. Broadcaster  — lock replaced with a plain list copy (no await on hot path).
#  6. YoloV8DetTRT.run  — letterbox + tensor prep moved OUT of TRTEngine.infer
#     so the inference method only does memcpy + execute + memcpy.
#     (See also trt_infer.py changes.)

import asyncio
import logging
import re
import threading
import queue
import os
import time
from typing import Any, Dict, Optional, List

try:
    from channels.channel_config import VideoChannelConfig
    from channels.channel import VideoChannel, RTSPEvent
except Exception:
    from .channels.channel_config import VideoChannelConfig
    from .channels.channel import VideoChannel, RTSPEvent

logger = logging.getLogger(__name__)
_DONE = object()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_jpeg_bytes(frame_bgr, *, max_edge: int = 960, jpeg_quality: int = 75) -> Optional[bytes]:
    if frame_bgr is None:
        return None
    try:
        import cv2
        height, width = frame_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return None
        scale = min(float(max_edge) / float(max(height, width)), 1.0)
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
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


def _detect_total_memory_mb() -> Optional[int]:
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return max(0, int(parts[1]) // 1024)
    except Exception:
        return None
    return None


def _default_auto_worker_cap(total_mem_mb: Optional[int] = None) -> int:
    if total_mem_mb is None:
        total_mem_mb = _detect_total_memory_mb()

    if total_mem_mb is None:
        return 2
    if int(total_mem_mb) <= 4608:
        return 1
    if int(total_mem_mb) <= 8192:
        return 2
    if int(total_mem_mb) <= 16384:
        return 3
    return 4


def _default_infer_result_timeout_s(total_mem_mb: Optional[int] = None) -> float:
    if total_mem_mb is None:
        total_mem_mb = _detect_total_memory_mb()

    if total_mem_mb is None:
        return 1.5
    if int(total_mem_mb) <= 4608:
        return 2.0
    if int(total_mem_mb) <= 8192:
        return 1.5
    return 1.0


# ---------------------------------------------------------------------------
# FIX 5: Broadcaster — no asyncio.Lock on the hot broadcast path
# ---------------------------------------------------------------------------

class Broadcaster:
    """
    Lock-free broadcast for the hot path.

    subscribe/unsubscribe still use a lock (they're rare).
    Subscribers may optionally scope themselves to a single camera_uuid so a
    per-camera SSE consumer does not queue unrelated detections.
    broadcast() does a single atomic list snapshot under the GIL —
    no await, no contention.
    """
    def __init__(self):
        self._subscribers = []          # [(asyncio.Queue, Optional[str])]
        self._sub_lock = asyncio.Lock() # only for subscribe/unsubscribe

    async def subscribe(self, camera_uuid=None):
        q = asyncio.Queue(maxsize=200)
        camera_key = None if camera_uuid is None else str(camera_uuid)
        async with self._sub_lock:
            self._subscribers = self._subscribers + [(q, camera_key)]  # new list = atomic replace
        return q

    async def unsubscribe(self, q):
        async with self._sub_lock:
            self._subscribers = [(sub_q, camera_key) for (sub_q, camera_key) in self._subscribers if sub_q is not q]

    async def broadcast(self, msg):
        # Snapshot is a single attribute read — atomic under GIL, no lock needed.
        msg_camera_uuid = None
        if isinstance(msg, dict) and msg.get("camera_uuid") is not None:
            msg_camera_uuid = str(msg.get("camera_uuid"))

        for q, camera_key in self._subscribers:
            if camera_key is not None and camera_key != msg_camera_uuid:
                continue
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# FIX 3: CoalescingBuffer — lock only guards set membership, not _latest
# ---------------------------------------------------------------------------

class CoalescingBuffer(object):
    """
    Keeps only the newest frame per camera.

    All access is from a single asyncio event-loop thread, so no locks are
    needed — CPython dict/set operations are GIL-atomic and the asyncio
    event loop is single-threaded.
    """
    def __init__(self, max_pending_keys=1000):
        self._latest = {}               # camera_uuid -> RTSPEvent
        self._pending = asyncio.Queue(maxsize=max_pending_keys)
        self._in_queue = set()

    async def put(self, ev):
        key = str(ev.camera_uuid)
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
            self._in_queue.discard(key)
            ev = self._latest.pop(key, None)
            if ev is not None:
                return ev


# ---------------------------------------------------------------------------
# FIX 1: InferenceWorker — unchanged; now used in a pool
# ---------------------------------------------------------------------------

class InferenceWorker(object):
    """
    One dedicated inference thread owning its own CUDA context + TRT engines.
    Never share across threads.
    """
    def __init__(self, loop, worker_id=0, max_q=2):
        self._loop = loop
        self._worker_id = int(worker_id)
        self._q = queue.Queue(maxsize=max_q)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="trt-infer-worker-{}".format(worker_id),
            daemon=True,
        )
        self._infer = None
        self._thread.start()

    def submit(self, bgr, meta, fut):
        try:
            self._q.put_nowait((bgr, meta, fut))
            return True
        except queue.Full:
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
        from trt_infer import build_default
        try:
            self._infer = build_default()
            logger.info("[worker-%d] TRTInfer ready", self._worker_id)
        except Exception as e:
            logger.exception("[worker-%d] Failed to init TRT: %s", self._worker_id, e)
            self._infer = None

        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.05)
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
            else:
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


# ---------------------------------------------------------------------------
# FIX 1 (cont.): InferenceWorkerPool — N workers, cameras pinned by hash
# ---------------------------------------------------------------------------

class InferenceWorkerPool(object):
    """
    Wraps N InferenceWorker threads.

    Camera uuid is hashed to a worker index so the same camera always goes
    to the same worker (preserves ordering per-camera, avoids lock contention
    between workers sharing a queue).
    """
    def __init__(self, loop, num_workers, max_q_per_worker=2):
        self._loop = loop
        self._max_q_per_worker = max_q_per_worker
        self._workers = []
        self.ensure_size(num_workers)

    def _pick(self, camera_uuid):
        # Stable assignment: same camera always → same worker
        return self._workers[hash(str(camera_uuid)) % len(self._workers)]

    def ensure_size(self, num_workers):
        target = max(1, int(num_workers))
        current = len(self._workers)
        if target <= current:
            return current
        for i in range(current, target):
            self._workers.append(
                InferenceWorker(self._loop, worker_id=i, max_q=self._max_q_per_worker)
            )
        logger.info("InferenceWorkerPool: %d workers", len(self._workers))
        return len(self._workers)

    def submit(self, bgr, meta, fut):
        worker = self._pick(meta.get("camera_uuid", ""))
        return worker.submit(bgr, meta, fut)

    def stop(self):
        for w in self._workers:
            w.stop()

    def join(self, timeout=2.0):
        for w in self._workers:
            w.join(timeout)


# ---------------------------------------------------------------------------
# FIX 2: _handle_result + fire-and-forget dispatch
# ---------------------------------------------------------------------------

class SimpleInferencePipeline(object):
    def __init__(self, out_queue_max=None, infer_q_max=None):
        if out_queue_max is None:
            out_queue_max = _env_int("PIPELINE_OUT_QUEUE_MAX", 500, minimum=10)
        if infer_q_max is None:
            # 1 = drop immediately when worker is busy; avoids queuing stale frames
            infer_q_max = _env_int("INFER_QUEUE_MAX", 1, minimum=1)

        self._detected_mem_mb = _detect_total_memory_mb()
        auto_worker_cap = _default_auto_worker_cap(self._detected_mem_mb)
        infer_timeout_default = _default_infer_result_timeout_s(self._detected_mem_mb)

        # Number of parallel inference threads (one per camera is a good default)
        self._num_workers = _env_int("INFER_NUM_WORKERS", 0, minimum=0)
        # 0 → auto-size to number of cameras (capped at INFER_NUM_WORKERS_MAX)
        self._num_workers_max = _env_int("INFER_NUM_WORKERS_MAX", auto_worker_cap, minimum=1)

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
        self._snapshot_min_interval_ms = _env_int("SNAPSHOT_MIN_INTERVAL_MS", 500, minimum=0)   # matches .env.example SNAPSHOT_MIN_INTERVAL_MS=500
        self._snapshot_max_edge = _env_int("SNAPSHOT_MAX_EDGE", 960, minimum=64)
        self._snapshot_jpeg_quality = _env_int("SNAPSHOT_JPEG_QUALITY", 75, minimum=1)
        self._snapshot_on_detection_only = _env_bool("SNAPSHOT_ON_DETECTION_ONLY", True)    # matches .env.example SNAPSHOT_ON_DETECTION_ONLY=true
        self._emit_empty_detections = _env_bool("EMIT_EMPTY_DETECTIONS", False)
        self._infer_result_timeout_s = _env_float("INFER_RESULT_TIMEOUT_S", infer_timeout_default, minimum=0.0)
        self._infer_error_log_interval_s = _env_float("INFER_ERROR_LOG_INTERVAL_S", 10.0, minimum=0.0)
        self._last_infer_error_sig = {}
        self._last_infer_error_ts = {}

        self._lock = asyncio.Lock()
        self._inference_task = None
        self._infer_pool = None         # created on start()
        self._infer_q_max = int(infer_q_max)
        # Cap how many frames can be in-flight at once. Must be >= number of
        # cameras so every camera gets a fair shot at inference each cycle.
        # Worker queue rejection (INFER_QUEUE_MAX) is the real memory guard —
        # it only holds 2 frames (1 processing + 1 queued). This cap is a
        # safety net against runaway inflight growth, not the primary throttle.
        self._max_inflight = _env_int("MAX_INFLIGHT_FRAMES", 8, minimum=1)
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

        # FIX 2: track in-flight futures so we can cancel on shutdown.
        # Only accessed from the single event-loop thread — no lock needed.
        self._inflight = {}             # (camera_uuid, seq) -> asyncio.Future

        logger.info(
            "[pipeline] config mem_total_mb=%s workers=%s worker_cap=%d infer_timeout_s=%.2f max_inflight=%d",
            self._detected_mem_mb if self._detected_mem_mb is not None else "unknown",
            "auto" if self._num_workers == 0 else int(self._num_workers),
            int(self._num_workers_max),
            float(self._infer_result_timeout_s),
            int(self._max_inflight),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Channel management
    # ------------------------------------------------------------------

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
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if old_ch is not None:
            try:
                await old_ch.stop()
            except Exception:
                pass

        if should_start:
            async with self._lock:
                if camera_key in self._channels and not self._closing:
                    if self._infer_pool is not None and self._num_workers == 0:
                        desired_workers = min(max(1, len(self._channels)), self._num_workers_max)
                        self._infer_pool.ensure_size(desired_workers)
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
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if ch is not None:
            try:
                await ch.stop()
            except Exception:
                pass

        async with self._latest_lock:
            self._latest.pop(camera_key, None)
            self._latest_snapshots.pop(camera_key, None)
            self._latest_snapshot_ts_ms.pop(camera_key, None)
        self._last_infer_error_sig.pop(camera_key, None)
        self._last_infer_error_ts.pop(camera_key, None)

    def list_channels(self):
        return list(self._channels.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._closing = False

            loop = asyncio.get_event_loop()

            num_workers = self._num_workers
            if num_workers == 0:
                num_workers = min(max(1, len(self._channels)), self._num_workers_max)

            self._infer_pool = InferenceWorkerPool(
                loop=loop,
                num_workers=num_workers,
                max_q_per_worker=self._infer_q_max,
            )
            logger.info(
                "[pipeline] starting camera_count=%d worker_count=%d auto_workers=%s",
                len(self._channels),
                len(self._infer_pool._workers),
                self._num_workers == 0,
            )

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
            except asyncio.CancelledError:
                pass
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
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if self._infer_pool is not None:
            self._infer_pool.stop()
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self._infer_pool.join, 2.0)
            except Exception:
                pass

        async with self._lock:
            self._channel_tasks = {}
            self._channels = {}
            self._started = False
        self._last_infer_error_sig = {}
        self._last_infer_error_ts = {}

        await self._put_out(_DONE)

    # ------------------------------------------------------------------
    # Internal tasks
    # ------------------------------------------------------------------

    async def _put_out(self, ev):
        if self._out_q.full():
            try:
                self._out_q.get_nowait()
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
        self._channel_tasks[camera_key] = asyncio.ensure_future(
            self._pump_channel(camera_key, ch)
        )

    async def _pump_channel(self, camera_key, ch):
        try:
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

    # ------------------------------------------------------------------
    # FIX 2: fire-and-forget inference dispatch
    # ------------------------------------------------------------------

    async def _pump_inference(self):
        """
        Dispatch frames to the worker pool without awaiting each one.
        Results come back asynchronously via _handle_result() callbacks,
        so we can immediately fetch the next frame from the buffer.

        NOTE: _inflight is only touched from this event-loop thread
        (put here, popped by call_soon callbacks / watchdog coroutines),
        so no lock is needed — plain dict ops are GIL-atomic.
        """
        loop = asyncio.get_event_loop()

        try:
            while not self._closing:
                rtsp_ev = await self._buffer.get()
                self._stats["frames_in"] += 1

                if self._log_every_n and (int(getattr(rtsp_ev, "seq", 0)) % self._log_every_n == 0):
                    logger.debug(
                        "[pipeline] frame camera=%s seq=%s",
                        rtsp_ev.camera_uuid, rtsp_ev.seq,
                    )

                bgr = getattr(rtsp_ev, "frame", None)
                # Release frame from the event so GC can free it once we're done
                rtsp_ev.frame = None

                if bgr is None:
                    self._stats["infer_fail"] += 1
                    result = {
                        "type": "InferenceFailedEvent",
                        "camera_uuid": str(rtsp_ev.camera_uuid),
                        "frame_ts_ms": int(rtsp_ev.ts_ms),
                        "frame_seq": int(rtsp_ev.seq),
                        "reason": "No frame data (emit_format=raw required)",
                    }
                    self._latest[str(rtsp_ev.camera_uuid)] = result
                    await self.broadcaster.broadcast(result)
                    await self._put_out(result)
                    continue

                if self._infer_pool is None:
                    bgr = None
                    continue

                # Back-pressure: if too many frames are already in flight,
                # drop this one to prevent unbounded memory growth.
                if len(self._inflight) >= self._max_inflight:
                    bgr = None
                    self._stats["infer_dropped"] += 1
                    continue

                camera_uuid_str = str(rtsp_ev.camera_uuid)
                meta = {
                    "camera_uuid": camera_uuid_str,
                    "channel_id": getattr(rtsp_ev, "channel_id", None),
                    "frame_ts_ms": int(rtsp_ev.ts_ms),
                    "frame_seq": int(rtsp_ev.seq),
                    "_bgr_ref": bgr,
                    "_ts_ms": int(rtsp_ev.ts_ms),
                }

                fut = loop.create_future()
                inflight_key = (camera_uuid_str, int(rtsp_ev.seq))

                # No lock — single event-loop thread
                self._inflight[inflight_key] = fut

                ok = self._infer_pool.submit(bgr, meta, fut)

                if not ok:
                    # Worker queue full — clean up immediately
                    self._inflight.pop(inflight_key, None)
                    meta.pop("_bgr_ref", None)
                    bgr = None
                    self._stats["infer_dropped"] += 1
                else:
                    # Release local ref — worker + meta["_bgr_ref"] still hold it
                    bgr = None

                    # Fire-and-forget: wire result handler to future done callback
                    def _on_done(f, _meta=meta, _key=inflight_key):
                        asyncio.ensure_future(self._handle_result(f.result(), _meta))
                        loop.call_soon(self._drop_inflight, _key)

                    fut.add_done_callback(_on_done)

                    # Lightweight watchdog — no frame reference
                    if self._infer_result_timeout_s > 0.0:
                        watchdog_meta = {
                            "camera_uuid": camera_uuid_str,
                            "frame_ts_ms": meta["frame_ts_ms"],
                            "frame_seq": meta["frame_seq"],
                        }
                        asyncio.ensure_future(
                            self._watchdog_future(fut, inflight_key, watchdog_meta)
                        )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._put_out({"type": "InferencePumpFailed", "reason": str(e)})

    def _drop_inflight(self, key):
        self._inflight.pop(key, None)

    async def _watchdog_future(self, fut, key, meta):
        """Cancel + resolve a future that hasn't completed within the timeout."""
        await asyncio.sleep(self._infer_result_timeout_s)
        if fut.done():
            return
        timeout_result = {
            "type": "InferenceFailedEvent",
            "camera_uuid": meta.get("camera_uuid", "unknown"),
            "frame_ts_ms": meta.get("frame_ts_ms", 0),
            "frame_seq": meta.get("frame_seq", 0),
            "reason": "Inference timed out",
        }
        try:
            if not fut.done():
                fut.set_result(timeout_result)
        except Exception:
            pass
        self._inflight.pop(key, None)

    async def _handle_result(self, result, meta):
        """
        Process one inference result.  Called from the done-callback so it runs
        concurrently for different cameras — no camera serialises another.
        """
        camera_uuid = str(meta.get("camera_uuid", "unknown"))
        bgr = meta.pop("_bgr_ref", None)
        ts_ms = int(meta.get("_ts_ms", 0))
        frame_seq = int(meta.get("frame_seq", 0))

        if isinstance(result, dict) and result.get("type") == "InferenceFailedEvent":
            self._stats["infer_fail"] += 1
            reason = str(result.get("reason", "") or "")
            if "queue full" in reason.lower():
                self._stats["infer_dropped"] += 1
            if self._should_log_infer_failure(camera_uuid, reason):
                logger.warning(
                    "[pipeline] Inference failed camera=%s seq=%s reason=%s",
                    camera_uuid, frame_seq, reason or "unknown",
                )
        else:
            self._stats["infer_ok"] += 1
            detections = []
            if isinstance(result, dict):
                detections = result.get("detections", []) or []
            self._stats["detections_total"] += len(detections)
            has_detections = bool(detections)

            # FIX 4: snapshot encoding offloaded to executor (non-blocking)
            if self._snapshot_enabled and bgr is not None:
                if (not self._snapshot_on_detection_only) or has_detections:
                    asyncio.ensure_future(
                        self._cache_snapshot(camera_uuid=camera_uuid, frame_bgr=bgr, ts_ms=ts_ms)
                    )

        # Release frame reference now that snapshot is scheduled.
        # The executor lambda already captured its own ref; we don't need ours.
        bgr = None

        # Atomic dict write — GIL-safe, no lock needed here
        self._latest[camera_uuid] = result

        should_emit = True
        if isinstance(result, dict) and result.get("type") != "InferenceFailedEvent":
            detections = result.get("detections", []) or []
            should_emit = self._emit_empty_detections or bool(detections)

        if should_emit:
            await self.broadcaster.broadcast(result)
            await self._put_out(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def events(self):
        if not self._started:
            await self.start()
        while True:
            item = await self._out_q.get()
            if item is _DONE:
                break
            yield item

    def peek_latest(self, camera_uuid):
        # Plain dict read — GIL-safe, no event-loop hop needed.
        return self._latest.get(str(camera_uuid))

    async def get_latest(self, camera_uuid):
        return self.peek_latest(camera_uuid)

    def peek_latest_snapshot(self, camera_uuid):
        # Reads a single dict entry; fine for health/debug paths that should not
        # block on the pipeline loop.
        return self._latest_snapshots.get(str(camera_uuid))

    async def get_latest_snapshot(self, camera_uuid):
        return self.peek_latest_snapshot(camera_uuid)

    def peek_stats(self) -> Dict[str, Any]:
        stats = dict(self._stats)
        stats.update({
            "channel_count": len(self._channels),
            "latest_cache_size": len(self._latest),
            "snapshot_cache_size": len(self._latest_snapshots),
            "infer_q_per_worker": int(self._infer_q_max),
            "out_queue_max": int(self._out_q.maxsize),
            "inflight_count": len(self._inflight),
            "max_inflight": int(self._max_inflight),
            "num_workers": len(self._infer_pool._workers) if self._infer_pool else 0,
            "workers_auto": self._num_workers == 0,
            "worker_cap": int(self._num_workers_max),
            "infer_timeout_s": float(self._infer_result_timeout_s),
            "mem_total_mb": self._detected_mem_mb,
        })
        return stats

    async def get_stats(self) -> Dict[str, Any]:
        return self.peek_stats()

    # ------------------------------------------------------------------
    # FIX 4: snapshot encode in executor so event loop is never stalled
    # ------------------------------------------------------------------

    async def _cache_snapshot(self, *, camera_uuid: str, frame_bgr, ts_ms: int) -> None:
        camera_key = str(camera_uuid)
        snapshot_ts_ms = int(ts_ms)

        # Quick check without lock (GIL-safe dict read)
        last_ts = int(self._latest_snapshot_ts_ms.get(camera_key, 0) or 0)
        if self._snapshot_min_interval_ms and last_ts and (snapshot_ts_ms - last_ts) < self._snapshot_min_interval_ms:
            return

        loop = asyncio.get_event_loop()
        max_edge = self._snapshot_max_edge
        jpeg_quality = self._snapshot_jpeg_quality
        # Encode in executor; capture frame_bgr in the lambda then release
        # our local reference so the caller's frame can be GC'd sooner.
        encoded = await loop.run_in_executor(
            None,
            lambda bgr=frame_bgr: _encode_jpeg_bytes(
                bgr,
                max_edge=max_edge,
                jpeg_quality=jpeg_quality,
            ),
        )
        # Release frame reference — executor is done with it
        frame_bgr = None
        if not encoded:
            return

        # Atomic store — GIL-safe dict writes, no lock needed
        self._latest_snapshots[camera_key] = encoded
        self._latest_snapshot_ts_ms[camera_key] = snapshot_ts_ms
