# pipeline.py
#
# Frame path: capture thread (one per camera) -> event loop -> FramePool ->
# dispatcher -> inference worker thread (TensorRT) -> Broadcaster -> SSE.
#
# Design points that matter:
#  * FramePool is a SHARED pool with one FIFO deque per camera. Cameras add
#    frames asynchronously; a batch is drawn from whatever is pooled, so the
#    GPU never waits for a slow camera and one camera may contribute several
#    frames when others are quiet. Overload is shed inside the pool, per
#    camera, so a spike never blanks every camera at once.
#  * The dispatcher waits for a free worker slot BEFORE drawing a batch. This
#    is what keeps the pool (not the dispatcher) in charge of what to drop.
#  * One periodic sweeper times out stuck frames instead of one watchdog task
#    per frame, and a result that lands after its timeout is still delivered
#    rather than discarded.
#  * Detection results are handled synchronously on the loop; only the JPEG
#    snapshot encode is offloaded to an executor.
#
# See ARCHITECTURE.md in this directory for the full walkthrough.

import asyncio
import collections
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

    def broadcast(self, msg):
        # Synchronous: it only does put_nowait, never awaits. Keeping it a
        # coroutine forced the result path to spawn a task per frame.
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
# FramePool — shared, fair, multi-frame-per-camera staging for batching
# ---------------------------------------------------------------------------

class FramePool(object):
    """
    Shared pool of pending frames, one FIFO deque per camera.

    Replaces the old CoalescingBuffer (which kept exactly ONE frame per camera
    and therefore could never fill a 10-wide batch: batch fullness was bounded
    by how many DISTINCT cameras happened to produce a frame inside the linger
    window). Here cameras push frames in asynchronously and a batch is drawn
    from whatever is pooled, so one camera can contribute several frames when
    others are quiet — the GPU never idles waiting for a slow camera.

    Eviction when the pool is full:
      1. Frames older than `max_age_s` are always dropped (stale frames are
         worse than useless — the tracker rejects out-of-date boxes).
      2. Otherwise the camera holding the MOST frames loses its OLDEST frame.
         That is the fairness rule: a fast camera that has hogged the pool pays
         for a new camera's frame, not the quiet cameras.
      3. If every camera holds exactly one frame there is no over-represented
         camera to charge, so the globally oldest frame is dropped.

    All access is from the single asyncio event-loop thread — no locks needed.
    Per-camera FIFO order is load-bearing: the cloud tracker discards
    out-of-order frame_seq, so frames of one camera must stay ordered from
    pool -> batch -> future resolution.
    """

    def __init__(self, capacity=30, max_age_s=0.7):
        self._frames = {}                 # camera_key -> deque[RTSPEvent]
        self._total = 0
        self._capacity = max(1, int(capacity))
        self._max_age_s = max(0.0, float(max_age_s))
        self._data_evt = asyncio.Event()
        self.evicted_total = 0
        self.expired_total = 0

    # -- internals ----------------------------------------------------------

    def _drop_expired(self):
        """Drop frames older than max_age_s. Deques are FIFO so the head is the
        oldest — stop scanning a camera as soon as its head is fresh."""
        if self._max_age_s <= 0.0 or self._total == 0:
            return
        cutoff_ms = (time.time() - self._max_age_s) * 1000.0
        for dq in self._frames.values():
            while dq and float(dq[0].ts_ms) < cutoff_ms:
                dq.popleft()
                self._total -= 1
                self.expired_total += 1

    def _evict_one(self):
        """Free exactly one slot using the fairness rule above."""
        victim_key = None
        victim_len = 0
        victim_ts = 0.0
        for key, dq in self._frames.items():
            if not dq:
                continue
            head_ts = float(dq[0].ts_ms)
            # Most frames wins; tie-break on the oldest head so eviction is
            # deterministic rather than dict-iteration-order dependent.
            if victim_key is None or len(dq) > victim_len or (len(dq) == victim_len and head_ts < victim_ts):
                victim_key = key
                victim_len = len(dq)
                victim_ts = head_ts

        if victim_key is None:
            return

        # When every camera holds exactly one frame the scan above degenerates
        # to "oldest head wins", which is exactly the desired fallback: nobody
        # is over-represented, so the globally oldest frame is the one to drop.
        self._frames[victim_key].popleft()
        self._total -= 1
        self.evicted_total += 1

    # -- producer side ------------------------------------------------------

    def put(self, ev):
        """Add one frame. Called from the event loop (capture thread -> loop)."""
        self._drop_expired()
        if self._total >= self._capacity:
            self._evict_one()

        key = str(ev.camera_uuid)
        dq = self._frames.get(key)
        if dq is None:
            # A newly added camera starts contributing immediately — no
            # scheduler registration, no rebalancing step.
            dq = collections.deque()
            self._frames[key] = dq
        dq.append(ev)
        self._total += 1
        self._data_evt.set()

    def discard_camera(self, camera_key):
        dq = self._frames.pop(str(camera_key), None)
        if dq:
            self._total -= len(dq)

    # -- consumer side ------------------------------------------------------

    async def get_batch(self, max_n, linger_s=0.0):
        """
        Draw up to `max_n` frames, round-robin across cameras, oldest first.

        Never waits for all cameras: it returns whatever is pooled. The linger
        is a small top-up window used ONLY when the pool is underfilled (idle or
        cold start); in steady state the pool has already accumulated frames
        while the GPU was busy with the previous batch, so batches self-fill.
        """
        max_n = max(1, int(max_n))

        while True:
            self._drop_expired()
            if self._total > 0:
                break
            self._data_evt.clear()
            await self._data_evt.wait()

        if self._total < max_n and linger_s > 0.0:
            await asyncio.sleep(linger_s)
            self._drop_expired()

        # Round-robin: every camera with pending frames contributes one frame
        # before any camera contributes a second. This is what makes per-camera
        # detection FPS equal under load instead of first-come-first-served.
        evs = []
        while len(evs) < max_n:
            ready = [k for k, dq in self._frames.items() if dq]
            if not ready:
                break
            ready.sort(key=lambda k: self._frames[k][0].ts_ms)
            for key in ready:
                if len(evs) >= max_n:
                    break
                evs.append(self._frames[key].popleft())
                self._total -= 1

        # Drop entries for cameras that are now empty, so a long-running service
        # that cycles through many cameras does not keep scanning dead keys.
        if self._total == 0:
            self._frames.clear()

        return evs

    def depth(self):
        return self._total


# ---------------------------------------------------------------------------
# FIX 1: InferenceWorker — unchanged; now used in a pool
# ---------------------------------------------------------------------------

class InferenceWorker(object):
    """
    One dedicated inference thread owning its own CUDA context + TRT engines.
    Never share across threads.
    """
    def __init__(self, loop, worker_id=0, max_q=1, ready_cb=None, late_result_cb=None):
        self._loop = loop
        self._worker_id = int(worker_id)
        # Queue holds whole BATCH JOBS (each a list of (bgr, meta, fut)).
        # Depth 1 = one batch executing + one queued. The dispatcher waits for
        # capacity BEFORE drawing frames, so surplus load is shed inside the
        # FramePool (fairly, newest-first) instead of being dropped here.
        self._q = queue.Queue(maxsize=max(1, int(max_q)))
        self._ready_cb = ready_cb
        self._late_result_cb = late_result_cb
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="trt-infer-worker-{}".format(worker_id),
            daemon=True,
        )
        self._infer = None
        self._thread.start()

    def submit_job(self, job):
        """job: list of (bgr, meta, fut). Returns False if the queue is full."""
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
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

    def _fail_result(self, meta, reason):
        return {
            "type": "InferenceFailedEvent",
            "camera_uuid": str(meta.get("camera_uuid", "unknown")),
            "frame_ts_ms": int(meta.get("frame_ts_ms", 0)),
            "frame_seq": int(meta.get("frame_seq", 0)),
            "reason": reason,
        }

    def _run(self):
        from trt_infer import build_default
        try:
            self._infer = build_default()
            eng_max = int(getattr(self._infer, "max_batch", 1))
            logger.info(
                "[worker-%d] TRTInfer ready (engine_max_batch=%d)",
                self._worker_id, eng_max,
            )
            if eng_max <= 1:
                logger.warning(
                    "[worker-%d] engine max_batch=1 — NOT a dynamic-batch engine; "
                    "batches will be split to 1 frame each. Re-export a dynamic "
                    "engine (see .env.example) for real batching.",
                    self._worker_id,
                )
        except Exception as e:
            logger.exception("[worker-%d] Failed to init TRT: %s", self._worker_id, e)
            self._infer = None

        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.05)
            except queue.Empty:
                continue

            # A slot just freed up — tell the dispatcher it may draw the next
            # batch while this one runs on the GPU.
            if self._ready_cb is not None:
                try:
                    self._loop.call_soon_threadsafe(self._ready_cb)
                except Exception:
                    pass

            if job is None:        # stop sentinel
                continue

            # job is an already-assembled batch: [(bgr, meta, fut), ...]
            bgrs = [it[0] for it in job]
            metas = [it[1] for it in job]
            futs = [it[2] for it in job]

            if self._infer is None:
                results = [self._fail_result(m, "TRT inference not initialized") for m in metas]
            else:
                try:
                    results = self._infer.infer_multitask_batch(bgrs, metas)
                except Exception as e:
                    reason = "{}: {}".format(type(e).__name__, e)
                    results = [self._fail_result(m, reason) for m in metas]

            for fut, res, meta in zip(futs, results, metas):
                self._loop.call_soon_threadsafe(self._deliver_result, fut, res, meta)

    def _deliver_result(self, fut, res, meta):
        """
        Resolve the frame's future, or salvage a late result.

        The sweeper may already have timed this frame out. Discarding the real
        result in that case (as the old code did) threw away completed GPU work
        and left the camera with a gap — a direct cause of detections blinking
        out. If the detections did arrive, hand them to the pipeline anyway.
        """
        try:
            if not fut.done():
                fut.set_result(res)
                return
        except Exception:
            return

        if self._late_result_cb is None:
            return
        if not isinstance(res, dict) or res.get("type") == "InferenceFailedEvent":
            return
        try:
            self._late_result_cb(res, meta)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FIX 1 (cont.): InferenceWorkerPool — N workers, cameras pinned by hash
# ---------------------------------------------------------------------------

class InferenceWorkerPool(object):
    """
    Wraps N InferenceWorker threads.

    Batches are already assembled (cross-camera) by the dispatcher, so whole
    batch jobs are round-robined across workers. On a single GPU one worker is
    usually best (multiple CUDA contexts time-slice the GPU); extra workers only
    help if batches arrive faster than one worker can drain them.
    """
    def __init__(self, loop, num_workers, max_q_per_worker=1, ready_cb=None, late_result_cb=None):
        self._loop = loop
        self._max_q_per_worker = max_q_per_worker
        self._ready_cb = ready_cb
        self._late_result_cb = late_result_cb
        self._workers = []
        self._rr = 0
        self.ensure_size(num_workers)

    def ensure_size(self, num_workers):
        target = max(1, int(num_workers))
        current = len(self._workers)
        if target <= current:
            return current
        for i in range(current, target):
            self._workers.append(
                InferenceWorker(
                    self._loop,
                    worker_id=i,
                    max_q=self._max_q_per_worker,
                    ready_cb=self._ready_cb,
                    late_result_cb=self._late_result_cb,
                )
            )
        logger.info("InferenceWorkerPool: %d workers", len(self._workers))
        return len(self._workers)

    def has_capacity(self):
        """True if at least one worker can accept a batch right now."""
        for w in self._workers:
            if not w._q.full():
                return True
        return False

    def submit_batch(self, job):
        """
        Hand a whole batch job to a worker (round-robin). Tries every worker
        once; returns False only if all worker queues are full.
        """
        n = len(self._workers)
        if n == 0:
            return False
        for off in range(n):
            w = self._workers[(self._rr + off) % n]
            if w.submit_job(job):
                self._rr = (self._rr + off + 1) % n
                return True
        return False

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
    def __init__(self, infer_q_max=None):
        if infer_q_max is None:
            # Depth 1: one batch executing + one queued. Load is shed in the
            # FramePool, not here.
            infer_q_max = _env_int("INFER_QUEUE_MAX", 1, minimum=1)

        self._detected_mem_mb = _detect_total_memory_mb()
        auto_worker_cap = _default_auto_worker_cap(self._detected_mem_mb)

        # Number of parallel inference threads (one per camera is a good default)
        self._num_workers = _env_int("INFER_NUM_WORKERS", 0, minimum=0)
        # 0 → auto-size to number of cameras (capped at INFER_NUM_WORKERS_MAX)
        self._num_workers_max = _env_int("INFER_NUM_WORKERS_MAX", auto_worker_cap, minimum=1)

        self._channels = {}
        self._channel_tasks = {}
        self._closing = False
        self._started = False

        # Batched inference: the dispatcher draws up to INFER_MAX_BATCH frames
        # from the shared FramePool and runs them as one (B,3,H,W) GPU call.
        # Requires a dynamic-batch engine (see .env.example); falls back to B=1
        # with a fixed engine. INFER_MAX_BATCH must be <= the engine's maxShapes
        # batch.
        self._max_batch = _env_int("INFER_MAX_BATCH", 10, minimum=1)
        # Top-up window used only when the pool is underfilled (idle/cold start).
        self._batch_linger_s = _env_float("INFER_BATCH_LINGER_MS", 10.0, minimum=0.0) / 1000.0

        # Pool holds a few batches' worth of frames so a camera can contribute
        # more than one frame when others are quiet. Frames older than
        # FRAME_MAX_AGE_MS are dropped rather than inferred — a stale detection
        # is rejected by the tracker anyway and costs a GPU slot.
        self._frame_max_age_s = _env_float("FRAME_MAX_AGE_MS", 700.0, minimum=0.0) / 1000.0
        self._buffer = FramePool(
            capacity=_env_int("FRAME_POOL_CAP", self._max_batch * 3, minimum=1),
            max_age_s=self._frame_max_age_s,
        )

        self._latest = {}
        self._latest_snapshots = {}
        self._latest_snapshot_ts_ms = {}
        self._latest_lock = asyncio.Lock()
        self._snapshot_enabled = _env_bool("ENABLE_SNAPSHOT_CACHE", True)
        self._snapshot_min_interval_ms = _env_int("SNAPSHOT_MIN_INTERVAL_MS", 500, minimum=0)   # .env.example ships 1000
        self._snapshot_max_edge = _env_int("SNAPSHOT_MAX_EDGE", 960, minimum=64)
        self._snapshot_jpeg_quality = _env_int("SNAPSHOT_JPEG_QUALITY", 75, minimum=1)
        self._snapshot_on_detection_only = _env_bool("SNAPSHOT_ON_DETECTION_ONLY", True)    # matches .env.example SNAPSHOT_ON_DETECTION_ONLY=true
        # The cloud tracker ages tracks by frame arrival: it must see the frames
        # where an object is absent, otherwise a disappearance looks like a
        # stalled stream and tracks coast instead of expiring.
        self._emit_empty_detections = _env_bool("EMIT_EMPTY_DETECTIONS", True)
        # Failed/timed-out frames carry no detections. Forwarding them makes the
        # tracker treat a transient GPU hiccup as "everything vanished", so they
        # are logged and counted but not broadcast by default.
        self._emit_failed_events = _env_bool("EMIT_FAILED_EVENTS", False)
        # Leak guard, not a latency knob: a timed-out frame is no longer thrown
        # away (the worker still delivers a late result), so this can be
        # generous. The old memory-scaled 1.5s on an 8GB Orin Nano was tight
        # enough to time out frames the GPU was about to return.
        self._infer_result_timeout_s = _env_float("INFER_RESULT_TIMEOUT_S", 3.0, minimum=0.0)
        self._infer_error_log_interval_s = _env_float("INFER_ERROR_LOG_INTERVAL_S", 10.0, minimum=0.0)
        self._last_infer_error_sig = {}
        self._last_infer_error_ts = {}

        self._lock = asyncio.Lock()
        self._inference_task = None
        self._infer_pool = None         # created on start()
        self._infer_q_max = int(infer_q_max)

        # Safety net only: the dispatcher waits for worker capacity before
        # drawing a batch, so inflight is bounded by (queued + executing)
        # batches by construction. Exceeding this means something leaked.
        self._max_inflight = _env_int(
            "MAX_INFLIGHT_FRAMES", max(8, self._max_batch * 4), minimum=1
        )
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

        # Track in-flight futures so we can time them out and cancel on
        # shutdown. Only accessed from the single event-loop thread — no lock.
        self._inflight = {}             # (camera_uuid, seq) -> (Future, deadline)
        self._sweeper_task = None
        # Set by a worker when it dequeues a batch; the dispatcher waits on it
        # instead of assembling batches it would have to throw away.
        self._worker_free_evt = asyncio.Event()
        self._worker_free_evt.set()

        logger.info(
            "[pipeline] config mem_total_mb=%s workers=%s worker_cap=%d infer_timeout_s=%.2f "
            "max_inflight=%d max_batch=%d batch_linger_ms=%d pool_cap=%d frame_max_age_ms=%d",
            self._detected_mem_mb if self._detected_mem_mb is not None else "unknown",
            "auto" if self._num_workers == 0 else int(self._num_workers),
            int(self._num_workers_max),
            float(self._infer_result_timeout_s),
            int(self._max_inflight),
            int(self._max_batch),
            int(self._batch_linger_s * 1000),
            int(self._buffer._capacity),
            int(self._frame_max_age_s * 1000),
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

        self._buffer.discard_camera(camera_key)

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
                ready_cb=self._worker_free_evt.set,
                late_result_cb=self._handle_result,
            )
            logger.info(
                "[pipeline] starting camera_count=%d worker_count=%d auto_workers=%s",
                len(self._channels),
                len(self._infer_pool._workers),
                self._num_workers == 0,
            )

            self._inference_task = asyncio.ensure_future(self._pump_inference())
            if self._infer_result_timeout_s > 0.0:
                self._sweeper_task = asyncio.ensure_future(self._sweep_inflight())

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

            if self._sweeper_task is not None and not self._sweeper_task.done():
                self._sweeper_task.cancel()

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

        for task in (self._inference_task, self._sweeper_task):
            if task is None:
                continue
            try:
                await task
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

    # ------------------------------------------------------------------
    # Internal tasks
    # ------------------------------------------------------------------

    def _start_channel_task(self, camera_key):
        ch = self._channels.get(camera_key)
        if ch is None:
            return
        self._channel_tasks[camera_key] = asyncio.ensure_future(
            self._pump_channel(camera_key, ch)
        )

    async def _pump_channel(self, camera_key, ch):
        try:
            async for ev in ch.stream():
                if self._closing:
                    break
                if isinstance(ev, RTSPEvent):
                    if getattr(ev, "detection_enabled", True):
                        self._buffer.put(ev)
                else:
                    # Connected/disconnected notices: nothing consumes these on
                    # the edge, the cloud infers link state from detection flow.
                    logger.info(
                        "[pipeline] channel event camera=%s type=%s",
                        camera_key, getattr(ev, "type", type(ev).__name__),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("[pipeline] channel pump failed camera=%s reason=%s", camera_key, e)
        finally:
            try:
                await ch.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Inference dispatch — capacity-gated, never drops a whole batch
    # ------------------------------------------------------------------

    async def _pump_inference(self):
        """
        Draw a batch from the FramePool and hand it to a worker.

        The loop waits for a free worker slot BEFORE drawing frames. That
        inversion is what removed the old whole-batch drops: when the GPU falls
        behind, frames simply stay in the pool and the pool sheds the surplus
        per-camera (oldest of the most-represented camera first). Previously an
        overload dropped every camera's frame in the batch at once, blanking
        all panels simultaneously.

        _inflight is only touched from this event-loop thread, so no lock.
        """
        loop = asyncio.get_event_loop()

        try:
            while not self._closing:
                if self._infer_pool is None:
                    await asyncio.sleep(0.05)
                    continue

                # Wait for a worker slot. The timeout is a liveness fallback in
                # case a ready callback is ever missed.
                while not self._infer_pool.has_capacity():
                    self._worker_free_evt.clear()
                    try:
                        await asyncio.wait_for(self._worker_free_evt.wait(), timeout=0.25)
                    except asyncio.TimeoutError:
                        pass
                    if self._closing:
                        return

                events = await self._buffer.get_batch(self._max_batch, self._batch_linger_s)
                if not events:
                    continue

                self._stats["frames_in"] += len(events)

                job = []   # [(bgr, meta, fut), ...]
                for rtsp_ev in events:
                    camera_uuid_str = str(rtsp_ev.camera_uuid)
                    bgr = getattr(rtsp_ev, "frame", None)
                    rtsp_ev.frame = None   # let the event be GC'd

                    if bgr is None:
                        self._stats["infer_fail"] += 1
                        if self._should_log_infer_failure(camera_uuid_str, "no frame data"):
                            logger.warning(
                                "[pipeline] camera=%s produced an event with no frame "
                                "(emit_format=raw required)", camera_uuid_str,
                            )
                        continue

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
                    deadline = loop.time() + self._infer_result_timeout_s
                    self._inflight[inflight_key] = (fut, deadline)

                    # Resolve inline: _handle_result is synchronous, so a frame
                    # costs no extra event-loop task (the old code spawned two
                    # per frame — ~240 tasks/s at 10 cameras).
                    def _on_done(f, _meta=meta, _key=inflight_key):
                        self._inflight.pop(_key, None)
                        try:
                            self._handle_result(f.result(), _meta)
                        except Exception:
                            logger.exception("[pipeline] result handler failed")
                    fut.add_done_callback(_on_done)

                    job.append((bgr, meta, fut))

                if not job:
                    continue

                # Capacity was checked above, so this only fails on a race with
                # worker teardown. Resolve the futures immediately rather than
                # letting the sweeper hold their frame references for seconds.
                if not self._infer_pool.submit_batch(job):
                    self._stats["infer_dropped"] += len(job)
                    logger.warning("[pipeline] worker pool refused a batch of %d", len(job))
                    for (_bgr, _meta, _fut) in job:
                        if not _fut.done():
                            _fut.set_result({
                                "type": "InferenceFailedEvent",
                                "camera_uuid": _meta.get("camera_uuid", "unknown"),
                                "frame_ts_ms": _meta.get("frame_ts_ms", 0),
                                "frame_seq": _meta.get("frame_seq", 0),
                                "reason": "Worker pool refused the batch",
                            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("[pipeline] inference pump failed: %s", e)

    async def _sweep_inflight(self):
        """
        Resolve frames whose result never arrived.

        Replaces the old per-frame watchdog coroutine (one asyncio task + one
        timer per frame). A worker that overruns is a GPU stall, not a per-frame
        event, so one periodic sweep is enough — and if the real result lands
        afterwards the worker still delivers it via late_result_cb instead of
        throwing the finished work away.
        """
        interval = max(0.25, float(self._infer_result_timeout_s) / 4.0)
        loop = asyncio.get_event_loop()
        try:
            while not self._closing:
                await asyncio.sleep(interval)
                if not self._inflight:
                    continue
                now = loop.time()
                for key, (fut, deadline) in list(self._inflight.items()):
                    if deadline > now:
                        continue
                    self._inflight.pop(key, None)
                    if fut.done():
                        continue
                    camera_uuid, frame_seq = key
                    fut.set_result({
                        "type": "InferenceFailedEvent",
                        "camera_uuid": camera_uuid,
                        "frame_seq": frame_seq,
                        "frame_ts_ms": 0,
                        "reason": "Inference timed out",
                    })
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[pipeline] inflight sweeper failed")

    def _handle_result(self, result, meta):
        """
        Process one inference result. Runs synchronously inside the future's
        done-callback (or from the worker's late-result path).
        """
        camera_uuid = str(meta.get("camera_uuid", "unknown"))
        bgr = meta.pop("_bgr_ref", None)
        ts_ms = int(meta.get("_ts_ms", 0))
        frame_seq = int(meta.get("frame_seq", 0))

        failed = isinstance(result, dict) and result.get("type") == "InferenceFailedEvent"

        if failed:
            self._stats["infer_fail"] += 1
            reason = str(result.get("reason", "") or "")
            if "queue full" in reason.lower():
                self._stats["infer_dropped"] += 1
            if self._should_log_infer_failure(camera_uuid, reason):
                logger.warning(
                    "[pipeline] Inference failed camera=%s seq=%s reason=%s",
                    camera_uuid, frame_seq, reason or "unknown",
                )
            # A failure carries no detections. Forwarding it makes the cloud
            # tracker see "no objects" and age every track on this camera, which
            # is exactly the box-blinking we are removing.
            should_emit = self._emit_failed_events
        else:
            detections = []
            if isinstance(result, dict):
                detections = result.get("detections", []) or []
            self._stats["infer_ok"] += 1
            self._stats["detections_total"] += len(detections)
            # Empty frames still emit (when EMIT_EMPTY_DETECTIONS): the tracker
            # ages tracks by frame arrival, so it must see "object gone" frames.
            should_emit = self._emit_empty_detections or bool(detections)

            # JPEG encode is the one genuinely slow step — keep it off the loop.
            if self._snapshot_enabled and bgr is not None:
                if (not self._snapshot_on_detection_only) or detections:
                    asyncio.ensure_future(
                        self._cache_snapshot(camera_uuid=camera_uuid, frame_bgr=bgr, ts_ms=ts_ms)
                    )

        # Release frame reference now that the snapshot task holds its own.
        bgr = None

        # Atomic dict write — GIL-safe, no lock needed here
        self._latest[camera_uuid] = result

        if should_emit:
            self.broadcaster.broadcast(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            "pool_depth": self._buffer.depth(),
            "pool_capacity": int(self._buffer._capacity),
            "pool_evicted_total": int(self._buffer.evicted_total),
            "pool_expired_total": int(self._buffer.expired_total),
            "frame_max_age_ms": int(self._frame_max_age_s * 1000),
            "inflight_count": len(self._inflight),
            "max_inflight": int(self._max_inflight),
            "num_workers": len(self._infer_pool._workers) if self._infer_pool else 0,
            "workers_auto": self._num_workers == 0,
            "worker_cap": int(self._num_workers_max),
            "max_batch": int(self._max_batch),
            "batch_linger_ms": int(self._batch_linger_s * 1000),
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
