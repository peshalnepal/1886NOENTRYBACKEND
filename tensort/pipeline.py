"""Connect camera capture to ordered detection, result caching, and SSE.

The asyncio loop owns channels, frame queues, and result callbacks. CUDA runs on
a dedicated worker thread; JPEG encoding has its own bounded executor.
See README.md for the full flow and ARCHITECTURE.md for tuning details.
"""

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

if __package__:
    from .channels.channel import VideoChannel
    from .env_utils import env_bool, env_float, env_int
    from .frame_pool import FramePool as FramePool
    from .inference_worker import InferenceWorker as InferenceWorker, InferenceWorkerPool
    from .limits import CameraCapacityError, camera_limit
else:
    from channels.channel import VideoChannel
    from env_utils import env_bool, env_float, env_int
    from frame_pool import FramePool as FramePool
    from inference_worker import InferenceWorker as InferenceWorker, InferenceWorkerPool
    from limits import CameraCapacityError, camera_limit

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


# ---------------------------------------------------------------------------
# Detection event subscribers
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
                # A slow consumer needs recent detections, not an old backlog.
                q.get_nowait()
                q.put_nowait(msg)


# ---------------------------------------------------------------------------
# Camera lifecycle and inference dispatch
# ---------------------------------------------------------------------------

class SimpleInferencePipeline(object):
    def __init__(self, infer_q_max=None):
        if infer_q_max is None:
            # Depth 1: one batch executing + one queued. Load is shed in the
            # FramePool, not here.
            infer_q_max = env_int("INFER_QUEUE_MAX", 1, minimum=1)

        self._detected_mem_mb = _detect_total_memory_mb()

        # One ordered GPU queue avoids duplicate engines and cross-batch
        # reordering. Extra CUDA contexts do not add another GPU on an Orin.
        if env_int("INFER_NUM_WORKERS", 1) > 1:
            logger.warning("Using one inference worker to preserve per-camera result order")
        self._max_cameras = camera_limit()
        self._engine_max_batch = 0
        self._lock = asyncio.Lock()
        self._inference_task = None
        self._infer_pool = None
        self._infer_q_max = int(infer_q_max)

        self._channels = {}
        self._channel_tasks = {}
        self._closing = False
        self._started = False
        self._configured_max_batch = min(self._max_cameras,
                                         env_int("INFER_MAX_BATCH", self._max_cameras, minimum=1))
        self._max_batch = self._configured_max_batch
        # Top-up window used only when the pool is underfilled (idle/cold start).
        self._batch_linger_s = env_float("INFER_BATCH_LINGER_MS", 10.0, minimum=0.0) / 1000.0

        # Pool holds a few batches' worth of frames so a camera can contribute
        # more than one frame when others are quiet. Frames older than
        # FRAME_MAX_AGE_MS are dropped rather than inferred — a stale detection
        # is rejected by the tracker anyway and costs a GPU slot.
        self._frame_max_age_s = env_float("FRAME_MAX_AGE_MS", 700.0, minimum=0.0) / 1000.0
        self._buffer = FramePool(
            capacity=env_int("FRAME_POOL_CAP", self._max_batch * 2, minimum=self._max_cameras),
            max_age_s=self._frame_max_age_s,
        )

        # All three are plain dicts read/written from the loop thread and read
        # lock-free from Flask threads — single-key dict access is GIL-atomic.
        self._latest = {}
        self._latest_snapshots = {}
        self._latest_snapshot_ts_ms = {}
        self._snapshot_tasks = {}
        self._snapshot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")
        self._camera_metrics = {}
        self._channel_generation = {}
        self._snapshot_enabled = env_bool("ENABLE_SNAPSHOT_CACHE", True)
        self._snapshot_min_interval_ms = env_int("SNAPSHOT_MIN_INTERVAL_MS", 1000, minimum=0)
        self._snapshot_max_edge = env_int("SNAPSHOT_MAX_EDGE", 640, minimum=64)
        self._snapshot_jpeg_quality = min(100, env_int("SNAPSHOT_JPEG_QUALITY", 65, minimum=1))
        self._snapshot_on_detection_only = env_bool("SNAPSHOT_ON_DETECTION_ONLY", True)    # matches .env.example SNAPSHOT_ON_DETECTION_ONLY=true
        # The cloud tracker ages tracks by frame arrival: it must see the frames
        # where an object is absent, otherwise a disappearance looks like a
        # stalled stream and tracks coast instead of expiring.
        self._emit_empty_detections = env_bool("EMIT_EMPTY_DETECTIONS", True)
        # Failed/timed-out frames carry no detections. Forwarding them makes the
        # tracker treat a transient GPU hiccup as "everything vanished", so they
        # are logged and counted but not broadcast by default.
        self._emit_failed_events = env_bool("EMIT_FAILED_EVENTS", False)
        # Leak guard, not a latency knob: a timed-out frame is no longer thrown
        # away (the worker still delivers a late result), so this can be
        # generous. The old memory-scaled 1.5s on an 8GB Orin Nano was tight
        # enough to time out frames the GPU was about to return.
        self._infer_result_timeout_s = env_float("INFER_RESULT_TIMEOUT_S", 3.0, minimum=0.0)
        self._infer_error_log_interval_s = env_float("INFER_ERROR_LOG_INTERVAL_S", 10.0, minimum=0.0)
        self._last_infer_error_sig = {}
        self._last_infer_error_ts = {}


        self._stats = {
            "frames_in": 0,
            "infer_ok": 0,
            "infer_fail": 0,
            "infer_dropped": 0,
            "detections_total": 0,
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
            "[pipeline] config mem_total_mb=%s workers=1 infer_timeout_s=%.2f "
            "max_batch=%d batch_linger_ms=%d pool_cap=%d frame_max_age_ms=%d",
            self._detected_mem_mb if self._detected_mem_mb is not None else "unknown",
            float(self._infer_result_timeout_s),
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

    async def _stop_channel(self, camera_key):
        """Stop capture before releasing its slot or starting a replacement."""
        task = self._channel_tasks.pop(camera_key, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        channel = self._channels.get(camera_key)
        if channel is not None:
            await channel.stop()

    def _clear_camera_results(self, camera_key):
        self._buffer.discard_camera(camera_key)
        for cache in (self._latest, self._latest_snapshots, self._latest_snapshot_ts_ms,
                      self._last_infer_error_sig, self._last_infer_error_ts, self._camera_metrics):
            cache.pop(camera_key, None)

    async def add_channel(self, cfg):
        camera_key = str(cfg.camera_uuid)
        if not cfg.enabled:
            await self.remove_channel(camera_key)
            return
        new_channel = VideoChannel(cfg)
        async with self._lock:
            if camera_key not in self._channels and len(self._channels) >= self._max_cameras:
                raise CameraCapacityError("Device supports at most {} enabled cameras".format(self._max_cameras))
            await self._stop_channel(camera_key)
            self._channel_generation[camera_key] = object()
            self._clear_camera_results(camera_key)
            self._channels[camera_key] = new_channel
            if self._started and not self._closing:
                self._start_channel_task(camera_key)

    async def remove_channel(self, camera_uuid):
        camera_key = str(camera_uuid)
        async with self._lock:
            await self._stop_channel(camera_key)
            self._channels.pop(camera_key, None)
            self._channel_generation.pop(camera_key, None)
            self._clear_camera_results(camera_key)

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

            self._infer_pool = InferenceWorkerPool(
                loop=loop,
                max_q_per_worker=self._infer_q_max,
                ready_cb=self._worker_free_evt.set,
                late_result_cb=self._handle_result,
            )
            try:
                self._engine_max_batch = await self._infer_pool.wait_ready()
            except Exception:
                self._infer_pool.stop()
                self._started = False
                raise
            self._max_batch = min(self._configured_max_batch, self._engine_max_batch)
            logger.info(
                "[pipeline] starting camera_count=%d worker_count=1",
                len(self._channels),
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
        for fut, _deadline in list(self._inflight.values()):
            fut.cancel()
        self._inflight.clear()
        self._buffer = FramePool(self._buffer._capacity, self._frame_max_age_s)
        if self._snapshot_tasks:
            await asyncio.gather(*list(self._snapshot_tasks.values()), return_exceptions=True)
        self._snapshot_executor.shutdown(wait=False)

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
                if ev.detection_enabled:
                    self._buffer.put(ev)
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
    
    def _prepare_frame_job(self, event, loop):
        """Attach metadata and a completion callback to one captured frame."""
        camera_uuid = str(event.camera_uuid)
        bgr = getattr(event, "frame", None)
        event.frame = None   # let the event be GC'd

        if bgr is None:
            self._stats["infer_fail"] += 1
            if self._should_log_infer_failure(camera_uuid, "no frame data"):
                logger.warning(
                    "[pipeline] camera=%s produced an event with no frame",
                    camera_uuid,
                )
            return None

        meta = {
            "camera_uuid": camera_uuid,
            "channel_id": getattr(event, "channel_id", None),
            "frame_ts_ms": int(event.ts_ms),
            "frame_seq": int(event.seq),
            "_bgr_ref": bgr,
            "_generation": self._channel_generation.get(camera_uuid),
        }

        fut = loop.create_future()
        inflight_key = (camera_uuid, int(event.seq))
        deadline = loop.time() + self._infer_result_timeout_s
        self._inflight[inflight_key] = (fut, deadline)

        # Each call owns its metadata, so callbacks cannot mix camera frames.
        def on_done(completed):
            if self._inflight.get(inflight_key, (None,))[0] is completed:
                self._inflight.pop(inflight_key, None)
            if completed.cancelled():
                meta.pop("_bgr_ref", None)
                return
            try:
                self._handle_result(completed.result(), meta)
            except Exception:
                logger.exception("[pipeline] result handler failed")

        fut.add_done_callback(on_done)

        return bgr, meta, fut

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

                job = []
                for event in events:
                    frame_job = self._prepare_frame_job(event, loop)
                    if frame_job is not None:
                        job.append(frame_job)

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
        Process one inference result on the pipeline event loop.

        Both normal completion and late results are scheduled on this loop,
        so the GPU worker never writes these caches directly.
        """
        camera_uuid = str(meta.get("camera_uuid", "unknown"))
        bgr = meta.pop("_bgr_ref", None)
        # A replacement can reuse the UUID while old GPU work is still running.
        # Match the generation so old work cannot overwrite the new results.
        if self._closing or ("_generation" in meta and
                self._channel_generation.get(camera_uuid) is not meta["_generation"]):
            return
        ts_ms = int(meta.get("frame_ts_ms", 0))
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
            previous = self._camera_metrics.get(camera_uuid, {})
            self._camera_metrics[camera_uuid] = {
                "infer_ok": previous.get("infer_ok", 0) + 1,
                "last_result_ts_ms": int(time.time() * 1000),
                "frame_age_ms": max(0, int(time.time() * 1000) - ts_ms),
                "frame_seq": frame_seq,
            }
            # Empty frames still emit (when EMIT_EMPTY_DETECTIONS): the tracker
            # ages tracks by frame arrival, so it must see "object gone" frames.
            should_emit = self._emit_empty_detections or bool(detections)

            # JPEG encode is the one genuinely slow step — keep it off the loop.
            if self._snapshot_enabled and bgr is not None:
                if (not self._snapshot_on_detection_only) or detections:
                    last_ts = self._latest_snapshot_ts_ms.get(camera_uuid, 0)
                    if (camera_uuid not in self._snapshot_tasks and len(self._snapshot_tasks) < self._max_cameras and
                            ts_ms - last_ts >= self._snapshot_min_interval_ms):
                        task = asyncio.ensure_future(self._cache_snapshot(
                            camera_uuid=camera_uuid, frame_bgr=bgr, ts_ms=ts_ms,
                            generation=meta.get("_generation"),
                        ))
                        self._snapshot_tasks[camera_uuid] = task
                        task.add_done_callback(lambda _f, key=camera_uuid: self._snapshot_tasks.pop(key, None))

        # Release frame reference now that the snapshot task holds its own.
        bgr = None

        # Atomic dict write — GIL-safe, no lock needed here
        self._latest[camera_uuid] = result

        if should_emit:
            self.broadcaster.broadcast(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # All three are called straight from Flask threads: plain dict reads are
    # GIL-safe, so no event-loop hop (and no async wrapper) is needed.

    def peek_latest(self, camera_uuid):
        return self._latest.get(str(camera_uuid))

    def peek_latest_snapshot(self, camera_uuid):
        return self._latest_snapshots.get(str(camera_uuid))

    def peek_stats(self) -> Dict[str, Any]:
        stats = dict(self._stats)
        stats.update({
            "channel_count": len(self._channels),
            "max_cameras": self._max_cameras,
            "cameras": dict(self._camera_metrics),
            "capture": {key: ch.status() for key, ch in list(self._channels.items())},
            "snapshot_pending": len(self._snapshot_tasks),
            "latest_cache_size": len(self._latest),
            "snapshot_cache_size": len(self._latest_snapshots),
            "infer_q_per_worker": int(self._infer_q_max),
            "pool_depth": self._buffer.depth(),
            "pool_capacity": int(self._buffer._capacity),
            "pool_evicted_total": int(self._buffer.evicted_total),
            "pool_expired_total": int(self._buffer.expired_total),
            "frame_max_age_ms": int(self._frame_max_age_s * 1000),
            "inflight_count": len(self._inflight),
            "num_workers": len(self._infer_pool._workers) if self._infer_pool else 0,
            "workers_auto": False,
            "worker_cap": 1,
            "max_batch": int(self._max_batch),
            "configured_max_batch": self._configured_max_batch,
            "engine_max_batch": self._engine_max_batch,
            "inference_ready": bool(self._started and self._engine_max_batch and not self._closing),
            "batch_linger_ms": int(self._batch_linger_s * 1000),
            "infer_timeout_s": float(self._infer_result_timeout_s),
            "mem_total_mb": self._detected_mem_mb,
        })
        return stats

    # ------------------------------------------------------------------
    # Bounded snapshot encoding
    # ------------------------------------------------------------------

    async def _cache_snapshot(self, *, camera_uuid: str, frame_bgr, ts_ms: int, generation=None) -> None:
        camera_key = str(camera_uuid)
        snapshot_ts_ms = int(ts_ms)

        # Quick check without lock (GIL-safe dict read)
        last_ts = int(self._latest_snapshot_ts_ms.get(camera_key, 0) or 0)
        if self._snapshot_min_interval_ms and last_ts and (snapshot_ts_ms - last_ts) < self._snapshot_min_interval_ms:
            return

        loop = asyncio.get_event_loop()
        max_edge = self._snapshot_max_edge
        jpeg_quality = self._snapshot_jpeg_quality
        # JPEG encoding runs off the pipeline loop. Keep the frame alive until
        # the executor finishes, then drop this coroutine's reference.
        encoded = await loop.run_in_executor(
            self._snapshot_executor,
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
        if self._closing or (generation is not None and self._channel_generation.get(camera_key) is not generation):
            return

        # Atomic store — GIL-safe dict writes, no lock needed
        self._latest_snapshots[camera_key] = encoded
        self._latest_snapshot_ts_ms[camera_key] = snapshot_ts_ms
