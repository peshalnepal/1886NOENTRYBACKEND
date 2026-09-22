"""Ordered GPU execution. CUDA is created and released on its owner thread."""

import asyncio
import logging
import queue
import threading

logger = logging.getLogger(__name__)


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
        # FramePool (oldest frame from the largest backlog) instead of being dropped here.
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
        self.initialized = threading.Event()
        self.init_error = None
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
        try:
            if __package__:
                from .trt_infer import build_default
            else:
                from trt_infer import build_default
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
            self.init_error = "{}: {}".format(type(e).__name__, e)
            self._infer = None
        finally:
            self.initialized.set()

        try:
            self._process_jobs()
        finally:
            if self._infer is not None:
                try:
                    self._infer.close()
                except Exception:
                    logger.exception("Failed to release inference worker resources")
                self._infer = None

    def _process_jobs(self):
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
            result_futures = [it[2] for it in job]

            if self._infer is None:
                results = [self._fail_result(m, "TRT inference not initialized") for m in metas]
            else:
                try:
                    results = self._infer.infer_multitask_batch(bgrs, metas)
                except Exception as e:
                    reason = "{}: {}".format(type(e).__name__, e)
                    results = [self._fail_result(m, reason) for m in metas]

            for fut, res, meta in zip(result_futures, results, metas):
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
# Worker queue and readiness
# ---------------------------------------------------------------------------

class InferenceWorkerPool(object):
    """Own the single ordered GPU worker and expose queue readiness."""

    def __init__(self, loop, max_q_per_worker=1, ready_cb=None, late_result_cb=None):
        self._workers = [InferenceWorker(
            loop, max_q=max_q_per_worker,
            ready_cb=ready_cb, late_result_cb=late_result_cb,
        )]

    def has_capacity(self):
        """True if at least one worker can accept a batch right now."""
        for w in self._workers:
            if not w._q.full():
                return True
        return False

    async def wait_ready(self, timeout_s=60.0):
        deadline = asyncio.get_event_loop().time() + timeout_s
        while not all(w.initialized.is_set() for w in self._workers):
            if asyncio.get_event_loop().time() >= deadline:
                raise RuntimeError("TensorRT initialization timed out")
            await asyncio.sleep(0.05)
        errors = [w.init_error for w in self._workers if w.init_error]
        if errors:
            raise RuntimeError("TensorRT initialization failed: " + "; ".join(errors))
        return min(w._infer.max_batch for w in self._workers)

    def submit_batch(self, job):
        """Queue a batch; return False if the worker queue is full."""
        return self._workers[0].submit_job(job)

    def stop(self):
        for w in self._workers:
            w.stop()

    def join(self, timeout=2.0):
        for w in self._workers:
            w.join(timeout)
