"""
Dispatch / result-path tests (no GPU needed).

Covers the flicker fixes:
  * a result arriving after its timeout is still delivered, not discarded
  * failure events are not broadcast to the cloud tracker by default
  * empty detections ARE broadcast (the tracker needs them to age tracks)
  * the sweeper resolves stuck frames without a task-per-frame

Run:  python3 tests/test_dispatch.py     (from Backend/tensort)
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("database", MagicMock())
sys.modules.setdefault("database_orm", MagicMock())

from pipeline import InferenceWorker, SimpleInferencePipeline


def detections_event(camera="cam1", dets=None):
    return {
        "type": "DetectionsProducedEvent",
        "camera_uuid": camera,
        "frame_seq": 1,
        "frame_ts_ms": 0,
        "detections": dets if dets is not None else [],
    }


def failed_event(camera="cam1", reason="Inference timed out"):
    return {
        "type": "InferenceFailedEvent",
        "camera_uuid": camera,
        "frame_seq": 1,
        "frame_ts_ms": 0,
        "reason": reason,
    }


def test_late_result_is_salvaged():
    """
    The sweeper timed the frame out, then the GPU returned real detections.
    The old code dropped them on the floor (fut.done() -> discard), leaving a
    gap in the camera's detection stream. They must now be delivered.
    """
    async def run():
        loop = asyncio.get_event_loop()
        delivered = []

        worker = InferenceWorker.__new__(InferenceWorker)   # no thread/CUDA
        worker._loop = loop
        worker._late_result_cb = lambda res, meta: delivered.append(res)

        fut = loop.create_future()
        fut.set_result(failed_event())          # already timed out by the sweeper

        real = detections_event(dets=[{"cls_name": "person"}])
        worker._deliver_result(fut, real, {"camera_uuid": "cam1"})

        assert delivered == [real], delivered
        print("late-result-salvaged: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_late_failure_is_not_resent():
    """A late result that is itself a failure carries nothing useful."""
    async def run():
        loop = asyncio.get_event_loop()
        delivered = []

        worker = InferenceWorker.__new__(InferenceWorker)
        worker._loop = loop
        worker._late_result_cb = lambda res, meta: delivered.append(res)

        fut = loop.create_future()
        fut.set_result(failed_event())
        worker._deliver_result(fut, failed_event(reason="engine error"), {})

        assert delivered == []
        print("late-failure-not-resent: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_failure_events_are_not_broadcast_but_empties_are():
    """
    A failure looks like "no objects" to the cloud tracker and would age every
    track on that camera — that is the box-blinking. Empty detection frames are
    genuine and must still flow.
    """
    async def run():
        p = SimpleInferencePipeline()
        sent = []
        p.broadcaster.broadcast = lambda msg: sent.append(msg)

        p._handle_result(failed_event(), {"camera_uuid": "cam1"})
        assert sent == [], "failure must not be broadcast by default"

        empty = detections_event(dets=[])
        p._handle_result(empty, {"camera_uuid": "cam1"})
        assert sent == [empty], "empty detections must be broadcast for track aging"

        assert p.peek_stats()["infer_fail"] == 1, "failure still counted"
        print("failure-not-broadcast / empty-broadcast: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_sweeper_resolves_stuck_frames():
    """One periodic sweeper replaces a watchdog task per frame."""
    async def run():
        p = SimpleInferencePipeline()
        p._infer_result_timeout_s = 0.05
        loop = asyncio.get_event_loop()

        fut = loop.create_future()
        resolved = []
        fut.add_done_callback(lambda f: resolved.append(f.result()))

        p._inflight[("cam1", 7)] = (fut, loop.time() - 1.0)   # already overdue

        task = asyncio.ensure_future(p._sweep_inflight())
        await asyncio.sleep(0.3)
        p._closing = True
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert fut.done(), "sweeper should have resolved the overdue frame"
        assert resolved[0]["type"] == "InferenceFailedEvent"
        assert resolved[0]["camera_uuid"] == "cam1"
        assert ("cam1", 7) not in p._inflight
        print("sweeper-resolves-stuck-frames: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_worker_capacity_signalling():
    """has_capacity() drives the dispatcher's backpressure wait."""
    async def run():
        loop = asyncio.get_event_loop()
        from pipeline import InferenceWorkerPool
        import queue as _queue

        pool = InferenceWorkerPool.__new__(InferenceWorkerPool)
        w = InferenceWorker.__new__(InferenceWorker)
        w._q = _queue.Queue(maxsize=1)
        pool._workers = [w]

        assert pool.has_capacity() is True
        w._q.put_nowait(["job"])
        assert pool.has_capacity() is False
        w._q.get_nowait()
        assert pool.has_capacity() is True
        print("worker-capacity-signalling: OK")

    asyncio.get_event_loop().run_until_complete(run())


if __name__ == "__main__":
    test_late_result_is_salvaged()
    test_late_failure_is_not_resent()
    test_failure_events_are_not_broadcast_but_empties_are()
    test_sweeper_resolves_stuck_frames()
    test_worker_capacity_signalling()
    print("\nAll dispatch tests passed.")
