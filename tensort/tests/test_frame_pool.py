"""
FramePool behaviour tests (no GPU / no cameras needed).

Run:  python3 tests/test_frame_pool.py     (from Backend/tensort)
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pipeline.py imports channels.channel, which imports cv2.
sys.modules.setdefault("cv2", MagicMock())

from pipeline import FramePool


class FakeEvent(object):
    """Minimal stand-in for RTSPEvent: the pool only reads camera_uuid + ts_ms."""

    def __init__(self, camera_uuid, ts_ms):
        self.camera_uuid = camera_uuid
        self.ts_ms = ts_ms
        self.seq = 0
        self.frame = None


def now_ms():
    return time.time() * 1000.0


def test_evicts_from_most_represented_camera():
    """
    The user's rule: cam1 hogging the pool pays for a newcomer's frame, and it
    pays with its OLDEST frame — the quiet cameras are untouched.
    """
    pool = FramePool(capacity=8, max_age_s=10.0)
    base = now_ms()

    for i in range(5):                       # cam1 fills 5 slots
        pool.put(FakeEvent("cam1", base + i))
    for cam in ("cam2", "cam3", "cam4"):     # one frame each
        pool.put(FakeEvent(cam, base + 10))

    assert pool.depth() == 8, pool.depth()

    pool.put(FakeEvent("cam5", base + 20))   # pool full -> must evict

    assert pool.depth() == 8
    assert pool.evicted_total == 1
    assert len(pool._frames["cam1"]) == 4, "most-represented camera should shrink"
    # Its OLDEST frame went, not a recent one.
    assert pool._frames["cam1"][0].ts_ms == base + 1
    for cam in ("cam2", "cam3", "cam4", "cam5"):
        assert len(pool._frames[cam]) == 1, "quiet cameras must not be charged"
    print("evicts-from-most-represented: OK")


def test_evicts_globally_oldest_when_all_singletons():
    """When every camera holds one frame nobody is over-represented, so the
    globally oldest frame is the one that goes."""
    pool = FramePool(capacity=4, max_age_s=10.0)
    base = now_ms()

    pool.put(FakeEvent("cam1", base + 30))
    pool.put(FakeEvent("cam2", base + 10))   # oldest
    pool.put(FakeEvent("cam3", base + 20))
    pool.put(FakeEvent("cam4", base + 40))

    pool.put(FakeEvent("cam5", base + 50))

    assert pool.depth() == 4
    assert "cam2" not in pool._frames or len(pool._frames["cam2"]) == 0
    assert len(pool._frames["cam5"]) == 1
    print("evicts-globally-oldest: OK")


def test_expired_frames_are_dropped():
    """Frames older than max_age_s never reach the GPU."""
    pool = FramePool(capacity=10, max_age_s=0.5)
    old = now_ms() - 2000.0        # 2s old, well past the 500ms limit

    pool.put(FakeEvent("cam1", old))
    pool.put(FakeEvent("cam1", old))
    assert pool.depth() >= 1

    pool.put(FakeEvent("cam2", now_ms()))    # triggers the expiry sweep

    assert pool.depth() == 1, "only the fresh frame should survive"
    assert pool.expired_total == 2
    print("expired-frames-dropped: OK")


def test_batch_is_round_robin_and_does_not_wait_for_all():
    """
    A batch is drawn from whatever is pooled. cam1 has a backlog but every other
    camera still contributes before cam1 gets a second slot.
    """
    async def run():
        pool = FramePool(capacity=30, max_age_s=10.0)
        base = now_ms()

        for i in range(4):
            pool.put(FakeEvent("cam1", base + i))
        pool.put(FakeEvent("cam2", base + 5))
        pool.put(FakeEvent("cam3", base + 6))

        batch = await pool.get_batch(6, linger_s=0.0)

        assert len(batch) == 6
        first_three = [e.camera_uuid for e in batch[:3]]
        assert sorted(first_three) == ["cam1", "cam2", "cam3"], first_three
        # Remaining slots go to the only camera with a backlog.
        assert [e.camera_uuid for e in batch[3:]] == ["cam1", "cam1", "cam1"]
        assert pool.depth() == 0
        print("round-robin-drain: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_get_batch_returns_partial_rather_than_waiting():
    """Two cameras online must not block on the other eight."""
    async def run():
        pool = FramePool(capacity=30, max_age_s=10.0)
        pool.put(FakeEvent("cam1", now_ms()))
        pool.put(FakeEvent("cam2", now_ms()))

        t0 = time.time()
        batch = await pool.get_batch(10, linger_s=0.01)
        elapsed = time.time() - t0

        assert len(batch) == 2, "should return what is available"
        assert elapsed < 0.2, "must not wait for a full batch (took %.3fs)" % elapsed
        print("partial-batch-no-wait: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_new_camera_enters_immediately():
    """Adding a camera requires no registration step in the pool."""
    async def run():
        pool = FramePool(capacity=30, max_age_s=10.0)
        pool.put(FakeEvent("cam1", now_ms()))
        await pool.get_batch(10, linger_s=0.0)

        pool.put(FakeEvent("brand-new-cam", now_ms()))
        batch = await pool.get_batch(10, linger_s=0.0)
        assert [e.camera_uuid for e in batch] == ["brand-new-cam"]
        print("new-camera-enters-immediately: OK")

    asyncio.get_event_loop().run_until_complete(run())


def test_discard_camera():
    pool = FramePool(capacity=30, max_age_s=10.0)
    pool.put(FakeEvent("cam1", now_ms()))
    pool.put(FakeEvent("cam1", now_ms()))
    pool.put(FakeEvent("cam2", now_ms()))

    pool.discard_camera("cam1")

    assert pool.depth() == 1
    assert "cam1" not in pool._frames
    print("discard-camera: OK")


if __name__ == "__main__":
    test_evicts_from_most_represented_camera()
    test_evicts_globally_oldest_when_all_singletons()
    test_expired_frames_are_dropped()
    test_batch_is_round_robin_and_does_not_wait_for_all()
    test_get_batch_returns_partial_rather_than_waiting()
    test_new_camera_enters_immediately()
    test_discard_camera()
    print("\nAll FramePool tests passed.")
