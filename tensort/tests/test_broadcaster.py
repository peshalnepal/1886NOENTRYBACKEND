"""Slow consumers receive fresh state without starving quieter cameras."""

import asyncio
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("cv2", MagicMock())
from pipeline import Broadcaster


class BroadcasterTests(unittest.IsolatedAsyncioTestCase):
    def event(self, camera, seq, detections=None):
        return {"camera_uuid": camera, "frame_seq": seq, "detections": detections or []}

    async def test_scoped_subscriber_coalesces_and_keeps_empty_detections(self):
        broadcaster = Broadcaster()
        subscriber = await broadcaster.subscribe("first")
        for seq in range(250):
            broadcaster.broadcast(self.event("first", seq, [{"class": "person"}]))
            broadcaster.broadcast(self.event("other", seq))
        broadcaster.broadcast(self.event("first", 250))
        self.assertEqual(subscriber.qsize(), 1)
        self.assertEqual(await subscriber.get(), self.event("first", 250))

    async def test_busy_camera_cannot_displace_other_cameras(self):
        broadcaster = Broadcaster()
        with patch.dict(os.environ, {"MAX_CAMERAS": "3"}):
            subscriber = await broadcaster.subscribe()
        for camera in ("first", "quiet", "third"):
            broadcaster.broadcast(self.event(camera, 0))
        for seq in range(1, 250):
            broadcaster.broadcast(self.event("first", seq))
        self.assertEqual(subscriber.qsize(), 3)
        self.assertEqual(await subscriber.get(), self.event("first", 249))
        # Even while the busy camera keeps producing, the quiet one is next.
        broadcaster.broadcast(self.event("first", 250))
        self.assertEqual(await subscriber.get(), self.event("quiet", 0))
        self.assertEqual(await subscriber.get(), self.event("third", 0))

    async def test_expired_pending_result_waits_for_fresh_frame(self):
        broadcaster = Broadcaster()
        subscriber = await broadcaster.subscribe("camera")
        broadcaster.broadcast(self.event("camera", 1), deadline=time.monotonic() - 1)
        pending = asyncio.create_task(subscriber.get())
        await asyncio.sleep(0)
        self.assertFalse(pending.done())
        broadcaster.broadcast(self.event("camera", 2), deadline=time.monotonic() + 1)
        self.assertEqual(await asyncio.wait_for(pending, 1), self.event("camera", 2))

    async def test_camera_churn_keeps_queue_bounded_and_unsubscribe_stops_delivery(self):
        broadcaster = Broadcaster()
        with patch.dict(os.environ, {"MAX_CAMERAS": "2"}):
            subscriber = await broadcaster.subscribe()
        for seq in range(20):
            broadcaster.broadcast(self.event(str(seq), seq))
        self.assertEqual(subscriber.qsize(), 2)
        await broadcaster.unsubscribe(subscriber)
        broadcaster.broadcast(self.event("new", 21))
        self.assertEqual((await subscriber.get())["camera_uuid"], "18")
        self.assertEqual((await subscriber.get())["camera_uuid"], "19")


if __name__ == "__main__":
    unittest.main()
