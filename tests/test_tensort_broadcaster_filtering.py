import asyncio
import sys
import unittest


sys.path.insert(0, "/home/peshal/1886NOENTRY/Backend/tensort")
import pipeline as tensort_pipeline  # noqa: E402


class BroadcasterFilteringTests(unittest.IsolatedAsyncioTestCase):
    async def test_camera_specific_subscriber_receives_only_matching_messages(self):
        broadcaster = tensort_pipeline.Broadcaster()

        q_all = await broadcaster.subscribe()
        q_cam1 = await broadcaster.subscribe("cam-1")

        broadcaster.broadcast({"camera_uuid": "cam-2", "frame_seq": 1})
        broadcaster.broadcast({"camera_uuid": "cam-1", "frame_seq": 2})

        first_global = await asyncio.wait_for(q_all.get(), timeout=1.0)
        second_global = await asyncio.wait_for(q_all.get(), timeout=1.0)
        cam1_only = await asyncio.wait_for(q_cam1.get(), timeout=1.0)

        self.assertEqual(first_global["camera_uuid"], "cam-2")
        self.assertEqual(second_global["camera_uuid"], "cam-1")
        self.assertEqual(cam1_only["camera_uuid"], "cam-1")

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(q_cam1.get(), timeout=0.05)


if __name__ == "__main__":
    unittest.main()
