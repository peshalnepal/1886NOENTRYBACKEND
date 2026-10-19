"""Frame callbacks preserve per-camera metadata and release cancelled frames."""

import asyncio
import os
import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("cv2", MagicMock())
from pipeline import SimpleInferencePipeline


class FrameJobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pipeline = SimpleInferencePipeline()
        self.pipeline._snapshot_enabled = False

    async def asyncTearDown(self):
        self.pipeline._snapshot_executor.shutdown(wait=True)

    def event(self, camera, frame):
        return types.SimpleNamespace(camera_uuid=camera, channel_id=camera,
                                     ts_ms=int(time.time() * 1000), seq=1, frame=frame)

    async def test_callbacks_keep_camera_metadata_and_timestamps(self):
        loop = asyncio.get_running_loop()
        jobs = [self.pipeline._prepare_frame_job(self.event(camera, object()), loop)
                for camera in ("first", "second")]
        for _, meta, future in reversed(jobs):
            future.set_result({"type": "DetectionsProducedEvent",
                               "camera_uuid": meta["camera_uuid"], "detections": []})
        await asyncio.sleep(0)
        for camera in ("first", "second"):
            self.assertEqual(self.pipeline.peek_latest(camera)["camera_uuid"], camera)
            self.assertLess(self.pipeline.peek_stats()["cameras"][camera]["frame_age_ms"], 1000)
        self.assertEqual(self.pipeline._inflight, {})

    async def test_cancelled_job_releases_frame_without_caching_result(self):
        event = self.event("camera", object())
        _, meta, future = self.pipeline._prepare_frame_job(event, asyncio.get_running_loop())
        self.assertIsNone(event.frame)
        future.cancel()
        await asyncio.sleep(0)
        self.assertNotIn("_bgr_ref", meta)
        self.assertIsNone(self.pipeline.peek_latest("camera"))
        self.assertEqual(self.pipeline._inflight, {})

    async def test_worker_deadline_respects_frame_age_and_result_timeout(self):
        loop = asyncio.get_running_loop()
        for age, max_age, timeout, expected in (
            (0.75, 1.0, 4.0, 100.25),
            (1.2, 1.0, 4.0, 99.8),
            (0.75, 0.0, 4.0, 104.0),
            (0.0, 1.0, 0.2, 100.2),
            (0.75, 1.0, 0.0, 100.25),
            (0.0, 0.0, 0.0, float("inf")),
        ):
            with self.subTest(age=age, max_age=max_age, timeout=timeout):
                self.pipeline._frame_max_age_s = max_age
                self.pipeline._infer_result_timeout_s = timeout
                event = self.event("camera", object())
                event.ts_ms = int((1000.0 - age) * 1000)
                with patch("pipeline.time.time", return_value=1000.0), \
                        patch("pipeline.time.monotonic", return_value=100.0):
                    _, meta, future = self.pipeline._prepare_frame_job(event, loop)
                self.assertAlmostEqual(meta["_infer_deadline"], expected)
                future.cancel()
                await asyncio.sleep(0)

    async def test_missing_frame_is_counted_without_creating_job(self):
        job = self.pipeline._prepare_frame_job(self.event("camera", None), asyncio.get_running_loop())
        self.assertIsNone(job)
        self.assertEqual(self.pipeline.peek_stats()["infer_fail"], 1)
        self.assertEqual(self.pipeline._inflight, {})

    async def test_expired_result_does_not_replace_cache_or_emit_empty_frame(self):
        previous = {"camera_uuid": "camera", "detections": [{"class": "person"}]}
        self.pipeline._latest["camera"] = previous
        self.pipeline._emit_failed_events = True
        subscriber = await self.pipeline.broadcaster.subscribe("camera")
        for result in (
            {"type": "DetectionsProducedEvent", "camera_uuid": "camera", "detections": []},
            {"type": "InferenceFailedEvent", "camera_uuid": "camera", "reason": "Frame expired"},
        ):
            self.pipeline._handle_result(result, {
                "camera_uuid": "camera", "frame_ts_ms": int(time.time() * 1000),
                "frame_seq": 1, "_infer_deadline": time.monotonic() - 1,
            })
        self.assertIs(self.pipeline.peek_latest("camera"), previous)
        self.assertEqual(subscriber.qsize(), 0)
        self.assertEqual(self.pipeline.peek_stats()["infer_dropped"], 2)

    async def test_fresh_empty_result_still_clears_live_detections(self):
        subscriber = await self.pipeline.broadcaster.subscribe("camera")
        result = {"type": "DetectionsProducedEvent", "camera_uuid": "camera", "detections": []}
        self.pipeline._handle_result(result, {
            "camera_uuid": "camera", "frame_ts_ms": int(time.time() * 1000),
            "frame_seq": 1, "_infer_deadline": time.monotonic() + 1,
        })
        self.assertIs(await subscriber.get(), result)
        self.assertIs(self.pipeline.peek_latest("camera"), result)


if __name__ == "__main__":
    unittest.main()
