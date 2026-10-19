"""Exercise queue expiration on a real worker thread with a fake detector."""

import asyncio
import os
import queue
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_worker import InferenceWorker


class WorkerFrameAgeTests(unittest.IsolatedAsyncioTestCase):
    async def run_job(self, remaining_budgets):
        loop = asyncio.get_running_loop()
        worker = InferenceWorker.__new__(InferenceWorker)
        worker._loop = loop
        worker._q = queue.Queue(maxsize=1)
        worker._stop = threading.Event()
        worker._ready_cb = None
        worker._infer = MagicMock()
        worker._infer.infer_multitask_batch.side_effect = lambda frames, metas: [
            {"type": "DetectionsProducedEvent", "camera_uuid": m["camera_uuid"],
             "frame_seq": m["frame_seq"]} for m in metas]
        job = []
        for i, budget in enumerate(remaining_budgets):
            meta = {"camera_uuid": "camera-{}".format(i), "frame_seq": i, "frame_ts_ms": 123}
            if budget is not None:
                meta["_infer_deadline"] = time.monotonic() + budget
            job.append((object(), meta, loop.create_future()))
        worker._q.put_nowait(job)
        thread = threading.Thread(target=worker._process_jobs, daemon=True)
        thread.start()
        try:
            results = await asyncio.wait_for(asyncio.gather(*(item[2] for item in job)), 2)
        finally:
            worker.stop()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual([r["camera_uuid"] for r in results], [j[1]["camera_uuid"] for j in job])
        self.assertEqual([r["frame_seq"] for r in results], list(range(len(job))))
        return worker, job, results

    async def test_mixed_batch_preserves_fresh_rows_and_camera_identity(self):
        worker, job, results = await self.run_job([-10, 60, -10, 60])
        frames, metas = worker._infer.infer_multitask_batch.call_args.args
        self.assertEqual(frames, [job[1][0], job[3][0]])
        self.assertEqual([m["frame_seq"] for m in metas], [1, 3])
        self.assertEqual([r["type"] for r in results], [
            "InferenceFailedEvent", "DetectionsProducedEvent",
            "InferenceFailedEvent", "DetectionsProducedEvent"])
        self.assertEqual(results[0]["reason"], "Frame expired before inference")

    async def test_fully_expired_batch_resolves_without_inference(self):
        worker, _, results = await self.run_job([-10, -10])
        worker._infer.infer_multitask_batch.assert_not_called()
        self.assertTrue(all(r["type"] == "InferenceFailedEvent" for r in results))

    async def test_legacy_job_without_deadline_still_runs(self):
        worker, _, results = await self.run_job([None])
        worker._infer.infer_multitask_batch.assert_called_once()
        self.assertEqual(results[0]["type"], "DetectionsProducedEvent")

    async def test_result_expiring_during_inference_is_not_delivered_as_detection(self):
        worker = InferenceWorker.__new__(InferenceWorker)
        future = asyncio.get_running_loop().create_future()
        meta = {"camera_uuid": "camera", "_infer_deadline": time.monotonic() - 1}
        result = {"type": "DetectionsProducedEvent", "detections": [{"class": "person"}]}
        worker._deliver_result(future, result, meta)
        self.assertEqual(future.result()["type"], "InferenceFailedEvent")
        self.assertEqual(future.result()["reason"], "Frame expired after inference")
        expired_result = future.result()
        worker._deliver_result(future, result, meta)
        self.assertIs(future.result(), expired_result)


if __name__ == "__main__":
    unittest.main()
