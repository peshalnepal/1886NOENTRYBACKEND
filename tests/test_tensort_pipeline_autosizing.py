import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tensort"))
import pipeline as tensort_pipeline  # noqa: E402


class _FakeVideoChannel(object):
    def __init__(self, cfg):
        self.cfg = cfg

    async def stop(self):
        return None

    async def stream(self, event_queue=None):
        if False:
            yield event_queue


class _FakeInferenceWorkerPool(object):
    def __init__(self, loop, num_workers, max_q_per_worker=1, ready_cb=None, late_result_cb=None):
        self._loop = loop
        self._workers = [object() for _ in range(max(1, int(num_workers)))]

    def ensure_size(self, num_workers):
        while len(self._workers) < int(num_workers):
            self._workers.append(object())
        return len(self._workers)

    def has_capacity(self):
        return True

    async def wait_ready(self):
        return 8

    def stop(self):
        return None

    def join(self, timeout=2.0):
        return None


class TensortPipelineAutosizingTests(unittest.TestCase):
    def test_legacy_auto_settings_use_one_ordered_worker(self):
        with patch.dict(os.environ, {"INFER_NUM_WORKERS": "0", "INFER_NUM_WORKERS_MAX": "4"}):
            pipe = tensort_pipeline.SimpleInferencePipeline()
        self.assertEqual(pipe._num_workers, 1)
        self.assertEqual(pipe._num_workers_max, 1)

    def test_infer_timeout_is_a_flat_leak_guard(self):
        # A timed-out frame is no longer discarded (a late result is still
        # delivered), so the timeout is a leak guard rather than a
        # memory-scaled latency knob.
        pipe = tensort_pipeline.SimpleInferencePipeline()
        self.assertAlmostEqual(pipe._infer_result_timeout_s, 3.0)

    def test_pipeline_keeps_one_worker_as_camera_count_grows(self):
        async def scenario():
            with patch.dict(
                os.environ,
                {
                    "INFER_NUM_WORKERS": "0",
                    "INFER_NUM_WORKERS_MAX": "4",
                },
                clear=False,
            ):
                with patch.object(tensort_pipeline, "InferenceWorkerPool", _FakeInferenceWorkerPool):
                    with patch.object(tensort_pipeline, "VideoChannel", _FakeVideoChannel):
                        pipe = tensort_pipeline.SimpleInferencePipeline()
                        try:
                            await pipe.start()
                            self.assertEqual(len(pipe._infer_pool._workers), 1)

                            await pipe.add_channel(types.SimpleNamespace(camera_uuid="cam-1", enabled=True))
                            self.assertEqual(len(pipe._infer_pool._workers), 1)

                            await pipe.add_channel(types.SimpleNamespace(camera_uuid="cam-2", enabled=True))
                            self.assertEqual(len(pipe._infer_pool._workers), 1)

                            await pipe.add_channel(types.SimpleNamespace(camera_uuid="cam-3", enabled=True))
                            self.assertEqual(len(pipe._infer_pool._workers), 1)
                        finally:
                            await pipe.shutdown()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
