import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, "/home/peshal/1886NOENTRY/Backend/tensort")
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

    def stop(self):
        return None

    def join(self, timeout=2.0):
        return None


class TensortPipelineAutosizingTests(unittest.TestCase):
    def test_auto_worker_cap_defaults_are_nano_safe(self):
        self.assertEqual(tensort_pipeline._default_auto_worker_cap(4096), 1)
        self.assertEqual(tensort_pipeline._default_auto_worker_cap(8192), 2)
        self.assertEqual(tensort_pipeline._default_auto_worker_cap(16384), 3)
        self.assertEqual(tensort_pipeline._default_auto_worker_cap(32768), 4)

    def test_infer_timeout_is_a_flat_leak_guard(self):
        # A timed-out frame is no longer discarded (a late result is still
        # delivered), so the timeout is a leak guard rather than a
        # memory-scaled latency knob.
        pipe = tensort_pipeline.SimpleInferencePipeline()
        self.assertAlmostEqual(pipe._infer_result_timeout_s, 3.0)

    def test_pipeline_starts_small_and_grows_with_camera_count(self):
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

                            await pipe.add_channel(types.SimpleNamespace(camera_uuid="cam-1"))
                            self.assertEqual(len(pipe._infer_pool._workers), 1)

                            await pipe.add_channel(types.SimpleNamespace(camera_uuid="cam-2"))
                            self.assertEqual(len(pipe._infer_pool._workers), 2)

                            await pipe.add_channel(types.SimpleNamespace(camera_uuid="cam-3"))
                            self.assertEqual(len(pipe._infer_pool._workers), 3)
                        finally:
                            await pipe.shutdown()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
