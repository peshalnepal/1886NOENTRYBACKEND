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
    def __init__(self, loop, max_q_per_worker=1, ready_cb=None):
        self._loop = loop
        self._workers = [object()]

    def has_capacity(self):
        return True

    async def wait_ready(self):
        return 8

    def stop(self):
        return None

    def join(self, timeout=2.0):
        return None


class TensortPipelineAutosizingTests(unittest.TestCase):
    def test_default_inference_deadline_is_three_seconds(self):
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
