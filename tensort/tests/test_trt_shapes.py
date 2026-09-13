"""Exercise TensorRT shape negotiation and error paths using a fake CUDA driver."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("tensorrt", MagicMock())
sys.modules.setdefault("pycuda", MagicMock())
sys.modules.setdefault("pycuda.driver", MagicMock())
import trt_infer


class FakeContext:
    def __init__(self):
        self.shape = None
        self.reject_shape = False
        self.execute_ok = True

    def set_input_shape(self, name, shape):
        if self.reject_shape:
            return False
        self.shape = shape
        return True

    def get_tensor_shape(self, name):
        if name == "images":
            return self.shape
        batch, _channels, height, width = self.shape
        anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
        return batch, 84, anchors

    def set_tensor_address(self, *args):
        return True

    def execute_async_v3(self, **kwargs):
        return self.execute_ok


class FakeEngine:
    num_io_tensors = 2

    def __init__(self):
        self.context = FakeContext()

    def create_execution_context(self):
        return self.context

    def get_tensor_name(self, index):
        return ("images", "output0")[index]

    def get_tensor_dtype(self, name):
        return np.float32

    def get_tensor_shape(self, name):
        return (-1, 3, -1, -1) if name == "images" else (-1, 84, -1)

    def get_tensor_mode(self, name):
        return "input" if name == "images" else "output"

    def get_tensor_profile_shape(self, *args):
        return (1, 3, 320, 320), (4, 3, 640, 640), (8, 3, 640, 640)


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.fake_engine = FakeEngine()
        self.fake_cuda = MagicMock()
        self.fake_cuda.pagelocked_empty.side_effect = np.zeros
        # Device pointers need only be integer-convertible here.
        self.fake_cuda.mem_alloc.return_value = MagicMock(__int__=lambda _: 1)
        self.fake_trt = MagicMock()
        self.fake_trt.TensorIOMode.INPUT = "input"
        self.fake_trt.nptype.side_effect = lambda dtype: dtype
        self.fake_trt.Runtime.return_value.deserialize_cuda_engine.return_value = self.fake_engine
        self.patches = [patch.object(trt_infer, "cuda", self.fake_cuda),
                        patch.object(trt_infer, "trt", self.fake_trt),
                        patch.object(trt_infer, "CudaContext", MagicMock())]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.file = tempfile.NamedTemporaryFile()
        self.addCleanup(self.file.close)

    def engine(self):
        return trt_infer.TRTEngine(self.file.name, input_chw=(3, 320, 320))

    def test_dynamic_spatial_and_anchor_axes_use_resolved_shape(self):
        engine = self.engine()
        self.assertEqual(engine.input_view.shape, (8, 3, 320, 320))
        self.assertEqual(engine._max_sizes[1], 8 * 84 * 2100)
        for size in (8, 1, 4, 2):
            output = engine.infer_prepared(size, copy_outputs=False)[0]
            self.assertEqual(output.shape, (size, 84, 2100))
            self.assertTrue(np.shares_memory(output, engine._host_bufs[1]))

    def test_convenience_output_owns_copy(self):
        engine = self.engine()
        output = engine.infer_prepared(1)[0]
        self.assertFalse(np.shares_memory(output, engine._host_bufs[1]))

    def test_failed_execution_is_not_returned_as_old_detections(self):
        engine = self.engine()
        self.fake_engine.context.execute_ok = False
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            engine.infer_prepared(1)

    def test_rejected_input_shape_does_not_execute(self):
        engine = self.engine()
        self.fake_engine.context.reject_shape = True
        with self.assertRaisesRegex(RuntimeError, "rejected input"):
            engine.infer_prepared(1)

    def test_deserialization_failure_has_actionable_message(self):
        self.fake_trt.Runtime.return_value.deserialize_cuda_engine.return_value = None
        with self.assertRaisesRegex(RuntimeError, "rebuild it on this Jetson"):
            self.engine()

    def test_data_dependent_output_is_rejected_before_allocation(self):
        self.fake_engine.context.get_tensor_shape = lambda _: (-1, 84, -1)
        with self.assertRaisesRegex(RuntimeError, "unresolved output"):
            self.engine()
        self.fake_cuda.mem_alloc.assert_not_called()

    def test_partial_batch_profile_is_required(self):
        self.fake_engine.get_tensor_profile_shape = lambda *args: ((4, 3, 320, 320), (4, 3, 640, 640), (8, 3, 640, 640))
        with self.assertRaisesRegex(RuntimeError, "accept batch 1"):
            self.engine()


if __name__ == "__main__":
    unittest.main(verbosity=2)
