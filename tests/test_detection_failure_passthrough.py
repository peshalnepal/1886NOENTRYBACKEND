import unittest
import uuid

from domain.model_pipeline import ModelPipeline, ObjDetectResponse
from routes.camera_routes import _resp_to_detection_out


class DetectionFailurePassthroughTests(unittest.TestCase):
    def test_detection_out_preserves_failure_metadata(self):
        resp = ObjDetectResponse(
            camera_uuid=str(uuid.uuid4()),
            frame_ts_ms=1234567890,
            frame_seq=7,
            event_type="InferenceFailedEvent",
            reason="Inference timed out",
            detections=(),
        )

        out = _resp_to_detection_out(resp, normalize=False)

        self.assertEqual(out.event_type, "InferenceFailedEvent")
        self.assertEqual(out.reason, "Inference timed out")
        self.assertEqual(out.detections, [])


if __name__ == "__main__":
    unittest.main()
