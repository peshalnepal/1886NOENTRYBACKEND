"""
Car/truck label confusion must not split one vehicle into two objects.

The detector's label for a four-wheeled vehicle is unstable: the same van reads
"car" on one frame and "truck" on the next depending on viewing angle. That
showed up two ways, both looking like "the same car tracked twice":

  * one frame carrying BOTH a car box and a truck box on one vehicle, which
    class-scoped NMS would not suppress;
  * the label alternating across frames, which broke same-class association so
    the track id ping-ponged 1,2,1,2.

Classes that are NOT plausibly confusable (person, motorcycle) must still never
merge — grouping them would hide a pedestrian standing beside a vehicle.
"""

import unittest

from application.services.tracker import (
    ByteTrackLite,
    MultiCameraByteTrack,
    ROI,
    ROIAlertEngine,
    _same_object_class,
    nms_payload_detections,
)


def _payload_det(x, y, w=80.0, h=80.0, *, conf=0.8, cls_name="car"):
    return {
        "box": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
        "cls_name": cls_name,
        "conf": conf,
    }


def _flip_labels(n_frames, period=3):
    """car on most frames, truck on every `period`-th — a realistic wobble."""
    return ["truck" if i % period == 0 else "car" for i in range(n_frames)]


class ConfusableClassGroupTests(unittest.TestCase):
    def test_vehicle_labels_are_interchangeable(self):
        self.assertTrue(_same_object_class("car", "truck"))
        self.assertTrue(_same_object_class("truck", "bus"))
        self.assertTrue(_same_object_class("car", "car"))

    def test_unrelated_classes_are_not_interchangeable(self):
        self.assertFalse(_same_object_class("car", "person"))
        self.assertFalse(_same_object_class("car", "motorcycle"))
        self.assertFalse(_same_object_class("person", "motorcycle"))

    def test_unknown_classes_only_match_themselves(self):
        self.assertTrue(_same_object_class("forklift", "forklift"))
        self.assertFalse(_same_object_class("forklift", "car"))


class CrossClassNmsTests(unittest.TestCase):
    def test_car_and_truck_box_on_one_vehicle_collapse(self):
        out = nms_payload_detections([
            _payload_det(500, 300, conf=0.6, cls_name="car"),
            _payload_det(505, 303, conf=0.55, cls_name="truck"),
        ])
        self.assertEqual(len(out), 1)

    def test_highest_confidence_label_survives_the_collapse(self):
        out = nms_payload_detections([
            _payload_det(500, 300, conf=0.4, cls_name="car"),
            _payload_det(505, 303, conf=0.9, cls_name="truck"),
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["cls_name"], "truck")

    def test_person_beside_a_car_is_never_suppressed(self):
        for other in ("person", "motorcycle"):
            with self.subTest(other=other):
                out = nms_payload_detections([
                    _payload_det(500, 300, conf=0.9, cls_name="car"),
                    _payload_det(502, 302, conf=0.8, cls_name=other),
                ])
                self.assertEqual(len(out), 2)


class ClassFlipAssociationTests(unittest.TestCase):
    def test_label_flip_keeps_one_track_id(self):
        tracker = MultiCameraByteTrack()
        seen = set()
        for i, label in enumerate(_flip_labels(9)):
            out = tracker.update_from_event({
                "camera_uuid": "cam1",
                "frame_ts_ms": i * 4000,
                "detections": [_payload_det(500, 300, cls_name=label)],
            })
            seen |= {t["track_id"] for t in out["tracks"]}
        self.assertEqual(len(seen), 1, f"label flip split the vehicle: {sorted(seen)}")

    def test_reported_class_follows_the_weight_of_evidence(self):
        """
        The first frame's guess must not be frozen for the life of the track:
        cls_name drives ROI class filtering and the notification text.
        """
        tracker = MultiCameraByteTrack()
        reported = None
        for i, label in enumerate(_flip_labels(9)):   # "car" on 6 of 9 frames
            out = tracker.update_from_event({
                "camera_uuid": "cam1",
                "frame_ts_ms": i * 4000,
                "detections": [_payload_det(500, 300, cls_name=label)],
            })
            if out["tracks"]:
                reported = out["tracks"][0]["cls_name"]
        self.assertEqual(reported, "car")

    def test_confident_majority_outweighs_a_hesitant_flip(self):
        tracker = ByteTrackLite()
        import numpy as np

        def det(cls_name, conf):
            return {
                "bbox": np.array([500, 300, 580, 380], dtype="float32"),
                "cls_name": cls_name,
                "conf": conf,
            }

        tracker.update([det("car", 0.9)], ts_s=0.0)
        tracker.update([det("car", 0.9)], ts_s=1.0)
        out = tracker.update([det("truck", 0.25)], ts_s=2.0)
        self.assertEqual(out["tracks"][0]["cls_name"], "car")

    def test_person_and_car_at_the_same_spot_stay_two_tracks(self):
        """Association grouping must not let a person absorb a car's track."""
        tracker = MultiCameraByteTrack()
        final = []
        for i in range(6):
            out = tracker.update_from_event({
                "camera_uuid": "cam1",
                "frame_ts_ms": i * 2000,
                "detections": [
                    _payload_det(500, 300, cls_name="car"),
                    _payload_det(505, 305, 40.0, 90.0, cls_name="person"),
                ],
            })
            final = out["tracks"]
        self.assertEqual(len(final), 2)
        self.assertEqual(
            {t["cls_name"] for t in final}, {"car", "person"}
        )


class ClassFlipRoiTests(unittest.TestCase):
    def test_roi_alert_does_not_repeat_as_the_label_wobbles(self):
        """
        The ROI engine keys on track_id. Before grouping, every flip minted a new
        id, so a single vehicle could re-alert repeatedly. With one stable id it
        must alert at most once.
        """
        tracker = MultiCameraByteTrack()
        engine = ROIAlertEngine()
        roi = ROI(
            roi_id="r1",
            points=[(400, 200), (800, 200), (800, 600), (400, 600)],
            allowed_classes=("car",),
        )

        total = 0
        for i, label in enumerate(_flip_labels(10)):
            out = tracker.update_from_event({
                "camera_uuid": "cam1",
                "frame_ts_ms": i * 4000,
                "detections": [_payload_det(500, 300, cls_name=label)],
            })
            total += len(
                engine.process("cam1", 1920, 1080, out["tracks"], [roi], ts_ms=i * 4000)
            )

        self.assertLessEqual(total, 1, f"vehicle alerted {total} times")


if __name__ == "__main__":
    unittest.main()
