"""
Cross-class duplicate suppression on the edge (no GPU needed).

One vehicle scored as BOTH "car" and "truck" leaves this service as two boxes.
Neither NMS path in trt_infer.py stops that on its own:

  * ``nms_xyxy`` is class-agnostic on geometry, but it runs on a single argmax
    score per box — it never sees the same object emitted under two labels.
  * The end-to-end (yolo26) branch never calls ``nms_xyxy`` at all: its NMS is
    baked into the engine at export time and is PER-CLASS by construction. This
    is the ACTIVE path for the yolo26* engines in models/.

Downstream that duplicate is two drawn boxes and two track ids on an object
that never moved. These tests pin the suppression, and the opposite direction —
two genuinely distinct vehicles must both survive.

Run:  python3 tests/test_cross_class_dedupe.py     (from Backend/tensort)
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU/runtime deps are absent off-Jetson; the functions under test are pure.
sys.modules.setdefault("cv2", MagicMock())
sys.modules.setdefault("tensorrt", MagicMock())
sys.modules.setdefault("pycuda", MagicMock())
sys.modules.setdefault("pycuda.driver", MagicMock())
sys.modules.setdefault("pycuda.autoinit", MagicMock())

from trt_infer import _same_object_class, suppress_cross_class_duplicates


def _det(x, y, w=80, h=80, *, cls_name="car", conf=0.8):
    return {
        "cls_name": cls_name,
        "conf": conf,
        "box": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
    }


def _names(dets):
    return sorted(d["cls_name"] for d in dets)


class ConfusableClassGroupTests(unittest.TestCase):
    def test_vehicle_labels_are_interchangeable(self):
        for a, b in (("car", "truck"), ("truck", "bus"), ("van", "car")):
            self.assertTrue(_same_object_class(a, b), "%s/%s" % (a, b))

    def test_unrelated_labels_are_not(self):
        for a, b in (("person", "car"), ("motorcycle", "truck"), ("person", "bus")):
            self.assertFalse(_same_object_class(a, b), "%s/%s" % (a, b))

    def test_identical_labels_always_match(self):
        self.assertTrue(_same_object_class("person", "person"))


class CrossClassSuppressionTests(unittest.TestCase):
    def test_car_and_truck_on_one_vehicle_collapse(self):
        """The headline case: the same truck scored under two labels."""
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, cls_name="car", conf=0.60),
                _det(504, 302, cls_name="truck", conf=0.55),
            ]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["cls_name"], "car")  # highest confidence wins

    def test_highest_confidence_survives_regardless_of_order(self):
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, cls_name="car", conf=0.40),
                _det(503, 301, cls_name="truck", conf=0.85),
            ]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["cls_name"], "truck")

    def test_offset_duplicate_is_suppressed(self):
        """A leaked duplicate is rarely pixel-identical; it is offset a little."""
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, cls_name="car", conf=0.9),
                _det(510, 305, cls_name="truck", conf=0.6),
            ]
        )
        self.assertEqual(len(out), 1)

    def test_three_way_cluster_collapses_to_one(self):
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, cls_name="car", conf=0.9),
                _det(505, 302, cls_name="truck", conf=0.7),
                _det(509, 304, cls_name="bus", conf=0.5),
            ]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["cls_name"], "car")


class PreservesDistinctObjectsTests(unittest.TestCase):
    def test_person_in_front_of_car_both_survive(self):
        """Not confusable: merging these would hide a pedestrian."""
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, cls_name="car", conf=0.9),
                _det(505, 305, 40, 70, cls_name="person", conf=0.8),
            ]
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(_names(out), ["car", "person"])

    def test_two_separate_vehicles_both_survive(self):
        out = suppress_cross_class_duplicates(
            [
                _det(100, 100, cls_name="car", conf=0.9),
                _det(900, 600, cls_name="truck", conf=0.8),
            ]
        )
        self.assertEqual(len(out), 2)

    def test_small_car_inside_a_trucks_box_survives(self):
        """Mostly-contained but very different in size — a real distinct object."""
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, 300, 240, cls_name="truck", conf=0.9),
                _det(540, 460, 70, 60, cls_name="car", conf=0.7),
            ]
        )
        self.assertEqual(len(out), 2)

    def test_occluding_vehicles_both_survive(self):
        """The case that forbids an aggressive threshold (IoU ~0.6 is real)."""
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, 100, 100, cls_name="car", conf=0.9),
                _det(530, 320, 100, 100, cls_name="car", conf=0.8),
            ]
        )
        self.assertEqual(len(out), 2)


class InputHandlingTests(unittest.TestCase):
    def test_empty_and_single_inputs_pass_through(self):
        self.assertEqual(suppress_cross_class_duplicates([]), [])
        one = [_det(1, 1)]
        self.assertEqual(suppress_cross_class_duplicates(one), one)

    def test_input_order_is_preserved(self):
        dets = [
            _det(100, 100, cls_name="person", conf=0.5),
            _det(900, 600, cls_name="car", conf=0.9),
            _det(1500, 200, cls_name="motorcycle", conf=0.7),
        ]
        self.assertEqual(_names(suppress_cross_class_duplicates(dets)), _names(dets))
        self.assertEqual(
            [d["cls_name"] for d in suppress_cross_class_duplicates(dets)],
            [d["cls_name"] for d in dets],
        )

    def test_surviving_detection_keeps_its_wire_shape(self):
        """The cloud reads box/box_norm/cls_name/conf — do not reshape."""
        d = _det(500, 300, cls_name="car", conf=0.9)
        d["box_norm"] = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
        out = suppress_cross_class_duplicates([d, _det(504, 302, cls_name="truck", conf=0.5)])
        self.assertEqual(len(out), 1)
        self.assertEqual(set(out[0].keys()), {"cls_name", "conf", "box", "box_norm"})
        self.assertEqual(set(out[0]["box"].keys()), {"x1", "y1", "x2", "y2"})

    def test_degenerate_zero_area_boxes_do_not_crash(self):
        out = suppress_cross_class_duplicates(
            [
                _det(500, 300, 0, 0, cls_name="car", conf=0.9),
                _det(500, 300, 0, 0, cls_name="truck", conf=0.5),
            ]
        )
        self.assertLessEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
