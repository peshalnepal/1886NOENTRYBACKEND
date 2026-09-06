"""
Ingestion-side NMS over raw edge detections.

A box leaked past the edge's own NMS reaches the cloud as a second detection on
one object. It then shows up TWICE: once as a raw drawn box on the live overlay,
and once as a second track id. Suppressing it inside the tracker fixes only the
ids — the overlay is built from the raw payload — so the pass has to run at
ingestion, before the payload fans out.

Thresholds are deliberately conservative: two vehicles occluding each other are
geometrically indistinguishable from a loosely-offset duplicate, and hiding a
real vehicle is far worse than drawing one box too many.
"""

import unittest

from application.services.tracker import nms_payload_detections


def _det(x, y, w=80.0, h=80.0, *, conf=0.8, cls_name="car"):
    return {
        "box": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
        "cls_name": cls_name,
        "conf": conf,
    }


class NmsSuppressesDuplicatesTests(unittest.TestCase):
    def test_exact_duplicate_is_suppressed(self):
        out = nms_payload_detections([_det(500, 300, conf=0.9), _det(500, 300, conf=0.6)])
        self.assertEqual(len(out), 1)

    def test_offset_duplicate_is_suppressed(self):
        out = nms_payload_detections([_det(500, 300, conf=0.9), _det(510, 305, conf=0.6)])
        self.assertEqual(len(out), 1)

    def test_cluster_collapses_to_the_strongest_box(self):
        out = nms_payload_detections([
            _det(500, 300, conf=0.5),
            _det(506, 303, conf=0.95),
            _det(512, 306, conf=0.7),
        ])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["conf"]), 0.95)

    def test_duplicate_with_size_wobble_is_suppressed(self):
        out = nms_payload_detections([
            _det(500, 300, conf=0.9),
            _det(508, 304, 95.0, 92.0, conf=0.5),
        ])
        self.assertEqual(len(out), 1)

    def test_surviving_detection_keeps_its_payload_shape(self):
        """Downstream reads box/cls_name/conf off the dict — don't reshape it."""
        out = nms_payload_detections([_det(500, 300, conf=0.9), _det(505, 302, conf=0.4)])
        self.assertEqual(len(out), 1)
        kept = out[0]
        self.assertIn("box", kept)
        self.assertEqual(
            set(kept["box"]), {"x1", "y1", "x2", "y2"}
        )
        self.assertEqual(kept["cls_name"], "car")


class NmsPreservesDistinctObjectsTests(unittest.TestCase):
    def test_adjacent_vehicles_both_survive(self):
        out = nms_payload_detections([_det(500, 300), _det(580, 300)])
        self.assertEqual(len(out), 2)

    def test_occluding_vehicles_both_survive(self):
        """
        The case that forbids an aggressive threshold: a real car must not be
        suppressed just because it overlaps another. Two vehicles cannot occupy
        the same ground, so even in steep perspective their boxes top out around
        50px of an 80px width (IoU ~0.46) — that must survive.
        """
        for overlap_px in (30, 40, 50):
            with self.subTest(overlap_px=overlap_px):
                out = nms_payload_detections(
                    [_det(500, 300), _det(500 + 80 - overlap_px, 300)]
                )
                self.assertEqual(len(out), 2)

    def test_near_coincident_boxes_are_treated_as_duplicates(self):
        """
        The deliberate cutoff. At 60px overlap of an 80px box the two boxes are
        75% coincident — in a real scene that is a duplicate, not two cars, so
        classic NMS suppressing it is correct. Documented here so the boundary
        is a decision rather than an accident.
        """
        out = nms_payload_detections([_det(500, 300), _det(520, 300)])
        self.assertEqual(len(out), 1)

    def test_car_inside_truck_box_survives(self):
        out = nms_payload_detections([
            _det(500, 300, 160.0, 120.0, cls_name="truck"),
            _det(520, 320, 80.0, 80.0, cls_name="car"),
        ])
        self.assertEqual(len(out), 2)

    def test_different_classes_never_suppress_each_other(self):
        out = nms_payload_detections([
            _det(500, 300, cls_name="car"),
            _det(500, 300, cls_name="person"),
        ])
        self.assertEqual(len(out), 2)


class NmsInputHandlingTests(unittest.TestCase):
    def test_empty_and_single_inputs_pass_through(self):
        self.assertEqual(nms_payload_detections([]), [])
        self.assertEqual(len(nms_payload_detections([_det(1, 1)])), 1)

    def test_non_list_input_is_tolerated(self):
        self.assertEqual(nms_payload_detections(None), [])
        self.assertEqual(nms_payload_detections({"box": {}}), [])

    def test_malformed_entries_are_passed_through_not_dropped(self):
        """
        This is a de-duplication pass, not a validation pass — dropping entries
        here would silently lose detections the tracker could still sanitize.
        """
        dets = [{"cls_name": "car", "conf": 0.9}, _det(500, 300)]
        self.assertEqual(len(nms_payload_detections(dets)), 2)

    def test_inverted_box_coordinates_are_handled(self):
        flipped = {
            "box": {"x1": 580, "y1": 380, "x2": 500, "y2": 300},
            "cls_name": "car",
            "conf": 0.6,
        }
        out = nms_payload_detections([_det(500, 300, conf=0.9), flipped])
        self.assertEqual(len(out), 1, "inverted duplicate should still suppress")

    def test_input_order_is_preserved(self):
        dets = [_det(100, 100, conf=0.5), _det(900, 900, conf=0.9), _det(105, 103, conf=0.4)]
        out = nms_payload_detections(dets)
        xs = [d["box"]["x1"] for d in out]
        self.assertEqual(xs, sorted(xs))

    def test_original_list_is_not_mutated(self):
        dets = [_det(500, 300, conf=0.9), _det(505, 302, conf=0.4)]
        nms_payload_detections(dets)
        self.assertEqual(len(dets), 2)


if __name__ == "__main__":
    unittest.main()
