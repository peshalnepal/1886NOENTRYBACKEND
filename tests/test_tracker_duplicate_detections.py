"""
Duplicate detections must not become duplicate tracks.

A box leaked past the edge's NMS lands on an object that already has a track.
Association has no reason to reject it (it overlaps the track well), so it
spawns a SECOND track and the object is drawn with two ids at once — including
on a vehicle that never moved, which no amount of association-gate tuning can
explain or fix. The duplicate has to be collapsed on input.

The opposite error is worse: merging two genuinely distinct vehicles hides a
real object. These tests pin both directions.
"""

import unittest

import numpy as np

from application.services.tracker import ByteTrackLite, _dedupe_detections


def _det(x, y, w=80.0, h=80.0, *, conf=0.8, cls_name="car"):
    return {
        "bbox": np.array([x, y, x + w, y + h], dtype="float32"),
        "cls_name": cls_name,
        "conf": conf,
    }


def _ids_over(frames, interval=4.0, **kw):
    tracker = ByteTrackLite(**kw)
    seen = set()
    for i, dets in enumerate(frames):
        out = tracker.update(dets, ts_s=i * interval)
        seen |= {t["track_id"] for t in out["tracks"]}
    return seen


class DuplicateDetectionCollapseTests(unittest.TestCase):
    def test_stationary_car_with_duplicate_box_gets_one_id(self):
        """The reported symptom: a car that never moved, drawn with two ids."""
        frames = [[_det(500, 300), _det(510, 305)] for _ in range(6)]
        self.assertEqual(len(_ids_over(frames)), 1)

    def test_duplicate_with_size_wobble_is_collapsed(self):
        """A leaked duplicate is rarely pixel-identical; it breathes a little."""
        frames = [
            [_det(500, 300, conf=0.9), _det(508, 304, 95.0, 92.0, conf=0.5)]
            for _ in range(6)
        ]
        self.assertEqual(len(_ids_over(frames)), 1)

    def test_three_way_duplicate_collapses_to_one(self):
        frames = [
            [
                _det(500, 300, conf=0.9),
                _det(506, 303, conf=0.7),
                _det(512, 306, conf=0.5),
            ]
            for _ in range(6)
        ]
        self.assertEqual(len(_ids_over(frames)), 1)

    def test_highest_confidence_box_survives(self):
        kept = _dedupe_detections(
            [_det(500, 300, conf=0.4), _det(506, 303, conf=0.9)]
        )
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(float(kept[0]["conf"]), 0.9)

    def test_dedupe_preserves_input_order(self):
        dets = [
            _det(100, 100, conf=0.5),
            _det(900, 900, conf=0.9),
            _det(105, 103, conf=0.4),
        ]
        kept = _dedupe_detections(dets)
        xs = [float(d["bbox"][0]) for d in kept]
        self.assertEqual(xs, sorted(xs), "dedupe must not reorder detections")

    def test_dedupe_can_be_disabled(self):
        frames = [[_det(500, 300), _det(510, 305)] for _ in range(6)]
        self.assertEqual(len(_ids_over(frames, dedupe_overlap=0.0)), 2)


class DistinctObjectsAreNotMergedTests(unittest.TestCase):
    def test_two_cars_parked_side_by_side_keep_two_ids(self):
        frames = [[_det(500, 300), _det(580, 300)] for _ in range(6)]
        self.assertEqual(len(_ids_over(frames)), 2)

    def test_partially_occluding_cars_keep_two_ids(self):
        frames = [[_det(500, 300), _det(540, 300)] for _ in range(6)]
        self.assertEqual(len(_ids_over(frames)), 2)

    def test_car_inside_a_truck_box_is_not_collapsed(self):
        """
        Fully contained, so overlap-over-smaller-area is 1.0 — only the size
        ratio distinguishes this from a duplicate.
        """
        frames = [
            [
                _det(500, 300, 160.0, 120.0, cls_name="truck"),
                _det(520, 320, 80.0, 80.0, cls_name="car"),
            ]
            for _ in range(6)
        ]
        self.assertEqual(len(_ids_over(frames)), 2)

    def test_different_classes_are_never_merged(self):
        """A person standing against a car overlaps heavily but is not it."""
        frames = [
            [
                _det(500, 300, cls_name="car"),
                _det(505, 305, 40.0, 90.0, cls_name="person"),
            ]
            for _ in range(6)
        ]
        self.assertEqual(len(_ids_over(frames)), 2)

    def test_two_cars_crossing_do_not_merge_permanently(self):
        """
        Cars pass each other: boxes converge, overlap, then separate. Both must
        still be present once they part, even if they briefly looked like one.
        """
        tracker = ByteTrackLite()
        left, right = 100.0, 900.0
        final = []
        for i in range(10):
            out = tracker.update(
                [_det(left, 300), _det(right, 300)], ts_s=i * 1.0
            )
            final = out["tracks"]
            left += 90.0
            right -= 90.0
        self.assertEqual(len(final), 2, "a car was lost after the crossing")


if __name__ == "__main__":
    unittest.main()
