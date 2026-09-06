"""
Tracker behaviour when one Jetson round-robins many cameras.

Per-camera delivery drops to ~0.1-0.5 FPS (dt of seconds, often irregular).
Association and confirmation both used to assume ~10 FPS, so a car got a fresh
track id on every frame, never reached min_hits, and never confirmed — and
since only confirmed tracks are drawn, no box was ever rendered.
"""

import unittest

import numpy as np

from application.services.tracker import ByteTrackLite


def _det(x1, y1, x2, y2, *, conf=0.8, cls_name="car"):
    return {
        "bbox": np.array([x1, y1, x2, y2], dtype="float32"),
        "cls_name": cls_name,
        "conf": conf,
    }


def _drive(tracker, *, interval, step, frames=10, size=80.0, x0=100.0):
    """Run a box moving `step` px every `interval` s; return the ids seen."""
    ids = set()
    x = x0
    for i in range(frames):
        out = tracker.update(
            [_det(x, 200.0, x + size, 200.0 + size)], ts_s=i * interval
        )
        ids |= {t["track_id"] for t in out["tracks"]}
        x += step
    return ids


class SlowFpsIdStabilityTests(unittest.TestCase):
    def test_steady_object_keeps_one_id_at_slow_frame_rates(self):
        """The headline regression: one car, one id, at seconds-per-frame."""
        for interval, step in ((1.0, 200.0), (2.0, 400.0), (4.0, 600.0)):
            with self.subTest(interval=interval):
                ids = _drive(ByteTrackLite(), interval=interval, step=step)
                self.assertEqual(
                    len(ids), 1,
                    f"expected a single id at {interval}s/frame, got {sorted(ids)}",
                )

    def test_slow_object_becomes_drawable(self):
        """
        Only confirmed tracks with misses==0 are rendered (see _live_tracks in
        application/services/pipeline.py). At 4s/frame, min_hits=3 alone would
        mean a 12s wait; confirm_max_s must get a box on screen well before it.
        """
        tracker = ByteTrackLite()
        x = 100.0
        drawable_at = None
        for i in range(6):
            t = i * 4.0
            out = tracker.update([_det(x, 200.0, x + 80.0, 280.0)], ts_s=t)
            if any(d["confirmed"] and d["misses"] == 0 for d in out["tracks"]):
                drawable_at = t
                break
            x += 400.0

        self.assertIsNotNone(drawable_at, "track never became drawable")
        self.assertLessEqual(drawable_at, 8.0)

    def test_association_gate_scales_with_elapsed_time(self):
        """
        The same displacement is a different question at different frame rates:
        320px between two 0.1s frames is a teleport (new id), but across a 4s
        gap it is ordinary travel (same id). A size-only gate cannot tell them
        apart and produced the per-frame id churn.
        """
        fast = ByteTrackLite(min_hits=2, max_misses=0)
        fast.update([_det(0, 0, 100, 60)], ts_s=0.0)
        out_fast_prev = fast.update([_det(8, 0, 108, 60)], ts_s=0.1)
        established = out_fast_prev["tracks"][0]["track_id"]
        out_fast = fast.update([_det(320, 0, 420, 60)], ts_s=0.2)
        self.assertEqual(len(out_fast["tracks"]), 1)
        self.assertNotEqual(
            out_fast["tracks"][0]["track_id"], established,
            "a 320px jump between 0.1s frames must not keep the id",
        )

        slow = ByteTrackLite(min_hits=2, max_misses=0)
        slow.update([_det(0, 0, 100, 60)], ts_s=0.0)
        out_slow_prev = slow.update([_det(8, 0, 108, 60)], ts_s=4.0)
        established_slow = out_slow_prev["tracks"][0]["track_id"]
        out_slow = slow.update([_det(320, 0, 420, 60)], ts_s=8.0)
        self.assertEqual(len(out_slow["tracks"]), 1)
        self.assertEqual(
            out_slow["tracks"][0]["track_id"], established_slow,
            "the same jump across a 4s gap is ordinary travel — keep the id",
        )

    def test_distance_gate_is_not_truncated_by_the_accept_threshold(self):
        """
        The distance cost is 1.0 + 0.5*(dist/gate), which stays under _ACCEPT
        across the whole gate. It used to be 1.0 + dist/gate against an _ACCEPT
        of 1.6, so everything past 0.6*gate was admitted by the gate and then
        silently discarded — the effective gate was 60% of the configured one.
        """
        tracker = ByteTrackLite()
        # 80px box, 4s gap -> scale = 2.5 + 1.5*4 = 8.5, gate = 680px.
        # 500px sits inside the gate but beyond the old 0.6*gate = 408px cut.
        ids = _drive(tracker, interval=4.0, step=500.0, frames=6)
        self.assertEqual(len(ids), 1, f"gate truncated, ids={sorted(ids)}")

    def test_unrelated_distant_objects_are_not_merged(self):
        """
        The widened gate must not link genuinely different objects. Two cars far
        apart, both present every frame, keep two distinct ids.
        """
        tracker = ByteTrackLite()
        left, right = 100.0, 1600.0
        seen = set()
        for i in range(8):
            out = tracker.update(
                [
                    _det(left, 200.0, left + 80.0, 280.0),
                    _det(right, 200.0, right + 80.0, 280.0),
                ],
                ts_s=i * 2.0,
            )
            seen |= {t["track_id"] for t in out["tracks"]}
            left += 60.0
            right += 60.0
        self.assertEqual(len(seen), 2, f"expected exactly 2 ids, got {sorted(seen)}")

    def test_gate_growth_is_capped(self):
        """
        After a very long gap the gate must stop growing, otherwise a stale
        track would swallow any detection anywhere in the frame.
        """
        tracker = ByteTrackLite()
        tracker.update([_det(0.0, 200.0, 80.0, 280.0)], ts_s=0.0)
        # 60s later, a detection on the far side of the frame is not the
        # same object: cap is 12 * 80px = 960px.
        out = tracker.update([_det(3000.0, 200.0, 3080.0, 280.0)], ts_s=60.0)
        self.assertEqual(len(out["tracks"]), 1)
        self.assertNotEqual(out["tracks"][0]["track_id"], 1)


class SlowFpsFlickerTests(unittest.TestCase):
    def test_dim_and_dropped_frames_keep_the_id_at_slow_rates(self):
        """
        Combined flicker case at 2s/frame: two low-confidence frames (stage-2
        rescue) and one frame with no detection at all (coasting) must not
        change the id.
        """
        tracker = ByteTrackLite()
        x = 100.0
        ids = []
        for i in range(12):
            if i == 8:
                dets = []                      # detector missed the object
            else:
                conf = 0.25 if i in (5, 6) else 0.8
                dets = [_det(x, 200.0, x + 80.0, 280.0, conf=conf)]
            out = tracker.update(dets, ts_s=i * 2.0)
            ids.append({t["track_id"] for t in out["tracks"]})
            x += 200.0

        non_empty = [s for s in ids if s]
        self.assertTrue(non_empty)
        self.assertEqual(
            set().union(*non_empty), {1},
            f"id changed across dim/dropped frames: {ids}",
        )

    def test_cold_start_links_the_very_first_rematch(self):
        """
        On frame 2 the interval EMA has no sample yet, so the distance fallback
        used to stay disabled for exactly the frame where a cold track (zero
        velocity, so the predicted box equals the old one) needs it most. That
        lost the first re-match and started the churn cycle.
        """
        tracker = ByteTrackLite()
        out1 = tracker.update([_det(100.0, 200.0, 180.0, 280.0)], ts_s=0.0)
        first_id = out1["tracks"][0]["track_id"]
        out2 = tracker.update([_det(500.0, 200.0, 580.0, 280.0)], ts_s=4.0)

        self.assertEqual(len(out2["tracks"]), 1)
        self.assertEqual(out2["tracks"][0]["track_id"], first_id)


if __name__ == "__main__":
    unittest.main()
