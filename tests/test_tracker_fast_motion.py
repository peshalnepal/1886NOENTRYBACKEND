import unittest

import numpy as np

from application.services.tracker import ByteTrackLite


def _det(x1, y1, x2, y2, *, conf=0.9, cls_name="car"):
    return {
        "bbox": np.array([x1, y1, x2, y2], dtype="float32"),
        "cls_name": cls_name,
        "conf": conf,
    }


class ByteTrackLiteFastMotionTests(unittest.TestCase):
    def test_fast_jump_does_not_return_old_missed_track(self):
        tracker = ByteTrackLite(min_hits=2, max_misses=0)

        tracker.update([_det(0, 0, 100, 60)], ts_s=0.0)
        out2 = tracker.update([_det(8, 0, 108, 60)], ts_s=3.0)
        self.assertEqual(len(out2["tracks"]), 1)
        old_id = out2["tracks"][0]["track_id"]
        self.assertTrue(out2["tracks"][0]["confirmed"])

        out3 = tracker.update([_det(320, 0, 420, 60)], ts_s=6.0)

        self.assertEqual(len(out3["tracks"]), 1)
        self.assertNotEqual(out3["tracks"][0]["track_id"], old_id)
        self.assertEqual(out3["tracks"][0]["misses"], 0)

    def test_coasting_tracks_are_not_emitted_even_when_kept_internally(self):
        tracker = ByteTrackLite(min_hits=2, max_misses=2)

        tracker.update([_det(0, 0, 100, 60)], ts_s=0.0)
        out2 = tracker.update([_det(8, 0, 108, 60)], ts_s=1.0)
        old_id = out2["tracks"][0]["track_id"]

        out3 = tracker.update([_det(320, 0, 420, 60)], ts_s=2.0)

        emitted_ids = {t["track_id"] for t in out3["tracks"]}
        internal_ids = {t.track_id for t in tracker._tracks}
        self.assertNotIn(old_id, emitted_ids)
        self.assertIn(old_id, internal_ids)

    def test_emit_coasting_tracks_opt_in_keeps_old_behavior(self):
        tracker = ByteTrackLite(min_hits=2, max_misses=2, emit_coasting_tracks=True)

        tracker.update([_det(0, 0, 100, 60)], ts_s=0.0)
        out2 = tracker.update([_det(8, 0, 108, 60)], ts_s=1.0)
        old_id = out2["tracks"][0]["track_id"]

        out3 = tracker.update([_det(320, 0, 420, 60)], ts_s=2.0)

        old = [t for t in out3["tracks"] if t["track_id"] == old_id]
        self.assertEqual(len(old), 1)
        self.assertEqual(old[0]["misses"], 1)

    def test_same_timestamp_missed_track_is_not_emitted(self):
        tracker = ByteTrackLite(min_hits=1, max_misses=2)

        out1 = tracker.update([_det(0, 0, 100, 60)], ts_s=1.0)
        old_id = out1["tracks"][0]["track_id"]
        out2 = tracker.update([], ts_s=1.0)

        self.assertEqual(out2["tracks"], [])
        self.assertIn(old_id, {t.track_id for t in tracker._tracks})

    def test_velocity_update_uses_observed_camera_interval(self):
        tracker = ByteTrackLite(min_hits=2)

        tracker.update([_det(0, 0, 100, 60)], ts_s=0.0)
        tracker.update([_det(30, 0, 130, 60)], ts_s=3.0)

        track = tracker._tracks[0]
        # alpha=0.4, so the first velocity estimate contributes 60%.
        # The center moved 30px over the observed 3s camera interval.
        self.assertAlmostEqual(float(track.vel[0]), 6.0, places=3)


if __name__ == "__main__":
    unittest.main()
