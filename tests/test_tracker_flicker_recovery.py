"""
Tracker behaviour at the current edge frame rate (~10 FPS, dt = 0.1s).

These lock in the flicker fixes: a detection that briefly dims must keep its
track id (stage-2 low-confidence rescue), and a detection that truly goes away
must expire rather than coast forever.
"""

import unittest

import numpy as np

from application.services.tracker import ByteTrackLite


def _det(x1, y1, x2, y2, *, conf=0.9, cls_name="person"):
    return {
        "bbox": np.array([x1, y1, x2, y2], dtype="float32"),
        "cls_name": cls_name,
        "conf": conf,
    }


DT = 0.1   # 10 FPS


class TrackerFlickerRecoveryTests(unittest.TestCase):
    def test_low_confidence_dip_keeps_the_same_track_id(self):
        """
        The core flicker case: an object drops to 0.25 confidence for two
        frames. With the old low_th=0.3 the edge's 0.2-conf boxes fell outside
        the rescue band and the track was lost, so the box blinked and a new id
        appeared. It must now survive with its id intact.
        """
        tracker = ByteTrackLite()
        t = 0.0
        x = 0.0

        for _ in range(5):                       # establish + confirm
            tracker.update([_det(x, 0, x + 100, 60, conf=0.8)], ts_s=t)
            t += DT
            x += 4.0

        out = tracker.update([_det(x, 0, x + 100, 60, conf=0.8)], ts_s=t)
        self.assertEqual(len(out["tracks"]), 1)
        original_id = out["tracks"][0]["track_id"]
        self.assertTrue(out["tracks"][0]["confirmed"])

        for _ in range(2):                       # the dim frames
            t += DT
            x += 4.0
            out = tracker.update([_det(x, 0, x + 100, 60, conf=0.25)], ts_s=t)
            self.assertEqual(len(out["tracks"]), 1, "track vanished on a dim frame")
            self.assertEqual(out["tracks"][0]["track_id"], original_id)

        t += DT
        x += 4.0
        out = tracker.update([_det(x, 0, x + 100, 60, conf=0.8)], ts_s=t)
        self.assertEqual(out["tracks"][0]["track_id"], original_id,
                         "confidence recovery must not spawn a new id")

    def test_track_expires_when_object_really_leaves(self):
        """Empty frames must still age tracks out — coasting forever would
        leave ghost boxes on screen."""
        tracker = ByteTrackLite()
        t = 0.0
        x = 0.0

        for _ in range(6):
            tracker.update([_det(x, 0, x + 100, 60, conf=0.8)], ts_s=t)
            t += DT
            x += 4.0

        for _ in range(20):                      # object gone; empty frames
            out = tracker.update([], ts_s=t)
            t += DT

        self.assertEqual(len(out["tracks"]), 0, "track should have expired")

    def test_low_th_matches_edge_conf_filter(self):
        """
        Guard rail: the edge (Backend/tensort) filters detections at CONF=0.20.
        If low_th ever rises above that, the stage-2 rescue band goes empty and
        the flicker returns.
        """
        self.assertLessEqual(ByteTrackLite().low_th, 0.20 + 1e-9)


if __name__ == "__main__":
    unittest.main()
