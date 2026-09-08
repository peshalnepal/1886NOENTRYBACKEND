"""ROI edge-trigger behaviour under detection loss and track-id churn.

The failure these guard against is a repeat `roi_enter` for an object that never
left the zone: a missed detection, a box jittering off the polygon edge, or a
failed re-association all used to re-arm the edge trigger.
"""

import unittest

from application.services.tracker import ROI, ROIAlertEngine


FRAME_W, FRAME_H = 1920, 1080

# Centre square, well clear of the frame edges.
ZONE = ROI(
    roi_id="zone",
    points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
    normalized=True,
)

SLOW_FPS = 4.0    # one Jetson round-robining ~20 cameras
FAST_FPS = 0.1    # dedicated pipeline


def car(track_id, inside, cls_name="car"):
    x = 0.5 if inside else 0.02
    return [{
        "track_id": track_id,
        "cls_name": cls_name,
        "conf": 0.9,
        "confirmed": True,
        "bbox": [x * FRAME_W, 0.5 * FRAME_H, (x + 0.05) * FRAME_W, 0.6 * FRAME_H],
    }]


class ROIAlertEngineTests(unittest.TestCase):
    def _run(self, frames, interval, engine=None):
        """Feed (track_id, inside) frames; None track_id means no tracks."""
        engine = engine or ROIAlertEngine()
        ts_ms = 0
        alerts = []
        for track_id, inside in frames:
            ts_ms += int(interval * 1000)
            tracks = [] if track_id is None else car(track_id, inside)
            alerts.extend(
                engine.process(
                    "cam", FRAME_W, FRAME_H, tracks, [ZONE],
                    ts_ms=ts_ms, frame_interval_s=interval,
                )
            )
        return alerts

    def test_box_jitter_off_zone_edge_does_not_realert(self):
        frames = [(7, True), (7, True), (7, False), (7, True), (7, False), (7, True)]
        self.assertEqual(len(self._run(frames, SLOW_FPS)), 1)

    def test_track_id_churn_does_not_realert(self):
        frames = [(7, True), (7, True), (12, True), (12, True), (19, True)]
        self.assertEqual(len(self._run(frames, SLOW_FPS)), 1)

    def test_detection_gap_does_not_realert(self):
        frames = [(7, True), (None, None), (None, None), (7, True), (7, True)]
        self.assertEqual(len(self._run(frames, SLOW_FPS)), 1)

    def test_genuine_exit_and_return_alerts_twice(self):
        # Long enough outside to clear exit_grace_s; cooldown disabled so the
        # re-alert guard does not mask the edge trigger under test.
        engine = ROIAlertEngine(realert_cooldown_s=0.0)
        frames = [(7, True)] + [(7, False)] * 4 + [(7, True)]
        self.assertEqual(len(self._run(frames, SLOW_FPS, engine)), 2)

    def test_single_frame_passthrough_alerts_on_slow_camera(self):
        # At 0.25 FPS a vehicle crossing the zone is visible on ONE frame; that
        # is all the evidence that will ever arrive.
        self.assertEqual(len(self._run([(7, False), (7, True), (7, False)], SLOW_FPS)), 1)

    def test_single_frame_flicker_does_not_alert_on_fast_camera(self):
        # At 10 FPS the two-frame rule still filters detector noise.
        frames = [(7, False), (7, True), (7, False), (7, False)]
        self.assertEqual(len(self._run(frames, FAST_FPS)), 0)

    def test_sustained_entry_alerts_on_fast_camera(self):
        frames = [(7, False), (7, True), (7, True), (7, True)]
        self.assertEqual(len(self._run(frames, FAST_FPS)), 1)

    def test_distinct_objects_in_different_places_both_alert(self):
        engine = ROIAlertEngine()
        first = engine.process(
            "cam", FRAME_W, FRAME_H, car(7, True), [ZONE],
            ts_ms=1000, frame_interval_s=SLOW_FPS,
        )
        # Same class, opposite corner of the zone: no overlap with the first, so
        # the re-alert guard must not suppress it.
        elsewhere = [{
            "track_id": 99, "cls_name": "car", "conf": 0.9, "confirmed": True,
            "bbox": [0.70 * FRAME_W, 0.25 * FRAME_H, 0.78 * FRAME_W, 0.35 * FRAME_H],
        }]
        second = engine.process(
            "cam", FRAME_W, FRAME_H, elsewhere, [ZONE],
            ts_ms=5000, frame_interval_s=SLOW_FPS,
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    def test_unconfirmed_tracks_never_alert(self):
        engine = ROIAlertEngine()
        tracks = car(7, True)
        tracks[0]["confirmed"] = False
        alerts = engine.process(
            "cam", FRAME_W, FRAME_H, tracks, [ZONE],
            ts_ms=1000, frame_interval_s=SLOW_FPS,
        )
        self.assertEqual(alerts, [])

    def test_state_is_bounded_over_a_long_run(self):
        engine = ROIAlertEngine()
        ts_ms = 0
        for i in range(200):
            ts_ms += int(SLOW_FPS * 1000)
            engine.process(
                "cam", FRAME_W, FRAME_H, car(i, True), [ZONE],
                ts_ms=ts_ms, frame_interval_s=SLOW_FPS,
            )
        # Every id is unique and short-lived, so the TTL must reclaim them
        # rather than growing one entry per track forever.
        self.assertLess(len(engine._in_roi), 20)
        self.assertLess(len(engine._last_seen_s), 20)

    def test_reset_camera_clears_every_map(self):
        engine = ROIAlertEngine()
        engine.process(
            "cam", FRAME_W, FRAME_H, car(7, True), [ZONE],
            ts_ms=1000, frame_interval_s=SLOW_FPS,
        )
        engine.reset_camera("cam")
        self.assertEqual(engine._in_roi, {})
        self.assertEqual(engine._inside_streak, {})
        self.assertEqual(engine._last_seen_s, {})
        self.assertEqual(engine._last_inside_s, {})
        self.assertEqual(engine._recent_alerts, {})


class ROIProjectionTests(unittest.TestCase):
    def test_normalized_points_survive_every_frame_size(self):
        from application.services.tracker import _roi_points_px

        roi = ROI(
            roi_id="z",
            points=[(0.1, 0.1), (0.4, 0.1), (0.4, 0.9), (0.1, 0.9)],
            normalized=True,
            frame_w=1920,
            frame_h=1080,
        )
        # A mismatched authored/live aspect ratio used to deform the polygon.
        for frame_w, frame_h in [(1920, 1080), (1280, 720), (640, 640)]:
            points = _roi_points_px(roi, frame_w, frame_h)
            xs = [p[0] / frame_w for p in points]
            ys = [p[1] / frame_h for p in points]
            self.assertAlmostEqual(min(xs), 0.1, places=6)
            self.assertAlmostEqual(max(xs), 0.4, places=6)
            self.assertAlmostEqual(min(ys), 0.1, places=6)
            self.assertAlmostEqual(max(ys), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
