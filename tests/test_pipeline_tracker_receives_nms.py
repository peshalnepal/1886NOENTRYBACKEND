"""
The tracker must receive the SAME de-duplicated detections the overlay draws.

``_payload_to_resp`` runs ``nms_payload_detections`` at ingestion, but the
tracker was fed ``dict(payload)`` — a SHALLOW copy, whose "detections" key still
pointed at the raw edge list. So the NMS result reached only ``resp.detections``
(the drawn overlay) and the tracker fell back to its own weaker internal dedupe
(``_dedupe_detections``, which has no plain-IoU arm).

The two paths then disagreed on the same frame: the overlay could suppress a
duplicate that the tracker still promoted into a second track, so one drawn box
carried two live ids indefinitely — on an object that never moved.

These tests pin the contract at the seam, which the tracker-only and NMS-only
suites cannot see.
"""

import asyncio
import unittest
from typing import Any, Dict, List

from application.services.pipeline import ModelPipeline


class _Cfg:
    def __init__(self, camera_uuid: str) -> None:
        self.camera_uuid = camera_uuid
        self.site_uuid = None
        self.device_uuid = None
        self.device_url = None
        self.enabled = True
        self.detection_enabled = True
        self.notification_enabled = False


class _Ch:
    """Minimal VideoChannel stand-in: only what _process_detection_payload reads."""

    def __init__(self, camera_uuid: str) -> None:
        self.config = _Cfg(camera_uuid)

    def key(self) -> str:
        return str(self.config.camera_uuid)


class _RecordingTracker:
    """Captures exactly what the pipeline handed the tracker."""

    def __init__(self) -> None:
        self.seen: List[List[Dict[str, Any]]] = []

    def update_from_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        self.seen.append(list(ev.get("detections") or []))
        return {"tracks": [], "events": []}

    def remove_camera(self, camera_uuid: str) -> None:
        pass


def _box(x: int, y: int, w: int = 80, h: int = 80) -> Dict[str, int]:
    return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}


def _payload(detections: List[Dict[str, Any]], cam: str) -> Dict[str, Any]:
    return {
        "camera_uuid": cam,
        "frame_ts_ms": 1_000,
        "frame_seq": 1,
        "frame_w": 1920,
        "frame_h": 1080,
        "detections": detections,
    }


CAM = "11111111-2222-3333-4444-555555555555"


class _EventRecordingTracker(_RecordingTracker):
    """Captures the FULL event dict, not just its detections."""

    def __init__(self) -> None:
        super().__init__()
        self.events: List[Dict[str, Any]] = []

    def update_from_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        self.events.append(dict(ev))
        return super().update_from_event(ev)


class TrackerReceivesFrameTimestampTests(unittest.TestCase):
    """
    The tracker must get the frame's OWN timestamp for either wire shape.

    The Jetson may send detections nested ({"payload": {...}}). _payload_to_resp
    unwraps that into a local variable, so the caller's dict still has no
    frame_ts_ms. Building the tracker event from that outer dict made
    update_from_event fall back to time.time(); the skew from the Jetson clock
    inflated dt enough that the staleness budget purged every track before it
    could confirm — and only confirmed tracks are drawn, so the overlay went
    empty while detections were still arriving.
    """

    def setUp(self) -> None:
        import uuid as _uuid

        self.pipeline = ModelPipeline(pipeline_id=_uuid.uuid4())
        self.tracker = _EventRecordingTracker()
        self.pipeline._tracker = self.tracker
        self.ch = _Ch(CAM)

    def _dets(self):
        return [{"box": _box(500, 300), "cls_name": "car", "conf": 0.9}]

    def test_flat_payload_forwards_the_frame_timestamp(self):
        asyncio.run(
            self.pipeline._process_detection_payload(
                CAM, self.ch, _payload(self._dets(), CAM)
            )
        )
        self.assertEqual(self.tracker.events[0]["frame_ts_ms"], 1_000)

    def test_nested_payload_forwards_the_frame_timestamp(self):
        """The regression: the outer dict has no frame_ts_ms at all."""
        nested = {"payload": _payload(self._dets(), CAM)}
        asyncio.run(self.pipeline._process_detection_payload(CAM, self.ch, nested))
        ev = self.tracker.events[0]
        self.assertEqual(ev["frame_ts_ms"], 1_000)
        self.assertEqual(ev["frame_seq"], 1)

    def test_nested_payload_still_delivers_detections(self):
        """A nested payload must not starve the tracker of boxes."""
        nested = {"payload": _payload(self._dets(), CAM)}
        asyncio.run(self.pipeline._process_detection_payload(CAM, self.ch, nested))
        self.assertEqual(len(self.tracker.seen[0]), 1)

    def test_tracker_never_falls_back_to_wall_clock(self):
        """
        Pin the actual failure mode. time.time() is ~1.7e12 ms; the frame stamp
        here is 1000. A wall-clock fallback is therefore unmistakable.
        """
        for payload in (
            _payload(self._dets(), CAM),
            {"payload": _payload(self._dets(), CAM)},
        ):
            tracker = _EventRecordingTracker()
            self.pipeline._tracker = tracker
            self.pipeline._last_seen.clear()
            asyncio.run(
                self.pipeline._process_detection_payload(CAM, self.ch, payload)
            )
            self.assertLess(int(tracker.events[0]["frame_ts_ms"]), 1_000_000)


class TrackerReceivesNmsedDetectionsTests(unittest.TestCase):
    def setUp(self) -> None:
        import uuid as _uuid

        self.pipeline = ModelPipeline(pipeline_id=_uuid.uuid4())
        self.tracker = _RecordingTracker()
        self.pipeline._tracker = self.tracker
        self.ch = _Ch(CAM)

    def _run(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        asyncio.run(
            self.pipeline._process_detection_payload(
                CAM, self.ch, _payload(detections, CAM)
            )
        )
        self.assertEqual(len(self.tracker.seen), 1, "tracker was not invoked once")
        return self.tracker.seen[0]

    def test_offset_duplicate_is_suppressed_before_the_tracker(self):
        """The headline bug: the extra box must not reach association at all."""
        got = self._run(
            [
                {"box": _box(500, 300), "cls_name": "car", "conf": 0.9},
                {"box": _box(510, 305), "cls_name": "car", "conf": 0.6},
            ]
        )
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(float(got[0]["conf"]), 0.9)

    def test_cross_class_duplicate_is_suppressed_before_the_tracker(self):
        """One truck scored car+truck: the confusable-group arm must collapse it."""
        got = self._run(
            [
                {"box": _box(500, 300), "cls_name": "car", "conf": 0.6},
                {"box": _box(504, 302), "cls_name": "truck", "conf": 0.55},
            ]
        )
        self.assertEqual(len(got), 1)

    def test_tracker_and_overlay_see_the_same_detections(self):
        """
        The real invariant. Whatever NMS decided, BOTH consumers must agree —
        disagreement is what let one drawn box carry two track ids.
        """
        dets = [
            {"box": _box(500, 300), "cls_name": "car", "conf": 0.9},
            {"box": _box(508, 304), "cls_name": "car", "conf": 0.6},
            {"box": _box(1200, 700), "cls_name": "person", "conf": 0.8},
        ]
        resp = self.pipeline._payload_to_resp(_payload(dets, CAM), self.ch)
        got = self._run(dets)
        self.assertEqual(got, list(resp.detections))

    def test_distinct_objects_still_reach_the_tracker(self):
        """The opposite failure: never starve the tracker of a real object."""
        got = self._run(
            [
                {"box": _box(100, 100), "cls_name": "car", "conf": 0.9},
                {"box": _box(900, 600), "cls_name": "car", "conf": 0.8},
                {"box": _box(1400, 200), "cls_name": "person", "conf": 0.7},
            ]
        )
        self.assertEqual(len(got), 3)

    def test_detections_keep_the_wire_shape_the_tracker_parses(self):
        """
        update_from_event reads d["box"]["x1"...]. resp.detections must still be
        in that wire format, not the tracker's internal float32 "bbox".
        """
        got = self._run([{"box": _box(500, 300), "cls_name": "car", "conf": 0.9}])
        self.assertIn("box", got[0])
        self.assertEqual(
            set(got[0]["box"].keys()) & {"x1", "y1", "x2", "y2"},
            {"x1", "y1", "x2", "y2"},
        )
        self.assertIn("cls_name", got[0])
        self.assertIn("conf", got[0])


if __name__ == "__main__":
    unittest.main()
