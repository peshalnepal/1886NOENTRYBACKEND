from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

# Optional Hungarian (best). If missing, we fallback to greedy.
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


BBox = np.ndarray


def _iou(a: BBox, b: BBox) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (row, col) assignment pairs minimizing cost.
    Uses Hungarian if available, else greedy.
    """
    if cost.size == 0:
        return []

    if _HAS_SCIPY:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))

    # greedy fallback
    pairs = [(i, j, float(cost[i, j])) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    pairs.sort(key=lambda x: x[2])
    used_r, used_c = set(), set()
    out: List[Tuple[int, int]] = []
    for i, j, _ in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out.append((i, j))
    return out


@dataclass
class Track:
    track_id: int
    cls_name: str
    bbox: BBox
    score: float
    start_ts: float
    last_ts: float
    last_update_ts: float

    hits: int = 1
    misses: int = 0
    confirmed: bool = False

    # very light motion model (OC-SORT-ish flavor): constant velocity on bbox coords
    vel: BBox = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def predict(self, now_ts: float) -> BBox:
        dt = max(1e-3, now_ts - self.last_ts)
        return self.bbox + self.vel * dt

    def update(self, det_bbox: BBox, det_score: float, now_ts: float, alpha: float = 0.85) -> None:
        dt = max(1e-3, now_ts - self.last_ts)
        new_vel = (det_bbox - self.bbox) / dt
        self.vel = alpha * self.vel + (1.0 - alpha) * new_vel

        self.bbox = det_bbox
        self.score = float(det_score)
        self.last_ts = now_ts
        self.last_update_ts = now_ts

        self.hits += 1
        self.misses = 0


class ByteTrackLite:
    """
    ByteTrack-style (no ReID):
      - split detections into high / low confidence
      - stage1: match tracks with high-conf dets
      - stage2: match remaining tracks with low-conf dets (helps with flicker)
      - create new tracks from unmatched high-conf dets
      - confirm after min_hits
    """
    def __init__(
        self,
        high_th: float = 0.6,
        low_th: float = 0.1,
        min_iou_high: float = 0.30,
        min_iou_low: float = 0.20,
        min_hits: int = 3,
        max_misses: int = 15,
        max_stale_s: float = 2.5,
        match_same_class: bool = True,
    ) -> None:
        self.high_th = float(high_th)
        self.low_th = float(low_th)
        self.min_iou_high = float(min_iou_high)
        self.min_iou_low = float(min_iou_low)
        self.min_hits = int(min_hits)
        self.max_misses = int(max_misses)
        self.max_stale_s = float(max_stale_s)
        self.match_same_class = bool(match_same_class)

        self._next_id = 1
        self._tracks: List[Track] = []

    def _purge(self, now_ts: float) -> None:
        kept: List[Track] = []
        for t in self._tracks:
            stale_s = now_ts - t.last_update_ts
            if stale_s > self.max_stale_s:
                continue
            if t.misses > self.max_misses:
                continue
            kept.append(t)
        self._tracks = kept

    def update(self, detections: List[Dict[str, Any]], ts_s: Optional[float] = None) -> Dict[str, Any]:
        now_ts = float(ts_s if ts_s is not None else time.time())
        self._purge(now_ts)
        hi = [d for d in detections if float(d["conf"]) >= self.high_th]
        lo = [d for d in detections if self.low_th <= float(d["conf"]) < self.high_th]

        events: List[Tuple[str, int]] = [] 
        pred = [t.predict(now_ts) for t in self._tracks]
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_hi = set(range(len(hi)))

        if self._tracks and hi:
            cost = np.ones((len(self._tracks), len(hi)), dtype=np.float32)
            for i, t in enumerate(self._tracks):
                for j, d in enumerate(hi):
                    if self.match_same_class and t.cls_name != d["cls_name"]:
                        continue
                    cost[i, j] = 1.0 - _iou(pred[i], d["bbox"])

            for ti, dj in _assign(cost):
                iou_val = 1.0 - float(cost[ti, dj])
                if iou_val < self.min_iou_high:
                    continue
                unmatched_tracks.discard(ti)
                unmatched_hi.discard(dj)

                t = self._tracks[ti]
                was_confirmed = t.confirmed
                t.update(hi[dj]["bbox"], float(hi[dj]["conf"]), now_ts)

                if (not was_confirmed) and (t.hits >= self.min_hits):
                    t.confirmed = True
                    events.append(("track_confirmed", t.track_id))

        # Stage 2: match remaining tracks to LOW detections
        unmatched_lo = set(range(len(lo)))
        if unmatched_tracks and lo:
            remaining_tracks = sorted(unmatched_tracks)
            cost2 = np.ones((len(remaining_tracks), len(lo)), dtype=np.float32)
            for ii, ti in enumerate(remaining_tracks):
                t = self._tracks[ti]
                for j, d in enumerate(lo):
                    if self.match_same_class and t.cls_name != d["cls_name"]:
                        continue
                    cost2[ii, j] = 1.0 - _iou(pred[ti], d["bbox"])

            for r_i, dj in _assign(cost2):
                ti = remaining_tracks[r_i]
                iou_val = 1.0 - float(cost2[r_i, dj])
                if iou_val < self.min_iou_low:
                    continue
                if ti not in unmatched_tracks or dj not in unmatched_lo:
                    continue

                unmatched_tracks.discard(ti)
                unmatched_lo.discard(dj)

                t = self._tracks[ti]
                was_confirmed = t.confirmed
                t.update(lo[dj]["bbox"], float(lo[dj]["conf"]), now_ts)

                if (not was_confirmed) and (t.hits >= self.min_hits):
                    t.confirmed = True
                    events.append(("track_confirmed", t.track_id))

        # Any tracks still unmatched -> miss++
        for ti in list(unmatched_tracks):
            self._tracks[ti].misses += 1

        # Create new tracks from unmatched HIGH detections only (ByteTrack behavior)
        for dj in sorted(unmatched_hi):
            d = hi[dj]
            tid = self._next_id
            self._next_id += 1
            self._tracks.append(
                Track(
                    track_id=tid,
                    cls_name=str(d["cls_name"]),
                    bbox=d["bbox"].copy(),
                    score=float(d["conf"]),
                    start_ts=now_ts,
                    last_ts=now_ts,
                    last_update_ts=now_ts,
                )
            )
            events.append(("track_created", tid))

        # Output
        out_tracks = []
        for t in self._tracks:
            out_tracks.append({
                "track_id": t.track_id,
                "cls_name": t.cls_name,
                "conf": t.score,
                "bbox": t.bbox.tolist(),
                "confirmed": t.confirmed,
                "hits": t.hits,
                "misses": t.misses,
                "age_s": now_ts - t.start_ts,
                "last_seen_s": now_ts - t.last_update_ts,
            })

        return {"events": events, "tracks": out_tracks}


# --------------------------
# ROI + Alerting (notify on confirmed + ROI enter)
# --------------------------

@dataclass(frozen=True)
class ROI:
    roi_id: str
    points: List[Tuple[float, float]]
    normalized: bool = False


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x <  (y - y1)*(x2 - x1) / (y2 - y1 + 1e-9) + x1)
        if cond:
            inside = not inside
    return inside


def _bbox_center(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _roi_points_px(roi: ROI, frame_w: int, frame_h: int) -> List[Tuple[float, float]]:
    if not roi.normalized:
        return roi.points
    return [(px * frame_w, py * frame_h) for (px, py) in roi.points]


class ROIAlertEngine:
    """
    Emits alert when:
      - track is confirmed AND
      - track enters ROI (edge trigger)
    Also avoids repeat notifications per (camera_uuid, roi_id, track_id).
    """
    def __init__(self) -> None:
        self._notified: Dict[Tuple[str, str, int], bool] = {}
        self._in_roi: Dict[Tuple[str, str, int], bool] = {}

    def process(
        self,
        camera_uuid: str,
        frame_w: int,
        frame_h: int,
        tracks: List[Dict[str, Any]],
        rois: List[ROI],
        ts_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        ts_ms = int(ts_ms if ts_ms is not None else time.time() * 1000)

        for roi in rois:
            poly = _roi_points_px(roi, frame_w, frame_h)

            for t in tracks:
                if not t.get("confirmed", False):
                    continue

                track_id = int(t["track_id"])
                key = (camera_uuid, roi.roi_id, track_id)

                cx, cy = _bbox_center(t["bbox"])
                inside = _point_in_poly(cx, cy, poly)

                prev_inside = self._in_roi.get(key, False)
                self._in_roi[key] = inside

                # ROI enter = edge: False -> True
                if inside and not prev_inside:
                    if not self._notified.get(key, False):
                        self._notified[key] = True
                        alerts.append({
                            "type": "roi_enter",
                            "ts_ms": ts_ms,
                            "camera_uuid": camera_uuid,
                            "roi_id": roi.roi_id,
                            "track_id": track_id,
                            "cls_name": t["cls_name"],
                            "conf": float(t["conf"]),
                            "bbox": t["bbox"],
                        })

        return alerts


    def reset_camera(self, camera_uuid: str) -> None:
        cam = str(camera_uuid)
        self._notified = {k: v for k, v in self._notified.items() if k[0] != cam}
        self._in_roi = {k: v for k, v in self._in_roi.items() if k[0] != cam}



class MultiCameraByteTrack:
    def __init__(self, **tracker_kwargs: Any) -> None:
        self._trackers: Dict[str, ByteTrackLite] = {}
        self._kwargs = tracker_kwargs

    def update(self, camera_uuid: str, detections: List[Dict[str, Any]], ts_ms: int) -> Dict[str, Any]:
        cam = str(camera_uuid)
        ts_s = float(ts_ms) / 1000.0
        
        if cam not in self._trackers:
            self._trackers[cam] = ByteTrackLite(**self._kwargs)
            
        return self._trackers[cam].update(detections, ts_s=ts_s)

    def update_from_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        cam = str(ev["camera_uuid"])
        ts_s = float(ev.get("frame_ts_ms", int(time.time() * 1000))) / 1000.0

        dets: List[Dict[str, Any]] = []
        for d in ev.get("detections", []):
            b = d["box"]
            dets.append({
                "bbox": np.array([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=np.float32),
                "cls_name": str(d.get("cls_name", "unknown")),
                "conf": float(d.get("conf", 0.0)),
            })

        if cam not in self._trackers:
            self._trackers[cam] = ByteTrackLite(**self._kwargs)

        return self._trackers[cam].update(dets, ts_s=ts_s)
    
    def remove_camera(self, camera_uuid: str) -> None:
        self._trackers.pop(str(camera_uuid), None)

    def reset(self) -> None:
        self._trackers.clear()
