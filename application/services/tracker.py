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

def _xyxy_to_cxcyah(b: BBox) -> np.ndarray:
    w = b[2] - b[0]                                # box width  = x2 - x1
    h = b[3] - b[1]                                # box height = y2 - y1
    cx = b[0] + 0.5 * w                            # center x   = left edge + half width
    cy = b[1] + 0.5 * h                            # center y   = top edge + half height
    a = w / max(h, 1e-6)                           # aspect ratio = w/h (guard against div-by-zero)
    return np.array([cx, cy, a, h], dtype=np.float32)

def _cxcyah_to_xyxy(s: np.ndarray) -> BBox:
    cx, cy, a, h = s                               # unpack state
    w = a * h                                      # recover width from aspect * height
    return np.array([cx - 0.5*w, cy - 0.5*h,       # x1, y1 (top-left)
                     cx + 0.5*w, cy + 0.5*h],      # x2, y2 (bottom-right)
                    dtype=np.float32)

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

    # velocity is now 3D: [vcx, vcy, vh]. Aspect ratio is held constant — see predict().
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def predict(self, now_ts: float, max_dt: float = 0.5) -> BBox:
        # dt is clamped so a noisy velocity estimate can't fling the predicted
        # box across the frame. ``max_dt`` is supplied by the tracker and scales
        # with the observed frame interval: at 0.1-0.25 FPS we still extrapolate
        # ~one frame ahead instead of the old fixed 0.15s, which made prediction
        # a no-op for slow/irregular streams.
        dt = min(max(1e-3, now_ts - self.last_ts), max(1e-3, max_dt))
        state = _xyxy_to_cxcyah(self.bbox)                 # convert current bbox into (cx, cy, a, h)
        state[0] += self.vel[0] * dt                       # advance center x by vcx*dt
        state[1] += self.vel[1] * dt                       # advance center y by vcy*dt
        state[3] += self.vel[2] * dt                       # advance height by vh*dt (aspect 'a' is untouched)
        return _cxcyah_to_xyxy(state)                      # convert back to xyxy for IoU computation

    def update(self, det_bbox: BBox, det_score: float, now_ts: float, alpha: float = 0.4, max_dt: float = 0.5) -> None:
        dt = min(max(1e-3, now_ts - self.last_ts), max(1e-3, max_dt))           # time since last observation
        old = _xyxy_to_cxcyah(self.bbox)                   # previous state in cxcyah
        new = _xyxy_to_cxcyah(det_bbox)                    # new detection in cxcyah
        new_vel = np.array([
            (new[0] - old[0]) / dt,                        # vcx = Δcx / dt
            (new[1] - old[1]) / dt,                        # vcy = Δcy / dt
            (new[3] - old[3]) / dt,                        # vh  = Δh  / dt  (no va — aspect ignored)
        ], dtype=np.float32)
        self.vel = alpha * self.vel + (1.0 - alpha) * new_vel
        self.bbox = det_bbox
        self.score = float(det_score)
        self.last_ts = now_ts
        self.last_update_ts = now_ts
        self.hits += 1
        self.misses = 0

    def on_miss(self, decay: float = 0.9) -> None:
        self.misses += 1                                   # count this missed frame
        self.vel *= decay                                  # damp velocity — confidence in our extrapolation drops with each miss


class ByteTrackLite:
    """
    ByteTrack-style (no ReID):
      - split detections into high / low confidence
      - stage1: match tracks with high-conf dets
      - stage2: match remaining tracks with low-conf dets (helps with flicker)
      - create new tracks from unmatched high-conf dets
      - confirm after min_hits

    Low / irregular FPS robustness
    ------------------------------
    This tracker is designed to run on streams where a single Jetson cycles
    through ~20 cameras, so each camera may only deliver one frame every 4-10s
    (0.1-0.25 FPS), and the interval jitters. To cope:

      * Every time-based threshold self-tunes to the *observed* inter-frame
        interval (an EMA of dt), instead of assuming ~30 FPS. A fixed 0.1s
        staleness budget used to purge every track on the very next frame, so
        nothing ever survived long enough to confirm.
      * Velocity prediction is allowed to extrapolate ~one frame ahead rather
        than a fixed 0.15s.

    Association stays IoU-only (overlap-based). By default the public output is
    realtime-only: tracks that did not match a detection on the current frame
    are kept only inside the tracker, never emitted as drawable boxes. This
    prevents old bboxes from trailing fast-moving objects while still allowing
    callers to raise ``max_misses`` for short internal occlusion tolerance.
    """
    def __init__(
        self,
        high_th: float = 0.5,
        low_th: float = 0.3,
        min_iou_high: float = 0.40,
        min_iou_low: float = 0.20,
        min_hits: int = 2,
        max_misses: int = 4,
        max_stale_s: Optional[float] = None,
        stale_frames: float = 3.0,
        predict_horizon_frames: float = 1.5,
        match_same_class: bool = True,
        emit_coasting_tracks: bool = False,
    ) -> None:
        self.high_th = float(high_th)
        self.low_th = float(low_th)
        self.min_iou_high = float(min_iou_high)
        self.min_iou_low = float(min_iou_low)
        self.min_hits = int(min_hits)
        # max_misses = 0: drop a track as soon as it misses a frame. Raise it
        # only if you want short internal occlusion tolerance. Public output
        # still suppresses missed tracks unless emit_coasting_tracks=True.
        self.max_misses = int(max_misses)
        # max_stale_s: absolute-time staleness budget. Leave as None (default) to
        # derive it from the observed frame interval so the tracker self-tunes to
        # whatever (possibly very low / irregular) FPS each camera delivers. Set
        # an explicit value only to hard-override that behaviour.
        self.max_stale_s = float(max_stale_s) if max_stale_s is not None else None
        # How many frames a track may coast (when deriving max_stale_s) and how
        # far ahead predict() may extrapolate, both measured in frame intervals.
        self.stale_frames = float(stale_frames)
        self.predict_horizon_frames = float(predict_horizon_frames)
        self.match_same_class = bool(match_same_class)
        self.emit_coasting_tracks = bool(emit_coasting_tracks)

        self._next_id = 1
        self._tracks: List[Track] = []

        # Observed inter-frame interval (EMA, seconds) and the timestamp of the
        # previous update(), used to make all thresholds frame-rate adaptive.
        self._dt_ema: Optional[float] = None
        self._last_now_ts: Optional[float] = None

    def _frame_interval(self) -> float:
        """Best estimate of the current inter-frame interval (seconds)."""
        return float(self._dt_ema) if self._dt_ema is not None else 0.0

    def _purge(self, now_ts: float) -> None:
        interval = self._frame_interval()
        if self.max_stale_s is not None:
            stale_budget = self.max_stale_s
        else:
            # Allow ~stale_frames frames of coasting, with a 1.5x jitter margin
            # and a small floor so high-FPS streams still behave sensibly. This
            # is the key low-FPS fix: a fixed 0.1s budget purged every track on
            # the next frame (5-10s later) before it could ever confirm.
            stale_budget = max(0.5, interval * self.stale_frames * 1.5)
        kept: List[Track] = []
        for t in self._tracks:
            stale_s = now_ts - t.last_update_ts
            if stale_s > stale_budget:
                continue
            if t.misses > self.max_misses:
                continue
            kept.append(t)
        self._tracks = kept

    def update(self, detections: List[Dict[str, Any]], ts_s: Optional[float] = None) -> Dict[str, Any]:
        now_ts = float(ts_s if ts_s is not None else time.time())

        # Track the observed inter-frame interval so every time-based threshold
        # self-tunes to the actual delivery rate (which for a Jetson cycling
        # through ~20 cameras can be one frame every 4-10s). The pipeline only
        # forwards forward-progressing frames, so dt is normally positive; guard
        # anyway against the rare Jetson-restart timestamp regression.
        if self._last_now_ts is not None:
            raw_dt = now_ts - self._last_now_ts
            if raw_dt > 0:
                self._dt_ema = raw_dt if self._dt_ema is None else (0.7 * self._dt_ema + 0.3 * raw_dt)
        self._last_now_ts = now_ts

        self._purge(now_ts)
        hi = [d for d in detections if float(d["conf"]) >= self.high_th]                 # high-conf dets
        lo = [d for d in detections if self.low_th <= float(d["conf"]) < self.high_th]   # low-conf dets

        events: List[Tuple[str, int]] = []
        # Extrapolate/update up to ~predict_horizon_frames of motion (bounded by
        # the observed interval) rather than a fixed sub-second window.
        max_track_dt = max(0.15, self._frame_interval() * self.predict_horizon_frames)
        pred = [t.predict(now_ts, max_track_dt) for t in self._tracks]                   # predicted bbox per track

        # split tracks: confirmed get priority access to high-conf detections
        confirmed_idxs = [i for i, t in enumerate(self._tracks) if t.confirmed]
        tentative_idxs = [i for i, t in enumerate(self._tracks) if not t.confirmed]
        matched_track_idxs: set = set()                                                  # tracks that got a det this frame
        live_track_ids: set[int] = set()                                                  # track IDs updated/created this frame

        def _match(track_idxs, det_idxs, det_list, min_iou):
            """Hungarian match between a subset of tracks and a subset of detections."""
            if not track_idxs or not det_idxs:
                return []
            track_idxs = list(track_idxs)
            det_idxs = list(det_idxs)
            # 1e6 = "impossible pair" — much safer than 1.0 because Hungarian won't pick it as locally optimal
            cost = np.full((len(track_idxs), len(det_idxs)), 1e6, dtype=np.float32)
            for ii, ti in enumerate(track_idxs):
                t = self._tracks[ti]
                for jj, dj in enumerate(det_idxs):
                    d = det_list[dj]
                    if self.match_same_class and t.cls_name != d["cls_name"]:
                        continue                                                          # leave at 1e6 → effectively forbidden
                    cost[ii, jj] = 1.0 - _iou(pred[ti], d["bbox"])                       # standard 1 - IoU cost
            matches = []
            for ii, jj in _assign(cost):
                if cost[ii, jj] > 1.0 - min_iou:                                          # IoU too low → reject this pairing
                    continue
                matches.append((track_idxs[ii], det_idxs[jj]))                            # remap to original indices
            return matches

        unmatched_hi = set(range(len(hi)))                                                # all high-conf dets up for grabs

        # ─── Stage 1a: confirmed tracks ↔ high-conf detections (priority pass) ───
        for ti, dj in _match(confirmed_idxs, unmatched_hi, hi, self.min_iou_high):
            matched_track_idxs.add(ti)
            unmatched_hi.discard(dj)                                                      # this det is taken
            self._tracks[ti].update(hi[dj]["bbox"], float(hi[dj]["conf"]), now_ts, max_dt=max_track_dt)
            live_track_ids.add(self._tracks[ti].track_id)
            # no confirmation event needed — track was already confirmed

        # ─── Stage 1b: tentative tracks ↔ remaining high-conf detections ───
        for ti, dj in _match(tentative_idxs, unmatched_hi, hi, self.min_iou_high):
            matched_track_idxs.add(ti)
            unmatched_hi.discard(dj)
            t = self._tracks[ti]
            t.update(hi[dj]["bbox"], float(hi[dj]["conf"]), now_ts, max_dt=max_track_dt)
            live_track_ids.add(t.track_id)
            if t.hits >= self.min_hits:                                                   # graduate to confirmed if enough hits
                t.confirmed = True
                events.append(("track_confirmed", t.track_id))

        # ─── Stage 2: any still-unmatched track ↔ low-conf detections (flicker rescue) ───
        still_unmatched = [i for i in range(len(self._tracks)) if i not in matched_track_idxs]
        unmatched_lo = set(range(len(lo)))
        for ti, dj in _match(still_unmatched, unmatched_lo, lo, self.min_iou_low):
            matched_track_idxs.add(ti)
            unmatched_lo.discard(dj)
            t = self._tracks[ti]
            was_confirmed = t.confirmed
            t.update(lo[dj]["bbox"], float(lo[dj]["conf"]), now_ts, max_dt=max_track_dt)
            live_track_ids.add(t.track_id)
            if (not was_confirmed) and t.hits >= self.min_hits:
                t.confirmed = True
                events.append(("track_confirmed", t.track_id))

        # ─── Tracks that got nothing this frame: bump misses, decay velocity ───
        for ti in range(len(self._tracks)):
            if ti not in matched_track_idxs:
                self._tracks[ti].on_miss()                                                # misses++ AND vel *= 0.9

        # ─── New tracks from leftover high-conf dets (unchanged behavior) ───
        for dj in sorted(unmatched_hi):
            d = hi[dj]
            tid = self._next_id
            self._next_id += 1
            self._tracks.append(Track(
                track_id=tid,
                cls_name=str(d["cls_name"]),
                bbox=d["bbox"].copy(),
                score=float(d["conf"]),
                start_ts=now_ts,
                last_ts=now_ts,
                last_update_ts=now_ts,
            ))
            live_track_ids.add(tid)
            events.append(("track_created", tid))

        # Apply the miss/stale budget before returning. Without this, a track
        # that just missed is emitted for one extra frame and can be drawn
        # beside the newly-created track for a fast object that jumped ahead.
        self._purge(now_ts)

        # Output only tracks backed by a detection on this update. Coasting
        # tracks keep their previous bbox, so emitting them is the visible stale
        # box bug for fast objects that jumped to a new location.
        out_tracks = []
        for t in self._tracks:
            if not self.emit_coasting_tracks and t.track_id not in live_track_ids:
                continue
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
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None
    anchor: str = "bbox"
    allowed_classes: Optional[Tuple[str, ...]] = None
    enter_after_n: int = 2
    
    
def _bbox_anchor(b: List[float], mode: str) -> Tuple[float, float]:
    """Pick the point on the bbox that represents 'where the object is'."""
    x1, y1, x2, y2 = b
    if mode == "center":
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)  # geometric center
    return ((x1 + x2) * 0.5, max(y1, y2))

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


def _point_in_rect(x: float, y: float, rect: List[float]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _cross(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (
        min(a[0], c[0]) - 1e-9 <= b[0] <= max(a[0], c[0]) + 1e-9
        and min(a[1], c[1]) - 1e-9 <= b[1] <= max(a[1], c[1]) + 1e-9
    )


def _segments_intersect(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> bool:
    d1 = _cross(a1, a2, b1)
    d2 = _cross(a1, a2, b2)
    d3 = _cross(b1, b2, a1)
    d4 = _cross(b1, b2, a2)

    if (d1 > 0 > d2 or d1 < 0 < d2) and (d3 > 0 > d4 or d3 < 0 < d4):
        return True

    if abs(d1) < 1e-9 and _on_segment(a1, b1, a2):
        return True
    if abs(d2) < 1e-9 and _on_segment(a1, b2, a2):
        return True
    if abs(d3) < 1e-9 and _on_segment(b1, a1, b2):
        return True
    if abs(d4) < 1e-9 and _on_segment(b1, a2, b2):
        return True

    return False


def _bbox_intersects_poly(b: List[float], poly: List[Tuple[float, float]]) -> bool:
    if len(poly) < 3:
        return False

    x1, y1, x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    rect = [x1, y1, x2, y2]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if any(_point_in_poly(px, py, poly) for (px, py) in corners):
        return True

    if any(_point_in_rect(px, py, rect) for (px, py) in poly):
        return True

    rect_edges = [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]
    poly_edges = [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
    for rect_edge in rect_edges:
        for poly_edge in poly_edges:
            if _segments_intersect(rect_edge[0], rect_edge[1], poly_edge[0], poly_edge[1]):
                return True

    return False


def _contain_rect(outer_w: float, outer_h: float, inner_aspect: float) -> Tuple[float, float, float, float]:
    outer_aspect = float(outer_w) / max(float(outer_h), 1e-9)
    if outer_aspect > inner_aspect:
        active_h = float(outer_h)
        active_w = active_h * inner_aspect
        offset_x = (float(outer_w) - active_w) / 2.0
        offset_y = 0.0
    else:
        active_w = float(outer_w)
        active_h = active_w / max(inner_aspect, 1e-9)
        offset_x = 0.0
        offset_y = (float(outer_h) - active_h) / 2.0
    return offset_x, offset_y, active_w, active_h


def _roi_points_px(roi: ROI, frame_w: int, frame_h: int) -> List[Tuple[float, float]]:
    if not roi.normalized:
        return roi.points
    if roi.frame_w and roi.frame_h and (int(roi.frame_w) != int(frame_w) or int(roi.frame_h) != int(frame_h)):
        src_w = float(roi.frame_w)
        src_h = float(roi.frame_h)
        dst_w = float(frame_w)
        dst_h = float(frame_h)
        dst_aspect = dst_w / max(dst_h, 1e-9)
        offset_x, offset_y, active_w, active_h = _contain_rect(src_w, src_h, dst_aspect)
        out: List[Tuple[float, float]] = []
        for (px, py) in roi.points:
            src_x = float(px) * src_w
            src_y = float(py) * src_h
            dst_x_norm = (src_x - offset_x) / max(active_w, 1e-9)
            dst_y_norm = (src_y - offset_y) / max(active_h, 1e-9)
            out.append(
                (
                    max(0.0, min(1.0, dst_x_norm)) * dst_w,
                    max(0.0, min(1.0, dst_y_norm)) * dst_h,
                )
            )
        return out
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
        self._in_roi: Dict[Tuple[str, str, int], bool] = {}
        self._inside_streak: Dict[Tuple[str, str, int], int] = {}
        self._outside_streak: Dict[Tuple[str, str, int], int] = {}
        
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

        # set of track_ids alive this frame for this camera — used for cleanup at the end
        active_ids = {int(t["track_id"]) for t in tracks}

        for roi in rois:
            poly = _roi_points_px(roi, frame_w, frame_h)

            for t in tracks:
                if not t.get("confirmed", False):
                    continue
                if roi.allowed_classes is not None and t["cls_name"] not in roi.allowed_classes:
                    continue

                track_id = int(t["track_id"])
                key = (camera_uuid, roi.roi_id, track_id)

                if roi.anchor == "bbox":
                    raw_inside = _bbox_intersects_poly(t["bbox"], poly)   # any-overlap
                else:
                    ax, ay = _bbox_anchor(t["bbox"], roi.anchor)
                    raw_inside = _point_in_poly(ax, ay, poly)

                in_streak = self._inside_streak.get(key, 0)
                out_streak = self._outside_streak.get(key, 0)
                if raw_inside:
                    in_streak += 1
                    out_streak = 0
                else:
                    out_streak += 1
                    in_streak = 0
                self._inside_streak[key] = in_streak
                self._outside_streak[key] = out_streak
                prev_inside = self._in_roi.get(key, False)
                if prev_inside:
                    inside = out_streak < roi.enter_after_n
                else:
                    inside = in_streak >= roi.enter_after_n
                self._in_roi[key] = inside

                if inside and not prev_inside:
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
        self._in_roi = {k: v for k, v in self._in_roi.items()
                        if k[0] != camera_uuid or k[2] in active_ids}
        self._inside_streak = {k: v for k, v in self._inside_streak.items()
                               if k[0] != camera_uuid or k[2] in active_ids}
        self._outside_streak = {k: v for k, v in self._outside_streak.items()
                                if k[0] != camera_uuid or k[2] in active_ids}

        return alerts

    def cleanup_dead_tracks(self, camera_uuid: str, active_track_ids) -> None:
        """
        Manual cleanup hook. Only needed if you filter `tracks` upstream of
        process() (e.g. drop low-confidence tracks before passing them in),
        because process()'s auto-cleanup uses whatever it received as the
        source of truth for 'alive'.
        """
        cam = str(camera_uuid)
        active = {int(i) for i in active_track_ids}
        self._in_roi = {k: v for k, v in self._in_roi.items()
                        if k[0] != cam or k[2] in active}
        self._inside_streak = {k: v for k, v in self._inside_streak.items()
                               if k[0] != cam or k[2] in active}
        self._outside_streak = {k: v for k, v in self._outside_streak.items()
                                if k[0] != cam or k[2] in active}

    def reset_camera(self, camera_uuid: str) -> None:
        """Wipe all ROI state for a camera (use on stream restart / reconfig)."""
        cam = str(camera_uuid)
        self._in_roi = {k: v for k, v in self._in_roi.items() if k[0] != cam}
        self._inside_streak = {k: v for k, v in self._inside_streak.items() if k[0] != cam}
        self._outside_streak = {k: v for k, v in self._outside_streak.items() if k[0] != cam}



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
