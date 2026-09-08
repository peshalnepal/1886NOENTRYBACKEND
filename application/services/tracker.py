from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


BBox = np.ndarray


# Labels the detector flips between on one object. Only add genuinely
# confusable classes — grouping "person" with a vehicle would merge a pedestrian
# standing beside a car into it.
CONFUSABLE_CLASS_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("car", "truck", "bus", "van"),
)

_CLASS_GROUP: Dict[str, int] = {
    name: gi
    for gi, group in enumerate(CONFUSABLE_CLASS_GROUPS)
    for name in group
}


def _same_object_class(a: str, b: str) -> bool:
    """True when two labels could plausibly describe the same object."""
    if a == b:
        return True
    ga, gb = _CLASS_GROUP.get(a), _CLASS_GROUP.get(b)
    return ga is not None and ga == gb


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
    """(row, col) pairs minimizing cost. Hungarian if scipy is present, else greedy."""
    if cost.size == 0:
        return []

    # Both solvers break on NaN/inf; 1e6 reads as "forbidden" instead.
    if not np.all(np.isfinite(cost)):
        cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

    if _HAS_SCIPY:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))

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

def _sanitize_detections(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop malformed detections and normalize the rest.

    Guarantees a float32 bbox with x1<=x2, y1<=y2, a clamped conf and a string
    cls_name, so one bad entry from the edge cannot crash the whole frame.
    """
    if not detections:
        return []
    clean: List[Dict[str, Any]] = []
    for d in detections:
        if not isinstance(d, dict) or "bbox" not in d:
            continue
        try:
            bbox = np.asarray(d["bbox"], dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if bbox.size != 4 or not np.all(np.isfinite(bbox)):
            continue
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        try:
            conf = float(d.get("conf", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if not np.isfinite(conf):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        clean.append({
            "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
            "cls_name": str(d.get("cls_name", "unknown")),
            "conf": conf,
        })
    return clean


def nms_payload_detections(
    detections: Any,
    iou_thr: float = 0.55,
    overlap_thr: float = 0.70,
    size_ratio_thr: float = 0.65,
) -> List[Dict[str, Any]]:
    """Greedy NMS over raw edge detections, in their wire format.

    Runs at ingestion, before the payload fans out to the overlay and the tracker
    separately, so a leaked duplicate is kept out of both. Highest confidence in
    a cluster survives; only same-class boxes suppress each other. Unparseable
    boxes pass through — validation belongs to ``_sanitize_detections``.
    """
    if not isinstance(detections, list) or len(detections) < 2:
        return list(detections) if isinstance(detections, list) else []

    parsed: List[Optional[np.ndarray]] = []
    confs: List[float] = []
    for d in detections:
        box = d.get("box") if isinstance(d, dict) else None
        arr: Optional[np.ndarray] = None
        if isinstance(box, dict):
            try:
                x1, y1 = float(box["x1"]), float(box["y1"])
                x2, y2 = float(box["x2"]), float(box["y2"])
            except (KeyError, TypeError, ValueError):
                arr = None
            else:
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                candidate = np.array([x1, y1, x2, y2], dtype=np.float32)
                if np.all(np.isfinite(candidate)):
                    arr = candidate
        parsed.append(arr)
        try:
            confs.append(float(d.get("conf", 0.0)) if isinstance(d, dict) else 0.0)
        except (TypeError, ValueError):
            confs.append(0.0)

    # Unparseable boxes are never suppressed and never suppress others.
    order = sorted(
        (i for i, a in enumerate(parsed) if a is not None),
        key=lambda i: -confs[i],
    )
    suppressed: set = set()
    for pos, i in enumerate(order):
        if i in suppressed:
            continue
        di = detections[i]
        cls_i = str(di.get("cls_name", "unknown"))
        for j in order[pos + 1:]:
            if j in suppressed:
                continue
            if not _same_object_class(
                str(detections[j].get("cls_name", "unknown")), cls_i
            ):
                continue
            box_i, box_j = parsed[i], parsed[j]
            if _iou(box_i, box_j) >= iou_thr:
                suppressed.add(j)
                continue
            overlap, size_ratio = _overlap_min(box_i, box_j)
            if overlap >= overlap_thr and size_ratio >= size_ratio_thr:
                suppressed.add(j)

    if not suppressed:
        return list(detections)
    return [d for i, d in enumerate(detections) if i not in suppressed]


def _overlap_min(a: BBox, b: BBox) -> Tuple[float, float]:
    """Return (intersection / smaller area, smaller area / larger area).

    Plain IoU cannot separate a loose duplicate (~0.49) from two occluding
    vehicles (~0.60); adding a size-similarity ratio does, since a duplicate is
    both mostly-contained AND about the same size.
    """
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    smaller = min(area_a, area_b)
    larger = max(area_a, area_b)
    if smaller <= 0.0 or larger <= 0.0:
        return 0.0, 0.0
    return inter / smaller, smaller / larger


def _dedupe_detections(
    detections: List[Dict[str, Any]],
    overlap_thr: float = 0.70,
    size_ratio_thr: float = 0.65,
) -> List[Dict[str, Any]]:
    """Collapse near-duplicate boxes on one object into a single detection.

    Second line of defence behind ``nms_payload_detections``, for callers that
    feed ByteTrackLite directly. A pair must be BOTH mostly-overlapping AND
    similar in size: overlap alone merges a car inside a truck's box, size alone
    merges two cars parked side by side. Thresholds stay conservative — two boxes
    on one car is a milder failure than dropping a real vehicle.
    """
    if len(detections) < 2:
        return detections

    order = sorted(range(len(detections)), key=lambda i: -float(detections[i]["conf"]))
    kept: List[int] = []
    for i in order:
        di = detections[i]
        for j in kept:
            dj = detections[j]
            if not _same_object_class(di["cls_name"], dj["cls_name"]):
                continue
            overlap, size_ratio = _overlap_min(di["bbox"], dj["bbox"])
            if overlap >= overlap_thr and size_ratio >= size_ratio_thr:
                break                                  # duplicate of a stronger box
        else:
            kept.append(i)
    if len(kept) == len(detections):
        return detections
    return [detections[i] for i in sorted(kept)]


def _xyxy_to_cxcyah(b: BBox) -> np.ndarray:
    w = b[2] - b[0]
    h = b[3] - b[1]
    cx = b[0] + 0.5 * w
    cy = b[1] + 0.5 * h
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)

def _cxcyah_to_xyxy(s: np.ndarray) -> BBox:
    cx, cy, a, h = s
    w = a * h
    return np.array([cx - 0.5*w, cy - 0.5*h,
                     cx + 0.5*w, cy + 0.5*h],
                    dtype=np.float32)

@dataclass
class Track:
    track_id: int
    cls_name: str
    bbox: BBox
    score: float
    start_ts: float
    last_ts: float

    hits: int = 1
    misses: int = 0
    confirmed: bool = False

    # Confidence-weighted votes per label; cls_name follows the running winner
    # rather than whichever label the first frame carried.
    cls_votes: Dict[str, float] = field(default_factory=dict)

    # [vcx, vcy, vh]. Aspect ratio is held constant — see predict().
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def predict(self, now_ts: float, max_dt: float = 0.5) -> BBox:
        # Clamped so a noisy velocity can't fling the box across the frame.
        dt = min(max(1e-3, now_ts - self.last_ts), max(1e-3, max_dt))
        state = _xyxy_to_cxcyah(self.bbox)
        state[0] += self.vel[0] * dt
        state[1] += self.vel[1] * dt
        # Floored: a shrinking object over a long interval would otherwise give
        # an inverted box and a garbage IoU.
        state[3] = max(1e-3, state[3] + self.vel[2] * dt)
        return _cxcyah_to_xyxy(state)

    def vote_class(self, cls_name: str, det_score: float) -> None:
        """Record one label observation and adopt the running winner.

        Only called for labels that passed the association class check, so a
        track never drifts to an unrelated class.
        """
        name = str(cls_name)
        if not self.cls_votes:
            self.cls_votes[self.cls_name] = max(1e-6, float(self.score))
        self.cls_votes[name] = self.cls_votes.get(name, 0.0) + max(1e-6, float(det_score))
        self.cls_name = max(self.cls_votes.items(), key=lambda kv: kv[1])[0]

    def update(self, det_bbox: BBox, det_score: float, now_ts: float, alpha: float = 0.4) -> None:
        # Over ACTUAL elapsed time, not predict()'s clamp: dividing by the
        # smaller clamped dt would inflate velocity after a coast.
        elapsed = max(1e-3, now_ts - self.last_ts)
        old = _xyxy_to_cxcyah(self.bbox)
        new = _xyxy_to_cxcyah(det_bbox)
        new_vel = np.array([
            (new[0] - old[0]) / elapsed,
            (new[1] - old[1]) / elapsed,
            (new[3] - old[3]) / elapsed,
        ], dtype=np.float32)
        self.vel = alpha * self.vel + (1.0 - alpha) * new_vel
        self.bbox = det_bbox
        self.score = float(det_score)
        self.last_ts = now_ts
        self.hits += 1
        self.misses = 0

    def on_miss(self, decay: float = 0.9) -> None:
        self.misses += 1
        self.vel *= decay


class ByteTrackLite:
    """ByteTrack-style association (no ReID), IoU-first.

    Delivered frame rate varies by two orders of magnitude — ~10 FPS on a
    dedicated pipeline, ~0.1-0.25 FPS when one Jetson round-robins 20 cameras —
    so every time-based threshold self-tunes to the observed interval.

    NOTE: low_th must stay at or above the edge's CONF filter (see
    Backend/tensort/.env.example), or the stage-2 rescue band is empty and a
    briefly-dimmer detection drops the track instead of re-linking it.
    """
    def __init__(
        self,
        high_th: float = 0.5,
        low_th: float = 0.20,          # keep >= the edge CONF filter
        min_iou_high: float = 0.40,
        min_iou_low: float = 0.20,
        min_hits: int = 3,
        confirm_max_s: float = 1.5,    # tentative-track age cap; still needs 2 hits
        max_misses: int = 8,           # main anti-ID-churn knob
        max_stale_s: Optional[float] = None,
        stale_frames: float = 8.0,
        predict_horizon_frames: float = 1.5,
        match_same_class: bool = True,
        emit_coasting_tracks: bool = False,
        assoc_center_dist: bool = True,
        assoc_dist_scale: float = 2.5,
        assoc_dist_min_interval: float = 0.25,
        assoc_dist_per_s: float = 1.5,      # gate widens with time to move
        assoc_dist_max_scale: float = 12.0, # ceiling, or it links unrelated objects
        dedupe_overlap: float = 0.70,       # 0 disables dedupe
        dedupe_size_ratio: float = 0.65,
    ) -> None:
        self.high_th = float(high_th)
        self.low_th = float(low_th)
        self.min_iou_high = float(min_iou_high)
        self.min_iou_low = float(min_iou_low)
        self.min_hits = int(min_hits)
        self.confirm_max_s = float(confirm_max_s)
        self.max_misses = int(max_misses)
        # None derives the staleness budget from the observed frame interval.
        self.max_stale_s = float(max_stale_s) if max_stale_s is not None else None
        self.stale_frames = float(stale_frames)
        self.predict_horizon_frames = float(predict_horizon_frames)
        self.match_same_class = bool(match_same_class)
        self.emit_coasting_tracks = bool(emit_coasting_tracks)
        # Rescues low FPS, where an object moves more than its own size between
        # frames and IoU alone spawns a new id every frame.
        self.assoc_center_dist = bool(assoc_center_dist)
        self.assoc_dist_scale = float(assoc_dist_scale)
        self.assoc_dist_min_interval = float(assoc_dist_min_interval)
        self.assoc_dist_per_s = float(assoc_dist_per_s)
        self.assoc_dist_max_scale = float(assoc_dist_max_scale)
        self.dedupe_overlap = float(dedupe_overlap)
        self.dedupe_size_ratio = float(dedupe_size_ratio)

        self._next_id = 1
        self._tracks: List[Track] = []

        self._dt_ema: Optional[float] = None
        self._last_now_ts: Optional[float] = None

    def _frame_interval(self) -> float:
        """Observed inter-frame interval in seconds, 0.0 until known."""
        return float(self._dt_ema) if self._dt_ema is not None else 0.0

    def _should_confirm(self, t: Track, now_ts: float) -> bool:
        """Whether a tentative track has earned confirmation.

        min_hits alone assumes a steady rate; at 0.25 FPS three hits is a 12s
        wait and only confirmed tracks are drawn. The age path still requires a
        second sighting, so one-frame noise never confirms.
        """
        if t.hits >= self.min_hits:
            return True
        if self.confirm_max_s <= 0.0:
            return False
        return t.hits >= 2 and (now_ts - t.start_ts) >= self.confirm_max_s

    def _purge(self, now_ts: float) -> None:
        interval = self._frame_interval()
        if self.max_stale_s is not None:
            stale_budget = self.max_stale_s
        else:
            # ~stale_frames of coasting, plus a jitter margin and a floor.
            stale_budget = max(0.5, interval * self.stale_frames * 1.5)
        kept: List[Track] = []
        for t in self._tracks:
            stale_s = now_ts - t.last_ts
            if stale_s > stale_budget:
                continue
            if t.misses > self.max_misses:
                continue
            kept.append(t)
        self._tracks = kept

    def update(self, detections: List[Dict[str, Any]], ts_s: Optional[float] = None) -> Dict[str, Any]:
        now_ts = float(ts_s if ts_s is not None else time.time())
        if not np.isfinite(now_ts):
            now_ts = time.time()

        # Dedupe before association, or the extra box becomes a second track.
        detections = _sanitize_detections(detections)
        if self.dedupe_overlap > 0.0:
            detections = _dedupe_detections(
                detections, self.dedupe_overlap, self.dedupe_size_ratio
            )

        # Guard against a Jetson-restart timestamp regression.
        if self._last_now_ts is not None:
            raw_dt = now_ts - self._last_now_ts
            if raw_dt > 0:
                self._dt_ema = raw_dt if self._dt_ema is None else (0.7 * self._dt_ema + 0.3 * raw_dt)
        self._last_now_ts = now_ts

        # Seed before association runs: otherwise the interval is still 0.0 on
        # the first re-match, exactly when a cold track (zero velocity, so no
        # useful prediction) needs the distance fallback most.
        if self._dt_ema is None and self._tracks:
            gap = now_ts - max(t.last_ts for t in self._tracks)
            if gap > 0:
                self._dt_ema = gap

        self._purge(now_ts)
        hi = [d for d in detections if float(d["conf"]) >= self.high_th]
        lo = [d for d in detections if self.low_th <= float(d["conf"]) < self.high_th]

        events: List[Tuple[str, int]] = []
        max_track_dt = max(0.15, self._frame_interval() * self.predict_horizon_frames)
        pred = [t.predict(now_ts, max_track_dt) for t in self._tracks]

        # Confirmed tracks get priority access to high-conf detections.
        confirmed_idxs = [i for i, t in enumerate(self._tracks) if t.confirmed]
        tentative_idxs = [i for i, t in enumerate(self._tracks) if not t.confirmed]
        matched_track_idxs: set = set()
        live_track_ids: set[int] = set()

        use_dist = self.assoc_center_dist and self._frame_interval() >= self.assoc_dist_min_interval
        # cost:  IoU [0, 1-min_iou) | distance [1, 1.5) | forbidden 1e6
        #
        # _ACCEPT must stay at the TOP of the distance band so the gate is the
        # only rejection test; lower it and pairings the gate admitted are
        # silently discarded, narrowing the effective gate.
        _ACCEPT = 1.5

        def _match(track_idxs, det_idxs, det_list, min_iou):
            """Hungarian match between a subset of tracks and detections."""
            if not track_idxs or not det_idxs:
                return []
            track_idxs = list(track_idxs)
            det_idxs = list(det_idxs)
            # 1e6, not 1.0: Hungarian could pick 1.0 as locally optimal.
            cost = np.full((len(track_idxs), len(det_idxs)), 1e6, dtype=np.float32)
            for ii, ti in enumerate(track_idxs):
                t = self._tracks[ti]
                pb = pred[ti]
                pcx = 0.5 * (float(pb[0]) + float(pb[2]))
                pcy = 0.5 * (float(pb[1]) + float(pb[3]))
                ph = float(pb[3]) - float(pb[1])
                # A track coasting through misses had longer to move, so it
                # earns a proportionally wider gate below.
                gap_s = max(0.0, now_ts - t.last_ts)
                for jj, dj in enumerate(det_idxs):
                    d = det_list[dj]
                    if self.match_same_class and not _same_object_class(
                        t.cls_name, d["cls_name"]
                    ):
                        continue                                                          # leave at 1e6 → forbidden
                    iou = _iou(pb, d["bbox"])
                    if iou >= min_iou:
                        cost[ii, jj] = 1.0 - iou
                        continue
                    if not use_dist:
                        continue                                                          # pure-IoU regime: reject
                    db = d["bbox"]
                    dcx = 0.5 * (float(db[0]) + float(db[2]))
                    dcy = 0.5 * (float(db[1]) + float(db[3]))
                    dw = float(db[2]) - float(db[0])
                    dh = float(db[3]) - float(db[1])
                    if ph <= 0.0 or dh <= 0.0:
                        continue
                    ratio = dh / ph
                    if ratio < 0.5 or ratio > 2.0:                                        # very different scales
                        continue
                    # Size alone is not enough: the same car needs a ~200px gate
                    # at 10 FPS and ~800px after a 4s round-robin gap.
                    size = 0.5 * (dw + dh)
                    scale = min(
                        self.assoc_dist_scale + self.assoc_dist_per_s * gap_s,
                        self.assoc_dist_max_scale,
                    )
                    gate = size * scale
                    if gate <= 0.0:
                        continue
                    dist = float(np.hypot(dcx - pcx, dcy - pcy))
                    if dist < gate:
                        cost[ii, jj] = 1.0 + 0.5 * (dist / gate)
            matches = []
            for ii, jj in _assign(cost):
                if cost[ii, jj] >= _ACCEPT:                                               # forbidden or out-of-gate
                    continue
                matches.append((track_idxs[ii], det_idxs[jj]))
            return matches

        unmatched_hi = set(range(len(hi)))

        # ─── Stage 1a: confirmed tracks ↔ high-conf detections (priority pass) ───
        for ti, dj in _match(confirmed_idxs, unmatched_hi, hi, self.min_iou_high):
            matched_track_idxs.add(ti)
            unmatched_hi.discard(dj)
            self._tracks[ti].vote_class(hi[dj]["cls_name"], float(hi[dj]["conf"]))
            self._tracks[ti].update(hi[dj]["bbox"], float(hi[dj]["conf"]), now_ts)
            live_track_ids.add(self._tracks[ti].track_id)

        # ─── Stage 1b: tentative tracks ↔ remaining high-conf detections ───
        for ti, dj in _match(tentative_idxs, unmatched_hi, hi, self.min_iou_high):
            matched_track_idxs.add(ti)
            unmatched_hi.discard(dj)
            t = self._tracks[ti]
            t.vote_class(hi[dj]["cls_name"], float(hi[dj]["conf"]))
            t.update(hi[dj]["bbox"], float(hi[dj]["conf"]), now_ts)
            live_track_ids.add(t.track_id)
            if self._should_confirm(t, now_ts):
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
            t.vote_class(lo[dj]["cls_name"], float(lo[dj]["conf"]))
            t.update(lo[dj]["bbox"], float(lo[dj]["conf"]), now_ts)
            live_track_ids.add(t.track_id)
            if (not was_confirmed) and self._should_confirm(t, now_ts):
                t.confirmed = True
                events.append(("track_confirmed", t.track_id))

        # ─── Unmatched tracks: bump misses, decay velocity ───
        for ti in range(len(self._tracks)):
            if ti not in matched_track_idxs:
                self._tracks[ti].on_miss()

        # ─── New tracks from leftover high-conf dets ───
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
            ))
            live_track_ids.add(tid)
            events.append(("track_created", tid))

        # Before returning, or a just-missed track is emitted for one extra
        # frame beside the new track of the object that jumped ahead.
        self._purge(now_ts)

        # Coasting tracks keep their old bbox, so emitting them leaves a stale
        # box behind a fast-moving object.
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
                "last_seen_s": now_ts - t.last_ts,
            })
        return {"events": events, "tracks": out_tracks}

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
    """The point on the bbox representing where the object is."""
    x1, y1, x2, y2 = b
    if mode == "center":
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
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


def _roi_points_px(roi: ROI, frame_w: int, frame_h: int) -> List[Tuple[float, float]]:
    """Project ROI points into pixel coordinates of the current frame.

    Normalized points are fractions of the FULL frame, so scaling by the live
    frame size is the whole conversion — do NOT add a letterbox remap.
    roi.frame_w/frame_h are provenance, used only to rescale pixel ROIs.
    """
    if not roi.normalized:
        if roi.frame_w and roi.frame_h:
            sx = float(frame_w) / max(float(roi.frame_w), 1e-9)
            sy = float(frame_h) / max(float(roi.frame_h), 1e-9)
            if sx != 1.0 or sy != 1.0:
                return [(float(px) * sx, float(py) * sy) for (px, py) in roi.points]
        return roi.points
    return [(px * frame_w, py * frame_h) for (px, py) in roi.points]


class ROIAlertEngine:
    """Alerts when a confirmed track enters an ROI (edge trigger).

    A dropped frame and a re-minted track id both look like a fresh entry, so
    repeats are suppressed three ways: state retires on a TTL rather than on
    absence, leaving needs exit_grace_s of continuous outside readings, and a
    recent alert for the same class in the same place blocks the next.
    """

    def __init__(
        self,
        state_ttl_s: float = 30.0,
        exit_grace_s: float = 10.0,
        realert_cooldown_s: float = 60.0,
        realert_iou: float = 0.3,
    ) -> None:
        self._in_roi: Dict[Tuple[str, str, int], bool] = {}
        self._inside_streak: Dict[Tuple[str, str, int], int] = {}
        self._last_seen_s: Dict[Tuple[str, str, int], float] = {}
        self._last_inside_s: Dict[Tuple[str, str, int], float] = {}
        # Keyed WITHOUT track_id, so a re-identified object does not re-alert.
        self._recent_alerts: Dict[Tuple[str, str, str], List[Tuple[float, List[float]]]] = {}
        # Retire on a timer, NOT on absence from `tracks`: only tracks matched
        # this frame are emitted, so absence would re-arm the edge trigger.
        self.state_ttl_s = float(state_ttl_s)
        self.exit_grace_s = float(exit_grace_s)
        self.realert_cooldown_s = float(realert_cooldown_s)
        self.realert_iou = float(realert_iou)

    def _entry_threshold(self, roi: ROI, frame_interval_s: float) -> int:
        """Consecutive inside-frames needed to trigger.

        enter_after_n assumes frames are close together. At one frame every
        4-10s, requiring two means the object must stay 8s+ and be detected on
        both — a vehicle driving through never alerts.
        """
        n = int(roi.enter_after_n)
        if n <= 1:
            return 1
        if frame_interval_s >= 1.0:
            return 1
        return n

    def _recently_alerted(
        self, camera_uuid: str, roi_id: str, cls_name: str, bbox: Any, now_s: float
    ) -> bool:
        """Whether an equivalent alert already fired here recently.

        Track ids are not stable, so this matches on (class, place, time)
        instead: a car in the zone alerts once however many ids it is given.
        """
        if self.realert_cooldown_s <= 0.0:
            return False
        key = (camera_uuid, roi_id, str(cls_name))
        recent = self._recent_alerts.get(key)
        if not recent:
            return False
        cutoff = now_s - self.realert_cooldown_s
        try:
            box = np.asarray(bbox, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return False
        if box.size != 4:
            return False
        for (ts, prev_box) in recent:
            if ts < cutoff:
                continue
            if _iou(box, np.asarray(prev_box, dtype=np.float32)) >= self.realert_iou:
                return True
        return False

    def _record_alert(
        self, camera_uuid: str, roi_id: str, cls_name: str, bbox: Any, now_s: float
    ) -> None:
        if self.realert_cooldown_s <= 0.0:
            return
        key = (camera_uuid, roi_id, str(cls_name))
        try:
            box = [float(v) for v in np.asarray(bbox, dtype=np.float32).reshape(-1)]
        except (TypeError, ValueError):
            return
        if len(box) != 4:
            return
        cutoff = now_s - self.realert_cooldown_s
        kept = [(ts, b) for (ts, b) in self._recent_alerts.get(key, []) if ts >= cutoff]
        kept.append((now_s, box))
        self._recent_alerts[key] = kept[-16:]

    def process(
        self,
        camera_uuid: str,
        frame_w: int,
        frame_h: int,
        tracks: List[Dict[str, Any]],
        rois: List[ROI],
        ts_ms: Optional[int] = None,
        frame_interval_s: float = 0.0,
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        ts_ms = int(ts_ms if ts_ms is not None else time.time() * 1000)
        now_s = ts_ms / 1000.0

        for roi in rois:
            poly = _roi_points_px(roi, frame_w, frame_h)
            enter_after_n = self._entry_threshold(roi, frame_interval_s)

            for t in tracks:
                if not t.get("confirmed", False):
                    continue
                if roi.allowed_classes is not None and t["cls_name"] not in roi.allowed_classes:
                    continue

                track_id = int(t["track_id"])
                key = (camera_uuid, roi.roi_id, track_id)
                self._last_seen_s[key] = now_s

                if roi.anchor == "bbox":
                    raw_inside = _bbox_intersects_poly(t["bbox"], poly)
                else:
                    ax, ay = _bbox_anchor(t["bbox"], roi.anchor)
                    raw_inside = _point_in_poly(ax, ay, poly)

                if raw_inside:
                    in_streak = self._inside_streak.get(key, 0) + 1
                    self._last_inside_s[key] = now_s
                else:
                    in_streak = 0
                self._inside_streak[key] = in_streak

                prev_inside = self._in_roi.get(key, False)
                if prev_inside:
                    # Time-based, never frame-based: a box jittering off the
                    # polygon edge for one frame must not re-arm the trigger.
                    since_inside = now_s - self._last_inside_s.get(key, now_s)
                    inside = raw_inside or since_inside < self.exit_grace_s
                else:
                    inside = in_streak >= enter_after_n
                self._in_roi[key] = inside

                if inside and not prev_inside:
                    if self._recently_alerted(
                        camera_uuid, roi.roi_id, t["cls_name"], t["bbox"], now_s
                    ):
                        continue
                    self._record_alert(
                        camera_uuid, roi.roi_id, t["cls_name"], t["bbox"], now_s
                    )
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

        self._expire_state(camera_uuid, now_s)
        return alerts

    def _expire_state(self, camera_uuid: str, now_s: float) -> None:
        if self.state_ttl_s > 0.0:
            cutoff = now_s - self.state_ttl_s
            stale = [
                k
                for k, seen in self._last_seen_s.items()
                if k[0] == camera_uuid and seen < cutoff
            ]
            for k in stale:
                self._last_seen_s.pop(k, None)
                self._in_roi.pop(k, None)
                self._inside_streak.pop(k, None)
                self._last_inside_s.pop(k, None)

        if self.realert_cooldown_s > 0.0:
            alert_cutoff = now_s - self.realert_cooldown_s
            for k in [k for k in self._recent_alerts if k[0] == camera_uuid]:
                kept = [(ts, b) for (ts, b) in self._recent_alerts[k] if ts >= alert_cutoff]
                if kept:
                    self._recent_alerts[k] = kept
                else:
                    self._recent_alerts.pop(k, None)

    def reset_camera(self, camera_uuid: str) -> None:
        """Wipe ROI state for a camera, on stream restart or reconfig."""
        cam = str(camera_uuid)
        self._in_roi = {k: v for k, v in self._in_roi.items() if k[0] != cam}
        self._inside_streak = {k: v for k, v in self._inside_streak.items() if k[0] != cam}
        self._last_seen_s = {k: v for k, v in self._last_seen_s.items() if k[0] != cam}
        self._last_inside_s = {k: v for k, v in self._last_inside_s.items() if k[0] != cam}
        self._recent_alerts = {k: v for k, v in self._recent_alerts.items() if k[0] != cam}



class MultiCameraByteTrack:
    def __init__(self, **tracker_kwargs: Any) -> None:
        self._trackers: Dict[str, ByteTrackLite] = {}
        self._kwargs = tracker_kwargs

    def update_from_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        cam = str(ev["camera_uuid"])
        try:
            ts_s = float(ev.get("frame_ts_ms") or int(time.time() * 1000)) / 1000.0
        except (TypeError, ValueError):
            ts_s = time.time()

        dets: List[Dict[str, Any]] = []
        for d in ev.get("detections", []) or []:
            b = d.get("box") if isinstance(d, dict) else None
            if not isinstance(b, dict):
                continue
            try:
                bbox = np.array([b["x1"], b["y1"], b["x2"], b["y2"]], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue
            dets.append({
                "bbox": bbox,
                "cls_name": str(d.get("cls_name", "unknown")),
                "conf": float(d.get("conf", 0.0)),
            })

        if cam not in self._trackers:
            self._trackers[cam] = ByteTrackLite(**self._kwargs)

        return self._trackers[cam].update(dets, ts_s=ts_s)
    
    def frame_interval_s(self, camera_uuid: str) -> float:
        """Observed inter-frame interval for one camera, 0.0 if not yet known."""
        tr = self._trackers.get(str(camera_uuid))
        return tr._frame_interval() if tr is not None else 0.0

    def remove_camera(self, camera_uuid: str) -> None:
        self._trackers.pop(str(camera_uuid), None)

    def reset(self) -> None:
        self._trackers.clear()
