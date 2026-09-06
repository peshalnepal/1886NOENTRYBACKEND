"""Shared overlay-detection normalization helpers.

Normalize detection objects from various sources (Jetson event payloads,
tracker output, ROI alerts) into a uniform overlay schema, and accumulate
them with dedup rules suitable for rendering.

Public entry point used externally:
  - _append_overlay_detection (consumed by domain/model_pipeline.py,
    application/services/notification.py, application/services/clip_storage.py,
    routes/clips_routes.py)

Everything else is internal and may change freely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.coercions import coerce_positive_int


# --------------------------------- access ----------------------------------

class _Source:
    """Uniform read access to either a dict or an attribute-bag object."""
    __slots__ = ("_dict", "_obj")

    def __init__(self, raw: Any) -> None:
        if isinstance(raw, dict):
            self._dict, self._obj = raw, None
        else:
            self._dict, self._obj = None, raw

    def get(self, name: str) -> Any:
        if self._dict is not None:
            return self._dict.get(name)
        return getattr(self._obj, name, None)

    def first(self, *names: str) -> Any:
        """First truthy value among the given field names."""
        for n in names:
            v = self.get(n)
            if v:
                return v
        return None


# --------------------------------- boxes -----------------------------------

_BOX_KEYS = ("x1", "y1", "x2", "y2")
_NORM_KEYS = ("x", "y", "w", "h")


def _normalize_overlay_box(raw: Any) -> Optional[Dict[str, int]]:
    """Coerce a bounding box (dict, list/tuple, or object) to ints, or None."""
    if raw is None:
        return None

    if isinstance(raw, dict):
        values = [raw.get(k) for k in _BOX_KEYS]
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        values = list(raw[:4])
    else:
        values = [getattr(raw, k, None) for k in _BOX_KEYS]

    try:
        x1, y1, x2, y2 = (int(v) for v in values)
    except (TypeError, ValueError):
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _normalize_box_norm(raw: Any) -> Optional[Dict[str, float]]:
    """Coerce a normalized box (dict or x/y/w/h object) to floats, or None."""
    if raw is None:
        return None

    if isinstance(raw, dict):
        values = [raw.get(k) for k in _NORM_KEYS]
    elif all(hasattr(raw, k) for k in _NORM_KEYS):
        values = [getattr(raw, k) for k in _NORM_KEYS]
    else:
        return None

    try:
        return dict(zip(_NORM_KEYS, (float(v) for v in values)))
    except (TypeError, ValueError):
        return None


# ------------------------------ detections ---------------------------------

def _normalize_overlay_detection(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single detection to the overlay schema, or None if no box."""
    if raw is None:
        return None

    src = _Source(raw)

    box = _normalize_overlay_box(src.first("box", "bbox"))
    if box is None:
        return None

    try:
        conf = float(src.get("conf") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    out: Dict[str, Any] = {
        "cls_name": str(src.first("cls_name", "class_name", "class") or "obj"),
        "conf": conf,
        "box": box,
    }

    box_norm = _normalize_box_norm(src.get("box_norm"))
    if box_norm is not None:
        out["box_norm"] = box_norm

    try:
        out["track_id"] = int(src.get("track_id"))
    except (TypeError, ValueError):
        pass

    return out


# ---------------------------- dedup base key -------------------------------

def _overlay_detection_base_key(d: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity of a detection ignoring its track_id."""
    box = d["box"]
    return (d["cls_name"], d["conf"], box["x1"], box["y1"], box["x2"], box["y2"])


# ---------------- public API: procedural dedup append ----------------------
# Signature and behavior are deliberately unchanged — this function is
# imported by multiple modules. Do not modify without updating all callers.

def _append_overlay_detection(
    detections: List[Dict[str, Any]],
    raw_detection: Any,
    *,
    seen_exact: set,
    tracked_bases: set,
    untracked_indexes: Dict[Tuple[Any, ...], int],
) -> None:
    normalized = _normalize_overlay_detection(raw_detection)
    if normalized is None:
        return

    base_key = _overlay_detection_base_key(normalized)
    track_id = normalized.get("track_id")
    exact_key = base_key + (track_id,)
    if exact_key in seen_exact:
        return

    if track_id is None:
        if base_key in tracked_bases:
            return
        untracked_indexes.setdefault(base_key, len(detections))
        detections.append(normalized)
        seen_exact.add(exact_key)
        return

    untracked_index = untracked_indexes.pop(base_key, None)
    if untracked_index is not None:
        detections[untracked_index] = normalized
    else:
        detections.append(normalized)
    tracked_bases.add(base_key)
    seen_exact.add(exact_key)

def normalize_overlay_frame(
    *,
    camera_uuid: Optional[str],
    frame_ts_ms: Any,
    frame_seq: Any,
    frame_w: Any,
    frame_h: Any,
    detections: Any,
) -> Optional[Dict[str, Any]]:
    """Build one overlay frame, or None if it carries no usable detections."""
    try:
        ts_ms = int(frame_ts_ms)
        seq = int(frame_seq)
    except (TypeError, ValueError):
        return None

    normalized_detections: List[Dict[str, Any]] = []
    seen_exact: set[Tuple[Any, ...]] = set()
    tracked_bases: set[Tuple[Any, ...]] = set()
    untracked_indexes: Dict[Tuple[Any, ...], int] = {}
    for raw_detection in list(detections or []):
        _append_overlay_detection(
            normalized_detections,
            raw_detection,
            seen_exact=seen_exact,
            tracked_bases=tracked_bases,
            untracked_indexes=untracked_indexes,
        )

    if not normalized_detections:
        return None

    frame: Dict[str, Any] = {
        "frame_ts_ms": ts_ms,
        "frame_seq": seq,
        "detections": normalized_detections,
    }
    if camera_uuid:
        frame["camera_uuid"] = str(camera_uuid)
    for key, value in (("frame_w", frame_w), ("frame_h", frame_h)):
        normalized = coerce_positive_int(value)
        if normalized is not None:
            frame[key] = normalized
    return frame


def normalize_overlay_frame_dict(
    raw_frame: Any, *, default_camera_uuid: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Dict-shaped adapter over `normalize_overlay_frame`."""
    if not isinstance(raw_frame, dict):
        return None
    return normalize_overlay_frame(
        camera_uuid=str(raw_frame.get("camera_uuid") or default_camera_uuid or "").strip(),
        frame_ts_ms=raw_frame.get("frame_ts_ms"),
        frame_seq=raw_frame.get("frame_seq"),
        frame_w=raw_frame.get("frame_w"),
        frame_h=raw_frame.get("frame_h"),
        detections=raw_frame.get("detections"),
    )
