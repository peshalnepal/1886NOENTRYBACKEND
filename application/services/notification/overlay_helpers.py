"""Frame/overlay normalization helpers used by NotificationService.

Extracted verbatim from the former monolithic application/services/notification.py.
These helpers normalize detection overlays into a canonical dict shape that the
clip-history, prerecord, and email pipelines can consume safely.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from domain.events import DetectionsProducedEvent
from application.services.overlay_normalize import (
    _append_overlay_detection,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_overlay_frame(
    *,
    camera_uuid: Optional[str],
    frame_ts_ms: Any,
    frame_seq: Any,
    frame_w: Any,
    frame_h: Any,
    detections: Any,
) -> Optional[Dict[str, Any]]:
    try:
        ts_ms = int(frame_ts_ms)
    except (TypeError, ValueError):
        return None
    try:
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
        "frame_ts_ms": int(ts_ms),
        "frame_seq": int(seq),
        "detections": normalized_detections,
    }
    if camera_uuid:
        frame["camera_uuid"] = str(camera_uuid)
    normalized_w = _coerce_positive_int(frame_w)
    normalized_h = _coerce_positive_int(frame_h)
    if normalized_w is not None:
        frame["frame_w"] = normalized_w
    if normalized_h is not None:
        frame["frame_h"] = normalized_h
    return frame


def _merge_overlay_frames(*sources: Any) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for source in sources:
        if not isinstance(source, list):
            continue
        for raw_frame in source:
            if not isinstance(raw_frame, dict):
                continue
            normalized = _normalize_overlay_frame(
                camera_uuid=raw_frame.get("camera_uuid"),
                frame_ts_ms=raw_frame.get("frame_ts_ms"),
                frame_seq=raw_frame.get("frame_seq"),
                frame_w=raw_frame.get("frame_w"),
                frame_h=raw_frame.get("frame_h"),
                detections=raw_frame.get("detections"),
            )
            if normalized is None:
                continue
            merged[(normalized["frame_ts_ms"], normalized["frame_seq"])] = normalized

    return sorted(
        merged.values(),
        key=lambda item: (int(item.get("frame_ts_ms", 0)), int(item.get("frame_seq", 0))),
    )


def _select_overlay_reference_frame(
    frames: List[Dict[str, Any]],
    *,
    preferred_ts_ms: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not frames:
        return None
    if preferred_ts_ms is None:
        return frames[-1]
    return min(
        frames,
        key=lambda item: (
            abs(int(item.get("frame_ts_ms", 0)) - int(preferred_ts_ms)),
            abs(int(item.get("frame_seq", 0))),
            int(item.get("frame_ts_ms", 0)),
        ),
    )


def _build_overlay_payload_from_frames(
    *,
    camera_uuid: str,
    frames: List[Dict[str, Any]],
    preferred_ts_ms: Optional[int] = None,
    clip_start_time: Optional[datetime] = None,
    clip_end_time: Optional[datetime] = None,
    timeline_source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    merged_frames = _merge_overlay_frames(frames)
    reference = _select_overlay_reference_frame(merged_frames, preferred_ts_ms=preferred_ts_ms)
    if reference is None:
        return None

    payload: Dict[str, Any] = {
        "camera_uuid": str(camera_uuid),
        "frame_ts_ms": int(reference["frame_ts_ms"]),
        "frame_seq": int(reference["frame_seq"]),
        "detections": list(reference.get("detections") or []),
        "frames": merged_frames,
    }
    if reference.get("frame_w") is not None:
        payload["frame_w"] = int(reference["frame_w"])
    if reference.get("frame_h") is not None:
        payload["frame_h"] = int(reference["frame_h"])
    if clip_start_time is not None:
        payload["clip_start_time"] = clip_start_time.astimezone(timezone.utc).isoformat()
    if clip_end_time is not None:
        payload["clip_end_time"] = clip_end_time.astimezone(timezone.utc).isoformat()
    if timeline_source:
        payload["timeline_source"] = str(timeline_source)
    return payload


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_overlay_payload(
    det_ev: DetectionsProducedEvent,
    *,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> Dict[str, Any]:
    frame = _normalize_overlay_frame(
        camera_uuid=str(det_ev.camera_uuid),
        frame_ts_ms=det_ev.frame_ts_ms,
        frame_seq=det_ev.frame_seq,
        frame_w=frame_w,
        frame_h=frame_h,
        detections=list(det_ev.detections or []),
    )
    if frame is None:
        payload: Dict[str, Any] = {
            "camera_uuid": str(det_ev.camera_uuid),
            "frame_ts_ms": int(det_ev.frame_ts_ms),
            "frame_seq": int(det_ev.frame_seq),
            "detections": [],
            "frames": [],
        }
        if frame_w is not None:
            payload["frame_w"] = int(frame_w)
        if frame_h is not None:
            payload["frame_h"] = int(frame_h)
        return payload

    return _build_overlay_payload_from_frames(
        camera_uuid=str(det_ev.camera_uuid),
        frames=[frame],
        preferred_ts_ms=int(det_ev.frame_ts_ms),
        timeline_source="event",
    ) or {
        "camera_uuid": str(det_ev.camera_uuid),
        "frame_ts_ms": int(det_ev.frame_ts_ms),
        "frame_seq": int(det_ev.frame_seq),
        "detections": [],
        "frames": [],
    }
