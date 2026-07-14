# application/services/detection_stream.py
"""
Detection serialization + Server-Sent-Event generators.

Extracted from `routes/camera_routes.py` so the authenticated per-camera stream
and the anonymous public-wall stream share one implementation rather than two
copies that drift apart.

Nothing here does authorization — callers resolve the pipeline(s) they are
allowed to read and hand them in.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from application.services.pipeline import _live_tracks, _overlay_payload_from_resp
from core.schemas import BoxNorm, BoxPx, DetectionItemOut, DetectionOut

logger = logging.getLogger(__name__)

HEARTBEAT_EVENT = "event: heartbeat\ndata: {}\n\n"


def normalize_box_px(
    box: Dict[str, Any], frame_w: Optional[int], frame_h: Optional[int]
) -> Optional[BoxNorm]:
    if not frame_w or not frame_h:
        return None

    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    return BoxNorm(
        x=max(0.0, min(1.0, x1 / frame_w)),
        y=max(0.0, min(1.0, y1 / frame_h)),
        w=max(0.0, min(1.0, w / frame_w)),
        h=max(0.0, min(1.0, h / frame_h)),
    )


def resp_to_detection_out(resp: Any, *, normalize: bool) -> DetectionOut:
    fw = getattr(resp, "frame_w", None)
    fh = getattr(resp, "frame_h", None)

    merged = _overlay_payload_from_resp(
        resp,
        fallback_detections=_live_tracks(list(getattr(resp, "tracks", ()) or ())),
    ).get("detections") or []

    items: List[DetectionItemOut] = []
    for d in merged:
        box = d.get("box") or {}
        if not all(k in box for k in ("x1", "y1", "x2", "y2")):
            continue

        box_px = BoxPx(x1=box["x1"], y1=box["y1"], x2=box["x2"], y2=box["y2"])
        tid = d.get("track_id")
        items.append(
            DetectionItemOut(
                box=box_px,
                cls_name=str(d.get("cls_name") or ""),
                conf=float(d.get("conf") or 0.0),
                box_norm=normalize_box_px(box, fw, fh) if normalize else None,
                track_id=int(tid) if tid is not None else None,
            )
        )

    return DetectionOut(
        camera_uuid=str(resp.camera_uuid),
        frame_ts_ms=int(resp.frame_ts_ms),
        frame_seq=int(resp.frame_seq),
        event_type=str(getattr(resp, "event_type", None)) if getattr(resp, "event_type", None) is not None else None,
        reason=str(getattr(resp, "reason", None)) if getattr(resp, "reason", None) is not None else None,
        inference_ms=int(resp.inference_ms) if getattr(resp, "inference_ms", None) is not None else None,
        model_id=str(getattr(resp, "model_id", None)) if getattr(resp, "model_id", None) is not None else None,
        frame_w=int(fw) if fw is not None else None,
        frame_h=int(fh) if fh is not None else None,
        detections=items,
        pose=getattr(resp, "pose", None),
    )


def detection_event(resp: Any, *, normalize: bool) -> str:
    payload = resp_to_detection_out(resp, normalize=normalize).model_dump()
    return f"event: detection\ndata: {json.dumps(payload)}\n\n"


async def stream_camera_detections(
    *,
    pipeline: Any,
    camera_uuid: str,
    after_ts_ms: int = 0,
    after_seq: int = 0,
    timeout_ms: int = 30000,
    normalize: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]],
):
    """SSE for a single camera, read off one pipeline's detection store."""
    cam_key = str(camera_uuid)
    last_ts = int(after_ts_ms)
    last_seq = int(after_seq)

    initial = await pipeline.get_latest_detection(cam_key)
    if initial is not None and last_ts == 0 and last_seq > 0:
        last_ts = int(initial.frame_ts_ms)
    if initial is not None:
        its, isq = int(initial.frame_ts_ms), int(initial.frame_seq)
        if its > last_ts or (its == last_ts and isq > last_seq):
            last_ts, last_seq = its, isq
            yield detection_event(initial, normalize=normalize)

    while True:
        if await is_disconnected():
            return

        resp = await pipeline.detect_store.wait_new(
            cam_key,
            after_ts_ms=last_ts,
            after_seq=last_seq,
            timeout_ms=int(timeout_ms),
        )

        if resp is None:
            yield HEARTBEAT_EVENT
            continue

        last_ts = int(resp.frame_ts_ms)
        last_seq = int(resp.frame_seq)
        yield detection_event(resp, normalize=normalize)


async def stream_multi_camera_detections(
    *,
    cameras: Iterable[Tuple[str, Any]],
    timeout_ms: int = 30000,
    normalize: bool = False,
    is_disconnected: Callable[[], Awaitable[bool]],
    still_authorized: Optional[Callable[[], Awaitable[bool]]] = None,
    reauthorize_every_s: float = 60.0,
):
    """One SSE carrying every camera on a wall, fanning in across pipelines.

    ``cameras`` is (camera_uuid, pipeline) pairs — cameras on a multi-site wall can
    belong to different owners and therefore different pipelines. Each camera is
    waited on individually against its own pipeline's detection store, which is
    keyed by camera_uuid; the pipeline-wide `detection_hub` is deliberately NOT
    used, because it does no filtering and would emit detections for cameras that
    are not on this wall.

    Fanning in server-side keeps a viewer to ONE connection. Opening one stream per
    camera instead would exhaust the browser's six-connections-per-origin budget on
    the wall alone and stall every other request the page makes.

    ``still_authorized`` is re-checked every ``reauthorize_every_s`` so a revoked
    share link drops viewers who are already connected — without it, revocation
    would only stop new ones.
    """
    cursors: Dict[str, Tuple[int, int]] = {}
    pipelines: Dict[str, Any] = {}

    for camera_uuid, pipeline in cameras:
        cam_key = str(camera_uuid)
        pipelines[cam_key] = pipeline
        cursors[cam_key] = (0, 0)

    if not pipelines:
        # Nothing to read (e.g. no pipeline is loaded for any owner). Hold the
        # connection open with heartbeats so the client does not hot-reconnect.
        while True:
            if await is_disconnected():
                return
            if still_authorized is not None and not await still_authorized():
                return
            await asyncio.sleep(min(reauthorize_every_s, timeout_ms / 1000.0))
            yield HEARTBEAT_EVENT

    # Prime each camera with whatever the pipeline already holds, so a viewer sees
    # boxes immediately instead of waiting for the next frame.
    for cam_key, pipeline in pipelines.items():
        try:
            initial = await pipeline.get_latest_detection(cam_key)
        except Exception:
            logger.debug("Priming detection failed for %s", cam_key, exc_info=True)
            continue

        if initial is None:
            continue

        cursors[cam_key] = (int(initial.frame_ts_ms), int(initial.frame_seq))
        yield detection_event(initial, normalize=normalize)

    tasks: Dict[asyncio.Task, str] = {}
    last_authorized_at = asyncio.get_running_loop().time()

    def _spawn(cam_key: str) -> None:
        last_ts, last_seq = cursors[cam_key]
        task = asyncio.create_task(
            pipelines[cam_key].detect_store.wait_new(
                cam_key,
                after_ts_ms=last_ts,
                after_seq=last_seq,
                timeout_ms=int(timeout_ms),
            )
        )
        tasks[task] = cam_key

    try:
        for cam_key in pipelines:
            _spawn(cam_key)

        while True:
            if await is_disconnected():
                return

            done, _pending = await asyncio.wait(
                tasks.keys(),
                timeout=reauthorize_every_s,
                return_when=asyncio.FIRST_COMPLETED,
            )

            now = asyncio.get_running_loop().time()
            if still_authorized is not None and (now - last_authorized_at) >= reauthorize_every_s:
                if not await still_authorized():
                    return
                last_authorized_at = now

            if not done:
                yield HEARTBEAT_EVENT
                continue

            # A camera whose wait_new() timed out resolves to None. If every
            # completed wait was a timeout we still have to say something, or the
            # connection goes silent and intermediaries reap it as idle.
            emitted = False

            for task in done:
                cam_key = tasks.pop(task)

                try:
                    resp = task.result()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Detection wait failed for %s", cam_key, exc_info=True)
                    resp = None

                if resp is not None:
                    cursors[cam_key] = (int(resp.frame_ts_ms), int(resp.frame_seq))
                    emitted = True
                    yield detection_event(resp, normalize=normalize)

                _spawn(cam_key)

            if not emitted:
                yield HEARTBEAT_EVENT
    finally:
        for task in tasks:
            task.cancel()
