# agents/domain/model_pipeline.py
"""
Azure-side pipeline.

- NO RTSP ingest
- NO frame streaming

It keeps a registry of Camera Channels and maintains an in-memory "latest detection"
cache per camera by polling Jetson: /detection/{camera_uuid}
"""

import base64
import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.channels.channel import VideoChannel, VideoChannelConfig
from application.services.tracker import MultiCameraByteTrack, ROI, ROIAlertEngine
from application.services.notification import NotificationMessage, NotificationService
from core.database_orm import Site
from application.repositories.notification_repository import (
    NotificationRepository,
    CameraContext,
)


logger = logging.getLogger(__name__)

ROIProvider = Callable[[str], Awaitable[List[ROI]]]
SessionFactory = Callable[[], AsyncSession]
@dataclass(frozen=True)
class ObjDetectResponse:
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int

    event_type: Optional[str] = None
    reason: Optional[str] = None
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None

    site_uuid: Optional[str] = None
    device_uuid: Optional[str] = None

    detections: Tuple[Any, ...] = ()
    pose: Optional[Any] = None
    inference_ms: Optional[int] = None
    model_id: Optional[str] = None
    image_url: Optional[str] = None

    # NEW: tracking + alert scaffolding
    tracks: Tuple[Dict[str, Any], ...] = ()
    track_events: Tuple[Tuple[str, int], ...] = ()
    alerts: Tuple[Dict[str, Any], ...] = ()


def _coerce_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _normalize_track_id(value: Any) -> Optional[int]:
    return _coerce_int(value)


def _normalize_overlay_box(raw_box: Any) -> Optional[Dict[str, int]]:
    if isinstance(raw_box, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(key in raw_box for key in keys):
            return None
        values = tuple(_coerce_int(raw_box.get(key)) for key in keys)
        if any(value is None for value in values):
            return None
        x1, y1, x2, y2 = values
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        values = tuple(_coerce_int(raw_box[idx]) for idx in range(4))
        if any(value is None for value in values):
            return None
        x1, y1, x2, y2 = values
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    if all(hasattr(raw_box, key) for key in ("x1", "y1", "x2", "y2")):
        values = tuple(_coerce_int(getattr(raw_box, key, None)) for key in ("x1", "y1", "x2", "y2"))
        if any(value is None for value in values):
            return None
        x1, y1, x2, y2 = values
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    return None


def _normalize_box_norm(raw: Any) -> Optional[Dict[str, float]]:
    if isinstance(raw, dict):
        raw_dict = raw
    elif hasattr(raw, "x") and hasattr(raw, "y") and hasattr(raw, "w") and hasattr(raw, "h"):
        raw_dict = {"x": raw.x, "y": raw.y, "w": raw.w, "h": raw.h}
    else:
        return None
    try:
        x = float(raw_dict["x"])
        y = float(raw_dict["y"])
        w = float(raw_dict["w"])
        h = float(raw_dict["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _normalize_overlay_detection(raw_detection: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw_detection, dict):
        box = _normalize_overlay_box(raw_detection.get("box") or raw_detection.get("bbox"))
        cls_name = str(raw_detection.get("cls_name") or raw_detection.get("class") or "obj")
        conf = float(raw_detection.get("conf", 0.0) or 0.0)
        raw_box_norm = raw_detection.get("box_norm")
        track_id = _normalize_track_id(raw_detection.get("track_id"))
    else:
        box = _normalize_overlay_box(getattr(raw_detection, "box", None) or getattr(raw_detection, "bbox", None))
        cls_name = str(
            getattr(raw_detection, "cls_name", None)
            or getattr(raw_detection, "class_name", None)
            or getattr(raw_detection, "class", None)
            or "obj"
        )
        conf = float(getattr(raw_detection, "conf", 0.0) or 0.0)
        raw_box_norm = getattr(raw_detection, "box_norm", None)
        track_id = _normalize_track_id(getattr(raw_detection, "track_id", None))

    if box is None:
        return None

    result: Dict[str, Any] = {
        "cls_name": cls_name,
        "conf": conf,
        "box": box,
    }
    box_norm = _normalize_box_norm(raw_box_norm)
    if box_norm is not None:
        result["box_norm"] = box_norm
    if track_id is not None:
        result["track_id"] = track_id
    return result


def _overlay_detection_base_key(normalized: Dict[str, Any]) -> Tuple[Any, ...]:
    box = normalized["box"]
    return (
        normalized["cls_name"],
        normalized["conf"],
        box["x1"],
        box["y1"],
        box["x2"],
        box["y2"],
    )


def _append_overlay_detection(
    detections: List[Dict[str, Any]],
    raw_detection: Any,
    *,
    seen_exact: set[Tuple[Any, ...]],
    tracked_bases: set[Tuple[Any, ...]],
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


def _overlay_payload_from_resp(
    resp: ObjDetectResponse,
    *,
    fallback_detections: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    detections: List[Dict[str, Any]] = []
    seen_exact: set[Tuple[Any, ...]] = set()
    tracked_bases: set[Tuple[Any, ...]] = set()
    untracked_indexes: Dict[Tuple[Any, ...], int] = {}

    for raw_detection in list(resp.detections or ()) + list(fallback_detections or ()):
        _append_overlay_detection(
            detections,
            raw_detection,
            seen_exact=seen_exact,
            tracked_bases=tracked_bases,
            untracked_indexes=untracked_indexes,
        )

    return {
        "frame_ts_ms": int(resp.frame_ts_ms),
        "frame_seq": int(resp.frame_seq),
        "frame_w": _coerce_int(resp.frame_w),
        "frame_h": _coerce_int(resp.frame_h),
        "detections": detections,
    }


class ObjDetectStorePort(Protocol):
    async def put(self, resp: ObjDetectResponse) -> None: ...
    async def get_latest(self, camera_uuid: str) -> Optional[ObjDetectResponse]: ...
    async def wait_new(
        self,
        camera_uuid: str,
        *,
        after_ts_ms: int,
        after_seq: int,
        timeout_ms: int
    ) -> Optional[ObjDetectResponse]: ...


class InMemoryObjDetectStore:
    """Latest-only per camera_uuid with an asyncio.Event per camera."""

    def __init__(self):
        self._latest: Dict[str, ObjDetectResponse] = {}
        self._signal: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def _evt(self, camera_uuid: str) -> asyncio.Event:
        async with self._lock:
            e = self._signal.get(camera_uuid)
            if e is None:
                e = asyncio.Event()
                self._signal[camera_uuid] = e
            return e

    async def put(self, resp: ObjDetectResponse) -> None:
        key = str(resp.camera_uuid)
        async with self._lock:
            self._latest[key] = resp
        (await self._evt(key)).set()

    async def get_latest(self, camera_uuid: str) -> Optional[ObjDetectResponse]:
        async with self._lock:
            return self._latest.get(str(camera_uuid))

    async def forget(self, camera_uuid: str) -> None:
        key = str(camera_uuid)
        async with self._lock:
            self._latest.pop(key, None)
            evt = self._signal.get(key)
            if evt is not None:
                evt.clear()

    async def wait_new(
        self,
        camera_uuid: str,
        *,
        after_ts_ms: int,
        after_seq: int,
        timeout_ms: int
    ) -> Optional[ObjDetectResponse]:
        key = str(camera_uuid)
        evt = await self._evt(key)
        timeout_s = max(0.0, timeout_ms / 1000.0)

        while True:
            try:
                await asyncio.wait_for(evt.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                return None

            async with self._lock:
                resp = self._latest.get(key)
                evt.clear()
            if resp is not None:
                if resp.frame_ts_ms > after_ts_ms:
                    return resp
                if resp.frame_ts_ms == after_ts_ms and resp.frame_seq > after_seq:
                    return resp


class DetectionHub:
    """
    Pub/Sub hub for all detections across all cameras.
    Used for the global SSE stream.
    """
    def __init__(self, max_q: int = 100):
        self._subs: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._max_q = max_q

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_q)
        async with self._lock:
            self._subs.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subs.discard(q)

    async def publish(self, resp: ObjDetectResponse) -> None:
        async with self._lock:
            subs = list(self._subs)
        if not subs:
            return
        for q in subs:
            try:
                q.put_nowait(resp)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(resp)
                except Exception:
                    pass


# -------------------------
# ModelPipeline (registry + pollers)
# -------------------------
class ModelPipeline:
    """
    Azure runtime registry + Jetson detection polling.

    Important:
    - webrtc_url is treated as immutable (cannot be patched/edited here).
    """

    def __init__(
        self,
        pipeline_id: UUID,
        channels: Optional[List[VideoChannel]] = None,
        *,
        detect_store: Optional[ObjDetectStorePort] = None,
        roi_provider: Optional[ROIProvider] = None,          # NEW
        notification_service: Optional[NotificationService] = None,  # NEW

    ):
        self.pipeline_id = pipeline_id
        self.detect_store = detect_store or InMemoryObjDetectStore()
        self.detection_hub = DetectionHub()
        self._last_seen: Dict[str, Tuple[int, int]] = {}     
        self._last_ok_s: Dict[str, float] = {}

        self._channels: Dict[str, VideoChannel] = {}
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._last_seq: Dict[str, int] = {}
        self._tracker: MultiCameraByteTrack = MultiCameraByteTrack()
        self._roi_engine = ROIAlertEngine()                 
        self._roi_provider: ROIProvider = roi_provider or self._no_rois 
        self._notification_service = notification_service    
        self._last_detection_summary_s: Dict[str, float] = {}
        self._detection_summary_cooldown_s = max(
            0.0,
            float(os.getenv("DETECTION_ALERT_COOLDOWN_S", "8.0")),
        )
        self._notif_repo = NotificationRepository()
        self._cam_ctx_cache: Dict[str, Tuple[Optional[CameraContext], float]] = {}
        self._cam_ctx_cache_lock = asyncio.Lock()
        self._cam_ctx_ttl_s = 60.0 
        self._cam_ctx_miss_ttl_s = max(
            0.0,
            float(os.getenv("CAMERA_CONTEXT_MISS_CACHE_TTL_S", "5.0")),
        )
        self._cam_ctx_inflight: Dict[str, asyncio.Future] = {}
        self._cam_ctx_lookup_limit = asyncio.Semaphore(
            max(
                1,
                int(os.getenv("CAMERA_CONTEXT_MAX_CONCURRENT_DB_LOOKUPS", "4")),
            )
        )
        self._session_factory: Optional[SessionFactory] = None
        self._site_cache: Dict[str, Tuple[str, float]] = {}
        self._site_cache_lock = asyncio.Lock()
        self._site_trigger_mode_cache: Dict[str, Tuple[str, float]] = {}
        self._site_trigger_mode_cache_lock = asyncio.Lock()
        self._site_trigger_mode_ttl_s = max(
            5.0,
            float(os.getenv("SITE_TRIGGER_MODE_CACHE_TTL_S", "30.0")),
        )
        self._device_fetch_limits: Dict[str, asyncio.Semaphore] = {}
        self._max_concurrent_fetch_per_device = max(
            1,
            int(os.getenv("DETECTION_MAX_CONCURRENT_FETCH_PER_DEVICE", "8")),
        )
        self._startup_jitter_ms = max(
            0,
            int(os.getenv("DETECTION_STARTUP_JITTER_MS", "100")),
        )
        self._lock = asyncio.Lock()
        self._started = False
        self._closing = False

        for ch in (channels or []):
            self._channels[ch.key()] = ch

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._started = True
            self._closing = False
            items = list(self._channels.items())

        for key, ch in items:
            self._ensure_poller(key, ch)

    async def shutdown(self) -> None:
        async with self._lock:
            if not self._started:
                return
            self._closing = True
            tasks = list(self._poll_tasks.values())
            self._poll_tasks.clear()
            self._device_fetch_limits.clear()
            self._channels.clear()
            self._started = False
            self._last_seen.clear()
            self._last_ok_s.clear()
            self._last_seq.clear()
            self._last_detection_summary_s.clear()
        for t in tasks:
            if t and not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Poller task failed during shutdown")

    def _is_new_detection(self, key: str, resp: ObjDetectResponse) -> bool:
        prev_ts, prev_seq = self._last_seen.get(key, (0, -1))
        cur_ts, cur_seq = int(resp.frame_ts_ms), int(resp.frame_seq)

        # normal monotonic case
        if cur_ts > prev_ts:
            return True
        if cur_ts == prev_ts and cur_seq > prev_seq:
            return True

        # exact duplicate frame (same ts and seq): not new, and not a restart.
        if cur_ts == prev_ts and cur_seq == prev_seq:
            return False

        # stale/backward frame but no actual regression (older seq with newer ts is handled above)
        regressed = (cur_ts < prev_ts) or (cur_ts == prev_ts and cur_seq < prev_seq)
        if not regressed:
            return False

        # regression case: if we had a gap (disconnect), assume Jetson restarted -> accept & reset
        last_ok = self._last_ok_s.get(key, 0.0)
        if (time.monotonic() - last_ok) > 5.0:
            logger.warning(
                "Detected possible Jetson restart (ts/seq regressed). Resetting last_seen. camera=%s prev=(%s,%s) cur=(%s,%s)",
                key, prev_ts, prev_seq, cur_ts, cur_seq
            )
            return True

        return False

    async def _no_rois(self, camera_uuid: str) -> List[ROI]:
        return []

    def set_notification_service(self, svc: NotificationService) -> None:
        self._notification_service = svc

    def set_roi_provider(self, roi_provider: ROIProvider) -> None:
        self._roi_provider = roi_provider

    def set_session_factory(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory
        if self._notification_service is not None:
            try:
                self._notification_service.set_session_factory(session_factory)
            except Exception:
                logger.exception("Failed to set session factory on NotificationService")

    def invalidate_camera_roi_state(self, camera_uuid: str) -> None:
        """
        Clear per-camera ROI edge-trigger state so ROI edits take effect immediately.
        """
        self._roi_engine.reset_camera(str(camera_uuid))

    async def _get_site_trigger_mode(self, site_uuid: Optional[str]) -> str:
        """
        Return the trigger_mode from the site's multi_camera_prerecord settings.

        Values: "roi_enter" | "any_detection"

        - "roi_enter"      → only emit ROI-enter notifications
        - "any_detection"  → emit all notification types (default / backward-compatible)

        Result is cached per site_uuid for _site_trigger_mode_ttl_s seconds.
        Falls back to "any_detection" (permissive) when the setting cannot be loaded.
        """
        if not site_uuid:
            return "roi_enter"

        key = str(site_uuid)
        now = time.monotonic()

        async with self._site_trigger_mode_cache_lock:
            cached = self._site_trigger_mode_cache.get(key)
            if cached and (now - cached[1]) < self._site_trigger_mode_ttl_s:
                return cached[0]

        sf = self._session_factory
        if sf is None:
            return "roi_enter"

        try:
            su = UUID(key)
        except Exception:
            return "roi_enter"

        trigger_mode = "roi_enter"
        try:
            from core.database_orm import SiteSettings
            async with sf() as db:
                res = await db.execute(
                    select(SiteSettings.config).where(SiteSettings.site_uuid == su)
                )
                config = res.scalar_one_or_none()
                if isinstance(config, dict):
                    notification_rule = config.get("notification")
                    notif_raw = None
                    if isinstance(notification_rule, dict):
                        notif_raw = notification_rule.get("trigger_mode")
                    if notif_raw is None:
                        legacy = config.get("multi_camera_prerecord") or {}
                        if isinstance(legacy, dict):
                            notif_raw = legacy.get("trigger_mode")
                    raw = str(notif_raw or "roi_enter").strip().lower()
                    if raw in {"roi_enter", "any_detection"}:
                        trigger_mode = raw
        except Exception:
            logger.exception("Failed to load site trigger_mode site_uuid=%s", site_uuid)
            trigger_mode = "roi_enter"

        async with self._site_trigger_mode_cache_lock:
            self._site_trigger_mode_cache[key] = (trigger_mode, time.monotonic())

        return trigger_mode

    def invalidate_site_trigger_mode_cache(self, site_uuid: str) -> None:
        """
        Invalidate the cached trigger_mode for a site (e.g., after settings are saved).
        """
        key = str(site_uuid)
        async def _clear():
            async with self._site_trigger_mode_cache_lock:
                self._site_trigger_mode_cache.pop(key, None)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_clear())
            else:
                loop.run_until_complete(_clear())
        except Exception:
            # Fallback: clear synchronously (lock may not be acquired but best-effort)
            self._site_trigger_mode_cache.pop(key, None)

    async def _get_site_name(self, site_uuid: Optional[str]) -> str:
        if not site_uuid:
            return "Unknown Site"

        key = str(site_uuid)
        now = time.monotonic()

        async with self._site_cache_lock:
            cached = self._site_cache.get(key)
            if cached and (now - cached[1]) < 300.0:  # 5 min TTL
                return cached[0]

        sf = self._session_factory
        if sf is None:
            return "Unknown Site"

        try:
            su = UUID(str(site_uuid))
        except Exception:
            return "Unknown Site"

        try:
            async with sf() as db:
                res = await db.execute(select(Site.name).where(Site.site_uuid == su))
                name = res.scalar_one_or_none() or "Unknown Site"
        except Exception:
            logger.exception("Failed to lookup site name for site_uuid=%s", site_uuid)
            name = "Unknown Site"

        async with self._site_cache_lock:
            self._site_cache[key] = (name, now)

        return name

    async def add_channel(self, ch: VideoChannel) -> None:
        key = ch.key()
        async with self._lock:
            self._channels[key] = ch
            started = self._started and (not self._closing)

        if started:
            self._ensure_poller(key, ch)

    async def edit_channel(self, ch: VideoChannel) -> None:
        """
        Replace the channel (e.g., device_url changed).
        Preserves webrtc_url immutability by refusing changes if attempted.
        """
        key = ch.key()
        async with self._lock:
            old = self._channels.get(key)

            if old is not None:
                old_cfg = getattr(old, "config", None)
                new_cfg = getattr(ch, "config", None)
                if old_cfg is not None and new_cfg is not None:
                    if getattr(old_cfg, "webrtc_url", None) != getattr(new_cfg, "webrtc_url", None):
                        raise ValueError("webrtc_url is immutable; do not change it on edit.")

            self._channels[key] = ch

            old_task = self._poll_tasks.pop(key, None)
            started = self._started and (not self._closing)

        if old_task and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Old poller failed during edit camera=%s", key)

        if started:
            self._ensure_poller(key, ch)

    async def remove_channel(self, camera_uuid: str) -> bool:
        key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.pop(key, None)
            t = self._poll_tasks.pop(key, None)
            self._last_seq.pop(key, None)
            self._last_seen.pop(key, None)
            self._last_ok_s.pop(key, None)
            self._last_detection_summary_s.pop(key, None)
        if isinstance(self.detect_store, InMemoryObjDetectStore):
            await self.detect_store.forget(key)
        async with self._cam_ctx_cache_lock:
            self._cam_ctx_cache.pop(key, None)
            future = self._cam_ctx_inflight.pop(key, None)
            if future is not None and not future.done():
                future.cancel()
        site_uuid = getattr(getattr(ch, "config", None), "site_uuid", None) if ch is not None else None
        if site_uuid is not None:
            async with self._site_cache_lock:
                self._site_cache.pop(str(site_uuid), None)
        self._tracker.remove_camera(key)
        self._roi_engine.reset_camera(key)


        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Poller failed during remove camera=%s", key)

        return True

    def list_channel_ids(self) -> List[str]:
        return list(self._channels.keys())

    async def get_channel_config(self, camera_uuid: str) -> Optional[VideoChannelConfig]:
        key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.get(key)
            return getattr(ch, "config", None) if ch else None

    async def patch_channel_config(self, camera_uuid: str, patch: dict) -> bool:
        """
        Patch runtime config (NO DB write here).

        Rules:
        - webrtc_url cannot be changed
        - camera_uuid cannot be changed
        """
        key = str(camera_uuid)
        patch = patch or {}

        if "webrtc_url" in patch:
            # ignore or reject; I’m rejecting to prevent hidden bugs
            raise ValueError("webrtc_url is immutable; remove it from patch.")
        if "camera_uuid" in patch:
            raise ValueError("camera_uuid cannot be patched.")

        async with self._lock:
            ch = self._channels.get(key)
            if ch is None:
                return False
            cfg = ch.config

            data = {
                "camera_uuid": cfg.camera_uuid,
                "rtsp_url": cfg.rtsp_url,
                "webrtc_url": cfg.webrtc_url,
                "site_uuid": cfg.site_uuid,
                "device_uuid": cfg.device_uuid,
                "device_url": cfg.device_url,
                "enabled": cfg.enabled,
                "poll_interval_ms": cfg.poll_interval_ms,
                "request_timeout_s": cfg.request_timeout_s,
                "detection_path_template": cfg.detection_path_template,
            }

            for k, v in patch.items():
                if v is not None and k in data:
                    data[k] = v

            new_cfg = VideoChannelConfig(**data)
            self._channels[key] = VideoChannel(new_cfg)

            # restart poller if running
            old_task = self._poll_tasks.pop(key, None)
            started = self._started and (not self._closing)

        if old_task and not old_task.done():
            old_task.cancel()
            try:
                await old_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Old poller failed during patch camera=%s", key)

        if started:
            async with self._lock:
                ch2 = self._channels.get(key)
            if ch2 is not None:
                self._ensure_poller(key, ch2)

        return True

    async def put_detection(self, resp: ObjDetectResponse) -> None:
        await self.detect_store.put(resp)

    async def get_latest_detection(self, camera_uuid: str) -> Optional[ObjDetectResponse]:
        return await self.detect_store.get_latest(str(camera_uuid))

    async def refresh_detection_once(self, camera_uuid: str) -> Optional[ObjDetectResponse]:
        """
        On-demand pull from Jetson once (useful if the cache is empty).
        """
        key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.get(key)
        if ch is None:
            return None

        sem = self._device_fetch_semaphore(getattr(ch.config, "device_url", None))
        async with sem:
            payload = await ch.fetch_detection_json()
        resp = self._payload_to_resp(payload, ch)
        if resp is None:
            return None

        if not self._is_new_detection(key, resp):
            return await self.detect_store.get_latest(key)

        now = time.monotonic()
        self._last_seen[key] = (int(resp.frame_ts_ms), int(resp.frame_seq))
        self._last_ok_s[key] = now
        self._last_seq[key] = int(resp.frame_seq)  # optional compat

        await self.detect_store.put(resp)
        return resp

    async def _get_camera_ctx(self, camera_uuid: str) -> Optional[CameraContext]:
        """
        Resolve user/site/device/camera names via NotificationRepository with TTL cache.
        """
        sf = self._session_factory
        if sf is None:
            return None

        key = str(camera_uuid)
        now = time.monotonic()
        leader = False
        pending: Optional[asyncio.Future] = None

        async with self._cam_ctx_cache_lock:
            cached = self._cam_ctx_cache.get(key)
            if cached and cached[1] > now:
                return cached[0]
            pending = self._cam_ctx_inflight.get(key)
            if pending is None:
                pending = asyncio.get_running_loop().create_future()
                self._cam_ctx_inflight[key] = pending
                leader = True

        try:
            cam_uuid = UUID(key)
        except Exception:
            if leader:
                async with self._cam_ctx_cache_lock:
                    future = self._cam_ctx_inflight.pop(key, None)
                    if future is not None and not future.done():
                        future.set_result(None)
            return None

        if not leader and pending is not None:
            return await pending

        ctx: Optional[CameraContext] = None
        cancelled = False
        try:
            async with self._cam_ctx_lookup_limit:
                async with sf() as db:
                    ctx = await self._notif_repo.get_camera_context(db, camera_uuid=cam_uuid)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("Failed to load CameraContext for camera=%s", key)
            ctx = None
        finally:
            async with self._cam_ctx_cache_lock:
                if not cancelled:
                    ttl_s = self._cam_ctx_ttl_s if ctx else self._cam_ctx_miss_ttl_s
                    if ttl_s > 0.0:
                        self._cam_ctx_cache[key] = (ctx, now + ttl_s)
                    else:
                        self._cam_ctx_cache.pop(key, None)
                future = self._cam_ctx_inflight.pop(key, None)
                if future is not None and not future.done():
                    if cancelled:
                        future.cancel()
                    else:
                        future.set_result(ctx)

        return ctx

    async def _persist_and_maybe_email(
        self,
        *,
        ctx: CameraContext,
        msg: NotificationMessage,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        svc = self._notification_service
        if svc is None:
            return
        await svc.enqueue_notification(msg, ctx, extra_payload=extra_payload)

    def _playback_enabled_for_channel(self, ch: VideoChannel) -> bool:
        cfg = getattr(ch, "config", None)
        return getattr(cfg, "enabled", True) is not False

    def _notifications_allowed_now(self, ch: VideoChannel) -> bool:
        cfg = getattr(ch, "config", None)
        if cfg is None:
            return True

        if getattr(cfg, "notification_enabled", True) is False:
            return False

        is_scheduled_now = getattr(cfg, "is_scheduled_now", None)
        if callable(is_scheduled_now):
            try:
                return bool(is_scheduled_now())
            except Exception:
                logger.exception("Failed to evaluate notification schedule camera=%s", ch.key())

        return True

    def _interesting_detection_classes(
        self,
        resp: ObjDetectResponse,
        svc: NotificationService,
    ) -> List[str]:
        interesting = getattr(svc, "interesting", None)
        cls_names: List[str] = []

        for det in list(resp.detections or ()):
            if not isinstance(det, dict):
                continue
            cls_name = str(det.get("cls_name") or "").strip()
            if not cls_name:
                continue
            if interesting and cls_name not in interesting:
                continue
            cls_names.append(cls_name)

        return sorted(set(cls_names))

    async def _process_detection_payload(self, key: str, ch: VideoChannel, payload: Dict[str, Any]) -> bool:
        resp = self._payload_to_resp(payload, ch)
        if resp is None:
            return False

        if not self._is_new_detection(key, resp):
            return False

        tracker_payload = dict(payload or {})
        tracker_payload["camera_uuid"] = str(resp.camera_uuid)
        tracker_out = self._tracker.update_from_event(tracker_payload)
        tracks = tuple(tracker_out.get("tracks", []) or [])
        track_events = tuple(tracker_out.get("events", []) or [])

        alerts: List[Dict[str, Any]] = []
        if resp.frame_w and resp.frame_h and tracks:
            rois = await self._roi_provider(str(resp.camera_uuid))
            alerts = self._roi_engine.process(
                camera_uuid=str(resp.camera_uuid),
                frame_w=int(resp.frame_w),
                frame_h=int(resp.frame_h),
                tracks=list(tracks),
                rois=rois,
                ts_ms=int(resp.frame_ts_ms),
            )

        resp2 = replace(
            resp,
            tracks=tracks,
            track_events=track_events,
            alerts=tuple(alerts),
        )

        now = time.monotonic()
        self._last_seen[key] = (int(resp2.frame_ts_ms), int(resp2.frame_seq))
        self._last_ok_s[key] = now
        self._last_seq[key] = int(resp2.frame_seq)

        await self.detect_store.put(resp2)
        await self.detection_hub.publish(resp2)

        svc = self._notification_service
        if svc is None:
            return True

        cam_uuid = str(resp2.camera_uuid)

        _cam_cfg = getattr(ch, "config", None)
        _cam_playback_override = getattr(_cam_cfg, "camera_playback_enabled", None)
        _do_playback = (
            _cam_playback_override is True
            or (
                _cam_playback_override is None
                and self._playback_enabled_for_channel(ch)
            )
        )
        if _do_playback:
            try:
                _prerecord_ok = (
                    _cam_playback_override is True          # explicit override bypasses site list
                    or await svc.is_camera_prerecord_eligible(cam_uuid)
                )
                if _prerecord_ok:
                    overlay_payload = _overlay_payload_from_resp(resp2)
                    await svc.record_detection_overlay_frame(
                        camera_uuid=cam_uuid,
                        frame_ts_ms=overlay_payload.get("frame_ts_ms"),
                        frame_seq=overlay_payload.get("frame_seq"),
                        frame_w=overlay_payload.get("frame_w"),
                        frame_h=overlay_payload.get("frame_h"),
                        detections=list(overlay_payload.get("detections") or []),
                    )
            except Exception:
                logger.exception("Failed to record detection overlay frame camera=%s", cam_uuid)

        if not self._notifications_allowed_now(ch):
            return True

        _cam_trigger_mode = getattr(_cam_cfg, "notification_trigger_mode", None)
        if _cam_trigger_mode is not None:
            allow_broad_notifications = (str(_cam_trigger_mode) == "any_detection")
        else:
            site_uuid_for_trigger = getattr(ch.config, "site_uuid", None)
            site_trigger_mode = await self._get_site_trigger_mode(
                str(site_uuid_for_trigger) if site_uuid_for_trigger else None
            )
            allow_broad_notifications = (site_trigger_mode == "any_detection")
        extra_payload: Optional[Dict[str, Any]] = None
        extra_payload_loaded = False

        async def _ensure_alert_extra_payload() -> Optional[Dict[str, Any]]:
            nonlocal extra_payload, extra_payload_loaded
            if not extra_payload_loaded:
                extra_payload = await self._build_alert_extra_payload(resp=resp2, ch=ch)
                extra_payload_loaded = True
            return extra_payload

        emitted_detail = False

        if allow_broad_notifications and getattr(svc, "notify_on_confirmed", False) and track_events:
            try:
                emitted_detail = (
                    await self._emit_item_detected_notifications(
                        resp2,
                        tracks,
                        track_events,
                        extra_payload=await _ensure_alert_extra_payload(),
                    )
                ) or emitted_detail
            except Exception:
                logger.exception("Failed to emit item-detected notifications camera=%s", cam_uuid)

        if getattr(svc, "notify_on_roi_enter", True) and alerts:
            try:
                emitted_detail = (
                    await self._emit_roi_alert_notifications(
                        resp2,
                        alerts,
                        extra_payload=await _ensure_alert_extra_payload(),
                    )
                ) or emitted_detail
            except Exception:
                logger.exception("Failed to emit ROI notifications camera=%s", cam_uuid)

        if not emitted_detail and allow_broad_notifications:
            summary_classes = self._interesting_detection_classes(resp2, svc)
            if summary_classes and self._reserve_detection_summary_alert(cam_uuid, summary_classes):
                try:
                    await self._emit_detection_summary_notification(
                        resp2,
                        extra_payload=await _ensure_alert_extra_payload(),
                    )
                except Exception:
                    logger.exception("Failed to emit detection-summary notification camera=%s", cam_uuid)

        return True

    def _device_fetch_semaphore(self, device_url: Optional[str]) -> asyncio.Semaphore:
        key = str(device_url or "").strip().lower()
        sem = self._device_fetch_limits.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrent_fetch_per_device)
            self._device_fetch_limits[key] = sem
        return sem

    def _reserve_detection_summary_alert(self, camera_uuid: str, cls_names: Optional[List[str]] = None) -> bool:
        """
        Check if detection alert should be emitted using per-class cooldown.
        Uses (camera_uuid, cls_name) tuple instead of just camera_uuid.
        This allows different object classes to bypass each other's cooldown.
        """
        cam = str(camera_uuid)
        cooldown_s = float(self._detection_summary_cooldown_s or 0.0)
        
        if not cls_names:
            # Fallback to old behavior if no classes provided
            now = time.monotonic()
            last = self._last_detection_summary_s.get(cam, 0.0)
            if cooldown_s > 0.0 and (now - last) < cooldown_s:
                return False
            self._last_detection_summary_s[cam] = now
            return True
        
        now = time.monotonic()
        
        # Check each class independently
        for cls_name in cls_names:
            key = (cam, str(cls_name))  # (camera_uuid, class_name) tuple
            last = self._last_detection_summary_s.get(key, 0.0)
            
            if cooldown_s > 0.0 and (now - last) < cooldown_s:
                # This class was seen recently, skip it
                continue
            
            # Class is not in cooldown, update and allow alert
            self._last_detection_summary_s[key] = now
            return True
        
        # All classes are in cooldown
        return False

    def _image_bytes_to_data_url(self, payload: bytes, content_type: Optional[str]) -> Optional[str]:
        if not payload:
            return None

        mime = str(content_type or "image/jpeg").split(";", 1)[0].strip().lower() or "image/jpeg"
        if not mime.startswith("image/"):
            mime = "image/jpeg"

        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def _build_alert_extra_payload(
        self,
        *,
        resp: ObjDetectResponse,
        ch: VideoChannel,
    ) -> Optional[Dict[str, Any]]:
        image_url = str(getattr(resp, "image_url", "") or "").strip()
        if image_url:
            return {"image_url": image_url}

        try:
            sem = self._device_fetch_semaphore(getattr(ch.config, "device_url", None))
            async with sem:
                snapshot = await ch.fetch_snapshot_bytes()
        except Exception:
            logger.exception("Failed to fetch alert snapshot camera=%s", resp.camera_uuid)
            return None

        if not snapshot:
            return None

        payload, content_type = snapshot
        data_url = self._image_bytes_to_data_url(payload, content_type)
        if not data_url:
            return None

        return {"image_url": data_url}

    async def _emit_roi_alert_notifications(
        self,
        resp: ObjDetectResponse,
        alerts: List[Dict[str, Any]],
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        svc = self._notification_service
        if svc is None:
            return False

        cam_uuid = str(resp.camera_uuid)

        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning("Skipping ROI alert publish because camera context was not found camera=%s", cam_uuid)
            return False

        site_name = ctx.site_name
        site_uuid_str = str(ctx.site_uuid)
        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        emitted = False

        for a in alerts:
            overlay_payload = _overlay_payload_from_resp(resp, fallback_detections=[a])
            title = f"ROI Alert ({a.get('type','roi')})"
            body = f"{a.get('cls_name','object')} entered ROI {a.get('roi_id')} (track {a.get('track_id')})"

            raw_track_id = a.get("track_id")
            try:
                track_id = int(raw_track_id) if raw_track_id is not None else None
            except Exception:
                track_id = None

            msg = NotificationMessage(
                user_id=int(ctx.user_id),
                id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-{a.get('roi_id')}-{a.get('track_id')}",
                ts_ms=int(resp.frame_ts_ms),
                camera_uuid=cam_uuid,
                site_uuid=site_uuid_str,
                site_name=site_name,
                title=title,
                body=body,
                alert_type="roi_enter",
                cls_names=[str(a.get("cls_name", "object"))],
                max_conf=float(a.get("conf", 0.0) or 0.0),
                frame_w=overlay_payload.get("frame_w"),
                frame_h=overlay_payload.get("frame_h"),
                frame_seq=overlay_payload.get("frame_seq"),
                detections=list(overlay_payload.get("detections") or []),
                roi_id=str(a.get("roi_id", "")),
                track_id=track_id,
                device_name=ctx.device_name if ctx else None,
                camera_name=ctx.camera_name if ctx else None,
                image_url=image_url,
            )

            await svc.hub.publish(msg)
            emitted = True

            persist_payload = {
                **overlay_payload,
                **(extra_payload or {}),
            }
            persist_payload["alert"] = a
            asyncio.create_task(
                self._persist_and_maybe_email(ctx=ctx, msg=msg, extra_payload=persist_payload),
                name=f"persist_roi_alert:{cam_uuid}",
            )
        
        return emitted

    async def _emit_item_detected_notifications(
        self,
        resp: ObjDetectResponse,
        tracks: Tuple[Dict[str, Any], ...],
        track_events: Tuple[Tuple[str, int], ...],
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        svc = self._notification_service
        if svc is None:
            return False

        confirmed_track_ids = [int(track_id) for (ev_type, track_id) in track_events if ev_type == "track_confirmed"]
        if not confirmed_track_ids:
            return False

        tracks_by_id: Dict[int, Dict[str, Any]] = {}
        for tr in tracks:
            try:
                tracks_by_id[int(tr.get("track_id"))] = tr
            except Exception:
                continue

        cam_uuid = str(resp.camera_uuid)
        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning("Skipping item-detected alert publish because camera context was not found camera=%s", cam_uuid)
            return False

        site_name = ctx.site_name
        site_uuid_str = str(ctx.site_uuid)
        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        emitted = False

        for track_id in confirmed_track_ids:
            tr = tracks_by_id.get(track_id)
            if tr is None:
                continue

            overlay_payload = _overlay_payload_from_resp(resp, fallback_detections=[tr])
            cls_name = str(tr.get("cls_name") or "object")
            conf = float(tr.get("conf", 0.0) or 0.0)

            msg = NotificationMessage(
                user_id=int(ctx.user_id),
                id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-track-{track_id}",
                ts_ms=int(resp.frame_ts_ms),
                camera_uuid=cam_uuid,
                site_uuid=site_uuid_str,
                site_name=site_name,
                title=f"Item Detected: {cls_name}",
                body=f"{cls_name} confirmed (track_id={track_id}, conf={conf:.2f})",
                alert_type="item_detected",
                cls_names=[cls_name],
                max_conf=conf,
                frame_w=overlay_payload.get("frame_w"),
                frame_h=overlay_payload.get("frame_h"),
                frame_seq=overlay_payload.get("frame_seq"),
                detections=list(overlay_payload.get("detections") or []),
                track_id=track_id,
                device_name=ctx.device_name if ctx else None,
                camera_name=ctx.camera_name if ctx else None,
                image_url=image_url,
            )

            await svc.hub.publish(msg)
            emitted = True

            persist_payload = {
                **overlay_payload,
                **(extra_payload or {}),
            }
            persist_payload["track"] = tr
            persist_payload["event"] = "track_confirmed"
            asyncio.create_task(
                self._persist_and_maybe_email(
                    ctx=ctx,
                    msg=msg,
                    extra_payload=persist_payload,
                ),
                name=f"persist_track_confirmed:{cam_uuid}:{track_id}",
            )

        return emitted

    async def _emit_detection_summary_notification(
        self,
        resp: ObjDetectResponse,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        svc = self._notification_service
        if svc is None:
            return False

        cam_uuid = str(resp.camera_uuid)
        raw_detections = list(resp.detections or [])
        if not raw_detections:
            return False

        interesting = getattr(svc, "interesting", None)
        filtered: List[Dict[str, Any]] = []
        classes: List[str] = []
        max_conf = 0.0
        for d in raw_detections:
            if not isinstance(d, dict):
                continue
            cls_name = str(d.get("cls_name") or "").strip()
            if not cls_name:
                continue
            if interesting and cls_name not in interesting:
                continue
            conf = float(d.get("conf", 0.0) or 0.0)
            max_conf = max(max_conf, conf)
            classes.append(cls_name)
            filtered.append(d)

        if not filtered:
            return False

        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning("Skipping detection-summary alert because camera context was not found camera=%s", cam_uuid)
            return False

        uniq_classes = sorted(set(classes))
        classes_text = ", ".join(uniq_classes)
        _verb = "is" if len(uniq_classes) == 1 else "are"
        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        overlay_payload = _overlay_payload_from_resp(resp)
        msg = NotificationMessage(
            user_id=int(ctx.user_id),
            id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-summary",
            ts_ms=int(resp.frame_ts_ms),
            camera_uuid=cam_uuid,
            site_uuid=str(ctx.site_uuid),
            site_name=ctx.site_name,
            title=f"Detection: {classes_text}",
            body=f"{int(max_conf * 100)}% chance that {classes_text} {_verb} being detected",
            alert_type="detection_summary",
            cls_names=uniq_classes,
            max_conf=float(max_conf),
            frame_w=overlay_payload.get("frame_w"),
            frame_h=overlay_payload.get("frame_h"),
            frame_seq=overlay_payload.get("frame_seq"),
            detections=list(overlay_payload.get("detections") or []),
            device_name=ctx.device_name,
            camera_name=ctx.camera_name,
            image_url=image_url,
        )

        await svc.hub.publish(msg)

        persist_payload = {
            **overlay_payload,
            **(extra_payload or {}),
        }
        persist_payload["detections"] = filtered[:20]
        persist_payload["event"] = "detection_summary"
        asyncio.create_task(
            self._persist_and_maybe_email(
                ctx=ctx,
                msg=msg,
                extra_payload=persist_payload,
            ),
            name=f"persist_detection_summary:{cam_uuid}",
        )
        return True
        
    async def _stream_loop(self, key: str, ch: VideoChannel) -> None:
        backoff_s = 0.5
        max_backoff_s = 5.0

        if self._startup_jitter_ms > 0:
            await asyncio.sleep(random.uniform(0.0, self._startup_jitter_ms / 1000.0))

        while True:
            async with self._lock:
                if self._closing or (not self._started):
                    return
                cur = self._channels.get(key)
                if cur is None:
                    return
                ch = cur

            cfg = ch.config
            if getattr(cfg, "detection_enabled", True) is False:
                await asyncio.sleep(0.5)
                continue

            try:
                got_any = False
                async for payload in ch.stream_detections():
                    got_any = True
                    await self._process_detection_payload(key, ch, payload)

                    async with self._lock:
                        if self._closing or (not self._started) or (self._channels.get(key) is None):
                            return

                # normal stream end -> reconnect
                if got_any:
                    backoff_s = 0.5

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Detection stream loop failed camera=%s", key)

            await asyncio.sleep(backoff_s)
            backoff_s = min(max_backoff_s, backoff_s * 2.0)
    def _ensure_poller(self, key: str, ch: VideoChannel) -> None:
        t = self._poll_tasks.get(key)
        if t is None or t.done():
            self._poll_tasks[key] = asyncio.create_task(
                self._stream_loop(key, ch),
                name="stream_jetson_detection:%s" % key,
            )
    def _payload_to_resp(self, payload: Optional[Dict[str, Any]], ch: VideoChannel) -> Optional[ObjDetectResponse]:
        if not payload or not isinstance(payload, dict):
            return None

        if "frame_seq" not in payload and "payload" in payload and isinstance(payload["payload"], dict):
            payload = payload["payload"]

        frame_seq = payload.get("frame_seq")
        frame_ts_ms = payload.get("frame_ts_ms")
        if frame_seq is None or frame_ts_ms is None:
            return None

        dets = payload.get("detections") or []
        pose = payload.get("pose")
        inf_ms = payload.get("inference_ms")
        model_id = payload.get("model_id")
        image_url = payload.get("image_url")
        event_type = payload.get("type")
        reason = payload.get("reason")

        frame_w = payload.get("frame_w") or payload.get("image_w") or payload.get("width")
        frame_h = payload.get("frame_h") or payload.get("image_h") or payload.get("height")

        expected_cam = str(ch.config.camera_uuid)
        payload_cam = payload.get("camera_uuid")
        if payload_cam is not None and str(payload_cam) != expected_cam:
            logger.warning(
                "Dropping detection payload with mismatched camera_uuid expected=%s got=%s",
                expected_cam,
                payload_cam,
            )
            return None

        cam = expected_cam

        return ObjDetectResponse(
            camera_uuid=cam,
            frame_ts_ms=int(frame_ts_ms),
            frame_seq=int(frame_seq),
            event_type=str(event_type) if event_type is not None else None,
            reason=str(reason) if reason is not None else None,
            frame_w=int(frame_w) if frame_w is not None else None,
            frame_h=int(frame_h) if frame_h is not None else None,
            site_uuid=ch.config.site_uuid,
            device_uuid=ch.config.device_uuid,
            detections=tuple(dets) if isinstance(dets, list) else (),
            pose=pose,
            inference_ms=int(inf_ms) if inf_ms is not None else None,
            model_id=str(model_id) if model_id is not None else None,
            image_url=str(image_url).strip() if image_url is not None else None,
        )
