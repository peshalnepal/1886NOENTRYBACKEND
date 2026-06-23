# application/services/pipeline.py
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
import uuid as _uuid
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from application.channels.channel import VideoChannel, VideoChannelConfig
from application.services.tracker import MultiCameraByteTrack, ROI, ROIAlertEngine
from application.services.notification import NotificationMessage, NotificationService
from application.repositories.notification_repository import CameraContext
from application.services.common import (
    CameraContextResolver,
    SiteArmStateResolver,
    SiteTriggerModeResolver,
)


logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]
TaskSpawner = Callable[[Awaitable[Any], str], asyncio.Task]
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


from application.services.overlay_normalize import (
    _append_overlay_detection,
)


def _humanize_roi_alert(cls_name: Any, ctx: CameraContext) -> Tuple[str, str]:
    """Build a human-readable title/body for an ROI-enter alert.

    Replaces the old debug-style ``"car entered ROI <uuid>-roi (track 112809)"``
    with operator-friendly copy like ``"Car entered the restricted zone at
    Front Gate · Main Site"``. The class word is preserved in the body so the
    downstream object-name inference (regex over the body) still works, and the
    location is drawn from the resolved camera/site names rather than raw UUIDs.
    The timestamp is rendered by the client from ``ts_ms``, so it is not
    duplicated in the text.
    """
    label = (str(cls_name or "object").strip() or "object")
    label = label[:1].upper() + label[1:]

    loc_parts = [p for p in (getattr(ctx, "camera_name", None), getattr(ctx, "site_name", None)) if p]
    location = ", ".join(loc_parts) if loc_parts else "the monitored area"

    title = f"{label} entered restricted zone"
    body = f"{label} entered the restricted zone at {location}"
    return title, body


def _humanize_item_detected(cls_name: Any, conf: float, ctx: CameraContext) -> Tuple[str, str]:
    """Build a human-readable title/body for a confirmed-detection alert.

    Replaces the old ``"car confirmed (track_id=112809, conf=0.87)"`` debug
    string with operator-friendly copy. Keeps the class word in the body so the
    downstream object-name inference still works.
    """
    label = (str(cls_name or "object").strip() or "object")
    label = label[:1].upper() + label[1:]

    loc_parts = [p for p in (getattr(ctx, "camera_name", None), getattr(ctx, "site_name", None)) if p]
    location = ", ".join(loc_parts) if loc_parts else "the monitored area"

    try:
        pct = max(0, min(100, int(round(float(conf) * 100))))
    except (TypeError, ValueError):
        pct = 0

    title = f"{label} detected"
    body = f"{label} detected at {location} ({pct}% confidence)"
    return title, body


def _live_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter raw tracker output down to the tracks that should actually be drawn
    on the live overlay (and recorded for playback).

    The tracker keeps "coasting" tracks alive for several frames after they stop
    matching a detection (occlusion / flicker tolerance). Those tracks have no
    real detection backing them on the current frame — their bbox is pure
    velocity extrapolation that, for fast objects, freezes in place and leaves a
    trail of stale ghost boxes behind the moving object.

    For rendering we only want tracks that:
      - were updated by a detection THIS frame (``misses == 0``), and
      - are confirmed (survived ``min_hits``), so a single spurious detection
        does not flash an ID'd box for one frame.

    The raw Jetson detections are still drawn separately, so a real object is
    never hidden by this filter — it only suppresses the ID'd ghost boxes.
    """
    out: List[Dict[str, Any]] = []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        if not t.get("confirmed", False):
            continue
        try:
            if int(t.get("misses", 0) or 0) > 0:
                continue
        except (TypeError, ValueError):
            continue
        out.append(t)
    return out


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

    try:
        frame_w = int(resp.frame_w)
    except (TypeError, ValueError):
        frame_w = None
    try:
        frame_h = int(resp.frame_h)
    except (TypeError, ValueError):
        frame_h = None

    return {
        "frame_ts_ms": int(resp.frame_ts_ms),
        "frame_seq": int(resp.frame_seq),
        "frame_w": frame_w,
        "frame_h": frame_h,
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
        notification_service: Optional[NotificationService] = None,  # NEW
        tracker_cfg: Optional[dict] = None,
        interesting_classes: Optional[Set[str]] = None,
        notify_on_confirmed: bool = False,
        notify_on_roi_enter: bool = True,
        task_spawner: Optional[TaskSpawner] = None,
    ):
        self.pipeline_id = pipeline_id
        self.detect_store = detect_store or InMemoryObjDetectStore()
        self.detection_hub = DetectionHub()
        self._last_seen: Dict[str, Tuple[int, int]] = {}
        self._last_ok_s: Dict[str, float] = {}
        # Clock-skew monitor: the Jetson stamps frame_ts_ms with its own wall
        # clock, and the clip window is sliced out of MediaMTX recordings stamped
        # with the Azure clock. If the Jetson clock drifts from UTC, every clip is
        # shifted and the event falls outside the captured window. We can't fix the
        # Jetson clock from here, but we CAN detect the drift: the Azure clock ≈ the
        # MediaMTX clock (both Azure/NTP), so (azure_now - frame_ts_ms) on arrival
        # is the one-way lag; its rolling floor approximates the clock offset.
        self._clock_skew_floor_ms: Dict[str, float] = {}
        self._clock_skew_floor_reset_s: Dict[str, float] = {}
        self._clock_skew_last_warn_s: Dict[str, float] = {}
        self._channels: Dict[str, VideoChannel] = {}
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._last_seq: Dict[str, int] = {}
        
        cfg = tracker_cfg or {}
        self._tracker = MultiCameraByteTrack(**cfg)
        self._roi_engine = ROIAlertEngine()
        
        self.interesting_classes = interesting_classes or {"person", "car", "motorcycle", "truck"}
        self.notify_on_confirmed = notify_on_confirmed
        self.notify_on_roi_enter = notify_on_roi_enter
        
        self._roi_cache: Dict[str, Tuple[float, List[ROI]]] = {}
        self._roi_cache_ttl_s = max(
            0.0,
            float(os.getenv("CAMERA_ROI_CACHE_TTL_S", "30.0")),
        )
        self._notification_service = notification_service
        self._last_detection_summary_s: Dict[str, float] = {}
        self._detection_summary_cooldown_s = max(
            0.0,
            float(os.getenv("DETECTION_ALERT_COOLDOWN_S", "8.0")),
        )
        self._ctx_resolver = CameraContextResolver(
            hit_ttl_s=60.0,
            miss_ttl_s=float(os.getenv("CAMERA_CONTEXT_MISS_CACHE_TTL_S", "5.0")),
            max_concurrent_lookups=int(
                os.getenv("CAMERA_CONTEXT_MAX_CONCURRENT_DB_LOOKUPS", "4")
            ),
        )
        self._trigger_mode_resolver = SiteTriggerModeResolver(
            default="roi_enter",
            ttl_s=float(os.getenv("SITE_TRIGGER_MODE_CACHE_TTL_S", "30.0")),
        )
        self._arm_state_resolver = SiteArmStateResolver(
            ttl_s=float(os.getenv("SITE_ARM_STATE_CACHE_TTL_S", "10.0")),
        )
        self._session_factory: Optional[SessionFactory] = None
        self._device_fetch_limits: Dict[str, asyncio.Semaphore] = {}
        self._max_concurrent_fetch_per_device = max(
            1,
            int(os.getenv("DETECTION_MAX_CONCURRENT_FETCH_PER_DEVICE", "8")),
        )
        self._startup_jitter_ms = max(
            0,
            int(os.getenv("DETECTION_STARTUP_JITTER_MS", "100")),
        )
        # Warn when the estimated Jetson↔Azure clock offset exceeds this many ms.
        # The clip window absorbs sub-second offsets; multi-second skew shifts the
        # event out of the PRE/POST window, so default to 3s. Set 0 to disable.
        self._clock_skew_warn_ms = max(
            0,
            int(os.getenv("DETECTION_CLOCK_SKEW_WARN_MS", "3000")),
        )
        self._clock_skew_floor_window_s = max(
            30.0,
            float(os.getenv("DETECTION_CLOCK_SKEW_FLOOR_WINDOW_S", "300.0")),
        )
        self._clock_skew_warn_interval_s = max(
            5.0,
            float(os.getenv("DETECTION_CLOCK_SKEW_WARN_INTERVAL_S", "120.0")),
        )
        self._lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._task_spawner: TaskSpawner = task_spawner or (
            lambda coro, name: asyncio.create_task(coro, name=name)
        )

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

    def _monitor_clock_skew(self, key: str, frame_ts_ms: int) -> None:
        """Estimate and warn on Jetson↔Azure clock drift (warn-only, no behavior change).

        ``lag = azure_now_ms - frame_ts_ms`` is the one-way delay between the
        Jetson stamping a frame and Azure receiving the detection. Real lag is
        always positive and bounded below by the network/processing latency, so
        the *minimum* lag over a window is a good estimate of the constant clock
        offset: a near-zero/positive floor means the clocks agree; a large
        positive floor means the Jetson clock is BEHIND Azure (and the clip window
        will land in the past); a negative floor means the Jetson clock is AHEAD
        (the frame is stamped in the "future", which is physically impossible
        without skew, so it is a definitive signal). The floor is reset on a slow
        window so a corrected clock recovers instead of latching forever.
        """
        if self._clock_skew_warn_ms <= 0:
            return

        now_s = time.time()
        lag_ms = (now_s * 1000.0) - float(frame_ts_ms)

        reset_at = self._clock_skew_floor_reset_s.get(key, 0.0)
        floor = self._clock_skew_floor_ms.get(key)
        if floor is None or now_s >= reset_at:
            self._clock_skew_floor_ms[key] = lag_ms
            self._clock_skew_floor_reset_s[key] = now_s + self._clock_skew_floor_window_s
            floor = lag_ms
        elif lag_ms < floor:
            self._clock_skew_floor_ms[key] = lag_ms
            floor = lag_ms

        # A healthy floor is a small positive number (one-way latency). Flag only
        # when it drifts beyond the threshold in either direction.
        if -self._clock_skew_warn_ms <= floor <= self._clock_skew_warn_ms:
            return

        last_warn = self._clock_skew_last_warn_s.get(key, 0.0)
        if (now_s - last_warn) < self._clock_skew_warn_interval_s:
            return
        self._clock_skew_last_warn_s[key] = now_s

        direction = "AHEAD of" if floor < 0 else "BEHIND"
        logger.warning(
            "Jetson clock appears skewed camera=%s estimated_offset_ms=%d (Jetson clock is %s Azure/MediaMTX). "
            "Clip windows are anchored on the Jetson wall clock; multi-second skew shifts the event out of the "
            "captured PRE/POST window. Verify NTP/time sync on the Jetson host.",
            key,
            int(floor),
            direction,
        )

    def set_notification_service(self, svc: NotificationService) -> None:
        self._notification_service = svc

    def set_session_factory(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory
        self._ctx_resolver.set_session_factory(session_factory)
        self._trigger_mode_resolver.set_session_factory(session_factory)
        self._arm_state_resolver.set_session_factory(session_factory)
        if self._notification_service is not None:
            try:
                self._notification_service.set_session_factory(session_factory)
            except Exception:
                logger.exception("Failed to set session factory on NotificationService")

    def invalidate_camera_roi_state(self, camera_uuid: str) -> None:
        """
        Clear per-camera ROI edge-trigger state so ROI edits take effect immediately.
        """
        cam = str(camera_uuid)
        self._roi_engine.reset_camera(cam)
        self._roi_cache.pop(cam, None)

    def invalidate_site_trigger_mode_cache(self, site_uuid: str) -> None:
        """Invalidate the cached trigger_mode for a site (after settings are saved)."""
        self._trigger_mode_resolver.invalidate(str(site_uuid))

    def invalidate_site_arm_state(self, site_uuid: str) -> None:
        """Invalidate the cached arm/disarm override for a site so a fresh
        arm/disarm action gates notifications immediately instead of after TTL."""
        self._arm_state_resolver.invalidate(str(site_uuid))

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

        self._ctx_resolver.invalidate(key)

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
            self._clock_skew_floor_ms.pop(key, None)
            self._clock_skew_floor_reset_s.pop(key, None)
            self._clock_skew_last_warn_s.pop(key, None)
        if isinstance(self.detect_store, InMemoryObjDetectStore):
            await self.detect_store.forget(key)
        self._ctx_resolver.invalidate(key)
        self._tracker.remove_camera(key)
        self._roi_engine.reset_camera(key)
        self._roi_cache.pop(key, None)


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
        """Resolve user/site/device/camera names for a camera (TTL-cached)."""
        return await self._ctx_resolver.resolve(camera_uuid)

    async def _publish_and_persist(
        self, *, ctx: CameraContext, msg: NotificationMessage, overlay_payload: Dict[str, Any], extra_payload: Optional[Dict[str, Any]], persist_extra: Optional[Dict[str, Any]], task_name: str
    ) -> None:
        svc = self._notification_service
        if svc is None:
            return
        # If the site's org has an operator, hold the alert for approval: persist
        # it (the flusher marks it pending/invisible) but do NOT push it to the
        # end user's realtime stream. With no operator, publish immediately.
        if not await svc.requires_operator_approval(ctx.site_uuid):
            await svc.hub.publish(msg)
        persist_payload = {**overlay_payload, **(extra_payload or {}), **(persist_extra or {})}
        self._task_spawner(
            svc.enqueue_notification(msg, ctx, extra_payload=persist_payload),
            task_name,
        )

    def is_channel_enable(self, ch: VideoChannel) -> bool:
        cfg = getattr(ch, "config", None)
        return getattr(cfg, "enabled", True) is not False

    async def _notifications_allowed_now(self, ch: VideoChannel) -> bool:
        cfg = getattr(ch, "config", None)
        if cfg is None:
            return True

        if getattr(cfg, "notification_enabled", True) is False:
            return False

        # A site-level arm/disarm override wins over the schedule for every
        # camera in the site (until it clears at the next schedule boundary). With
        # no active override we fall back to the camera's own schedule.
        site_uuid = getattr(cfg, "site_uuid", None)
        override = await self._arm_state_resolver.resolve(
            str(site_uuid) if site_uuid else None
        )
        if override is not None:
            return bool(override)

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
    ) -> List[str]:
        cls_names: List[str] = []

        for det in list(resp.detections or ()):
            if not isinstance(det, dict):
                continue
            cls_name = str(det.get("cls_name") or "").strip()
            if not cls_name:
                continue
            if self.interesting_classes and cls_name not in self.interesting_classes:
                continue
            cls_names.append(cls_name)

        return sorted(set(cls_names))

    async def _process_detection_payload(self, key: str, ch: VideoChannel, payload: Dict[str, Any]) -> bool:
        resp = self._payload_to_resp(payload, ch)
        if resp is None:
            return False

        if not self._is_new_detection(key, resp):
            return False

        self._monitor_clock_skew(key, int(resp.frame_ts_ms))

        tracker_payload = dict(payload or {})
        tracker_payload["camera_uuid"] = str(resp.camera_uuid)
        tracker_out = self._tracker.update_from_event(tracker_payload)
        tracks = tuple(tracker_out.get("tracks", []) or [])
        track_events = tuple(tracker_out.get("events", []) or [])

        alerts: List[Dict[str, Any]] = []
        if resp.frame_w and resp.frame_h and tracks:
            rois = await self._fetch_rois(str(resp.camera_uuid))
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

        # Publish for the live overlay first. Anything slow (clip recording,
        # snapshot fetch, notification persistence) must happen AFTER this so
        # the SSE stream to the frontend sees the freshest frame ASAP.
        await self.detect_store.put(resp2)
        await self.detection_hub.publish(resp2)

        svc = self._notification_service
        if svc is None:
            return True

        # Fire-and-forget the post-publish work. Previously this block was
        # awaited inline, which blocked the per-camera Jetson SSE consumer
        # (the `async for payload in ch.stream_detections()` loop) on every
        # frame and pushed the overlay several frames behind the live video.
        # Spawning a task lets the consumer immediately read the next frame.
        self._task_spawner(
            self._post_publish_work(resp2, ch),
            f"detection_post_publish:{key}",
        )

        return True

    async def _post_publish_work(
        self,
        resp2: ObjDetectResponse,
        ch: VideoChannel,
    ) -> None:
        """
        Runs off the stream-consumer hot path. Handles:
          - overlay-frame recording for clip playback
          - alert snapshot fetch (was the worst per-frame stall on alert frames)
          - ROI / item-detected / detection-summary notifications

        Any exception here is logged but never propagated, since it's detached
        from the caller.
        """
        try:
            svc = self._notification_service
            if svc is None:
                return

            cam_uuid = str(resp2.camera_uuid)
            _cam_cfg = getattr(ch, "config", None)

            # Compute the base overlay payload exactly once for the recorder.
            # The per-alert notification emitters still build their own overlays
            # because each one appends a different alert/track-specific detection
            # on top of the base.
            base_overlay: Optional[Dict[str, Any]] = None

            _cam_playback_override = str(getattr(_cam_cfg, "camera_playback_enabled", "inherit") or "inherit")
            _do_playback = (
                _cam_playback_override == "always" or _cam_playback_override == "inherit"
            ) and self.is_channel_enable(ch)
            if _do_playback:
                try:
                    _prerecord_ok = (
                        _cam_playback_override == "always"       # explicit override bypasses site list
                        or await svc.is_camera_prerecord_eligible(cam_uuid)
                    )
                    if _prerecord_ok:
                        base_overlay = _overlay_payload_from_resp(resp2, fallback_detections=_live_tracks(list(resp2.tracks)))
                        await svc.record_detection_overlay_frame(
                            camera_uuid=cam_uuid,
                            frame_ts_ms=base_overlay.get("frame_ts_ms"),
                            frame_seq=base_overlay.get("frame_seq"),
                            frame_w=base_overlay.get("frame_w"),
                            frame_h=base_overlay.get("frame_h"),
                            detections=list(base_overlay.get("detections") or []),
                        )
                except Exception:
                    logger.exception("Failed to record detection overlay frame camera=%s", cam_uuid)

            if not await self._notifications_allowed_now(ch):
                return

            _cam_trigger_mode = str(getattr(_cam_cfg, "notification_trigger_mode", "inherit") or "inherit")
            if _cam_trigger_mode in ("roi_enter", "any_detection"):
                allow_broad_notifications = (_cam_trigger_mode == "any_detection")
            else:
                site_uuid_for_trigger = getattr(ch.config, "site_uuid", None)
                site_trigger_mode = await self._trigger_mode_resolver.resolve(
                    str(site_uuid_for_trigger) if site_uuid_for_trigger else None
                )
                allow_broad_notifications = (site_trigger_mode == "any_detection")

            # _build_alert_extra_payload may HTTP-fetch a snapshot from Jetson
            # (hundreds of ms). It was previously awaited on the stream loop;
            # now it only runs here, off the hot path, and still lazily.
            extra_payload: Optional[Dict[str, Any]] = None
            extra_payload_loaded = False

            async def _ensure_alert_extra_payload() -> Optional[Dict[str, Any]]:
                nonlocal extra_payload, extra_payload_loaded
                if not extra_payload_loaded:
                    extra_payload = await self._build_alert_extra_payload(resp=resp2, ch=ch)
                    extra_payload_loaded = True
                return extra_payload

            tracks = resp2.tracks
            track_events = resp2.track_events
            alerts = list(resp2.alerts or ())
            emitted_detail = False

            # ROI entry is the most specific operator event. Emit it first and
            # suppress broader same-frame detection notices so one event does
            # not create two pending notifications for the operator.
            if self.notify_on_roi_enter and alerts:
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

            if not emitted_detail and allow_broad_notifications and self.notify_on_confirmed and track_events:
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

            if not emitted_detail and allow_broad_notifications:
                summary_classes = self._interesting_detection_classes(resp2)
                if summary_classes and self._reserve_detection_summary_alert(cam_uuid, summary_classes):
                    try:
                        await self._emit_detection_summary_notification(
                            resp2,
                            extra_payload=await _ensure_alert_extra_payload(),
                        )
                    except Exception:
                        logger.exception("Failed to emit detection-summary notification camera=%s", cam_uuid)
        except Exception:
            logger.exception("post-publish detection work failed camera=%s", str(resp2.camera_uuid))

    @staticmethod
    def _parse_roi_points(raw: Any) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        if not isinstance(raw, (list, tuple)):
            return out
        for p in raw:
            try:
                if isinstance(p, dict):
                    x = float(p.get("x"))
                    y = float(p.get("y"))
                elif isinstance(p, (list, tuple)) and len(p) >= 2:
                    x = float(p[0])
                    y = float(p[1])
                else:
                    continue
            except (TypeError, ValueError):
                continue
            out.append((x, y))
        return out

    @staticmethod
    def _coerce_roi_normalized(raw: Any, points: List[Tuple[float, float]]) -> bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in ("true", "1", "yes"):
                return True
            if s in ("false", "0", "no"):
                return False
        # Heuristic: all coords within [0, 1] => normalized
        if points and all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for (x, y) in points):
            return True
        return False

    @staticmethod
    def _parse_roi_frame_size(row: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        def _maybe_int(v: Any) -> Optional[int]:
            try:
                iv = int(v)
                return iv if iv > 0 else None
            except (TypeError, ValueError):
                return None
        return (
            _maybe_int(row.get("frame_w") or row.get("image_w") or row.get("width")),
            _maybe_int(row.get("frame_h") or row.get("image_h") or row.get("height")),
        )

    @staticmethod
    def _clamp_unit_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for (x, y) in points]

    async def _fetch_rois(self, camera_uuid: str) -> List[ROI]:
        if not self._session_factory:
            return []

        now = time.monotonic()
        hit = self._roi_cache.get(camera_uuid)
        if hit and hit[0] > now:
            return hit[1]

        try:
            from core.database_orm import Camera
            from sqlalchemy import select

            cam_uuid = _uuid.UUID(str(camera_uuid))

            async with self._session_factory() as session:
                result = await session.execute(
                    select(Camera.roi).where(Camera.camera_uuid == cam_uuid)
                )
                row = result.scalar_one_or_none()

            if not row or not isinstance(row, dict):
                rois: List[ROI] = []
                self._roi_cache[camera_uuid] = (now + self._roi_cache_ttl_s, rois)
                return rois

            points = self._parse_roi_points(row.get("points", []))
            normalized = self._coerce_roi_normalized(row.get("normalized"), points)
            roi_frame_w, roi_frame_h = self._parse_roi_frame_size(row)

            if not points or len(points) < 3:
                rois = []
                self._roi_cache[camera_uuid] = (now + self._roi_cache_ttl_s, rois)
                return rois

            roi_points = self._clamp_unit_points(points) if normalized else points
            rois = [
                ROI(
                    roi_id=f"{camera_uuid}-roi",
                    points=roi_points,
                    normalized=normalized,
                    frame_w=roi_frame_w,
                    frame_h=roi_frame_h,
                )
            ]
            self._roi_cache[camera_uuid] = (now + self._roi_cache_ttl_s, rois)
            return rois

        except Exception as e:
            stale = self._roi_cache.get(camera_uuid)
            if stale:
                return stale[1]
            logger.warning("Failed to fetch ROI for %s: %s", camera_uuid, e)
            return []

    def _device_fetch_semaphore(self, device_url: Optional[str]) -> asyncio.Semaphore:
        key = str(device_url or "").strip().lower()
        sem = self._device_fetch_limits.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrent_fetch_per_device)
            self._device_fetch_limits[key] = sem
        return sem

    def _reserve_detection_summary_alert(self, camera_uuid: str, cls_names: List[str]) -> bool:
        """
        Check if detection alert should be emitted using per-class cooldown.
        Uses (camera_uuid, cls_name) tuple instead of just camera_uuid.
        This allows different object classes to bypass each other's cooldown.
        """
        cam = str(camera_uuid)
        cooldown_s = float(self._detection_summary_cooldown_s or 0.0)
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
        if self._notification_service is None:
            return False

        cam_uuid = str(resp.camera_uuid)
        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning("Skipping ROI alert publish because camera context was not found camera=%s", cam_uuid)
            return False

        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        emitted = False

        for a in alerts:
            overlay_payload = _overlay_payload_from_resp(resp, fallback_detections=[a])
            raw_track_id = a.get("track_id")
            track_id = int(raw_track_id) if raw_track_id is not None else None

            roi_title, roi_body = _humanize_roi_alert(a.get("cls_name"), ctx)

            msg = NotificationMessage(
                user_id=int(ctx.user_id),
                id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-{a.get('roi_id')}-{a.get('track_id')}",
                ts_ms=int(resp.frame_ts_ms),
                camera_uuid=cam_uuid,
                site_uuid=str(ctx.site_uuid),
                site_name=ctx.site_name,
                title=roi_title,
                body=roi_body,
                alert_type="roi_enter",
                cls_names=[str(a.get("cls_name", "object"))],
                max_conf=float(a.get("conf", 0.0) or 0.0),
                frame_w=overlay_payload.get("frame_w"),
                frame_h=overlay_payload.get("frame_h"),
                frame_seq=overlay_payload.get("frame_seq"),
                detections=list(overlay_payload.get("detections") or []),
                roi_id=str(a.get("roi_id", "")),
                track_id=track_id,
                device_name=ctx.device_name,
                camera_name=ctx.camera_name,
                image_url=image_url,
            )

            await self._publish_and_persist(
                ctx=ctx,
                msg=msg,
                overlay_payload=overlay_payload,
                extra_payload=extra_payload,
                persist_extra={"alert": a},
                task_name=f"persist_roi_alert:{cam_uuid}"
            )
            emitted = True
        
        return emitted

    async def _emit_item_detected_notifications(
        self,
        resp: ObjDetectResponse,
        tracks: Tuple[Dict[str, Any], ...],
        track_events: Tuple[Tuple[str, int], ...],
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._notification_service is None:
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

        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        emitted = False

        for track_id in confirmed_track_ids:
            tr = tracks_by_id.get(track_id)
            if tr is None:
                continue

            overlay_payload = _overlay_payload_from_resp(resp, fallback_detections=[tr])
            cls_name = str(tr.get("cls_name") or "object")
            conf = float(tr.get("conf", 0.0) or 0.0)

            item_title, item_body = _humanize_item_detected(cls_name, conf, ctx)

            msg = NotificationMessage(
                user_id=int(ctx.user_id),
                id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-track-{track_id}",
                ts_ms=int(resp.frame_ts_ms),
                camera_uuid=cam_uuid,
                site_uuid=str(ctx.site_uuid),
                site_name=ctx.site_name,
                title=item_title,
                body=item_body,
                alert_type="item_detected",
                cls_names=[cls_name],
                max_conf=conf,
                frame_w=overlay_payload.get("frame_w"),
                frame_h=overlay_payload.get("frame_h"),
                frame_seq=overlay_payload.get("frame_seq"),
                detections=list(overlay_payload.get("detections") or []),
                track_id=track_id,
                device_name=ctx.device_name,
                camera_name=ctx.camera_name,
                image_url=image_url,
            )

            await self._publish_and_persist(
                ctx=ctx,
                msg=msg,
                overlay_payload=overlay_payload,
                extra_payload=extra_payload,
                persist_extra={"track": tr, "event": "track_confirmed"},
                task_name=f"persist_track_confirmed:{cam_uuid}:{track_id}"
            )
            emitted = True

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

        filtered: List[Dict[str, Any]] = []
        classes: List[str] = []
        max_conf = 0.0
        for d in raw_detections:
            if not isinstance(d, dict):
                continue
            cls_name = str(d.get("cls_name") or "").strip()
            if not cls_name:
                continue
            if self.interesting_classes and cls_name not in self.interesting_classes:
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

        await self._publish_and_persist(
            ctx=ctx,
            msg=msg,
            overlay_payload=overlay_payload,
            extra_payload=extra_payload,
            persist_extra={"detections": filtered[:20], "event": "detection_summary"},
            task_name=f"persist_detection_summary:{cam_uuid}"
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
