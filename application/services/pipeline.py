# application/services/pipeline.py
"""The cloud-side detection pipeline.

It never ingests RTSP or streams frames itself: it keeps a registry of camera
channels, consumes each Jetson's detection stream, runs tracking and ROI rules
over the results, and caches the latest detection per camera for the live
overlay.
"""

import asyncio
import base64
import logging
import random
import time
import uuid as _uuid
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Set, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from application.channels.channel import VideoChannel, VideoChannelConfig
from application.repositories.notification_repository import CameraContext
from application.services.common import (
    CameraContextResolver,
    SiteArmStateResolver,
    SiteTriggerModeResolver,
)
from application.services.notification import NotificationMessage, NotificationService
from application.services.overlay_normalize import _append_overlay_detection
from application.services.tracker import (
    MultiCameraByteTrack,
    ROI,
    ROIAlertEngine,
    nms_payload_detections,
)
from core.env import env_float, env_int

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

    tracks: Tuple[Dict[str, Any], ...] = ()
    track_events: Tuple[Tuple[str, int], ...] = ()
    alerts: Tuple[Dict[str, Any], ...] = ()


def _alert_label(cls_name: Any) -> str:
    label = str(cls_name or "object").strip() or "object"
    return label[:1].upper() + label[1:]


def _alert_location(ctx: CameraContext) -> str:
    parts = [p for p in (ctx.camera_name, ctx.site_name) if p]
    return ", ".join(parts) if parts else "the monitored area"


# The class word stays in the body because downstream object-name inference
# runs a regex over that text. The timestamp is rendered client-side from
# `ts_ms`, so it is deliberately absent here.
def _humanize_roi_alert(cls_name: Any, ctx: CameraContext) -> Tuple[str, str]:
    label = _alert_label(cls_name)
    return (
        f"{label} entered restricted zone",
        f"{label} entered the restricted zone at {_alert_location(ctx)}",
    )


def _humanize_item_detected(
    cls_name: Any, conf: float, ctx: CameraContext
) -> Tuple[str, str]:
    label = _alert_label(cls_name)
    try:
        pct = max(0, min(100, round(float(conf) * 100)))
    except (TypeError, ValueError):
        pct = 0
    return (
        f"{label} detected",
        f"{label} detected at {_alert_location(ctx)} ({pct}% confidence)",
    )


def _live_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tracks that should actually be drawn on the live overlay.

    Coasting tracks (kept alive for a few frames after they stop matching a
    detection) have a pure velocity-extrapolated bbox, which trails stale ghost
    boxes behind a fast object. Only confirmed tracks matched this frame are
    drawn. Raw edge detections are drawn separately, so this never hides a real
    object.
    """
    out: List[Dict[str, Any]] = []
    for t in tracks:
        if not isinstance(t, dict) or not t.get("confirmed", False):
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
    seen_exact: Set[Tuple[Any, ...]] = set()
    tracked_bases: Set[Tuple[Any, ...]] = set()
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
        timeout_ms: int,
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
        timeout_ms: int,
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
    """Pub/sub hub for all detections across all cameras, used by the global stream."""

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
        for q in subs:
            try:
                q.put_nowait(resp)
            except asyncio.QueueFull:
                # Drop the oldest frame rather than block the producer.
                try:
                    q.get_nowait()
                    q.put_nowait(resp)
                except Exception:
                    pass


class ModelPipeline:
    """Azure runtime channel registry plus Jetson detection consumption.

    `webrtc_url` is immutable: `edit_channel` refuses a change to it.
    """

    def __init__(
        self,
        pipeline_id: UUID,
        channels: Optional[List[VideoChannel]] = None,
        *,
        detect_store: Optional[ObjDetectStorePort] = None,
        notification_service: Optional[NotificationService] = None,
        tracker_cfg: Optional[dict] = None,
        interesting_classes: Optional[Set[str]] = None,
        notify_on_confirmed: bool = False,
        notify_on_roi_enter: bool = True,
        task_spawner: Optional[TaskSpawner] = None,
    ):
        self.pipeline_id = pipeline_id
        self.detect_store = detect_store or InMemoryObjDetectStore()
        self.detection_hub = DetectionHub()

        self._channels: Dict[str, VideoChannel] = {}
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._last_seen: Dict[str, Tuple[int, int]] = {}
        self._last_ok_s: Dict[str, float] = {}

        # Clock-skew monitor: the Jetson stamps frame_ts_ms with its own wall
        # clock, while clip windows are sliced out of MediaMTX recordings stamped
        # with the Azure clock. Drift shifts every clip so the event falls outside
        # the captured window. We cannot fix the Jetson clock from here, but the
        # rolling floor of (azure_now - frame_ts_ms) approximates the offset.
        self._clock_skew_floor_ms: Dict[str, float] = {}
        self._clock_skew_floor_reset_s: Dict[str, float] = {}
        self._clock_skew_last_warn_s: Dict[str, float] = {}

        self._tracker = MultiCameraByteTrack(**(tracker_cfg or {}))
        self._nms_iou = env_float("DETECTION_NMS_IOU", 0.55, minimum=0.0)
        self._nms_overlap = env_float("DETECTION_NMS_OVERLAP", 0.70, minimum=0.0)
        self._nms_size_ratio = env_float("DETECTION_NMS_SIZE_RATIO", 0.65, minimum=0.0)
        self._roi_engine = ROIAlertEngine()

        self.interesting_classes = interesting_classes or {
            "person", "car", "motorcycle", "truck",
        }
        self.notify_on_confirmed = notify_on_confirmed
        self.notify_on_roi_enter = notify_on_roi_enter

        self._roi_cache: Dict[str, Tuple[float, List[ROI]]] = {}
        self._roi_cache_ttl_s = env_float("CAMERA_ROI_CACHE_TTL_S", 30.0, minimum=0.0)
        self._roi_cache_error_ttl_s = env_float(
            "CAMERA_ROI_CACHE_ERROR_TTL_S", 5.0, minimum=0.0
        )

        self._notification_service = notification_service
        self._last_detection_summary_s: Dict[Tuple[str, str], float] = {}
        self._detection_summary_cooldown_s = env_float(
            "DETECTION_ALERT_COOLDOWN_S", 8.0, minimum=0.0
        )

        self._ctx_resolver = CameraContextResolver(
            hit_ttl_s=60.0,
            miss_ttl_s=env_float("CAMERA_CONTEXT_MISS_CACHE_TTL_S", 5.0),
            max_concurrent_lookups=env_int(
                "CAMERA_CONTEXT_MAX_CONCURRENT_DB_LOOKUPS", 4
            ),
        )
        self._trigger_mode_resolver = SiteTriggerModeResolver(
            default="roi_enter",
            ttl_s=env_float("SITE_TRIGGER_MODE_CACHE_TTL_S", 30.0),
        )
        self._arm_state_resolver = SiteArmStateResolver(
            ttl_s=env_float("SITE_ARM_STATE_CACHE_TTL_S", 10.0),
        )

        self._session_factory: Optional[SessionFactory] = None
        self._device_fetch_limits: Dict[str, asyncio.Semaphore] = {}
        self._max_concurrent_fetch_per_device = env_int(
            "DETECTION_MAX_CONCURRENT_FETCH_PER_DEVICE", 8, minimum=1
        )
        self._startup_jitter_ms = env_int("DETECTION_STARTUP_JITTER_MS", 100, minimum=0)

        # The clip window absorbs sub-second offsets; multi-second skew pushes the
        # event outside the PRE/POST window. 0 disables the check.
        self._clock_skew_warn_ms = env_int(
            "DETECTION_CLOCK_SKEW_WARN_MS", 3000, minimum=0
        )
        self._clock_skew_floor_window_s = env_float(
            "DETECTION_CLOCK_SKEW_FLOOR_WINDOW_S", 300.0, minimum=30.0
        )
        self._clock_skew_warn_interval_s = env_float(
            "DETECTION_CLOCK_SKEW_WARN_INTERVAL_S", 120.0, minimum=5.0
        )

        self._lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._task_spawner: TaskSpawner = task_spawner or (
            lambda coro, name: asyncio.create_task(coro, name=name)
        )

        for ch in (channels or []):
            self._channels[ch.key()] = ch

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
            self._last_detection_summary_s.clear()
            self._clock_skew_floor_ms.clear()
            self._clock_skew_floor_reset_s.clear()
            self._clock_skew_last_warn_s.clear()
            self._roi_cache.clear()

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

    async def add_channel(self, ch: VideoChannel) -> None:
        key = ch.key()
        async with self._lock:
            self._channels[key] = ch
            started = self._started and (not self._closing)

        if started:
            self._ensure_poller(key, ch)

    async def edit_channel(self, ch: VideoChannel) -> None:
        """Replace a channel in place, for example after device_url changed."""
        key = ch.key()
        async with self._lock:
            old = self._channels.get(key)
            if old is not None:
                old_cfg = getattr(old, "config", None)
                new_cfg = getattr(ch, "config", None)
                if old_cfg is not None and new_cfg is not None:
                    if getattr(old_cfg, "webrtc_url", None) != getattr(
                        new_cfg, "webrtc_url", None
                    ):
                        raise ValueError(
                            "webrtc_url is immutable; do not change it on edit."
                        )

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
            self._channels.pop(key, None)
            t = self._poll_tasks.pop(key, None)
            self._last_seen.pop(key, None)
            self._last_ok_s.pop(key, None)
            self._clock_skew_floor_ms.pop(key, None)
            self._clock_skew_floor_reset_s.pop(key, None)
            self._clock_skew_last_warn_s.pop(key, None)
            # Cooldown keys are (camera_uuid, cls_name), so drop by prefix.
            for cooldown_key in [
                k for k in self._last_detection_summary_s if k[0] == key
            ]:
                self._last_detection_summary_s.pop(cooldown_key, None)

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

    def _ensure_poller(self, key: str, ch: VideoChannel) -> None:
        t = self._poll_tasks.get(key)
        if t is None or t.done():
            self._poll_tasks[key] = self._task_spawner(
                self._stream_loop(key, ch),
                f"stream_jetson_detection:{key}",
            )

    # ------------------------------------------------------------------
    # Registry accessors and cache invalidation
    # ------------------------------------------------------------------

    def list_channel_ids(self) -> List[str]:
        return list(self._channels.keys())

    async def get_channel_config(
        self, camera_uuid: str
    ) -> Optional[VideoChannelConfig]:
        key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.get(key)
            return getattr(ch, "config", None) if ch else None

    async def get_latest_detection(
        self, camera_uuid: str
    ) -> Optional[ObjDetectResponse]:
        return await self.detect_store.get_latest(str(camera_uuid))

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
                logger.exception(
                    "Failed to set session factory on NotificationService"
                )

    def invalidate_camera_roi_state(self, camera_uuid: str) -> None:
        """Clear per-camera ROI edge-trigger state so ROI edits apply immediately."""
        cam = str(camera_uuid)
        self._roi_engine.reset_camera(cam)
        self._roi_cache.pop(cam, None)

    def invalidate_site_trigger_mode_cache(self, site_uuid: str) -> None:
        self._trigger_mode_resolver.invalidate(str(site_uuid))

    def invalidate_site_arm_state(self, site_uuid: str) -> None:
        self._arm_state_resolver.invalidate(str(site_uuid))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

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

            if getattr(ch.config, "detection_enabled", True) is False:
                await asyncio.sleep(0.5)
                continue

            try:
                got_any = False
                async for payload in ch.stream_detections():
                    got_any = True
                    await self._process_detection_payload(key, ch, payload)

                    async with self._lock:
                        if (
                            self._closing
                            or (not self._started)
                            or (self._channels.get(key) is None)
                        ):
                            return

                if got_any:
                    backoff_s = 0.5

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Detection stream loop failed camera=%s", key)

            await asyncio.sleep(backoff_s)
            backoff_s = min(max_backoff_s, backoff_s * 2.0)

    async def refresh_detection_once(
        self, camera_uuid: str
    ) -> Optional[ObjDetectResponse]:
        """On-demand pull from the Jetson, useful when the cache is empty.

        Goes through the same processing path as the stream so an on-demand
        frame is not silently skipped by the tracker later.
        """
        key = str(camera_uuid)
        async with self._lock:
            ch = self._channels.get(key)
        if ch is None:
            return None

        sem = self._device_fetch_semaphore(getattr(ch.config, "device_url", None))
        async with sem:
            payload = await ch.fetch_detection_json()

        await self._process_detection_payload(key, ch, payload)
        return await self.detect_store.get_latest(key)

    def _payload_to_resp(
        self, payload: Optional[Dict[str, Any]], ch: VideoChannel
    ) -> Optional[ObjDetectResponse]:
        if not payload or not isinstance(payload, dict):
            return None

        if (
            "frame_seq" not in payload
            and "payload" in payload
            and isinstance(payload["payload"], dict)
        ):
            payload = payload["payload"]

        frame_seq = payload.get("frame_seq")
        frame_ts_ms = payload.get("frame_ts_ms")
        if frame_seq is None or frame_ts_ms is None:
            return None

        # Suppress duplicate boxes at ingestion, before the payload fans out to
        # the drawn overlay and the tracker. A box that leaks past the edge's NMS
        # otherwise shows up twice on one object: once as a raw drawn detection
        # and once as a second track id.
        dets = nms_payload_detections(
            payload.get("detections") or [],
            iou_thr=self._nms_iou,
            overlap_thr=self._nms_overlap,
            size_ratio_thr=self._nms_size_ratio,
        )

        frame_w = payload.get("frame_w") or payload.get("image_w") or payload.get("width")
        frame_h = payload.get("frame_h") or payload.get("image_h") or payload.get("height")
        event_type = payload.get("type")
        reason = payload.get("reason")
        inf_ms = payload.get("inference_ms")
        model_id = payload.get("model_id")
        image_url = payload.get("image_url")

        expected_cam = str(ch.config.camera_uuid)
        payload_cam = payload.get("camera_uuid")
        if payload_cam is not None and str(payload_cam) != expected_cam:
            logger.warning(
                "Dropping detection payload with mismatched camera_uuid expected=%s got=%s",
                expected_cam,
                payload_cam,
            )
            return None

        return ObjDetectResponse(
            camera_uuid=expected_cam,
            frame_ts_ms=int(frame_ts_ms),
            frame_seq=int(frame_seq),
            event_type=str(event_type) if event_type is not None else None,
            reason=str(reason) if reason is not None else None,
            frame_w=int(frame_w) if frame_w is not None else None,
            frame_h=int(frame_h) if frame_h is not None else None,
            site_uuid=ch.config.site_uuid,
            device_uuid=ch.config.device_uuid,
            detections=tuple(dets) if isinstance(dets, list) else (),
            pose=payload.get("pose"),
            inference_ms=int(inf_ms) if inf_ms is not None else None,
            model_id=str(model_id) if model_id is not None else None,
            image_url=str(image_url).strip() if image_url is not None else None,
        )

    def _is_new_detection(self, key: str, resp: ObjDetectResponse) -> bool:
        prev_ts, prev_seq = self._last_seen.get(key, (0, -1))
        cur_ts, cur_seq = int(resp.frame_ts_ms), int(resp.frame_seq)

        if cur_ts > prev_ts:
            return True
        if cur_ts == prev_ts and cur_seq > prev_seq:
            return True
        if cur_ts == prev_ts and cur_seq == prev_seq:
            return False

        regressed = (cur_ts < prev_ts) or (cur_ts == prev_ts and cur_seq < prev_seq)
        if not regressed:
            return False

        # Regressed after a gap: assume the Jetson restarted, so accept the frame
        # and re-anchor rather than stalling forever.
        last_ok = self._last_ok_s.get(key, 0.0)
        if (time.monotonic() - last_ok) > 5.0:
            logger.warning(
                "Detected possible Jetson restart (ts/seq regressed). Resetting last_seen. "
                "camera=%s prev=(%s,%s) cur=(%s,%s)",
                key, prev_ts, prev_seq, cur_ts, cur_seq,
            )
            return True

        return False

    def _monitor_clock_skew(self, key: str, frame_ts_ms: int) -> None:
        """Estimate and warn on Jetson to Azure clock drift. Warn only, no behaviour change.

        Real lag is always positive and bounded below by network and processing
        latency, so the minimum lag over a window estimates the constant clock
        offset. A large positive floor means the Jetson clock is behind Azure; a
        negative floor means it is ahead, which is physically impossible without
        skew. The floor resets on a slow window so a corrected clock recovers.
        """
        if self._clock_skew_warn_ms <= 0:
            return

        now_s = time.time()
        lag_ms = (now_s * 1000.0) - float(frame_ts_ms)

        reset_at = self._clock_skew_floor_reset_s.get(key, 0.0)
        floor = self._clock_skew_floor_ms.get(key)
        if floor is None or now_s >= reset_at:
            floor = lag_ms
            self._clock_skew_floor_ms[key] = floor
            self._clock_skew_floor_reset_s[key] = now_s + self._clock_skew_floor_window_s
        elif lag_ms < floor:
            floor = lag_ms
            self._clock_skew_floor_ms[key] = floor

        if -self._clock_skew_warn_ms <= floor <= self._clock_skew_warn_ms:
            return

        last_warn = self._clock_skew_last_warn_s.get(key, 0.0)
        if (now_s - last_warn) < self._clock_skew_warn_interval_s:
            return
        self._clock_skew_last_warn_s[key] = now_s

        direction = "AHEAD of" if floor < 0 else "BEHIND"
        logger.warning(
            "Jetson clock appears skewed camera=%s estimated_offset_ms=%d "
            "(Jetson clock is %s Azure/MediaMTX). Clip windows are anchored on the "
            "Jetson wall clock; multi-second skew shifts the event out of the captured "
            "PRE/POST window. Verify NTP time sync on the Jetson host.",
            key,
            int(floor),
            direction,
        )

    async def _process_detection_payload(
        self, key: str, ch: VideoChannel, payload: Dict[str, Any]
    ) -> None:
        resp = self._payload_to_resp(payload, ch)
        if resp is None:
            return

        if not self._is_new_detection(key, resp):
            return

        self._monitor_clock_skew(key, int(resp.frame_ts_ms))

        tracker_out = self._tracker.update_from_event({
            "camera_uuid": str(resp.camera_uuid),
            "frame_ts_ms": int(resp.frame_ts_ms),
            "frame_seq": int(resp.frame_seq),
            "detections": list(resp.detections),
        })
        tracks = tuple(tracker_out.get("tracks", []) or [])
        track_events = tuple(tracker_out.get("events", []) or [])

        alerts: List[Dict[str, Any]] = []
        if resp.frame_w and resp.frame_h and tracks:
            alerts = self._roi_engine.process(
                camera_uuid=str(resp.camera_uuid),
                frame_w=int(resp.frame_w),
                frame_h=int(resp.frame_h),
                tracks=list(tracks),
                rois=await self._fetch_rois(str(resp.camera_uuid)),
                ts_ms=int(resp.frame_ts_ms),
            )

        resp2 = replace(
            resp,
            tracks=tracks,
            track_events=track_events,
            alerts=tuple(alerts),
        )

        self._last_seen[key] = (int(resp2.frame_ts_ms), int(resp2.frame_seq))
        self._last_ok_s[key] = time.monotonic()

        # Publish for the live overlay first. Clip recording, snapshot fetch and
        # notification persistence all run after this, off the stream-consumer
        # loop, so the overlay never falls behind the live video.
        await self.detect_store.put(resp2)
        await self.detection_hub.publish(resp2)

        if self._notification_service is not None:
            self._task_spawner(
                self._post_publish_work(resp2, ch),
                f"detection_post_publish:{key}",
            )

    # ------------------------------------------------------------------
    # Post-publish work (off the hot path)
    # ------------------------------------------------------------------

    async def _post_publish_work(
        self, resp: ObjDetectResponse, ch: VideoChannel
    ) -> None:
        try:
            if self._notification_service is None:
                return
            await self._record_playback_frame(resp, ch)
            if await self._notifications_allowed_now(ch):
                await self._emit_notifications(resp, ch)
        except Exception:
            logger.exception(
                "post-publish detection work failed camera=%s", resp.camera_uuid
            )

    async def _record_playback_frame(
        self, resp: ObjDetectResponse, ch: VideoChannel
    ) -> None:
        svc = self._notification_service
        if svc is None:
            return

        cam_uuid = str(resp.camera_uuid)
        override = str(
            getattr(ch.config, "camera_playback_enabled", "inherit") or "inherit"
        )
        if override not in ("always", "inherit") or not self.is_channel_enable(ch):
            return

        try:
            # An explicit "always" override bypasses the site eligibility list.
            eligible = override == "always" or await svc.is_camera_prerecord_eligible(
                cam_uuid
            )
            if not eligible:
                return

            overlay = _overlay_payload_from_resp(
                resp, fallback_detections=_live_tracks(list(resp.tracks))
            )
            await svc.record_detection_overlay_frame(
                camera_uuid=cam_uuid,
                frame_ts_ms=overlay.get("frame_ts_ms"),
                frame_seq=overlay.get("frame_seq"),
                frame_w=overlay.get("frame_w"),
                frame_h=overlay.get("frame_h"),
                detections=list(overlay.get("detections") or []),
            )
        except Exception:
            logger.exception(
                "Failed to record detection overlay frame camera=%s", cam_uuid
            )

    async def _emit_notifications(
        self, resp: ObjDetectResponse, ch: VideoChannel
    ) -> None:
        cam_uuid = str(resp.camera_uuid)
        allow_broad = await self._allow_broad_notifications(ch)

        # _build_alert_extra_payload may fetch a snapshot over HTTP, so it is
        # resolved lazily and at most once per frame.
        cached: List[Optional[Dict[str, Any]]] = []

        async def alert_extra() -> Optional[Dict[str, Any]]:
            if not cached:
                cached.append(await self._build_alert_extra_payload(resp=resp, ch=ch))
            return cached[0]

        # ROI entry is the most specific operator event. It wins, and suppresses
        # the broader same-frame notices so one event never creates two pending
        # notifications for the operator.
        emitted = False

        if self.notify_on_roi_enter and resp.alerts:
            try:
                emitted = await self._emit_roi_alert_notifications(
                    resp, list(resp.alerts), extra_payload=await alert_extra()
                )
            except Exception:
                logger.exception("Failed to emit ROI notifications camera=%s", cam_uuid)

        if not emitted and allow_broad and self.notify_on_confirmed and resp.track_events:
            try:
                emitted = await self._emit_item_detected_notifications(
                    resp, extra_payload=await alert_extra()
                )
            except Exception:
                logger.exception(
                    "Failed to emit item-detected notifications camera=%s", cam_uuid
                )

        if not emitted and allow_broad:
            summary_classes = self._interesting_detection_classes(resp)
            if summary_classes and self._reserve_detection_summary_alert(
                cam_uuid, summary_classes
            ):
                try:
                    await self._emit_detection_summary_notification(
                        resp, extra_payload=await alert_extra()
                    )
                except Exception:
                    logger.exception(
                        "Failed to emit detection-summary notification camera=%s",
                        cam_uuid,
                    )

    def is_channel_enable(self, ch: VideoChannel) -> bool:
        return getattr(getattr(ch, "config", None), "enabled", True) is not False

    async def _notifications_allowed_now(self, ch: VideoChannel) -> bool:
        cfg = getattr(ch, "config", None)
        if cfg is None:
            return True

        if getattr(cfg, "notification_enabled", True) is False:
            return False

        # A site-level arm/disarm override wins over the schedule for every camera
        # in the site until it clears at the next schedule boundary.
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
                logger.exception(
                    "Failed to evaluate notification schedule camera=%s", ch.key()
                )

        return True

    async def _allow_broad_notifications(self, ch: VideoChannel) -> bool:
        mode = str(
            getattr(ch.config, "notification_trigger_mode", "inherit") or "inherit"
        )
        if mode in ("roi_enter", "any_detection"):
            return mode == "any_detection"

        site_uuid = getattr(ch.config, "site_uuid", None)
        site_mode = await self._trigger_mode_resolver.resolve(
            str(site_uuid) if site_uuid else None
        )
        return site_mode == "any_detection"

    def _interesting_detection_classes(self, resp: ObjDetectResponse) -> List[str]:
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

    def _reserve_detection_summary_alert(
        self, camera_uuid: str, cls_names: List[str]
    ) -> bool:
        """Per-class cooldown, so a newly-appearing class is not silenced by another."""
        cam = str(camera_uuid)
        cooldown_s = float(self._detection_summary_cooldown_s or 0.0)
        now = time.monotonic()

        for cls_name in cls_names:
            key = (cam, str(cls_name))
            last = self._last_detection_summary_s.get(key, 0.0)
            if cooldown_s > 0.0 and (now - last) < cooldown_s:
                continue
            self._last_detection_summary_s[key] = now
            return True

        return False

    # ------------------------------------------------------------------
    # ROI lookup
    # ------------------------------------------------------------------

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
        # All coordinates inside [0, 1] means normalized.
        return bool(points) and all(
            0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for (x, y) in points
        )

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
    def _clamp_unit_points(
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        return [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for (x, y) in points]

    def _cache_rois(self, camera_uuid: str, rois: List[ROI], ttl_s: float) -> List[ROI]:
        self._roi_cache[camera_uuid] = (time.monotonic() + ttl_s, rois)
        return rois

    async def _fetch_rois(self, camera_uuid: str) -> List[ROI]:
        if not self._session_factory:
            return []

        hit = self._roi_cache.get(camera_uuid)
        if hit and hit[0] > time.monotonic():
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
                return self._cache_rois(camera_uuid, [], self._roi_cache_ttl_s)

            points = self._parse_roi_points(row.get("points", []))
            if len(points) < 3:
                return self._cache_rois(camera_uuid, [], self._roi_cache_ttl_s)

            normalized = self._coerce_roi_normalized(row.get("normalized"), points)
            roi_frame_w, roi_frame_h = self._parse_roi_frame_size(row)
            roi_points = self._clamp_unit_points(points) if normalized else points

            return self._cache_rois(
                camera_uuid,
                [
                    ROI(
                        roi_id=f"{camera_uuid}-roi",
                        points=roi_points,
                        normalized=normalized,
                        frame_w=roi_frame_w,
                        frame_h=roi_frame_h,
                    )
                ],
                self._roi_cache_ttl_s,
            )

        except Exception as e:
            stale = self._roi_cache.get(camera_uuid)
            if stale:
                return stale[1]
            logger.warning("Failed to fetch ROI for %s: %s", camera_uuid, e)
            # Short negative cache so a database outage does not re-query every frame.
            return self._cache_rois(camera_uuid, [], self._roi_cache_error_ttl_s)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def _device_fetch_semaphore(self, device_url: Optional[str]) -> asyncio.Semaphore:
        key = str(device_url or "").strip().lower()
        sem = self._device_fetch_limits.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrent_fetch_per_device)
            self._device_fetch_limits[key] = sem
        return sem

    @staticmethod
    def _image_bytes_to_data_url(
        payload: bytes, content_type: Optional[str]
    ) -> Optional[str]:
        if not payload:
            return None

        mime = str(content_type or "image/jpeg").split(";", 1)[0].strip().lower()
        if not mime.startswith("image/"):
            mime = "image/jpeg"

        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def _build_alert_extra_payload(
        self, *, resp: ObjDetectResponse, ch: VideoChannel
    ) -> Optional[Dict[str, Any]]:
        image_url = str(resp.image_url or "").strip()
        if image_url:
            return {"image_url": image_url}

        try:
            sem = self._device_fetch_semaphore(getattr(ch.config, "device_url", None))
            async with sem:
                snapshot = await ch.fetch_snapshot_bytes()
        except Exception:
            logger.exception(
                "Failed to fetch alert snapshot camera=%s", resp.camera_uuid
            )
            return None

        if not snapshot:
            return None

        data_url = self._image_bytes_to_data_url(*snapshot)
        return {"image_url": data_url} if data_url else None

    # ------------------------------------------------------------------
    # Notification emission
    # ------------------------------------------------------------------

    async def _publish_and_persist(
        self,
        *,
        ctx: CameraContext,
        msg: NotificationMessage,
        overlay_payload: Dict[str, Any],
        extra_payload: Optional[Dict[str, Any]],
        persist_extra: Optional[Dict[str, Any]],
        task_name: str,
    ) -> None:
        svc = self._notification_service
        if svc is None:
            return

        # If the site's org has an operator, hold the alert for approval: persist
        # it as pending but do not push it to the end user's realtime stream.
        if not await svc.requires_operator_approval(ctx.site_uuid):
            await svc.hub.publish(msg)

        persist_payload = {
            **overlay_payload,
            **(extra_payload or {}),
            **(persist_extra or {}),
        }
        self._task_spawner(
            svc.enqueue_notification(msg, ctx, extra_payload=persist_payload),
            task_name,
        )

    async def _emit_detail_notification(
        self,
        resp: ObjDetectResponse,
        ctx: CameraContext,
        *,
        overlay_source: Dict[str, Any],
        alert_id: str,
        title: str,
        body: str,
        alert_type: str,
        cls_name: str,
        conf: float,
        track_id: Optional[int] = None,
        roi_id: Optional[str] = None,
        image_url: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        persist_extra: Optional[Dict[str, Any]] = None,
        task_name: str,
    ) -> None:
        """Shared body for ROI-entry and item-detected notifications."""
        overlay_payload = _overlay_payload_from_resp(
            resp, fallback_detections=[overlay_source]
        )

        msg = NotificationMessage(
            user_id=int(ctx.user_id),
            id=alert_id,
            ts_ms=int(resp.frame_ts_ms),
            camera_uuid=str(resp.camera_uuid),
            site_uuid=str(ctx.site_uuid),
            site_name=ctx.site_name,
            title=title,
            body=body,
            alert_type=alert_type,
            cls_names=[cls_name],
            max_conf=conf,
            frame_w=overlay_payload.get("frame_w"),
            frame_h=overlay_payload.get("frame_h"),
            frame_seq=overlay_payload.get("frame_seq"),
            detections=list(overlay_payload.get("detections") or []),
            roi_id=roi_id,
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
            persist_extra=persist_extra,
            task_name=task_name,
        )

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
            logger.warning(
                "Skipping ROI alert publish because camera context was not found camera=%s",
                cam_uuid,
            )
            return False

        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        emitted = False

        for a in alerts:
            cls_name = str(a.get("cls_name", "object"))
            title, body = _humanize_roi_alert(a.get("cls_name"), ctx)
            raw_track_id = a.get("track_id")

            await self._emit_detail_notification(
                resp,
                ctx,
                overlay_source=a,
                alert_id=(
                    f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}"
                    f"-{a.get('roi_id')}-{raw_track_id}"
                ),
                title=title,
                body=body,
                alert_type="roi_enter",
                cls_name=cls_name,
                conf=float(a.get("conf", 0.0) or 0.0),
                track_id=int(raw_track_id) if raw_track_id is not None else None,
                roi_id=str(a.get("roi_id", "")),
                image_url=image_url,
                extra_payload=extra_payload,
                persist_extra={"alert": a},
                task_name=f"persist_roi_alert:{cam_uuid}",
            )
            emitted = True

        return emitted

    async def _emit_item_detected_notifications(
        self,
        resp: ObjDetectResponse,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if self._notification_service is None:
            return False

        confirmed_ids = [
            int(track_id)
            for (ev_type, track_id) in resp.track_events
            if ev_type == "track_confirmed"
        ]
        if not confirmed_ids:
            return False

        tracks_by_id: Dict[int, Dict[str, Any]] = {}
        for tr in resp.tracks:
            try:
                tracks_by_id[int(tr.get("track_id"))] = tr
            except (TypeError, ValueError):
                continue

        cam_uuid = str(resp.camera_uuid)
        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning(
                "Skipping item-detected alert publish because camera context was not "
                "found camera=%s",
                cam_uuid,
            )
            return False

        image_url = str((extra_payload or {}).get("image_url") or "").strip() or None
        emitted = False

        for track_id in confirmed_ids:
            tr = tracks_by_id.get(track_id)
            if tr is None:
                continue

            cls_name = str(tr.get("cls_name") or "object")
            conf = float(tr.get("conf", 0.0) or 0.0)
            title, body = _humanize_item_detected(cls_name, conf, ctx)

            await self._emit_detail_notification(
                resp,
                ctx,
                overlay_source=tr,
                alert_id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-track-{track_id}",
                title=title,
                body=body,
                alert_type="item_detected",
                cls_name=cls_name,
                conf=conf,
                track_id=track_id,
                image_url=image_url,
                extra_payload=extra_payload,
                persist_extra={"track": tr, "event": "track_confirmed"},
                task_name=f"persist_track_confirmed:{cam_uuid}:{track_id}",
            )
            emitted = True

        return emitted

    async def _emit_detection_summary_notification(
        self,
        resp: ObjDetectResponse,
        *,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._notification_service is None:
            return

        cam_uuid = str(resp.camera_uuid)
        filtered: List[Dict[str, Any]] = []
        classes: List[str] = []
        max_conf = 0.0

        for d in list(resp.detections or ()):
            if not isinstance(d, dict):
                continue
            cls_name = str(d.get("cls_name") or "").strip()
            if not cls_name:
                continue
            if self.interesting_classes and cls_name not in self.interesting_classes:
                continue
            max_conf = max(max_conf, float(d.get("conf", 0.0) or 0.0))
            classes.append(cls_name)
            filtered.append(d)

        if not filtered:
            return

        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning(
                "Skipping detection-summary alert because camera context was not found "
                "camera=%s",
                cam_uuid,
            )
            return

        uniq_classes = sorted(set(classes))
        classes_text = ", ".join(uniq_classes)
        verb = "is" if len(uniq_classes) == 1 else "are"
        overlay_payload = _overlay_payload_from_resp(resp)

        msg = NotificationMessage(
            user_id=int(ctx.user_id),
            id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-summary",
            ts_ms=int(resp.frame_ts_ms),
            camera_uuid=cam_uuid,
            site_uuid=str(ctx.site_uuid),
            site_name=ctx.site_name,
            title=f"Detection: {classes_text}",
            body=f"{int(max_conf * 100)}% chance that {classes_text} {verb} being detected",
            alert_type="detection_summary",
            cls_names=uniq_classes,
            max_conf=float(max_conf),
            frame_w=overlay_payload.get("frame_w"),
            frame_h=overlay_payload.get("frame_h"),
            frame_seq=overlay_payload.get("frame_seq"),
            detections=list(overlay_payload.get("detections") or []),
            device_name=ctx.device_name,
            camera_name=ctx.camera_name,
            image_url=str((extra_payload or {}).get("image_url") or "").strip() or None,
        )

        await self._publish_and_persist(
            ctx=ctx,
            msg=msg,
            overlay_payload=overlay_payload,
            extra_payload=extra_payload,
            persist_extra={"detections": filtered[:20], "event": "detection_summary"},
            task_name=f"persist_detection_summary:{cam_uuid}",
        )

    async def _get_camera_ctx(self, camera_uuid: str) -> Optional[CameraContext]:
        return await self._ctx_resolver.resolve(camera_uuid)