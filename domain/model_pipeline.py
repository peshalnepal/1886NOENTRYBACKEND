# agents/domain/model_pipeline.py
"""
Azure-side pipeline.

- NO RTSP ingest
- NO frame streaming

It keeps a registry of Camera Channels and maintains an in-memory "latest detection"
cache per camera by polling Jetson: /detection/{camera_uuid}
"""

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

    frame_w: Optional[int] = None
    frame_h: Optional[int] = None

    site_uuid: Optional[str] = None
    device_uuid: Optional[str] = None

    detections: Tuple[Any, ...] = ()
    pose: Optional[Any] = None
    inference_ms: Optional[int] = None
    model_id: Optional[str] = None

    # NEW: tracking + alert scaffolding
    tracks: Tuple[Dict[str, Any], ...] = ()
    track_events: Tuple[Tuple[str, int], ...] = ()
    alerts: Tuple[Dict[str, Any], ...] = ()


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
        self._cam_ctx_cache: Dict[str, Tuple[CameraContext, float]] = {}
        self._cam_ctx_cache_lock = asyncio.Lock()
        self._cam_ctx_ttl_s = 60.0 
        self._session_factory: Optional[SessionFactory] = None
        self._site_cache: Dict[str, Tuple[str, float]] = {}
        self._site_cache_lock = asyncio.Lock()
        self._device_fetch_limits: Dict[str, asyncio.Semaphore] = {}
        self._max_concurrent_fetch_per_device = max(
            1,
            int(os.getenv("DETECTION_MAX_CONCURRENT_FETCH_PER_DEVICE", "8")),
        )
        self._startup_jitter_ms = max(
            0,
            int(os.getenv("DETECTION_STARTUP_JITTER_MS", "500")),
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
            self._channels.pop(key, None)
            t = self._poll_tasks.pop(key, None)
            self._last_seq.pop(key, None)
            self._last_seen.pop(key, None)
            self._last_ok_s.pop(key, None)
            self._last_detection_summary_s.pop(key, None)
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

        async with self._cam_ctx_cache_lock:
            cached = self._cam_ctx_cache.get(key)
            if cached and cached[1] > now:
                return cached[0]

        try:
            cam_uuid = UUID(key)
        except Exception:
            return None

        try:
            async with sf() as db:
                ctx = await self._notif_repo.get_camera_context(db, camera_uuid=cam_uuid)
        except Exception:
            logger.exception("Failed to load CameraContext for camera=%s", key)
            ctx = None

        if ctx:
            async with self._cam_ctx_cache_lock:
                self._cam_ctx_cache[key] = (ctx, now + self._cam_ctx_ttl_s)

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

    def _ensure_poller(self, key: str, ch: VideoChannel) -> None:
        t = self._poll_tasks.get(key)
        if t is None or t.done():
            self._poll_tasks[key] = asyncio.create_task(
                self._poll_loop(key, ch),
                name="poll_jetson_detection:%s" % key,
            )

    def _device_fetch_semaphore(self, device_url: Optional[str]) -> asyncio.Semaphore:
        key = str(device_url or "").strip().lower()
        sem = self._device_fetch_limits.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self._max_concurrent_fetch_per_device)
            self._device_fetch_limits[key] = sem
        return sem

    def _reserve_detection_summary_alert(self, camera_uuid: str) -> bool:
        cam = str(camera_uuid)
        cooldown_s = float(self._detection_summary_cooldown_s or 0.0)
        now = time.monotonic()
        last = self._last_detection_summary_s.get(cam, 0.0)
        if cooldown_s > 0.0 and (now - last) < cooldown_s:
            return False
        self._last_detection_summary_s[cam] = now
        return True

    async def _emit_roi_alert_notifications(self, resp: ObjDetectResponse, alerts: List[Dict[str, Any]]) -> None:
        svc = self._notification_service
        if svc is None:
            return

        cam_uuid = str(resp.camera_uuid)

        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning("Skipping ROI alert publish because camera context was not found camera=%s", cam_uuid)
            return

        site_name = ctx.site_name
        site_uuid_str = str(ctx.site_uuid)

        for a in alerts:
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
                roi_id=str(a.get("roi_id", "")),
                track_id=track_id,
                device_name=ctx.device_name if ctx else None,
                camera_name=ctx.camera_name if ctx else None,
            )

            await svc.hub.publish(msg)

            asyncio.create_task(
                self._persist_and_maybe_email(ctx=ctx, msg=msg, extra_payload={"alert": a}),
                name=f"persist_roi_alert:{cam_uuid}",
            )
                
    async def _emit_item_detected_notifications(
        self,
        resp: ObjDetectResponse,
        tracks: Tuple[Dict[str, Any], ...],
        track_events: Tuple[Tuple[str, int], ...],
    ) -> None:
        svc = self._notification_service
        if svc is None:
            return

        confirmed_track_ids = [int(track_id) for (ev_type, track_id) in track_events if ev_type == "track_confirmed"]
        if not confirmed_track_ids:
            return

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
            return

        site_name = ctx.site_name
        site_uuid_str = str(ctx.site_uuid)

        for track_id in confirmed_track_ids:
            tr = tracks_by_id.get(track_id)
            if tr is None:
                continue

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
                track_id=track_id,
                device_name=ctx.device_name if ctx else None,
                camera_name=ctx.camera_name if ctx else None,
            )

            await svc.hub.publish(msg)

            asyncio.create_task(
                self._persist_and_maybe_email(
                    ctx=ctx,
                    msg=msg,
                    extra_payload={"track": tr, "event": "track_confirmed"},
                ),
                name=f"persist_track_confirmed:{cam_uuid}:{track_id}",
            )

    async def _emit_detection_summary_notification(self, resp: ObjDetectResponse) -> None:
        svc = self._notification_service
        if svc is None:
            return

        cam_uuid = str(resp.camera_uuid)
        raw_detections = list(resp.detections or [])
        if not raw_detections:
            return

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
            return

        ctx = await self._get_camera_ctx(cam_uuid)
        if ctx is None:
            logger.warning("Skipping detection-summary alert because camera context was not found camera=%s", cam_uuid)
            return

        uniq_classes = sorted(set(classes))
        classes_text = ", ".join(uniq_classes)
        msg = NotificationMessage(
            user_id=int(ctx.user_id),
            id=f"{cam_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-summary",
            ts_ms=int(resp.frame_ts_ms),
            camera_uuid=cam_uuid,
            site_uuid=str(ctx.site_uuid),
            site_name=ctx.site_name,
            title=f"Detection: {classes_text}",
            body=f"Detected {classes_text} (max_conf={max_conf:.2f})",
            alert_type="detection_summary",
            cls_names=uniq_classes,
            max_conf=float(max_conf),
            device_name=ctx.device_name,
            camera_name=ctx.camera_name,
        )

        await svc.hub.publish(msg)

        asyncio.create_task(
            self._persist_and_maybe_email(
                ctx=ctx,
                msg=msg,
                extra_payload={"detections": filtered[:20], "event": "detection_summary"},
            ),
            name=f"persist_detection_summary:{cam_uuid}",
        )

    async def _poll_loop(self, key: str, ch: VideoChannel) -> None:
        backoff_ms = 250
        max_backoff_ms = 8000
        empty_miss_count = 0
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
            if not cfg.enabled:
                await asyncio.sleep(0.5)
                continue
            
            if getattr(cfg, "detection_enabled", True) is False:
                await asyncio.sleep(0.5)
                continue

            try:
                sem = self._device_fetch_semaphore(getattr(cfg, "device_url", None))
                async with sem:
                    payload = await ch.stream()
                resp = self._payload_to_resp(payload, ch)
                if resp is None:
                    empty_miss_count = min(empty_miss_count + 1, 6)
                    base_sleep_s = max(0.25, float(cfg.poll_interval_ms) / 1000.0)
                    miss_sleep_s = min(5.0, base_sleep_s * (2 ** (empty_miss_count - 1)))
                    await asyncio.sleep(miss_sleep_s)
                    continue

                empty_miss_count = 0
                if not self._is_new_detection(key, resp):
                    await asyncio.sleep(max(0.05, float(cfg.poll_interval_ms) / 1000.0))
                    continue

                tracker_out = self._tracker.update_from_event(payload or {})
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
                if getattr(cfg, "notification_enabled", True) and self._notification_service:
                    if track_events:
                        asyncio.create_task(self._emit_item_detected_notifications(resp2, tracks, track_events))
                    if alerts:
                        asyncio.create_task(self._emit_roi_alert_notifications(resp2, alerts))
                    if (not track_events) and (not alerts) and resp2.detections:
                        cam_uuid = str(resp2.camera_uuid)
                        if self._reserve_detection_summary_alert(cam_uuid):
                            asyncio.create_task(self._emit_detection_summary_notification(resp2))
                backoff_ms = 250
                await asyncio.sleep(max(0.05, float(cfg.poll_interval_ms) / 1000.0))

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Detection poll failed camera=%s", key)
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, max_backoff_ms)

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

        # NEW: frame size (prefer explicit)
        frame_w = payload.get("frame_w") or payload.get("image_w") or payload.get("width")
        frame_h = payload.get("frame_h") or payload.get("image_h") or payload.get("height")

        cam = str(payload.get("camera_uuid") or ch.config.camera_uuid)

        return ObjDetectResponse(
            camera_uuid=cam,
            frame_ts_ms=int(frame_ts_ms),
            frame_seq=int(frame_seq),
            frame_w=int(frame_w) if frame_w is not None else None,
            frame_h=int(frame_h) if frame_h is not None else None,
            site_uuid=ch.config.site_uuid,
            device_uuid=ch.config.device_uuid,
            detections=tuple(dets) if isinstance(dets, list) else (),
            pose=pose,
            inference_ms=int(inf_ms) if inf_ms is not None else None,
            model_id=str(model_id) if model_id is not None else None,
        )
