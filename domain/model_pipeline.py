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
    async def wait_new(self, camera_uuid: str, *, after_seq: int, timeout_ms: int) -> Optional[ObjDetectResponse]: ...


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

    async def wait_new(self, camera_uuid: str, *, after_seq: int, timeout_ms: int) -> Optional[ObjDetectResponse]:
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

            if resp is not None and int(resp.frame_seq) > int(after_seq):
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

        self._channels: Dict[str, VideoChannel] = {}
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._last_seq: Dict[str, int] = {}
        self._tracker: MultiCameraByteTrack = MultiCameraByteTrack()
        self._roi_engine = ROIAlertEngine()                 
        self._roi_provider: ROIProvider = roi_provider or self._no_rois 
        self._notification_service = notification_service    
        self._session_factory: Optional[SessionFactory] = None
        self._site_cache: Dict[str, Tuple[str, float]] = {}
        self._site_cache_lock = asyncio.Lock()
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
            self._channels.clear()
            self._started = False

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


    async def _no_rois(self, camera_uuid: str) -> List[ROI]:
        return []

    def set_notification_service(self, svc: NotificationService) -> None:
        self._notification_service = svc

    def set_roi_provider(self, roi_provider: ROIProvider) -> None:
        self._roi_provider = roi_provider

    def set_session_factory(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory


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

    # ---------- detections ----------
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

        last = self._last_seq.get(key, 0)
        if int(resp.frame_seq) <= int(last):
            return await self.detect_store.get_latest(key)

        self._last_seq[key] = int(resp.frame_seq)
        await self.detect_store.put(resp)
        return resp

    def _ensure_poller(self, key: str, ch: VideoChannel) -> None:
        t = self._poll_tasks.get(key)
        if t is None or t.done():
            self._poll_tasks[key] = asyncio.create_task(
                self._poll_loop(key, ch),
                name="poll_jetson_detection:%s" % key,
            )
            
    async def _emit_roi_alert_notifications(self, resp: ObjDetectResponse, alerts: List[Dict[str, Any]]) -> None:
        svc = self._notification_service
        if svc is None:
            return

        site_name = await self._get_site_name(resp.site_uuid)
        cam_uuid = str(resp.camera_uuid)
        to_emails: List[str] = []
        if svc.email:
            try:
                to_emails = await svc._get_notification_emails(cam_uuid)
            except Exception:
                logger.exception("Failed loading notification recipients for camera=%s", cam_uuid)

        for a in alerts:
            title = f"ROI Alert ({a.get('type','roi')})"
            body = f"{a.get('cls_name','object')} entered ROI {a.get('roi_id')} (track {a.get('track_id')})"

            msg = NotificationMessage(
                id=f"{resp.camera_uuid}-{resp.frame_ts_ms}-{resp.frame_seq}-{a.get('roi_id')}-{a.get('track_id')}",
                ts_ms=int(resp.frame_ts_ms),
                camera_uuid=cam_uuid,
                site_name=site_name,
                title=title,
                body=body,
                cls_names=[str(a.get("cls_name", "object"))],
                max_conf=float(a.get("conf", 0.0) or 0.0),
            )

            # web notification
            await svc.hub.publish(msg)
            # email if configured
            if svc.email:
                try:
                    await svc.email.send(msg, to_emails=to_emails if to_emails else None)
                except Exception:
                    logger.exception("Failed sending ROI email camera=%s alert=%s", cam_uuid, a)
                
    async def _poll_loop(self, key: str, ch: VideoChannel) -> None:
        backoff_ms = 250
        max_backoff_ms = 8000
        empty_miss_count = 0

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
                payload = await ch.stream()
                resp = self._payload_to_resp(payload, ch)
                if resp is None:
                    empty_miss_count = min(empty_miss_count + 1, 6)
                    base_sleep_s = max(0.25, float(cfg.poll_interval_ms) / 1000.0)
                    miss_sleep_s = min(5.0, base_sleep_s * (2 ** (empty_miss_count - 1)))
                    await asyncio.sleep(miss_sleep_s)
                    continue

                empty_miss_count = 0
                prev = int(self._last_seq.get(key, 0))
                if int(resp.frame_seq) <= prev:
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

                self._last_seq[key] = int(resp2.frame_seq)
                await self.detect_store.put(resp2)
                await self.detection_hub.publish(resp2)

                if alerts and getattr(cfg, "notification_enabled", True) and self._notification_service:
                    await self._emit_roi_alert_notifications(resp2, alerts)

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
