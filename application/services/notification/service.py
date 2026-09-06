"""The `NotificationService` facade: one delivery surface over the flusher,
clip manager and deleter."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from core.env import env_float, env_int
from application.repositories.notification_repository import NotificationRepository, CameraContext
from application.services.alert_image_storage import AlertImageStorageService
from application.services.clip_storage import EventClipService
from application.services.notification.email_notifier import EmailNotifier
from application.services.notification.hub import WebNotificationHub
from application.services.notification.types import NotificationMessage

from application.services.notification.flusher import NotificationFlusher
from application.services.notification.clip_manager import ClipManager
from application.services.notification.deleter import NotificationDeleter

logger = logging.getLogger(__name__)


class NotificationService:
    def __init__(
        self,
        *,
        hub: WebNotificationHub,
        email: Optional[EmailNotifier] = None,
        clip_service: Optional[EventClipService] = None,
        image_service: Optional[AlertImageStorageService] = None,
    ):
        self.hub = hub
        self.email = email
        self._clip_service = clip_service or EventClipService()
        self._image_service = image_service or AlertImageStorageService()
        self._repo = NotificationRepository()
        self._session_factory = None
        
        self.clip_manager = ClipManager(
            clip_service=self._clip_service,
            repo=self._repo,
            session_factory=self._session_factory,
            image_service=self._image_service,
            # Per-camera overlay history is a fixed-size FIFO ring buffer (latest
            # in, oldest out) — retention is bounded by COUNT, not wall-clock time,
            # so a clip's PRE/POST window is always reconstructable regardless of
            # the variable per-camera detection cadence. Default and floor are
            # 10800 = 60*60*3 frames per camera.
            clip_overlay_history_max_frames=env_int("CLIP_OVERLAY_HISTORY_MAX_FRAMES_PER_CAMERA", 10800, minimum=10800),
            prerecord_eligible_ttl_s=env_float("PRERECORD_ELIGIBLE_CACHE_TTL_S", 30.0, minimum=5.0),
            # These timeouts MUST exceed the clip service's post-event tail wait
            # (POST_EVENT_S + 2s, ~32s by default) plus download/upload time, or
            # capture_pre_event_clip is cancelled mid-sleep and no clip is produced.
            # Capture runs on a background finalize task, so generous values are safe.
            site_prerecord_timeout_s=env_float("SITE_PRERECORD_TIMEOUT_S", 90.0, minimum=1.0),
            trigger_camera_timeout_s=env_float("TRIGGER_CAMERA_TIMEOUT_S", 90.0, minimum=1.0),
        )
        
        self.flusher = NotificationFlusher(
            repo=self._repo,
            session_factory=self._session_factory,
            hub=self.hub,
            email=self.email,
            image_service=self._image_service,
            clip_manager=self.clip_manager,
            buffer_max_items=env_int("NOTIFICATION_BUFFER_MAX_ITEMS", 100, minimum=1),
            buffer_max_age_s=env_float("NOTIFICATION_BUFFER_MAX_AGE_S", 60.0, minimum=1.0),
            buffer_poll_s=env_float("NOTIFICATION_BUFFER_POLL_S", 1.0, minimum=0.2),
            recipient_ttl_s=env_float("NOTIFICATION_RECIPIENT_CACHE_TTL_S", 60.0, minimum=1.0),
        )
        
        self.deleter = NotificationDeleter(
            repo=self._repo,
            session_factory=self._session_factory,
            image_service=self._image_service,
            buffer_poll_s=env_float("NOTIFICATION_BUFFER_POLL_S", 1.0, minimum=0.2),
        )
        self._bg_tasks: set[asyncio.Task] = set()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.flusher.start()
        self.deleter.start()
        self._started = True

    def set_session_factory(self, session_factory):
        self._session_factory = session_factory
        self.clip_manager._session_factory = session_factory
        self.flusher._session_factory = session_factory
        self.deleter._session_factory = session_factory
        if self._clip_service is not None:
            try:
                self._clip_service.set_session_factory(session_factory)
            except Exception:
                logger.exception("Failed to set session factory on EventClipService")

    async def shutdown(self) -> None:
        self._started = False
        await self.flusher.shutdown()
        await self.deleter.shutdown()

        for service, label in (
            (self._clip_service, "Event clip service"),
            (self._image_service, "Alert image service"),
        ):
            if service is None:
                continue
            try:
                await service.close()
            except Exception:
                logger.exception("%s shutdown failed", label)

    # --- Facade API for ModelPipeline and Routes ---
    
    async def enqueue_notification(self, msg: NotificationMessage, ctx: CameraContext, extra_payload: Optional[Dict[str, Any]] = None) -> None:
        await self.flusher.enqueue(msg, ctx, extra_payload)

    async def requires_operator_approval(self, site_uuid: Any) -> bool:
        """True when the site's org has an operator, meaning realtime alerts
        must be withheld from the end user until the operator approves them."""
        return await self.flusher.requires_operator_approval(site_uuid)

    def queue_approved_emails(
        self, *, notification_ids: List[int], site_uuids: List[uuid.UUID]
    ) -> None:
        """Fire-and-forget the approval email for the given notifications.

        Spawned from the approve route after the approval is committed, so the
        operator's request returns without waiting on SMTP. The task is tracked
        to keep a strong reference (asyncio only holds weak ones to live tasks).
        """
        if not notification_ids:
            return
        task = asyncio.create_task(
            self.flusher.email_approved_notifications(
                notification_ids=notification_ids, site_uuids=site_uuids
            ),
            name="email_approved_notifications",
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _load_notification_payloads(
        self, *, notification_ids: List[int], site_uuids: List[uuid.UUID]
    ) -> List[Any]:
        """Fetch the given notifications, scoped to the caller's sites."""
        if not notification_ids or self._session_factory is None:
            return []
        async with self._session_factory() as db:
            rows = await self._repo.list_notifications(
                db, ids=[int(i) for i in notification_ids], site_uuids=site_uuids or None
            )
        return [getattr(row, "payload", None) for row in rows]

    @staticmethod
    async def _delete_blobs(service, keys: List[str], *, label: str) -> None:
        """Best-effort blob deletion: one failure must not stop the rest."""
        if service is None:
            return
        for key in dict.fromkeys(k for k in keys if k):
            try:
                await service.delete_blob(blob_name=key)
            except Exception:
                logger.warning("Failed deleting %s blob %s", label, key, exc_info=True)

    async def set_clips_approval_for_notifications(
        self, *, notification_ids: List[int], site_uuids: List[uuid.UUID], approved: bool
    ) -> int:
        """Flip captured clips visible/hidden to match an operator's decision.

        Clips are matched to alerts by the external_id(s) embedded in each
        notification payload (there is no FK). Approval reveals the playback to
        the end user; rejection keeps it hidden. Returns rows updated.
        """
        from application.services.clip_storage import extract_notification_clip_external_ids

        payloads = await self._load_notification_payloads(
            notification_ids=notification_ids, site_uuids=site_uuids
        )
        external_ids = list(
            dict.fromkeys(
                ext
                for payload in payloads
                for ext in extract_notification_clip_external_ids(payload)
            )
        )
        if not external_ids:
            return 0

        return await self.clip_manager.set_clips_approval(
            external_ids=external_ids, approved=approved
        )

    async def purge_alert_media_for_notifications(
        self, *, notification_ids: List[int], site_uuids: List[uuid.UUID]
    ) -> None:
        """Delete the image and clip blobs behind rejected alerts.

        Storage keys are pulled from each notification payload — the same
        payload-driven extraction the retention service uses for expired alerts.
        The (now hidden) notification and VideoRecord rows stay put and are
        reaped later by retention.
        """
        from application.services.alert_image_storage import extract_image_storage_key
        from application.services.clip_storage import extract_notification_clip_storage_keys

        payloads = await self._load_notification_payloads(
            notification_ids=notification_ids, site_uuids=site_uuids
        )

        image_keys = [key for p in payloads if (key := extract_image_storage_key(p))]
        clip_keys = [k for p in payloads for k in extract_notification_clip_storage_keys(p)]

        await self._delete_blobs(self._image_service, image_keys, label="rejected alert image")
        await self._delete_blobs(self._clip_service, clip_keys, label="rejected alert clip")

    async def record_detection_overlay_frame(self, *, camera_uuid: str, frame_ts_ms: Any, frame_seq: Any, frame_w: Any = None, frame_h: Any = None, detections: Any = None) -> None:
        await self.clip_manager.record_detection_overlay_frame(
            camera_uuid=camera_uuid,
            frame_ts_ms=frame_ts_ms,
            frame_seq=frame_seq,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=detections,
        )

    async def handle_deletion_event(self, *, user_id: int, notification_ids: Optional[List[int]] = None, site_uuid: Optional[str] = None, camera_uuid: Optional[str] = None) -> Dict[str, Any]:
        return await self.deleter.handle_deletion_event(user_id=user_id, notification_ids=notification_ids, site_uuid=site_uuid, camera_uuid=camera_uuid)

    async def is_camera_prerecord_eligible(self, camera_uuid_str: str) -> bool:
        return await self.clip_manager.is_camera_prerecord_eligible(camera_uuid_str)

    async def purge_deleted_site_runtime_state(self, *, user_id: int, site_uuid: Optional[uuid.UUID] = None, camera_uuids: Optional[List[uuid.UUID]] = None) -> None:
        await self.flusher.purge_deleted_site(user_id=user_id, site_uuid=site_uuid, camera_uuids=camera_uuids)
        for cam_key in {str(value) for value in (camera_uuids or []) if value is not None}:
            await self.clip_manager.purge_deleted_camera(cam_key)
            
    def invalidate_recipient_cache(self, *, user_id: int, site_uuid: Optional[uuid.UUID] = None) -> None:
        self.flusher.invalidate_recipient_cache(user_id=user_id, site_uuid=site_uuid)
