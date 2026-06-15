"""Clip history, overlay payload, and prerecord manager.

Extracted from the former `_service_clip.py` and `_service_prerecord.py` mixins.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from application.repositories.notification_repository import CameraContext, NotificationRepository
from application.services.notification.overlay_helpers import (
    _build_overlay_payload_from_frames,
    _merge_overlay_frames,
    _normalize_overlay_frame,
    _parse_utc_datetime,
)
from application.services.notification.types import NotificationMessage, SitePrerecordPlan

logger = logging.getLogger(__name__)


class ClipManager:
    def __init__(
        self,
        clip_service,
        repo: NotificationRepository,
        session_factory,
        image_service,
        clip_overlay_history_ttl_s: float = 180.0,
        clip_overlay_history_max_frames: int = 10800,
        prerecord_eligible_ttl_s: float = 30.0,
        site_prerecord_timeout_s: float = 30.0,
        trigger_camera_timeout_s: float = 15.0,
    ):
        self._clip_service = clip_service
        self._repo = repo
        self._session_factory = session_factory
        self._image_service = image_service
        
        self._clip_overlay_history_ttl_s = clip_overlay_history_ttl_s
        self._clip_overlay_history_max_frames = clip_overlay_history_max_frames
        self._prerecord_eligible_ttl_s = prerecord_eligible_ttl_s
        self._site_prerecord_timeout_s = site_prerecord_timeout_s
        self._trigger_camera_timeout_s = trigger_camera_timeout_s
        
        self._overlay_history_by_camera: Dict[str, deque[Dict[str, Any]]] = {}
        self._overlay_history_lock = asyncio.Lock()
        self._prerecord_eligible_cache: Dict[str, Tuple[float, bool]] = {}
        
    def invalidate_prerecord_eligible_cache(self, camera_uuid_str: Optional[str] = None) -> None:
        if camera_uuid_str is None:
            self._prerecord_eligible_cache.clear()
        else:
            self._prerecord_eligible_cache.pop(str(camera_uuid_str), None)
            
    async def purge_deleted_camera(self, camera_uuid_str: str) -> None:
        async with self._overlay_history_lock:
            self._overlay_history_by_camera.pop(camera_uuid_str, None)
        self.invalidate_prerecord_eligible_cache(camera_uuid_str)

    async def is_camera_prerecord_eligible(self, camera_uuid_str: str, ctx: Optional[CameraContext] = None) -> bool:
        camera_uuid_key = str(camera_uuid_str)
        now = time.monotonic()
        hit = self._prerecord_eligible_cache.get(camera_uuid_key)
        if hit is not None and hit[0] > now:
            return hit[1]

        eligible = False
        try:
            if self._session_factory is not None:
                cam_uuid = uuid.UUID(camera_uuid_key)

                if ctx is None:
                    async with self._session_factory() as db:
                        ctx = await self._repo.get_camera_context(db, camera_uuid=cam_uuid)

                if ctx is not None:
                    async with self._session_factory() as db:
                        settings = await self._repo.get_site_prerecord_settings(
                            db,
                            user_id=int(ctx.user_id),
                            site_uuid=ctx.site_uuid,
                        )
                    eligible = bool(settings.enabled and cam_uuid in set(settings.camera_uuids))
        except Exception:
            logger.exception("Failed to check prerecord eligibility camera=%s", camera_uuid_str)

        self._prerecord_eligible_cache[camera_uuid_key] = (now + self._prerecord_eligible_ttl_s, eligible)
        return eligible

    async def materialize_alert_image_payload(self, *, msg: NotificationMessage, extra_payload: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
        payload = dict(extra_payload or {})
        image_url = str(payload.get("image_url") or msg.image_url or "").strip()
        if not image_url:
            return payload, msg.image_url, None

        image_service = self._image_service
        if image_service is None:
            return payload, image_url, str(payload.get("image_storage_key") or "").strip() or None

        if not image_url.startswith("data:"):
            return payload, image_url, str(payload.get("image_storage_key") or "").strip() or None

        try:
            stored = await image_service.store_image_data_url(
                image_data_url=image_url,
                camera_uuid=str(msg.camera_uuid),
                ts_ms=int(msg.ts_ms),
            )
        except Exception:
            logger.warning("Alert image upload failed for camera=%s msg_id=%s; keeping inline image payload", msg.camera_uuid, msg.id, exc_info=True)
            return payload, image_url, None
            
        if not stored:
            return payload, image_url, None

        stored_url = str(stored.get("image_url") or "").strip() or image_url
        stored_key = str(stored.get("image_storage_key") or "").strip() or None
        payload["image_url"] = stored_url
        if stored_key:
            payload["image_storage_key"] = stored_key
        return payload, stored_url, stored_key

    async def record_detection_overlay_frame(
        self,
        *,
        camera_uuid: str,
        frame_ts_ms: Any,
        frame_seq: Any,
        frame_w: Any = None,
        frame_h: Any = None,
        detections: Any = None,
    ) -> None:
        frame = _normalize_overlay_frame(
            camera_uuid=str(camera_uuid),
            frame_ts_ms=frame_ts_ms,
            frame_seq=frame_seq,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=detections,
        )
        if frame is None:
            return

        history_cutoff_ms = int(frame["frame_ts_ms"]) - int(self._clip_overlay_history_ttl_s * 1000.0)
        async with self._overlay_history_lock:
            bucket = self._overlay_history_by_camera.get(str(camera_uuid))
            if bucket is None:
                bucket = deque(maxlen=int(self._clip_overlay_history_max_frames))
                self._overlay_history_by_camera[str(camera_uuid)] = bucket

            while bucket and int(bucket[0].get("frame_ts_ms", 0)) < history_cutoff_ms:
                bucket.popleft()

            if bucket:
                last = bucket[-1]
                if (
                    int(last.get("frame_ts_ms", -1)) == int(frame["frame_ts_ms"])
                    and int(last.get("frame_seq", -1)) == int(frame["frame_seq"])
                ):
                    bucket[-1] = frame
                    return

            bucket.append(frame)

    def _build_clip_overlay_payload(self, *, msg: NotificationMessage, extra_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        payload = dict(extra_payload or {})

        detections = list(msg.detections or [])
        if not detections and isinstance(payload.get("detections"), list):
            detections = list(payload.get("detections") or [])

        frame_w = msg.frame_w if msg.frame_w is not None else payload.get("frame_w")
        frame_h = msg.frame_h if msg.frame_h is not None else payload.get("frame_h")
        frame_seq = msg.frame_seq if msg.frame_seq is not None else payload.get("frame_seq")

        if not detections and frame_w is None and frame_h is None:
            return None

        frame = _normalize_overlay_frame(
            camera_uuid=str(msg.camera_uuid),
            frame_ts_ms=msg.ts_ms,
            frame_seq=frame_seq or 0,
            frame_w=frame_w,
            frame_h=frame_h,
            detections=detections,
        )
        if frame is None:
            return None

        result = _build_overlay_payload_from_frames(
            camera_uuid=str(msg.camera_uuid),
            frames=[frame],
            preferred_ts_ms=int(msg.ts_ms),
            timeline_source="trigger_frame",
        )
        if result is not None and msg.alert_type:
            result["alert_type"] = str(msg.alert_type)
        return result

    async def _clip_overlay_frames_for_window(self, *, camera_uuid: str, start_time: Optional[datetime], end_time: Optional[datetime]) -> List[Dict[str, Any]]:
        if start_time is None or end_time is None:
            return []

        start_ts_ms = int(start_time.astimezone(timezone.utc).timestamp() * 1000.0)
        end_ts_ms = int(end_time.astimezone(timezone.utc).timestamp() * 1000.0)
        if end_ts_ms < start_ts_ms:
            start_ts_ms, end_ts_ms = end_ts_ms, start_ts_ms

        async with self._overlay_history_lock:
            bucket = list(self._overlay_history_by_camera.get(str(camera_uuid), ()))

        return [
            dict(frame)
            for frame in bucket
            if start_ts_ms <= int(frame.get("frame_ts_ms", -1)) <= end_ts_ms
        ]

    async def _trim_clip_overlay_history(self, *, camera_uuid: str, through_time: Optional[datetime]) -> None:
        if through_time is None:
            return

        through_ts_ms = int(through_time.astimezone(timezone.utc).timestamp() * 1000.0)
        async with self._overlay_history_lock:
            bucket = self._overlay_history_by_camera.get(str(camera_uuid))
            if not bucket:
                return

            while bucket and int(bucket[0].get("frame_ts_ms", 0)) <= through_ts_ms:
                bucket.popleft()

            if not bucket:
                self._overlay_history_by_camera.pop(str(camera_uuid), None)

    async def _finalize_captured_clip(
        self,
        *,
        camera_uuid: str,
        clip: Optional[Dict[str, Any]],
        preferred_ts_ms: Optional[int],
        fallback_overlay_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(clip, dict):
            return clip

        start_time = _parse_utc_datetime(clip.get("start_time"))
        end_time = _parse_utc_datetime(clip.get("end_time"))
        frames = await self._clip_overlay_frames_for_window(
            camera_uuid=str(camera_uuid),
            start_time=start_time,
            end_time=end_time,
        )

        fallback_frames = list(fallback_overlay_payload.get("frames") or []) if isinstance(fallback_overlay_payload, dict) else []
        if not fallback_frames and isinstance(fallback_overlay_payload, dict):
            fallback_frame = _normalize_overlay_frame(
                camera_uuid=str(camera_uuid),
                frame_ts_ms=fallback_overlay_payload.get("frame_ts_ms"),
                frame_seq=fallback_overlay_payload.get("frame_seq"),
                frame_w=fallback_overlay_payload.get("frame_w"),
                frame_h=fallback_overlay_payload.get("frame_h"),
                detections=fallback_overlay_payload.get("detections"),
            )
            if fallback_frame is not None:
                fallback_frames = [fallback_frame]

        overlay_payload = _build_overlay_payload_from_frames(
            camera_uuid=str(camera_uuid),
            frames=_merge_overlay_frames(frames, fallback_frames),
            preferred_ts_ms=preferred_ts_ms,
            clip_start_time=start_time,
            clip_end_time=end_time,
            timeline_source="clip_history",
        )

        if overlay_payload is None:
            return dict(clip)

        clip_service = self._clip_service
        external_id = str(clip.get("external_id") or "").strip()
        update_overlay = getattr(clip_service, "update_overlay_payload", None) if clip_service is not None else None
        if callable(update_overlay) and external_id:
            try:
                merged_overlay = await update_overlay(
                    camera_uuid=str(camera_uuid),
                    external_id=external_id,
                    overlay_payload=overlay_payload,
                )
                if isinstance(merged_overlay, dict):
                    overlay_payload = merged_overlay
            except Exception:
                logger.exception("Failed finalizing clip overlay camera=%s external_id=%s", camera_uuid, external_id)

        await self._trim_clip_overlay_history(camera_uuid=str(camera_uuid), through_time=end_time)

        finalized = dict(clip)
        finalized["overlay_payload"] = overlay_payload
        return finalized

    def _site_prerecord_trigger_matches(self, *, trigger_mode: str, alert_type: str) -> bool:
        normalized_mode = str(trigger_mode or "").strip().lower()
        normalized_alert = str(alert_type or "").strip().lower()

        if normalized_mode == "any_detection":
            return normalized_alert in {"roi_enter", "item_detected", "detection_summary"}

        return normalized_alert == "roi_enter"

    def _site_prerecord_clip_payload(self, *, camera_uuid: str, camera_ctx: CameraContext, clip: Dict[str, Any], trigger_camera_uuid: str) -> Dict[str, Any]:
        payload = dict(clip or {})
        payload["camera_uuid"] = str(camera_uuid)
        payload["camera_name"] = camera_ctx.camera_name
        payload["camera_code"] = camera_ctx.camera_code
        payload["site_uuid"] = str(camera_ctx.site_uuid)
        payload["is_trigger_camera"] = str(camera_uuid) == str(trigger_camera_uuid)
        return payload

    async def _load_site_prerecord_plan(self, *, msg: NotificationMessage, ctx: CameraContext) -> Optional[SitePrerecordPlan]:
        if self._session_factory is None:
            return None

        try:
            trigger_camera_uuid = uuid.UUID(str(msg.camera_uuid))
        except Exception:
            return None

        async with self._session_factory() as db:
            settings = await self._repo.get_site_prerecord_settings(
                db,
                user_id=int(ctx.user_id),
                site_uuid=ctx.site_uuid,
            )
            if not settings.enabled:
                return None
            if not self._site_prerecord_trigger_matches(
                trigger_mode=settings.trigger_mode,
                alert_type=msg.alert_type,
            ):
                return None

            selected_camera_uuids = list(settings.camera_uuids or [])
            if trigger_camera_uuid not in set(selected_camera_uuids):
                return None

            contexts_by_camera = await self._repo.list_camera_contexts(
                db,
                user_id=int(ctx.user_id),
                camera_uuids=selected_camera_uuids,
            )

        ordered_camera_uuids = [camera_uuid for camera_uuid in settings.camera_uuids if camera_uuid in contexts_by_camera]
        if not ordered_camera_uuids or trigger_camera_uuid not in contexts_by_camera:
            return None

        return SitePrerecordPlan(
            settings=settings,
            ordered_camera_uuids=ordered_camera_uuids,
            contexts_by_camera=contexts_by_camera,
        )

    async def set_clips_approval(self, *, external_ids: List[str], approved: bool) -> int:
        """Flip captured clips' visibility when an operator decides an alert."""
        clip_service = self._clip_service
        if clip_service is None:
            return 0
        setter = getattr(clip_service, "set_clips_approval", None)
        if not callable(setter):
            return 0
        try:
            return await setter(external_ids=external_ids, approved=approved)
        except Exception:
            logger.exception("Failed to set clip approval external_ids=%s approved=%s", external_ids, approved)
            return 0

    async def _capture_site_prerecord_clips(self, *, msg: NotificationMessage, ctx: CameraContext, plan: SitePrerecordPlan, trigger_clip: Optional[Dict[str, Any]] = None, requires_approval: bool = False) -> List[Dict[str, Any]]:
        clip_service = self._clip_service
        if clip_service is None:
            return []

        out: List[Dict[str, Any]] = []
        pending_camera_uuids: List[uuid.UUID] = []
        pending_tasks: List[asyncio.Future] = []
        trigger_camera_uuid_str = str(msg.camera_uuid)
        trigger_overlay_payload = self._build_clip_overlay_payload(msg=msg, extra_payload=None)

        for camera_uuid in plan.ordered_camera_uuids:
            camera_uuid_str = str(camera_uuid)
            camera_ctx = plan.contexts_by_camera.get(camera_uuid)
            if camera_ctx is None:
                continue

            if camera_uuid_str == trigger_camera_uuid_str and trigger_clip:
                out.append(self._site_prerecord_clip_payload(camera_uuid=camera_uuid_str, camera_ctx=camera_ctx, clip=trigger_clip, trigger_camera_uuid=msg.camera_uuid))
                continue

            pending_camera_uuids.append(camera_uuid)
            pending_tasks.append(
                clip_service.capture_pre_event_clip(
                    camera_uuid=camera_uuid_str,
                    ctx=camera_ctx,
                    event_ts_ms=msg.ts_ms,
                    trigger=f"site_prerecord:{msg.camera_uuid}:{plan.settings.trigger_mode}:{msg.alert_type}",
                    overlay_payload=(trigger_overlay_payload if camera_uuid_str == trigger_camera_uuid_str else None),
                    requires_approval=requires_approval,
                )
            )

        if pending_tasks:
            results = []
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*pending_tasks, return_exceptions=True),
                    timeout=self._site_prerecord_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning("Site prerecord clip capture timed out after %.1fs trigger_camera=%s pending_count=%s", self._site_prerecord_timeout_s, msg.camera_uuid, len(pending_camera_uuids))
                return out
            
            for camera_uuid, result in zip(pending_camera_uuids, results):
                camera_uuid_str = str(camera_uuid)
                if isinstance(result, Exception):
                    logger.exception("Failed capturing site prerecord clip trigger_camera=%s target_camera=%s", msg.camera_uuid, camera_uuid_str, exc_info=(type(result), result, result.__traceback__))
                    continue
                if not result:
                    continue
                camera_ctx = plan.contexts_by_camera.get(camera_uuid)
                if camera_ctx is None:
                    continue
                finalized_clip = await self._finalize_captured_clip(
                    camera_uuid=camera_uuid_str,
                    clip=result,
                    preferred_ts_ms=int(msg.ts_ms),
                    fallback_overlay_payload=(trigger_overlay_payload if camera_uuid_str == trigger_camera_uuid_str else None),
                )
                out.append(self._site_prerecord_clip_payload(camera_uuid=camera_uuid_str, camera_ctx=camera_ctx, clip=finalized_clip or result, trigger_camera_uuid=msg.camera_uuid))

        return out

    async def attach_clip_payload(self, *, msg: NotificationMessage, ctx: CameraContext, extra_payload: Optional[Dict[str, Any]], requires_approval: bool = False) -> Optional[Dict[str, Any]]:
        clip_service = self._clip_service
        if clip_service is None:
            return extra_payload

        plan = await self._load_site_prerecord_plan(msg=msg, ctx=ctx)
        if plan is None:
            return extra_payload

        overlay_payload = self._build_clip_overlay_payload(msg=msg, extra_payload=extra_payload)
        
        merged = dict(extra_payload or {})
        try:
            trigger_camera_uuid = uuid.UUID(str(msg.camera_uuid))
        except Exception:
            trigger_camera_uuid = None

        trigger_ctx = plan.contexts_by_camera.get(trigger_camera_uuid) if trigger_camera_uuid is not None else None
        trigger_mode = plan.settings.trigger_mode
        
        clip = None
        try:
            clip = await asyncio.wait_for(
                clip_service.capture_pre_event_clip(
                    camera_uuid=msg.camera_uuid,
                    ctx=trigger_ctx or ctx,
                    event_ts_ms=msg.ts_ms,
                    trigger=f"site_prerecord:{msg.camera_uuid}:{trigger_mode}:{msg.alert_type}",
                    overlay_payload=overlay_payload,
                    requires_approval=requires_approval,
                ),
                timeout=self._trigger_camera_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("Trigger camera clip capture timed out after %.1fs camera=%s", self._trigger_camera_timeout_s, msg.camera_uuid)
        except Exception:
            logger.exception("Trigger camera clip capture failed camera=%s", msg.camera_uuid)
        
        if clip:
            clip = await self._finalize_captured_clip(
                camera_uuid=str(msg.camera_uuid),
                clip=clip,
                preferred_ts_ms=int(msg.ts_ms),
                fallback_overlay_payload=overlay_payload,
            )
            merged["clip"] = clip

        if plan is None:
            return merged or extra_payload

        multi_clips = await self._capture_site_prerecord_clips(msg=msg, ctx=ctx, plan=plan, trigger_clip=clip, requires_approval=requires_approval)
        if multi_clips:
            merged["multi_camera_prerecordings"] = multi_clips

        return merged or extra_payload
