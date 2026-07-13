"""Notification batch inserter, WebSocket publisher, and email digest scheduler.

Extracted from the former `_service_flush.py` mixin.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.exc import OperationalError
from application.repositories.organization_repository import OrganizationRepository
from core.database_orm import Site
from sqlalchemy import select

from application.dtos import NotificationCreateDTO
from application.repositories.notification_repository import (
    CameraContext,
    NotificationRepository,
    dt_from_ts_ms,
)
from application.services.clip_storage import extract_notification_clip_external_ids
from application.services.notification.overlay_helpers import _json_safe
from application.services.notification.types import (
    BufferedNotification,
    NotificationMessage,
)

logger = logging.getLogger(__name__)


class NotificationFlusher:
    def __init__(
        self,
        repo: NotificationRepository,
        session_factory,
        hub,
        email,
        image_service,
        clip_manager,
        buffer_max_items: int = 100,
        buffer_max_age_s: float = 60.0,
        buffer_poll_s: float = 1.0,
        recipient_ttl_s: float = 60.0,
    ):
        self._repo = repo
        self._session_factory = session_factory
        self.hub = hub
        self.email = email
        self._image_service = image_service
        self._clip_manager = clip_manager
        
        self._buffer_max_items = buffer_max_items
        self._buffer_max_age_s = buffer_max_age_s
        self._buffer_poll_s = buffer_poll_s
        self._recipient_ttl_s = recipient_ttl_s
        
        self._recipient_cache: Dict[Tuple[int, str], Tuple[float, List[str]]] = {}
        # Per-site cache of the owning org's operator user-ids. A non-empty list
        # means new alerts are held for approval and routed live to those
        # operators. Keyed by site_uuid str -> (expiry_monotonic, operator_ids).
        self._approval_cache: Dict[str, Tuple[float, List[int]]] = {}
        
        self._buffer_lock = asyncio.Lock()
        self._flush_event = asyncio.Event()
        self._pending_by_user: Dict[int, Dict[str, Dict[str, List[BufferedNotification]]]] = {}
        self._pending_since: Dict[int, float] = {}
        self._active_flush_users: set[int] = set()
        self._closing = False
        
        self._flush_task: Optional[asyncio.Task] = None

    def start(self):
        if self._flush_task is None or self._flush_task.done():
            self._closing = False
            self._flush_task = asyncio.create_task(
                self._flush_loop(),
                name="notification_buffer_flush",
            )

    async def shutdown(self):
        self._closing = True
        self._flush_event.set()
        if self._flush_task is not None:
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Notification flush task shutdown failed")
        self._flush_task = None

    def invalidate_recipient_cache(self, *, user_id: int, site_uuid: Optional[uuid.UUID] = None) -> None:
        uid = int(user_id)
        if site_uuid is not None:
            self._recipient_cache.pop((uid, str(site_uuid)), None)
            return

        for key in list(self._recipient_cache.keys()):
            if key[0] == uid:
                self._recipient_cache.pop(key, None)
                
    async def purge_deleted_site(self, *, user_id: int, site_uuid: Optional[uuid.UUID], camera_uuids: Optional[List[uuid.UUID]]) -> None:
        uid = int(user_id)
        site_key = str(site_uuid) if site_uuid is not None else None
        camera_keys = {str(value) for value in (camera_uuids or []) if value is not None}

        async with self._buffer_lock:
            hierarchical = self._pending_by_user.get(uid)

            if hierarchical is not None:
                if site_key is not None:
                    hierarchical.pop(site_key, None)

                if camera_keys:
                    for raw_site_key in list(hierarchical.keys()):
                        per_camera = hierarchical.get(raw_site_key) or {}
                        for cam_key in list(per_camera.keys()):
                            if cam_key in camera_keys:
                                per_camera.pop(cam_key, None)
                        if not per_camera:
                            hierarchical.pop(raw_site_key, None)

                if hierarchical:
                    self._pending_by_user[uid] = hierarchical
                else:
                    self._pending_by_user.pop(uid, None)
                    self._pending_since.pop(uid, None)

        self.invalidate_recipient_cache(user_id=uid, site_uuid=site_uuid)

    async def enqueue(self, msg: NotificationMessage, ctx: CameraContext, extra_payload: Optional[Dict[str, Any]] = None) -> None:
        if not self._session_factory:
            return

        async with self._session_factory() as db:
            try:
                item = await self._prepare_notification_item(msg=msg, ctx=ctx, extra_payload=extra_payload)
                ok = await self._flush_user_batch(int(ctx.user_id), [item], db=db)
                if not ok:
                    await db.rollback()
                    logger.warning("Failed to persist notification immediately user=%s camera=%s msg_id=%s", ctx.user_id, msg.camera_uuid, msg.id)
                    return
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def _flush_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._flush_event.wait(), timeout=self._buffer_poll_s)
            except asyncio.TimeoutError:
                pass
            self._flush_event.clear()

            force_all = bool(self._closing)
            await self._flush_ready_users(force_all=force_all)
            if force_all:
                return

    async def _flush_ready_users(self, *, force_all: bool = False) -> None:
        now = time.monotonic()
        ready: Dict[int, List[BufferedNotification]] = {}

        async with self._buffer_lock:
            for user_id, hierarchical in list(self._pending_by_user.items()):
                if user_id in self._active_flush_users:
                    continue

                total_items = sum(len(alerts) for sites in hierarchical.values() for alerts in sites.values())
                
                if total_items == 0:
                    self._pending_by_user.pop(user_id, None)
                    self._pending_since.pop(user_id, None)
                    continue

                since = self._pending_since.get(user_id, now)
                if force_all or total_items >= self._buffer_max_items or (now - since) >= self._buffer_max_age_s:
                    items = self._flatten_user_alerts(hierarchical)
                    ready[user_id] = items
                    self._pending_by_user.pop(user_id, None)
                    self._pending_since.pop(user_id, None)
                    self._active_flush_users.add(user_id)

        if not ready:
            return

        results = await asyncio.gather(
            *[self._flush_user_batch(user_id, items) for user_id, items in ready.items()],
            return_exceptions=True,
        )

        for (user_id, items), result in zip(ready.items(), results):
            requeue_items = False
            if isinstance(result, Exception):
                logger.error("Notification batch flush failed user=%s", user_id, exc_info=(type(result), result, result.__traceback__))
                requeue_items = not force_all
            elif result is False:
                requeue_items = not force_all

            async with self._buffer_lock:
                self._active_flush_users.discard(user_id)

                if requeue_items:
                    hierarchical: Dict[str, Dict[str, List[BufferedNotification]]] = {}
                    for item in items:
                        site_uuid_str = str(item.ctx.site_uuid)
                        camera_uuid_str = str(item.msg.camera_uuid)
                        
                        if site_uuid_str not in hierarchical:
                            hierarchical[site_uuid_str] = {}
                        if camera_uuid_str not in hierarchical[site_uuid_str]:
                            hierarchical[site_uuid_str][camera_uuid_str] = []
                        
                        hierarchical[site_uuid_str][camera_uuid_str].append(item)
                    
                    self._pending_by_user[user_id] = hierarchical
                    self._pending_since[user_id] = time.monotonic() - self._buffer_max_age_s
                    self._flush_event.set()

                if user_id in self._pending_by_user:
                    total_items = sum(len(alerts) for sites in self._pending_by_user[user_id].values() for alerts in sites.values())
                    if total_items >= self._buffer_max_items:
                        self._flush_event.set()

    def _flatten_user_alerts(self, hierarchical: Dict[str, Dict[str, List[BufferedNotification]]]) -> List[BufferedNotification]:
        all_items: List[BufferedNotification] = []
        for sites in hierarchical.values():
            for alerts in sites.values():
                all_items.extend(alerts)
        all_items.sort(key=lambda x: int(x.msg.ts_ms))
        return all_items

    async def _get_recipients_for_sites_cached(self, *, user_id: int, site_uuids: List[uuid.UUID]) -> Dict[uuid.UUID, List[str]]:
        now = time.monotonic()
        out: Dict[uuid.UUID, List[str]] = {}
        missing: List[uuid.UUID] = []

        for site_uuid in site_uuids:
            cache_key = (int(user_id), str(site_uuid))
            hit = self._recipient_cache.get(cache_key)
            if hit and hit[0] > now:
                out[site_uuid] = list(hit[1])
            else:
                missing.append(site_uuid)

        if missing and self._session_factory:
            async with self._session_factory() as db:
                loaded = await self._repo.list_notification_emails_for_sites(
                    db,
                    user_id=int(user_id),
                    site_uuids=missing,
                    only_enabled=True,
                )

            for site_uuid in missing:
                emails = list(loaded.get(site_uuid, []))
                self._recipient_cache[(int(user_id), str(site_uuid))] = (now + self._recipient_ttl_s, emails)
                out[site_uuid] = list(emails)

        return out
    
    async def _operator_ids_for_sites(
        self, site_uuids: List[uuid.UUID]
    ) -> Dict[uuid.UUID, List[int]]:
        """For each site, the owning org's operator user-ids. An empty list means
        no operator, so alerts go straight to the end user. A non-empty list means
        alerts are held `pending` and routed live to those operators. Cached
        per-site with the recipient TTL."""
        now = time.monotonic()
        out: Dict[uuid.UUID, List[int]] = {}
        missing: List[uuid.UUID] = []
        for su in site_uuids:
            hit = self._approval_cache.get(str(su))
            if hit and hit[0] > now:
                out[su] = list(hit[1])
            else:
                missing.append(su)

        if missing and self._session_factory:

            org_repo = OrganizationRepository()
            async with self._session_factory() as db:
                for su in missing:
                    operator_ids: List[int] = []
                    try:
                        org_id = (
                            await db.execute(
                                select(Site.org_id).where(Site.site_uuid == su).limit(1)
                            )
                        ).scalar_one_or_none()
                        if org_id is not None:
                            operator_ids = await org_repo.list_operator_user_ids(db, org_id=int(org_id))
                    except Exception:
                        logger.warning("Operator-gate lookup failed for site=%s", su, exc_info=True)
                        operator_ids = []
                    self._approval_cache[str(su)] = (now + self._recipient_ttl_s, list(operator_ids))
                    out[su] = list(operator_ids)

        return out

    async def operator_user_ids_for_site(self, site_uuid: Any) -> List[int]:
        """Org operator user-ids for a single site (empty => no operator gate)."""
        if site_uuid is None:
            return []
        try:
            su = site_uuid if isinstance(site_uuid, uuid.UUID) else uuid.UUID(str(site_uuid))
        except Exception:
            return []
        result = await self._operator_ids_for_sites([su])
        return list(result.get(su, []))

    async def requires_operator_approval(self, site_uuid: Any) -> bool:
        """Whether a single site's org has an operator (so realtime alerts must
        be withheld from the end user until approved). Uses the cached lookup."""
        return bool(await self.operator_user_ids_for_site(site_uuid))
    
    async def _run_with_session(self, db: Optional[Any], fn: Callable[[Any], Any], retry: bool = False) -> Any:
        if db is not None:
            return await fn(db)

        for attempt in range(3 if retry else 1):
            try:
                async with self._session_factory() as _db:
                    result = await fn(_db)
                    await _db.commit()
                    await _db.refresh(result) if result is not None else None
                    return result
            except OperationalError as exc:
                if retry and attempt < 2 and "1205" in str(exc):
                    wait_s = 0.5 * (2 ** attempt)
                    await asyncio.sleep(wait_s)
                    continue
                raise
            except Exception:
                raise
    async def _flush_user_batch(self, user_id: int, items: List[BufferedNotification], db: Optional[Any] = None) -> bool:
        if not items:
            return True
        if not self._session_factory:
            return False

        # 1. Pre-calculate site-level operator data
        distinct_sites = list({item.ctx.site_uuid for item in items})
        operator_ids_by_site = await self._operator_ids_for_sites(distinct_sites)

        # 2. Materialize images concurrently in batches
        IMAGE_UPLOAD_CONCURRENCY = 10
        materialized: List[Tuple[Dict[str, Any], Optional[str], Optional[str]]] = []
        for batch_start in range(0, len(items), IMAGE_UPLOAD_CONCURRENCY):
            batch = items[batch_start : batch_start + IMAGE_UPLOAD_CONCURRENCY]
            results = await asyncio.gather(
                *(self._clip_manager.materialize_alert_image_payload(msg=item.msg, extra_payload=item.extra_payload) for item in batch),
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("Image materialization failed: %s", r)
                    materialized.append(({}, None, None))
                else:
                    materialized.append(r)

        # 3. Prepare creation DTOs and update items efficiently
        site_groups: Dict[uuid.UUID, List[int]] = defaultdict(list)
        create_rows: List[NotificationCreateDTO] = []

        for idx, item in enumerate(items):
            stored_extra, stored_url, stored_key = materialized[idx]
            
            # Determine operator presence for this specific site
            operator_ids = operator_ids_by_site.get(item.ctx.site_uuid) or []
            approval_status = "pending" if operator_ids else "approved"
            
            # Gather message updates to avoid multiple model_copy calls
            msg_updates = {}
            if stored_url:
                msg_updates["image_url"] = stored_url
            if stored_key:
                msg_updates["image_storage_key"] = stored_key

            if msg_updates or stored_extra != (item.extra_payload or {}):
                updated_msg = item.msg.model_copy(update=msg_updates) if msg_updates else item.msg
                item = BufferedNotification(msg=updated_msg, ctx=item.ctx, extra_payload=stored_extra)
                items[idx] = item

            # Prepare payload for DB
            msg_payload = item.msg.model_dump()
            site_groups[item.ctx.site_uuid].append(idx)
            
            create_rows.append(
                NotificationCreateDTO(
                    user_id=int(item.ctx.user_id),
                    site_uuid=item.ctx.site_uuid,
                    camera_uuid=uuid.UUID(item.msg.camera_uuid),
                    device_uuid=item.ctx.device_uuid,
                    event_type=item.msg.alert_type,
                    title=item.msg.title,
                    message=item.msg.body,
                    payload=_json_safe({"msg": msg_payload, "extra": stored_extra}),
                    detected_at=dt_from_ts_ms(item.msg.ts_ms),
                    status="created",
                    sent_at=None,
                    approval_status=approval_status,
                )
            )

        # 4. Persist to Database
        async def _do_create(_db):
            return await self._repo.create_notifications(_db, dtos=create_rows)

        try:
            rows = await self._run_with_session(db, _do_create, retry=True)
            if rows is None:
                return False
        except Exception:
            logger.exception("Failed to persist buffered notifications user=%s count=%s", user_id, len(items))
            return False

        # 5. Publish to Hub & Trigger Clip Finalization
        clip_finalize_targets: List[Tuple[int, BufferedNotification]] = []
        notification_ids_by_site: Dict[uuid.UUID, List[int]] = defaultdict(list)
        
        for idx, row in enumerate(rows):
            row_id = getattr(row, "id", None)
            if row_id is None:
                continue
                
            item = items[idx]
            notification_ids_by_site[item.ctx.site_uuid].append(int(row_id))
            
            operator_ids = operator_ids_by_site.get(item.ctx.site_uuid) or []
            approval_status = "pending" if operator_ids else "approved"
            
            # Single model_copy to prepare the published message
            published_msg = item.msg.model_copy(update={
                "db_id": int(row_id),
                "approval_status": approval_status
            })

            if operator_ids:
                await self.hub.publish_to_users(operator_ids, published_msg)
            else:
                await self.hub.publish(published_msg)
                
            if str(item.msg.clip_status or "") == "loading":
                clip_finalize_targets.append((int(row_id), item))

        for notification_id, item in clip_finalize_targets:
            asyncio.create_task(
                self._finalize_clip_for_notification(
                    notification_id=notification_id,
                    msg=item.msg.model_copy(update={"db_id": notification_id}),
                    ctx=item.ctx,
                    extra_payload=item.extra_payload,
                ),
                name=f"clip_finalize:{notification_id}",
            )

        # 6. Process Emails
        if not self.email:
            return True

        recipients_by_site = await self._get_recipients_for_sites_cached(
            user_id=int(user_id), 
            site_uuids=list(site_groups.keys())
        )
        
        sent_ids: List[int] = []
        failed_ids: List[int] = []
        sent_at = datetime.now(timezone.utc)

        for site_uuid, indices in site_groups.items():
            # Skip email if this site requires operator approval
            if operator_ids_by_site.get(site_uuid):
                continue
                
            recipients = recipients_by_site.get(site_uuid, [])
            if not recipients:
                continue

            # Only grab messages for rows that were successfully created
            site_messages = [items[idx].msg for idx in indices if idx < len(rows) and getattr(rows[idx], "id", None)]
            site_notif_ids = notification_ids_by_site.get(site_uuid, [])
            
            if not site_messages:
                continue

            try:
                await self.email.send_digest(site_messages, to_emails=recipients)
                sent_ids.extend(site_notif_ids)
            except Exception:
                logger.exception("Buffered email digest send failed user=%s site=%s count=%s", user_id, site_uuid, len(site_messages))
                failed_ids.extend(site_notif_ids)

        # 7. Update Email Status in DB
        if sent_ids or failed_ids:
            async def _do_updates(_db):
                if sent_ids:
                    await self._repo.mark_notifications_sent(_db, notification_ids=sent_ids, sent_at=sent_at)
                if failed_ids:
                    await self._repo.mark_notifications_failed(_db, notification_ids=failed_ids, sent_at=sent_at)

            try:
                await self._run_with_session(db, _do_updates, retry=False)
            except Exception:
                logger.exception("Failed to update buffered notification statuses user=%s", user_id)

        return True
    
    async def email_approved_notifications(
        self,
        *,
        notification_ids: List[int],
        site_uuids: List[uuid.UUID],
    ) -> None:
        if not self.email or not notification_ids or not self._session_factory:
            return

        site_scope = [s for s in (site_uuids or []) if s is not None]

        async with self._session_factory() as db:
            rows = await self._repo.list_notifications(
                db,
                ids=[int(i) for i in notification_ids],
                site_uuids=site_scope or None,
                approval_status="approved",
            )

        # 1. Group cleanly by a tuple of (owner_id, site_uuid)
        grouped_rows: Dict[Tuple[int, uuid.UUID], List[Any]] = defaultdict(list)
        for r in rows:
            grouped_rows[(int(r.user_id), r.site_uuid)].append(r)

        sent_ids: List[int] = []
        failed_ids: List[int] = []
        sent_at = datetime.now(timezone.utc)

        # 2. Iterate flatly instead of nested loops
        for (owner_id, site_uuid), site_rows in grouped_rows.items():
            # Note: _get_recipients_for_sites_cached takes a list of sites, so we wrap it
            recipients_by_site = await self._get_recipients_for_sites_cached(
                user_id=owner_id, site_uuids=[site_uuid]
            )
            recipients = recipients_by_site.get(site_uuid, [])
            
            if not recipients:
                continue

            messages: List[NotificationMessage] = []
            ids_for_site: List[int] = []
            
            for r in site_rows:
                payload = r.payload if isinstance(r.payload, dict) else {}
                msg_payload = payload.get("msg")
                if not isinstance(msg_payload, dict):
                    continue
                try:
                    messages.append(
                        NotificationMessage.model_validate({**msg_payload, "db_id": int(r.id)})
                    )
                    ids_for_site.append(int(r.id))
                except Exception:
                    logger.warning("Could not rebuild approved notification for email id=%s", r.id, exc_info=True)

            if not messages:
                continue

            try:
                await self.email.send_digest(messages, to_emails=recipients)
                sent_ids.extend(ids_for_site)
            except Exception:
                logger.exception("Approval email send failed owner=%s site=%s count=%s", owner_id, site_uuid, len(messages))
                failed_ids.extend(ids_for_site)

        if not sent_ids and not failed_ids:
            return

        try:
            async with self._session_factory() as db:
                if sent_ids:
                    await self._repo.mark_notifications_sent(db, notification_ids=sent_ids, sent_at=sent_at)
                if failed_ids:
                    await self._repo.mark_notifications_failed(db, notification_ids=failed_ids)
                await db.commit()
        except Exception:
            logger.exception("Failed to update approved notification email statuses ids=%s", notification_ids)
            
    async def _prepare_notification_item(self, msg: NotificationMessage, ctx: CameraContext, extra_payload: Optional[Dict[str, Any]] = None) -> BufferedNotification:
        # Clip capture is decoupled from notification persistence: we mark
        # clip_status="loading" up front so the row persists, WS publishes, and
        # the email goes out immediately. A background task captures the clip
        # and patches the row + republishes with clip_status="ready".
        if msg.clip_status is None:
            msg = msg.model_copy(update={"clip_status": "loading"})
        return BufferedNotification(msg=msg, ctx=ctx, extra_payload=extra_payload)

    async def _finalize_clip_for_notification(
        self,
        *,
        notification_id: int,
        msg: NotificationMessage,
        ctx: CameraContext,
        extra_payload: Optional[Dict[str, Any]],
    ) -> None:
        """Background task: capture the clip, patch the persisted row, republish.

        Runs after the notification has already been persisted, WS-published,
        and emailed. It waits for the POST_EVENT_S tail, downloads the clip,
        then updates `Notification.payload` with the resolved clip block and
        republishes the message with `clip_status="ready"` so the web UI can
        swap the spinner for the player.
        """
        # When the site's org has an operator, the captured playback must be
        # held invisible (like the alert itself) until the operator approves it,
        # so the end user never sees the clip before review.
        operator_ids = await self.operator_user_ids_for_site(ctx.site_uuid)
        requires_approval = bool(operator_ids)

        try:
            merged_extra = await self._clip_manager.attach_clip_payload(
                msg=msg, ctx=ctx, extra_payload=extra_payload,
                requires_approval=requires_approval,
            )
        except Exception:
            logger.exception("Background clip capture failed notif_id=%s camera=%s", notification_id, msg.camera_uuid)
            merged_extra = extra_payload

        clip_payload = merged_extra.get("clip") if isinstance(merged_extra, dict) else None
        clip_url = str(clip_payload.get("recording_url") or "").strip() if isinstance(clip_payload, dict) else ""
        clip_status = "ready" if clip_url else "unavailable"

        updated_msg = msg.model_copy(update={
            "clip_url": clip_url or None,
            "clip_status": clip_status,
            "db_id": int(notification_id),
        })

        if self._session_factory is not None:
            try:
                msg_payload = updated_msg.model_dump()
                new_payload = _json_safe({"msg": msg_payload, "extra": merged_extra})
                async with self._session_factory() as db:
                    await self._repo.update_notification_payload(
                        db, notification_id=int(notification_id), payload=new_payload,
                    )
                    await db.commit()
            except Exception:
                logger.exception("Failed to persist resolved clip notif_id=%s", notification_id)

        # Settle the approve-during-capture race: clips are captured ~tens of
        # seconds after the alert is persisted, so the operator may have already
        # decided the alert before the clip rows existed (the approve/reject
        # route's clip flip would have matched nothing). Re-read the current
        # approval state and reconcile the just-captured clips with it.
        if requires_approval and self._session_factory is not None:
            try:
                external_ids = extract_notification_clip_external_ids(
                    {"extra": merged_extra} if isinstance(merged_extra, dict) else None
                )
                if external_ids:
                    async with self._session_factory() as db:
                        rows = await self._repo.list_notifications(db, ids=[int(notification_id)])
                    current_status = str(getattr(rows[0], "approval_status", "pending")) if rows else "pending"
                    if current_status == "approved":
                        await self._clip_manager.set_clips_approval(external_ids=external_ids, approved=True)
                    elif current_status == "rejected":
                        await self._clip_manager.set_clips_approval(external_ids=external_ids, approved=False)
            except Exception:
                logger.exception("Failed to reconcile captured clip approval notif_id=%s", notification_id)

        try:
            # Route the clip-ready republish the same way as the original alert:
            # to operators while it's awaiting approval, to the end user otherwise.
            if operator_ids:
                await self.hub.publish_to_users(
                    operator_ids, updated_msg.model_copy(update={"approval_status": "pending"})
                )
            else:
                await self.hub.publish(updated_msg.model_copy(update={"approval_status": "approved"}))
        except Exception:
            logger.exception("Failed to republish clip-ready notification notif_id=%s", notification_id)
