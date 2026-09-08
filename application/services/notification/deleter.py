"""Notification deleter with bulk SQL hide support."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from application.repositories.notification_repository import NotificationRepository

logger = logging.getLogger(__name__)

# How many blob deletes to run concurrently against Azure Storage.
_BLOB_DELETE_CONCURRENCY = 10


class NotificationDeleter:
    def __init__(
        self,
        repo: NotificationRepository,
        session_factory,
        image_service,
        buffer_poll_s: float = 1.0,
    ):
        self._repo = repo
        self._session_factory = session_factory
        self._image_service = image_service
        self._buffer_poll_s = buffer_poll_s
        
        self._buffer_lock = asyncio.Lock()
        self._delete_event = asyncio.Event()
        self._pending_delete_ids_by_user: Dict[int, Set[int]] = {}
        self._closing = False
        self._delete_task: Optional[asyncio.Task] = None

    def start(self):
        if self._delete_task is None or self._delete_task.done():
            self._closing = False
            self._delete_task = asyncio.create_task(
                self._delete_loop(),
                name="notification_delete_queue",
            )

    async def shutdown(self):
        self._closing = True
        self._delete_event.set()
        if self._delete_task is not None:
            try:
                await self._delete_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Notification delete task shutdown failed")
        self._delete_task = None

    async def _delete_alert_blob_keys(self, storage_keys: List[str]) -> None:
        """Delete alert images in bounded-concurrency batches, best effort."""
        unique_keys = [
            key
            for key in dict.fromkeys(str(k or "").strip() for k in storage_keys)
            if key
        ]
        if not unique_keys or not self._image_service:
            return

        try:
            for i in range(0, len(unique_keys), _BLOB_DELETE_CONCURRENCY):
                batch = unique_keys[i : i + _BLOB_DELETE_CONCURRENCY]
                results = await asyncio.gather(
                    *(self._image_service.delete_blob(blob_name=key) for key in batch),
                    return_exceptions=True,
                )
                for key, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Failed deleting alert image blob %s after alert hide: %s",
                            key,
                            result,
                        )
        except Exception:
            logger.exception("Failed bulk-deleting alert images")

    async def _batch_hide_notifications_by_id(
        self, *, user_id: int, notification_ids: List[int], batch_size: int = 1000
    ) -> List[str]:
        """Hide the given notifications in batches, returning their image keys.

        The keys are collected before hiding so the caller can delete the blobs
        afterwards, off the request path.
        """
        sorted_ids = sorted({int(nid) for nid in notification_ids if nid and int(nid) > 0})
        if not sorted_ids:
            return []

        storage_keys: List[str] = []
        for i in range(0, len(sorted_ids), batch_size):
            batch_ids = sorted_ids[i : i + batch_size]
            try:
                async with self._session_factory() as db:
                    rows = await self._repo.list_notifications(
                        db, user_id=int(user_id), ids=batch_ids, only_visible=True
                    )

                    matched_ids: List[int] = []
                    for row in rows:
                        try:
                            parsed_id = int(row.id)
                        except (TypeError, ValueError):
                            continue
                        if parsed_id <= 0:
                            continue
                        matched_ids.append(parsed_id)

                        if isinstance(row.payload, dict):
                            key = str(row.payload.get("image_storage_key") or "").strip()
                            if key:
                                storage_keys.append(key)

                    if not matched_ids:
                        continue

                    await self._repo.set_notifications_visibility(
                        db, user_id=int(user_id), ids=matched_ids, visible=False
                    )
                    await db.commit()
            except Exception:
                logger.exception(
                    "Batch hide failed for user=%s batch_size=%s", user_id, len(batch_ids)
                )
                raise

        return storage_keys

    async def _bulk_hide_notifications_by_filter(self, *, user_id: int, site_uuid: Optional[uuid.UUID] = None, camera_uuid: Optional[uuid.UUID] = None) -> None:
        """Hide notifications via a single bulk SQL statement, and clean up blobs via a streaming query."""
        try:
            async with self._session_factory() as db:
                affected_count = await self._repo.hide_notifications_by_filter(
                    db,
                    user_id=user_id,
                    site_uuid=site_uuid,
                    camera_uuid=camera_uuid,
                )
                await db.commit()
                logger.info("Bulk hid %s notifications for user=%s site=%s camera=%s", affected_count, user_id, site_uuid, camera_uuid)

            if affected_count > 0:
                asyncio.create_task(
                    self._cleanup_blobs_by_filter(user_id=user_id, site_uuid=site_uuid, camera_uuid=camera_uuid),
                    name=f"cleanup_blobs_u{user_id}_s{site_uuid}_c{camera_uuid}",
                )
        except Exception:
            logger.exception("Bulk hide failed for user=%s site=%s camera=%s", user_id, site_uuid, camera_uuid)

    async def _cleanup_blobs_by_filter(self, *, user_id: int, site_uuid: Optional[uuid.UUID] = None, camera_uuid: Optional[uuid.UUID] = None) -> None:
        try:
            batch = []
            async with self._session_factory() as db:
                async for key in self._repo.iter_storage_keys_by_filter(
                    db,
                    user_id=user_id,
                    site_uuid=site_uuid,
                    camera_uuid=camera_uuid,
                    batch_size=5000,
                ):
                    batch.append(key)
                    if len(batch) >= 1000:
                        await self._delete_alert_blob_keys(batch)
                        batch.clear()
                        
            if batch:
                await self._delete_alert_blob_keys(batch)
        except Exception:
            logger.exception("Blob cleanup streaming failed")

    async def _flush_delete_queue(self, *, force_all: bool = False) -> None:
        if not self._session_factory:
            return

        ready: Dict[int, List[int]] = {}
        async with self._buffer_lock:
            for user_id, ids in list(self._pending_delete_ids_by_user.items()):
                clean_ids = sorted({int(x) for x in ids if int(x) > 0})
                if clean_ids:
                    ready[int(user_id)] = clean_ids
            self._pending_delete_ids_by_user.clear()

        if not ready:
            return

        for user_id, ids in ready.items():
            try:
                storage_keys = await self._batch_hide_notifications_by_id(
                    user_id=int(user_id),
                    notification_ids=ids,
                    batch_size=1000,
                )

                if storage_keys:
                    asyncio.create_task(self._delete_alert_blob_keys(storage_keys), name=f"delete_alert_blobs_{user_id}")

            except Exception:
                logger.exception("Failed to apply queued notification hide user=%s ids=%s", user_id, ids)
                if not force_all:
                    async with self._buffer_lock:
                        bucket = self._pending_delete_ids_by_user.setdefault(int(user_id), set())
                        bucket.update(ids)
                    self._delete_event.set()

    async def _delete_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._delete_event.wait(), timeout=self._buffer_poll_s)
            except asyncio.TimeoutError:
                pass

            self._delete_event.clear()
            force_all = bool(self._closing)
            await self._flush_delete_queue(force_all=force_all)
            if force_all:
                return

    async def handle_deletion_event(
        self,
        *,
        user_id: int,
        notification_ids: Optional[List[int]] = None,
        site_uuid: Optional[str] = None,
        camera_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        ids = sorted({int(raw_id) for raw_id in (notification_ids or []) if raw_id is not None and int(raw_id) > 0})
        
        if not ids and (site_uuid or camera_uuid) and self._session_factory:
            su = None
            try:
                if site_uuid:
                    su = uuid.UUID(site_uuid)
            except ValueError:
                pass
                
            cu = None
            try:
                if camera_uuid:
                    cu = uuid.UUID(camera_uuid)
            except ValueError:
                pass
                
            if su is not None or cu is not None:
                # Bulk-SQL path: hides by filter without loading ids into memory.
                await self._bulk_hide_notifications_by_filter(user_id=int(user_id), site_uuid=su, camera_uuid=cu)
                return {"ok": True, "bulk": True}

        if not ids:
            return {"ok": True, "deleted": 0}

        async with self._buffer_lock:
            bucket = self._pending_delete_ids_by_user.setdefault(int(user_id), set())
            bucket.update(ids)

        self._delete_event.set()

        return {
            "ok": True,
            "deleted": len(ids),
            "notification_ids": ids[:1000],
        }
