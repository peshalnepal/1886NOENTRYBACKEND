"""Age-based cleanup of stored alerts and clips.

Runs on a background loop (see `main._retention_cleanup_loop`). Blobs are
deleted first and the DB row only afterwards: a failed blob delete is logged and
the row is removed anyway, since a stale row is worse than an orphaned blob.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import NotificationRepository
from application.repositories.video_repository import VideoRepository
from application.services.alert_image_storage import (
    AlertImageStorageService,
    extract_image_storage_key,
)
from application.services.clip_storage import (
    EventClipService,
    extract_notification_clip_storage_keys,
)

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]


class RetentionService:
    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        clip_service: Optional[EventClipService] = None,
        image_service: Optional[AlertImageStorageService] = None,
    ) -> None:
        self._session_factory = session_factory
        self._clip_service = clip_service or EventClipService()
        self._image_service = image_service or AlertImageStorageService()

    async def close(self) -> None:
        for service in (self._clip_service, self._image_service):
            if service is not None:
                await service.close()

    async def purge(
        self,
        *,
        clip_retention_days: int,
        alert_retention_days: int,
        batch_size: int = 200,
    ) -> Dict[str, int]:
        return {
            "clips_deleted": await self._purge_clips(
                retention_days=clip_retention_days, batch_size=batch_size
            ),
            "alerts_deleted": await self._purge_alerts(
                retention_days=alert_retention_days, batch_size=batch_size
            ),
        }

    @staticmethod
    def _cutoff(retention_days: int) -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=int(retention_days))

    async def _delete_blob(self, service, key: str, *, label: str) -> None:
        """Delete one blob; a failure never blocks removing the DB row."""
        if not key or service is None:
            return
        try:
            await service.delete_blob(blob_name=key)
        except Exception:
            logger.warning(
                "Failed deleting expired %s blob %s; removing DB row anyway",
                label,
                key,
                exc_info=True,
            )

    async def _purge_alerts(self, *, retention_days: int, batch_size: int) -> int:
        if retention_days < 0:
            return 0

        cutoff = self._cutoff(retention_days)
        repo = NotificationRepository()
        total_deleted = 0

        while True:
            async with self._session_factory() as db:
                rows = await repo.list_notifications(
                    db,
                    detected_before=cutoff,
                    limit=max(1, int(batch_size)),
                    order_desc=False,
                )
                if not rows:
                    break

                batch = [
                    (
                        int(row.id),
                        extract_image_storage_key(getattr(row, "payload", None)),
                        extract_notification_clip_storage_keys(getattr(row, "payload", None)),
                    )
                    for row in rows
                    if getattr(row, "id", None) is not None
                ]

            if not batch:
                break

            for _row_id, image_key, clip_keys in batch:
                await self._delete_blob(self._image_service, image_key, label="alert image")
                for clip_key in clip_keys:
                    await self._delete_blob(
                        self._clip_service, clip_key, label="notification clip"
                    )

            ids = [row_id for row_id, _image_key, _clip_keys in batch]
            async with self._session_factory() as db:
                await repo.delete_notifications(db, ids=ids)
                await db.commit()

            total_deleted += len(ids)

        return total_deleted

    async def _purge_clips(self, *, retention_days: int, batch_size: int) -> int:
        if retention_days < 0:
            return 0

        cutoff = self._cutoff(retention_days)
        repo = VideoRepository()
        total_deleted = 0

        while True:
            async with self._session_factory() as db:
                rows = await repo.list_video_records(
                    db,
                    created_before=cutoff,
                    limit=max(1, int(batch_size)),
                    order_asc=True,
                )
                if not rows:
                    break

                batch = [
                    (int(row.id), str(getattr(row, "storage_key", "") or "").strip())
                    for row in rows
                    if getattr(row, "id", None) is not None
                ]

            if not batch:
                break

            for _row_id, storage_key in batch:
                await self._delete_blob(self._clip_service, storage_key, label="clip")

            ids = [row_id for row_id, _storage_key in batch]
            async with self._session_factory() as db:
                await repo.delete_video_records(db, ids=ids)
                await db.commit()

            total_deleted += len(ids)

        return total_deleted
