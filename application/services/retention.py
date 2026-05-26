import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import NotificationRepository
from application.repositories.video_repository import VideoRepository
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import (
    EventClipService,
    extract_notification_clip_storage_keys as _extract_notification_clip_storage_keys,
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
        if self._clip_service is not None:
            await self._clip_service.close()
        if self._image_service is not None:
            await self._image_service.close()

    async def purge(
        self,
        *,
        clip_retention_days: int,
        alert_retention_days: int,
        batch_size: int = 200,
    ) -> Dict[str, int]:
        clips_deleted = await self._purge_clips(
            retention_days=clip_retention_days,
            batch_size=batch_size,
        )
        alerts_deleted = await self._purge_alerts(
            retention_days=alert_retention_days,
            batch_size=batch_size,
        )
        return {
            "clips_deleted": clips_deleted,
            "alerts_deleted": alerts_deleted,
        }

    async def _purge_alerts(self, *, retention_days: int, batch_size: int) -> int:
        if retention_days < 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(retention_days))
        total_deleted = 0

        while True:
            async with self._session_factory() as db:
                rows = await NotificationRepository().list_notifications(
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
                        _extract_notification_clip_storage_keys(getattr(row, "payload", None)),
                    )
                    for row in rows
                    if getattr(row, "id", None) is not None
                ]

            for _row_id, image_key, clip_keys in batch:
                if image_key:
                    try:
                        await self._image_service.delete_blob(blob_name=image_key)
                    except Exception:
                        logger.warning(
                            "Failed deleting expired alert image blob %s; removing DB row anyway",
                            image_key,
                            exc_info=True,
                        )
                for clip_key in clip_keys:
                    try:
                        await self._clip_service.delete_blob(blob_name=clip_key)
                    except Exception:
                        logger.warning(
                            "Failed deleting expired notification clip blob %s; removing DB row anyway",
                            clip_key,
                            exc_info=True,
                        )

            ids = [row_id for row_id, _image_key, _clip_keys in batch]
            if not ids:
                break

            async with self._session_factory() as db:
                await NotificationRepository().delete_notifications(db, ids=ids)
                await db.commit()

            total_deleted += len(ids)

        return total_deleted

    async def _purge_clips(self, *, retention_days: int, batch_size: int) -> int:
        if retention_days < 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(retention_days))
        total_deleted = 0

        while True:
            async with self._session_factory() as db:
                rows = await VideoRepository().list_video_records(
                    db,
                    created_before=cutoff,
                    limit=max(1, int(batch_size)),
                    order_asc=True,
                )

                if not rows:
                    break

                batch = [
                    (
                        int(row.id),
                        str(getattr(row, "storage_key", "") or "").strip(),
                    )
                    for row in rows
                    if getattr(row, "id", None) is not None
                ]

            for _row_id, storage_key in batch:
                if storage_key:
                    try:
                        await self._clip_service.delete_blob(blob_name=storage_key)
                    except Exception:
                        logger.warning(
                            "Failed deleting expired clip blob %s; removing DB row anyway",
                            storage_key,
                            exc_info=True,
                        )

            ids = [row_id for row_id, _storage_key in batch]
            if not ids:
                break

            async with self._session_factory() as db:
                await VideoRepository().delete_video_records(db, ids=ids)
                await db.commit()

            total_deleted += len(ids)

        return total_deleted
