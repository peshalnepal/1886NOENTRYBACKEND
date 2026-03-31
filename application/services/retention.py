import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import EventClipService
from core.database_orm import Notification, VideoRecord


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
                rows = (
                    await db.execute(
                        select(Notification)
                        .where(Notification.detected_at < cutoff)
                        .order_by(Notification.id.asc())
                        .limit(max(1, int(batch_size)))
                    )
                ).scalars().all()

                if not rows:
                    break

                for row in rows:
                    storage_key = extract_image_storage_key(getattr(row, "payload", None))
                    if storage_key:
                        try:
                            await self._image_service.delete_blob(blob_name=storage_key)
                        except Exception:
                            logger.warning(
                                "Failed deleting expired alert image blob %s; removing DB row anyway",
                                storage_key,
                                exc_info=True,
                            )
                    await db.delete(row)

                await db.commit()
                total_deleted += len(rows)

        return total_deleted

    async def _purge_clips(self, *, retention_days: int, batch_size: int) -> int:
        if retention_days < 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=int(retention_days))
        total_deleted = 0

        while True:
            async with self._session_factory() as db:
                rows = (
                    await db.execute(
                        select(VideoRecord)
                        .where(VideoRecord.created_at < cutoff)
                        .order_by(VideoRecord.id.asc())
                        .limit(max(1, int(batch_size)))
                    )
                ).scalars().all()

                if not rows:
                    break

                for row in rows:
                    storage_key = str(getattr(row, "storage_key", "") or "").strip()
                    if storage_key:
                        try:
                            await self._clip_service.delete_blob(blob_name=storage_key)
                        except Exception:
                            logger.warning(
                                "Failed deleting expired clip blob %s; removing DB row anyway",
                                storage_key,
                                exc_info=True,
                            )
                    await db.delete(row)

                await db.commit()
                total_deleted += len(rows)

        return total_deleted
