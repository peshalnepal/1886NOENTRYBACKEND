# application/repositories/notification_repository.py

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import (
    Camera,
    Site,
    CameraDevice,
    Device,
    Notification,
    NotificationEmail,
)


def dt_from_ts_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _as_uuid(v: Any) -> Optional[uuid.UUID]:
    if v is None:
        return None
    if isinstance(v, uuid.UUID):
        return v
    return uuid.UUID(str(v))


@dataclass(frozen=True)
class CameraContext:
    user_id: int
    site_uuid: uuid.UUID
    site_name: str
    camera_name: Optional[str]
    device_uuid: Optional[uuid.UUID]
    device_name: Optional[str]


class NotificationRepository:
    """
    DB access for:
      - camera context (user/site/device)
      - notification emails (site-scoped)
      - notifications table inserts/updates

    Notes:
    - No commits here. Caller controls transaction boundaries.
    """

    async def get_camera_context(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> Optional[CameraContext]:
        cam_uuid = _as_uuid(camera_uuid)
        if cam_uuid is None:
            return None

        stmt = (
            select(
                Camera.user_id,
                Camera.site_uuid,
                Site.name,
                Camera.name,
                Camera.camera_code,
                Device.device_uuid,
                Device.name,
            )
            .select_from(Camera)
            .join(Site, Site.site_uuid == Camera.site_uuid)
            .outerjoin(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
            .outerjoin(Device, Device.device_uuid == CameraDevice.device_uuid)
            .where(Camera.camera_uuid == cam_uuid)
        )

        row = (await db.execute(stmt)).first()
        if not row:
            return None

        user_id, site_uuid, site_name, camera_name, camera_code, device_uuid, device_name = row
        display_camera_name = camera_name or camera_code

        return CameraContext(
            user_id=int(user_id),
            site_uuid=site_uuid,
            site_name=site_name or "Unknown Site",
            camera_name=display_camera_name,
            device_uuid=device_uuid,
            device_name=device_name,
        )

    async def list_notification_emails_for_site(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
        only_enabled: bool = True,
    ) -> List[str]:
        stmt = select(NotificationEmail.email).where(
            NotificationEmail.user_id == int(user_id),
            NotificationEmail.site_uuid == _as_uuid(site_uuid),
        )
        if only_enabled:
            stmt = stmt.where(NotificationEmail.is_enabled.is_(True))

        res = await db.execute(stmt)
        return [r[0] for r in res.all()]

    async def create_notification(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
        camera_uuid: Optional[uuid.UUID],
        device_uuid: Optional[uuid.UUID],
        event_type: str,
        title: Optional[str],
        message: Optional[str],
        payload: Optional[Dict[str, Any]],
        detected_at: datetime,
        status: str = "created",
        sent_at: Optional[datetime] = None,
    ) -> Notification:
        row = Notification(
            user_id=int(user_id),
            site_uuid=_as_uuid(site_uuid),
            camera_uuid=_as_uuid(camera_uuid),
            device_uuid=_as_uuid(device_uuid),
            event_type=event_type,
            title=title,
            message=message,
            payload=payload,
            detected_at=detected_at,
            status=status,
            sent_at=sent_at,
        )
        db.add(row)
        await db.flush()  # makes row.id available
        return row

    async def mark_notification_sent(
        self,
        db: AsyncSession,
        *,
        notification_id: int,
        sent_at: Optional[datetime] = None,
        status: str = "sent",
    ) -> None:
        stmt = (
            update(Notification)
            .where(Notification.id == int(notification_id))
            .values(
                status=status,
                sent_at=sent_at or datetime.now(timezone.utc),
            )
        )
        await db.execute(stmt)
        await db.flush()

    async def mark_notification_failed(
        self,
        db: AsyncSession,
        *,
        notification_id: int,
        status: str = "failed",
    ) -> None:
        stmt = (
            update(Notification)
            .where(Notification.id == int(notification_id))
            .values(status=status)
        )
        await db.execute(stmt)
        await db.flush()
