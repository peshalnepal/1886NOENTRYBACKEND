# application/repositories/notification_repository.py

import uuid
from collections import defaultdict
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
    SiteSettings,
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
    camera_code: Optional[str]
    camera_name: Optional[str]
    device_uuid: Optional[uuid.UUID]
    device_name: Optional[str]
    notification_trigger_mode: Optional[str] = None
    camera_playback_enabled: Optional[bool] = None


@dataclass(frozen=True)
class SitePrerecordSettings:
    enabled: bool
    camera_uuids: List[uuid.UUID]
    trigger_mode: str = "roi_enter"


def _normalize_trigger_mode(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value == "any_detection":
        return "any_detection"
    return "roi_enter"


def _normalize_uuid_list(raw: Any) -> List[uuid.UUID]:
    if not isinstance(raw, (list, tuple, set)):
        return []

    seen = set()
    out: List[uuid.UUID] = []
    for item in raw:
        try:
            parsed = _as_uuid(item)
        except Exception:
            parsed = None
        if parsed is None:
            continue
        key = str(parsed)
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)
    return out


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
                Camera.notification_trigger_mode,
                Camera.camera_playback_enabled,
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

        (
            user_id, site_uuid, site_name, camera_name, camera_code,
            device_uuid, device_name,
            notification_trigger_mode, camera_playback_enabled,
        ) = row
        display_camera_name = camera_name or camera_code

        return CameraContext(
            user_id=int(user_id),
            site_uuid=site_uuid,
            site_name=site_name or "Unknown Site",
            camera_code=str(camera_code) if camera_code else None,
            camera_name=display_camera_name,
            device_uuid=device_uuid,
            device_name=device_name,
            notification_trigger_mode=str(notification_trigger_mode) if notification_trigger_mode else None,
            camera_playback_enabled=bool(camera_playback_enabled) if camera_playback_enabled is not None else None,
        )

    async def list_camera_contexts(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        camera_uuids: List[uuid.UUID],
    ) -> Dict[uuid.UUID, CameraContext]:
        camera_uuid_values = _normalize_uuid_list(camera_uuids or [])
        if not camera_uuid_values:
            return {}

        stmt = (
            select(
                Camera.camera_uuid,
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
            .where(
                Camera.user_id == int(user_id),
                Camera.camera_uuid.in_(camera_uuid_values),
            )
        )

        rows = (await db.execute(stmt)).all()
        out: Dict[uuid.UUID, CameraContext] = {}
        for (
            camera_uuid,
            camera_user_id,
            site_uuid,
            site_name,
            camera_name,
            camera_code,
            device_uuid,
            device_name,
        ) in rows:
            display_camera_name = camera_name or camera_code
            out[camera_uuid] = CameraContext(
                user_id=int(camera_user_id),
                site_uuid=site_uuid,
                site_name=site_name or "Unknown Site",
                camera_code=str(camera_code) if camera_code else None,
                camera_name=display_camera_name,
                device_uuid=device_uuid,
                device_name=device_name,
            )
        return out

    async def get_site_prerecord_settings(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
    ) -> SitePrerecordSettings:
        stmt = select(SiteSettings.config).where(
            SiteSettings.user_id == int(user_id),
            SiteSettings.site_uuid == _as_uuid(site_uuid),
        )
        config = (await db.execute(stmt)).scalar_one_or_none()
        if not isinstance(config, dict):
            return SitePrerecordSettings(enabled=False, camera_uuids=[])

        block = config.get("multi_camera_prerecord")
        if not isinstance(block, dict):
            return SitePrerecordSettings(enabled=False, camera_uuids=[])

        return SitePrerecordSettings(
            enabled=bool(block.get("enabled")),
            camera_uuids=_normalize_uuid_list(block.get("camera_uuids")),
            trigger_mode=_normalize_trigger_mode(block.get("trigger_mode")),
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

    async def list_notification_emails_for_sites(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuids: List[uuid.UUID],
        only_enabled: bool = True,
    ) -> Dict[uuid.UUID, List[str]]:
        site_uuid_values = [_as_uuid(site_uuid) for site_uuid in (site_uuids or [])]
        site_uuid_values = [site_uuid for site_uuid in site_uuid_values if site_uuid is not None]
        if not site_uuid_values:
            return {}

        stmt = select(NotificationEmail.site_uuid, NotificationEmail.email).where(
            NotificationEmail.user_id == int(user_id),
            NotificationEmail.site_uuid.in_(site_uuid_values),
        )
        if only_enabled:
            stmt = stmt.where(NotificationEmail.is_enabled.is_(True))

        rows = (await db.execute(stmt)).all()
        grouped: Dict[uuid.UUID, List[str]] = defaultdict(list)
        for site_uuid, email in rows:
            grouped[site_uuid].append(email)

        return {site_uuid: emails for site_uuid, emails in grouped.items()}

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

    async def create_notifications(
        self,
        db: AsyncSession,
        *,
        rows: List[Dict[str, Any]],
    ) -> List[Notification]:
        notifications: List[Notification] = []
        for item in rows or []:
            row = Notification(
                user_id=int(item["user_id"]),
                site_uuid=_as_uuid(item["site_uuid"]),
                camera_uuid=_as_uuid(item.get("camera_uuid")),
                device_uuid=_as_uuid(item.get("device_uuid")),
                event_type=item["event_type"],
                title=item.get("title"),
                message=item.get("message"),
                payload=item.get("payload"),
                detected_at=item["detected_at"],
                status=item.get("status", "created"),
                sent_at=item.get("sent_at"),
            )
            notifications.append(row)

        if notifications:
            db.add_all(notifications)
            await db.flush()

        return notifications

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

    async def mark_notifications_sent(
        self,
        db: AsyncSession,
        *,
        notification_ids: List[int],
        sent_at: Optional[datetime] = None,
        status: str = "sent",
    ) -> None:
        ids = sorted({int(notification_id) for notification_id in (notification_ids or []) if int(notification_id) > 0})
        if not ids:
            return

        stmt = (
            update(Notification)
            .where(Notification.id.in_(ids))
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

    async def mark_notifications_failed(
        self,
        db: AsyncSession,
        *,
        notification_ids: List[int],
        status: str = "failed",
    ) -> None:
        ids = sorted({int(notification_id) for notification_id in (notification_ids or []) if int(notification_id) > 0})
        if not ids:
            return

        stmt = (
            update(Notification)
            .where(Notification.id.in_(ids))
            .values(status=status)
        )
        await db.execute(stmt)
        await db.flush()
