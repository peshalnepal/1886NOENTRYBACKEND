# application/repositories/notification_repository.py

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, delete, false, func, literal_column, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import (
    CameraContextDTO,
    NotificationCreateDTO,
    NotificationEmailCreateDTO,
    SitePrerecordSettingsDTO,
)
from application.repositories._helpers import as_uuid as _as_uuid, normalize_uuid_list
from core.database_orm import (
    Camera,
    Site,
    Device,
    Notification,
    NotificationEmail,
    SiteSettings,
    utc_now,
)

# Inter-function value objects live in application.dtos now; these aliases keep
# the historical import paths (`from ...notification_repository import CameraContext`)
# working for existing callers.
CameraContext = CameraContextDTO
SitePrerecordSettings = SitePrerecordSettingsDTO


def dt_from_ts_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _normalize_trigger_mode(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value == "any_detection":
        return "any_detection"
    return "roi_enter"


def _coerce_trigger_mode(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in ("roi_enter", "any_detection", "inherit"):
        return value
    return "inherit"


def _coerce_playback_mode(raw: Any) -> str:
    if raw is True:
        return "always"
    if raw is False:
        return "never"
    value = str(raw or "").strip().lower()
    if value in ("always", "never", "inherit"):
        return value
    return "inherit"


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
            .outerjoin(Device, Device.device_uuid == Camera.device_uuid)
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
            notification_trigger_mode=_coerce_trigger_mode(notification_trigger_mode),
            camera_playback_enabled=_coerce_playback_mode(camera_playback_enabled),
        )

    async def list_camera_contexts(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        camera_uuids: List[uuid.UUID],
    ) -> Dict[uuid.UUID, CameraContext]:
        camera_uuid_values = normalize_uuid_list(camera_uuids or [])
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
                Camera.notification_trigger_mode,
                Camera.camera_playback_enabled,
            )
            .select_from(Camera)
            .join(Site, Site.site_uuid == Camera.site_uuid)
            .outerjoin(Device, Device.device_uuid == Camera.device_uuid)
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
            notification_trigger_mode,
            camera_playback_enabled,
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
                notification_trigger_mode=_coerce_trigger_mode(notification_trigger_mode),
                camera_playback_enabled=_coerce_playback_mode(camera_playback_enabled),
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
            camera_uuids=normalize_uuid_list(block.get("camera_uuids")),
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

    @staticmethod
    def _notification_from_dto(dto: NotificationCreateDTO) -> Notification:
        return Notification(
            user_id=int(dto.user_id),
            site_uuid=_as_uuid(dto.site_uuid),
            camera_uuid=_as_uuid(dto.camera_uuid),
            device_uuid=_as_uuid(dto.device_uuid),
            event_type=dto.event_type,
            title=dto.title,
            message=dto.message,
            payload=dto.payload,
            detected_at=dto.detected_at or utc_now(),
            status=dto.status,
            sent_at=dto.sent_at,
            approval_status=dto.approval_status,
            visible=(dto.approval_status == "approved"),
        )

    async def create_notification(
        self,
        db: AsyncSession,
        *,
        dto: NotificationCreateDTO,
    ) -> Notification:
        """Insert one Notification from a `NotificationCreateDTO`."""
        row = self._notification_from_dto(dto)
        db.add(row)
        await db.flush()  # makes row.id available
        return row

    async def create_notifications(
        self,
        db: AsyncSession,
        *,
        dtos: List[NotificationCreateDTO],
    ) -> List[Notification]:
        """Batch-insert Notifications from a list of `NotificationCreateDTO`."""
        notifications = [self._notification_from_dto(dto) for dto in (dtos or [])]
        if notifications:
            db.add_all(notifications)
            await db.flush()
        return notifications

    async def update_notification_payload(
        self,
        db: AsyncSession,
        *,
        notification_id: int,
        payload: Dict[str, Any],
    ) -> None:
        """Overwrite the JSON payload column for a single notification row.

        Used after a clip is captured asynchronously: the row is created up
        front with clip_status=loading, then this updates it with the resolved
        recording_url and clip_status=ready.
        """
        nid = int(notification_id)
        if nid <= 0:
            return
        stmt = (
            update(Notification)
            .where(Notification.id == nid)
            .values(payload=payload)
        )
        await db.execute(stmt)
        await db.flush()

    async def append_note(
        self,
        db: AsyncSession,
        *,
        notification_id: int,
        text: str,
        author_id: int,
        author_name: Optional[str] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
    ) -> Optional[Notification]:
        """Append an operator note to a notification's payload.

        Notes accumulate as a list under ``payload["notes"]`` so they can later
        be rolled up into the per-organization end-of-day report. Scoped to
        ``site_uuids`` (the caller's readable sites) so an operator cannot annotate
        notifications outside their organization. Returns the updated row, or
        ``None`` when no in-scope notification matches.
        """
        nid = int(notification_id)
        if nid <= 0:
            return None

        note_text = str(text or "").strip()
        if not note_text:
            return None

        conds = self._notification_conditions(ids=[nid], site_uuids=site_uuids)
        stmt = select(Notification)
        if conds:
            stmt = stmt.where(and_(*conds))
        row = (await db.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None

        payload = dict(row.payload) if isinstance(row.payload, dict) else {}
        existing = payload.get("notes")
        notes = list(existing) if isinstance(existing, list) else []
        notes.append(
            {
                "text": note_text,
                "author_id": int(author_id),
                "author_name": str(author_name or "") or None,
                "created_at": utc_now().isoformat(),
            }
        )
        payload["notes"] = notes
        # Reassign (rather than mutate in place) so SQLAlchemy detects the JSON change.
        row.payload = payload
        await db.flush()
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

    # ------------------------------------------------------------------
    # Notification reads / lifecycle
    # ------------------------------------------------------------------
    def _notification_conditions(
        self,
        *,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        ids: Optional[List[int]] = None,
        event_types: Optional[List[str]] = None,
        approval_status: Optional[str] = None,
        only_visible: Optional[bool] = None,
        only_unread: Optional[bool] = None,
        detected_before: Optional[datetime] = None,
        before_id: Optional[int] = None,
    ) -> list:
        conds: list = []
        if user_id is not None:
            conds.append(Notification.user_id == int(user_id))
        if site_uuid is not None:
            conds.append(Notification.site_uuid == _as_uuid(site_uuid))
        if site_uuids is not None:
            clean_sites = [_as_uuid(s) for s in site_uuids if s is not None]
            # Empty allow-list -> match nothing (always-false predicate).
            conds.append(
                Notification.site_uuid.in_(clean_sites) if clean_sites else false()
            )
        if approval_status is not None:
            conds.append(Notification.approval_status == str(approval_status))
        if camera_uuid is not None:
            conds.append(Notification.camera_uuid == _as_uuid(camera_uuid))
        if ids is not None:
            conds.append(Notification.id.in_([int(i) for i in ids]))
        if event_types is not None:
            conds.append(Notification.event_type.in_(list(event_types)))
        if only_visible is True:
            conds.append(Notification.visible.is_(True))
        elif only_visible is False:
            conds.append(Notification.visible.is_(False))
        if only_unread is True:
            conds.append(Notification.read_at.is_(None))
        elif only_unread is False:
            conds.append(Notification.read_at.isnot(None))
        if detected_before is not None:
            conds.append(Notification.detected_at < detected_before)
        if before_id is not None:
            conds.append(Notification.id < int(before_id))
        return conds

    async def list_notifications(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        ids: Optional[List[int]] = None,
        event_types: Optional[List[str]] = None,
        approval_status: Optional[str] = None,
        only_visible: Optional[bool] = None,
        only_unread: Optional[bool] = None,
        detected_before: Optional[datetime] = None,
        before_id: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_desc: bool = True,
    ) -> List[Notification]:
        """Return Notification rows matching any combination of filters."""
        conds = self._notification_conditions(
            user_id=user_id, site_uuid=site_uuid, site_uuids=site_uuids,
            camera_uuid=camera_uuid, ids=ids,
            event_types=event_types, approval_status=approval_status,
            only_visible=only_visible, only_unread=only_unread,
            detected_before=detected_before, before_id=before_id,
        )
        stmt = select(Notification)
        if conds:
            stmt = stmt.where(and_(*conds))
        stmt = stmt.order_by(Notification.id.desc() if order_desc else Notification.id.asc())
        if offset is not None:
            stmt = stmt.offset(int(offset))
        if limit is not None:
            stmt = stmt.limit(int(limit))
        return (await db.execute(stmt)).scalars().all()

    async def list_notification_ids(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        order_desc: bool = True,
    ) -> List[int]:
        """Return just the Notification ids matching the given filters.

        Lighter than list_notifications when only ids are needed (e.g. snapshotting
        ids before a camera delete SET NULLs camera_uuid).
        """
        conds = self._notification_conditions(
            user_id=user_id, site_uuid=site_uuid, camera_uuid=camera_uuid,
        )
        stmt = select(Notification.id)
        if conds:
            stmt = stmt.where(and_(*conds))
        stmt = stmt.order_by(Notification.id.desc() if order_desc else Notification.id.asc())
        rows = (await db.execute(stmt)).scalars().all()
        return [int(r) for r in rows]

    async def count_notifications(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        approval_status: Optional[str] = None,
        only_visible: Optional[bool] = None,
        only_unread: Optional[bool] = None,
    ) -> int:
        """Count Notification rows matching any combination of filters."""
        conds = self._notification_conditions(
            user_id=user_id, site_uuid=site_uuid, site_uuids=site_uuids,
            camera_uuid=camera_uuid, approval_status=approval_status,
            only_visible=only_visible, only_unread=only_unread,
        )
        stmt = select(func.count(Notification.id))
        if conds:
            stmt = stmt.where(and_(*conds))
        return int((await db.execute(stmt)).scalar_one() or 0)

    async def mark_notifications_read(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        ids: Optional[List[int]] = None,
        read_at: Optional[datetime] = None,
    ) -> int:
        """Set read_at on a user's notifications. ids=None marks all. Returns count."""
        conds = self._notification_conditions(user_id=user_id, ids=ids, only_unread=True)
        result = await db.execute(
            update(Notification)
            .where(and_(*conds))
            .values(read_at=read_at or utc_now())
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def set_notifications_visibility(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        ids: List[int],
        visible: bool,
    ) -> int:
        """Show/hide a user's notifications by id. Returns affected row count."""
        if not ids:
            return 0
        conds = self._notification_conditions(user_id=user_id, ids=ids)
        if visible:
            conds.append(Notification.visible.is_(False))
        else:
            conds.append(Notification.visible.is_(True))
        result = await db.execute(
            update(Notification)
            .where(and_(*conds))
            .values(visible=bool(visible))
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    # ------------------------------------------------------------------
    # Operator approval workflow
    # ------------------------------------------------------------------
    async def set_approval(
        self,
        db: AsyncSession,
        *,
        ids: List[int],
        approval_status: str,
        visible: bool,
        approved_by: Optional[int] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        only_pending: bool = True,
    ) -> int:
        """Approve or reject pending notifications by id.

        Scoped to `site_uuids` (the operator's org sites) so an operator
        cannot act on another tenant's alerts. By default only rows still
        `pending` are affected. Returns the number of rows updated.
        """
        if not ids:
            return 0
        conds = self._notification_conditions(ids=ids, site_uuids=site_uuids)
        if only_pending:
            conds.append(Notification.approval_status == "pending")
        result = await db.execute(
            update(Notification)
            .where(and_(*conds))
            .values(
                approval_status=str(approval_status),
                visible=bool(visible),
                approved_by=approved_by,
                approved_at=utc_now(),
            )
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def delete_notifications(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        ids: Optional[List[int]] = None,
        site_uuid: Optional[uuid.UUID] = None,
        detected_before: Optional[datetime] = None,
    ) -> int:
        """Hard-delete Notification rows matching the given filters. Returns count."""
        conds = self._notification_conditions(
            user_id=user_id, ids=ids, site_uuid=site_uuid, detected_before=detected_before,
        )
        if not conds:
            raise ValueError("delete_notifications requires at least one filter")
        result = await db.execute(
            delete(Notification)
            .where(and_(*conds))
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def hide_notifications_by_filter(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: Optional[uuid.UUID] = None,
        camera_uuid: Optional[uuid.UUID] = None,
    ) -> int:
        filters = [Notification.user_id == user_id, Notification.visible == True]
        if site_uuid is not None:
            filters.append(Notification.site_uuid == site_uuid)
        if camera_uuid is not None:
            filters.append(Notification.camera_uuid == camera_uuid)

        stmt = update(Notification).where(*filters).values(visible=False)
        result = await db.execute(stmt)
        return result.rowcount

    async def iter_storage_keys_by_filter(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: Optional[uuid.UUID] = None,
        camera_uuid: Optional[uuid.UUID] = None,
        batch_size: int = 5000,
    ):
        filters = [Notification.user_id == user_id, Notification.visible == True]
        if site_uuid is not None:
            filters.append(Notification.site_uuid == site_uuid)
        if camera_uuid is not None:
            filters.append(Notification.camera_uuid == camera_uuid)

        stmt = select(Notification.payload).where(*filters)
        result = await db.stream(stmt.execution_options(yield_per=batch_size))
        
        async for row in result:
            payload = row[0]
            if not isinstance(payload, dict):
                continue
            key = str(payload.get("image_storage_key") or "").strip()
            if key:
                yield key


    async def aggregate_detections_over_time(
        self,
        db: AsyncSession,
        *,
        site_uuids: List[uuid.UUID],
        start: datetime,
        bucket_minutes: int,
        bucket_ms: int,
        site_uuid: Optional[uuid.UUID] = None,
        needs_payload_filter: bool = False,
        roi_only: bool = False,
        class_filter: Optional[str] = None,
        as_utc_fn=None,
        is_roi_fn=None,
        extract_classes_fn=None,
    ) -> Dict[int, int]:
        """
        Bucket a user's detections by time. Returns a dict of
        {bucket_start_ms: count}.

        When needs_payload_filter is False a pure-SQL time-bucket aggregation is
        used. Otherwise rows are streamed in batches and filtered in Python via
        the supplied is_roi_fn / extract_classes_fn callbacks.
        """
        su = _as_uuid(site_uuid)
        clean_sites = [_as_uuid(s) for s in (site_uuids or []) if s is not None]
        site_scope = Notification.site_uuid.in_(clean_sites) if clean_sites else false()

        if not needs_payload_filter:
            # ── Fast path: pure SQL aggregation, no payload scanning ──
            bucket_seconds = bucket_minutes * 60
            bucket_expr = literal_column(
                f"FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(detected_at) / {bucket_seconds}) * {bucket_seconds})"
            )
            stmt = (
                select(bucket_expr.label("bucket_start"), func.count().label("cnt"))
                .select_from(Notification.__table__)
                .where(
                    site_scope,
                    Notification.detected_at >= start,
                    Notification.visible.is_(True),
                )
            )
            if su:
                stmt = stmt.where(Notification.site_uuid == su)
            stmt = stmt.group_by(literal_column("bucket_start"))

            rows = (await db.execute(stmt)).all()
            counts: Dict[int, int] = {}
            for bucket_start_dt, cnt in rows:
                dt = as_utc_fn(bucket_start_dt)
                ts_ms_val = int(dt.timestamp() * 1000)
                counts[ts_ms_val] = int(cnt)
            return counts

        # ── Filtered path: stream rows in batches to avoid OOM ──
        stmt = select(
            Notification.detected_at,
            Notification.event_type,
            Notification.title,
            Notification.message,
            Notification.payload,
        ).where(
            site_scope,
            Notification.detected_at >= start,
            Notification.visible.is_(True),
        )
        if su:
            stmt = stmt.where(Notification.site_uuid == su)

        # Hint for ROI: most ROI notifications have "roi" in event_type or title
        if roi_only and not class_filter:
            stmt = stmt.where(
                Notification.event_type.contains("roi")
                | Notification.title.contains("roi")
                | Notification.title.contains("ROI")
            )

        BATCH_SIZE = 5000
        counts = {}
        offset = 0
        while True:
            batch_stmt = stmt.order_by(Notification.id).offset(offset).limit(BATCH_SIZE)
            rows = (await db.execute(batch_stmt)).all()
            if not rows:
                break

            for raw_dt, event_type, title, message, payload in rows:
                if roi_only and not is_roi_fn(event_type, title, message, payload):
                    continue
                if class_filter:
                    classes = extract_classes_fn(event_type, title, message, payload)
                    if class_filter not in classes:
                        continue

                dt = as_utc_fn(raw_dt)
                ts_ms_val = int(dt.timestamp() * 1000)
                bucket = ts_ms_val - (ts_ms_val % bucket_ms)
                counts[bucket] = counts.get(bucket, 0) + 1

            offset += BATCH_SIZE
            if len(rows) < BATCH_SIZE:
                break

        return counts

    async def get_camera_mode_row(
        self, db: AsyncSession, *, camera_uuid: uuid.UUID
    ):
        """
        Return the camera/site mode row used by the notifications camera-mode
        cache: (is_enabled, is_detection_enabled, is_notification_enabled,
        use_site_schedule, roi, SiteSettings.config, SiteSettings.id,
        Site.timezone). Returns None when the camera does not exist.
        """
        return (
            await db.execute(
                select(
                    Camera.is_enabled,
                    Camera.is_detection_enabled,
                    Camera.is_notification_enabled,
                    Camera.use_site_schedule,
                    Camera.roi,
                    SiteSettings.config,
                    SiteSettings.id,
                    Site.timezone,
                )
                .join(Site, Site.site_uuid == Camera.site_uuid)
                .outerjoin(SiteSettings, SiteSettings.site_uuid == Camera.site_uuid)
                .where(Camera.camera_uuid == _as_uuid(camera_uuid))
            )
        ).first()

    # ------------------------------------------------------------------
    # NotificationEmail CRUD
    # ------------------------------------------------------------------
    async def list_notification_email_rows(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        only_enabled: Optional[bool] = None,
    ) -> List[NotificationEmail]:
        """Return NotificationEmail rows, scoped by user and/or site(s)."""
        stmt = select(NotificationEmail)
        if user_id is not None:
            stmt = stmt.where(NotificationEmail.user_id == int(user_id))
        if site_uuids is not None:
            clean = [_as_uuid(s) for s in site_uuids if s is not None]
            stmt = stmt.where(
                NotificationEmail.site_uuid.in_(clean) if clean else false()
            )
        if site_uuid is not None:
            stmt = stmt.where(NotificationEmail.site_uuid == _as_uuid(site_uuid))
        if only_enabled is True:
            stmt = stmt.where(NotificationEmail.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(NotificationEmail.is_enabled.is_(False))
        stmt = stmt.order_by(NotificationEmail.created_at.desc())
        return (await db.execute(stmt)).scalars().all()

    async def get_notification_email(
        self,
        db: AsyncSession,
        *,
        email_id: Optional[int] = None,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        email: Optional[str] = None,
    ) -> Optional[NotificationEmail]:
        stmt = select(NotificationEmail)
        if email_id is not None:
            stmt = stmt.where(NotificationEmail.id == int(email_id))
        if user_id is not None:
            stmt = stmt.where(NotificationEmail.user_id == int(user_id))
        if site_uuid is not None:
            stmt = stmt.where(NotificationEmail.site_uuid == _as_uuid(site_uuid))
        if email is not None:
            stmt = stmt.where(NotificationEmail.email == email)
        return (await db.execute(stmt)).scalars().first()

    async def notification_email_exists(
        self, db: AsyncSession, *, user_id: int, site_uuid: uuid.UUID, email: str
    ) -> bool:
        row = await self.get_notification_email(
            db, user_id=user_id, site_uuid=site_uuid, email=email
        )
        return row is not None

    async def create_notification_email(
        self,
        db: AsyncSession,
        *,
        dto: NotificationEmailCreateDTO,
    ) -> NotificationEmail:
        """Insert a site-scoped recipient email from a DTO. Flush only; caller commits."""
        row = NotificationEmail(
            user_id=int(dto.user_id),
            site_uuid=_as_uuid(dto.site_uuid),
            email=dto.email,
            is_enabled=bool(dto.is_enabled),
        )
        db.add(row)
        await db.flush()
        return row

    async def delete_notification_email(
        self,
        db: AsyncSession,
        *,
        email_id: Optional[int] = None,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        email: Optional[str] = None,
    ) -> int:
        """Delete recipient email(s) by id or by (user, site, email). Returns count."""
        conds = []
        if email_id is not None:
            conds.append(NotificationEmail.id == int(email_id))
        if user_id is not None:
            conds.append(NotificationEmail.user_id == int(user_id))
        if site_uuid is not None:
            conds.append(NotificationEmail.site_uuid == _as_uuid(site_uuid))
        if email is not None:
            conds.append(NotificationEmail.email == email)
        if not conds:
            raise ValueError("delete_notification_email requires at least one filter")
        result = await db.execute(
            delete(NotificationEmail)
            .where(and_(*conds))
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0
