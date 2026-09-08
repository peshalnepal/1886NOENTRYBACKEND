"""
Camera + ChannelConfiguration persistence.

Owns the `camera` and `channel_configurations` tables, and the camera side of
`pipeline_cameras` membership. Device rows belong to DeviceRepository; site
rows belong to SiteRepository.

Transaction policy: never commits, only flushes. Caller owns the transaction.
"""

import logging
import uuid
from datetime import time
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from application.channels.channel_config import VideoChannelConfig
from application.dtos import CameraUpdateDTO, CameraUpsertDTO
from application.repositories._helpers import as_uuid as _as_uuid, model_patch
from application.repositories.device_repository import DeviceRepository
from core.coercions import coerce_playback_mode, coerce_trigger_mode
from core.database_orm import (
    Camera,
    ChannelConfiguration,
    Device,
    Pipeline,
    PipelineCamera,
    Site,
)

logger = logging.getLogger(__name__)

ChannelConfigLike = Union[Dict[str, Any], Any]

DAY_NAME_BY_VALUE = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

# Sunday -> Saturday order, while still respecting ORM mapping 0=Mon ... 6=Sun
SUNDAY_TO_SATURDAY = [6, 0, 1, 2, 3, 4, 5]

DEFAULT_START_TIME = time(0, 0, 0)
DEFAULT_END_TIME = time(23, 59, 59)

class ChannelRepository:
    """Camera + ChannelConfiguration + pipeline-membership consistency.

    Every camera must have exactly one device assigned.
    """


    @staticmethod
    def _with_relations(stmt, *, include_config: bool, include_device: bool):
        """Eager-load the optional camera relations (never lazy under async)."""
        opts = []
        if include_config:
            opts.append(selectinload(Camera.channel_configuration))
        if include_device:
            opts.append(selectinload(Camera.device))
        return stmt.options(*opts) if opts else stmt

    @staticmethod
    def _tristate_filter(stmt, column, wanted: Optional[bool]):
        """Filter on a nullable boolean column, or not at all when `wanted` is None."""
        if wanted is None:
            return stmt
        return stmt.where(column.is_(bool(wanted)))

    async def get_camera(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        include_config: bool = False,
        include_device: bool = False,
    ) -> Optional[Camera]:
        """Return a single Camera by uuid, optionally eager-loading relations."""
        stmt = self._with_relations(
            select(Camera).where(Camera.camera_uuid == _as_uuid(camera_uuid)),
            include_config=include_config,
            include_device=include_device,
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    async def get_camera_full(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> Optional[Tuple[Camera, Optional[ChannelConfiguration], Optional[uuid.UUID]]]:
        """Return (Camera, ChannelConfiguration, pipeline_id) for one camera."""
        cam = await self.get_camera(
            db, camera_uuid=camera_uuid, include_config=True, include_device=True
        )
        if cam is None:
            return None

        pipeline_id = (
            await db.execute(
                select(PipelineCamera.pipeline_id).where(
                    PipelineCamera.camera_uuid == cam.camera_uuid
                )
            )
        ).scalar_one_or_none()

        return cam, cam.channel_configuration, pipeline_id

    async def list_cameras(
        self,
        db: AsyncSession,
        *,
        camera_uuids: Optional[List[uuid.UUID]] = None,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        site_uuids: Optional[List[uuid.UUID]] = None,
        device_uuid: Optional[uuid.UUID] = None,
        device_uuids: Optional[List[uuid.UUID]] = None,
        pipeline_id: Optional[uuid.UUID] = None,
        only_enabled: Optional[bool] = None,
        only_detection_enabled: Optional[bool] = None,
        include_config: bool = False,
        include_device: bool = False,
        order_by_created: bool = True,
    ) -> List[Camera]:
        """Cameras matching any combination of the given filters.

        `only_enabled` / `only_detection_enabled` are tri-state: None means no
        filter, True/False match the column. An explicitly empty uuid list
        matches nothing and short-circuits to `[]`.
        """
        stmt = select(Camera)

        if pipeline_id is not None:
            stmt = stmt.join(
                PipelineCamera, PipelineCamera.camera_uuid == Camera.camera_uuid
            ).where(PipelineCamera.pipeline_id == _as_uuid(pipeline_id))

        # Each "in" filter: an empty list is a deliberate "match nothing".
        for values, column in (
            (camera_uuids, Camera.camera_uuid),
            (site_uuids, Camera.site_uuid),
            (device_uuids, Camera.device_uuid),
        ):
            if values is None:
                continue
            clean = [_as_uuid(v) for v in values if v is not None]
            if not clean:
                return []
            stmt = stmt.where(column.in_(clean))

        if user_id is not None:
            stmt = stmt.where(Camera.user_id == int(user_id))
        if org_id is not None:
            stmt = stmt.where(Camera.org_id == int(org_id))
        if site_uuid is not None:
            stmt = stmt.where(Camera.site_uuid == _as_uuid(site_uuid))
        if device_uuid is not None:
            stmt = stmt.where(Camera.device_uuid == _as_uuid(device_uuid))

        stmt = self._tristate_filter(stmt, Camera.is_enabled, only_enabled)
        stmt = self._tristate_filter(
            stmt, Camera.is_detection_enabled, only_detection_enabled
        )
        stmt = self._with_relations(
            stmt, include_config=include_config, include_device=include_device
        )

        if order_by_created:
            stmt = stmt.order_by(Camera.created_at.asc())

        return list((await db.execute(stmt)).scalars().all())

    async def list_cameras_with_device_details(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        only_enabled: Optional[bool] = None,
    ) -> List[dict]:
        """Return camera rows joined with their device, as flat dicts.

        Cameras are scoped by `site_uuid` (which already pins the org). The
        optional `user_id`/`org_id` filters narrow further when provided.
        """
        conds = [Camera.site_uuid == _as_uuid(site_uuid)]
        if user_id is not None:
            conds.append(Camera.user_id == int(user_id))
        if org_id is not None:
            conds.append(Camera.org_id == int(org_id))
        stmt = (
            select(Camera, Device)
            .join(Device, Device.device_uuid == Camera.device_uuid, isouter=True)
            .where(*conds)
            .order_by(Camera.created_at.asc())
        )
        stmt = self._tristate_filter(stmt, Camera.is_enabled, only_enabled)

        rows = (await db.execute(stmt)).all()

        return [
            {
                "camera_uuid": cam.camera_uuid,
                "camera_code": cam.camera_code,
                "site_uuid": cam.site_uuid,
                "source_url": cam.source_url,
                "webrtc_url": cam.webrtc_url,
                "is_enabled": cam.is_enabled,
                "is_detection_enabled": cam.is_detection_enabled,
                "is_notification_enabled": cam.is_notification_enabled,
                "roi": cam.roi,
                "device_uuid": getattr(dev, "device_uuid", None),
                "device_url": getattr(dev, "device_url", None),
            }
            for cam, dev in rows
        ]

    async def get_camera_roi(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> Any:
        """Return just the roi JSON for one camera (None if camera missing)."""
        return (
            await db.execute(
                select(Camera.roi).where(Camera.camera_uuid == _as_uuid(camera_uuid))
            )
        ).scalar_one_or_none()

    async def cameras_exist(
        self,
        db: AsyncSession,
        *,
        camera_uuids: List[uuid.UUID],
        user_id: Optional[int] = None,
    ) -> set:
        """Return the subset of camera_uuids that exist (optionally scoped to user)."""
        clean = [_as_uuid(c) for c in (camera_uuids or []) if c is not None]
        if not clean:
            return set()
        stmt = select(Camera.camera_uuid).where(Camera.camera_uuid.in_(clean))
        if user_id is not None:
            stmt = stmt.where(Camera.user_id == int(user_id))
        rows = (await db.execute(stmt)).scalars().all()
        return set(rows)

    async def list_detection_enabled_user_ids(self, db: AsyncSession) -> List[int]:
        """Return distinct user ids that own at least one detection-enabled camera."""
        rows = (
            await db.execute(
                select(Camera.user_id)
                .where(Camera.is_detection_enabled.is_(True))
                .distinct()
                .order_by(Camera.user_id.asc())
            )
        ).scalars().all()
        return [int(uid) for uid in rows]


    async def update_camera(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        dto: CameraUpdateDTO,
    ) -> int:
        """Update the Camera columns set on `dto`. Returns affected row count."""
        values: Dict[str, Any] = model_patch(dto)
        if not values:
            return 0
        result = await db.execute(
            update(Camera)
            .where(Camera.camera_uuid == _as_uuid(camera_uuid))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def disable_cameras(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        device_uuid: Optional[uuid.UUID] = None,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> int:
        """
        Bulk-disable cameras (is_enabled + is_detection_enabled = False) matching
        any combination of filters. Returns affected row count.
        """
        conds = []
        if user_id is not None:
            conds.append(Camera.user_id == int(user_id))
        if site_uuid is not None:
            conds.append(Camera.site_uuid == _as_uuid(site_uuid))
        if device_uuid is not None:
            conds.append(Camera.device_uuid == _as_uuid(device_uuid))
        if camera_uuids is not None:
            clean = [_as_uuid(c) for c in (camera_uuids or []) if c is not None]
            if not clean:
                return 0
            conds.append(Camera.camera_uuid.in_(clean))
        if not conds:
            raise ValueError("disable_cameras requires at least one filter")

        result = await db.execute(
            update(Camera)
            .where(*conds)
            .values(is_enabled=False, is_detection_enabled=False)
            .execution_options(synchronize_session=False)
        )
        await db.flush()
        return result.rowcount or 0

    async def set_camera_device(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        device_uuid: uuid.UUID,
    ) -> None:
        """Assign a device to a camera (the single device each camera must have)."""
        clean = _as_uuid(device_uuid)
        await DeviceRepository().ensure_device_exists(db, clean)
        await db.execute(
            update(Camera)
            .where(Camera.camera_uuid == _as_uuid(camera_uuid))
            .values(device_uuid=clean)
            .execution_options(synchronize_session=False)
        )
        await db.flush()

    async def delete_camera(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> None:
        """Delete a camera and its dependent pipeline-membership + channel config."""
        cam_uuid = _as_uuid(camera_uuid)
        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == cam_uuid))
        await db.execute(
            delete(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == cam_uuid)
        )
        await db.execute(delete(Camera).where(Camera.camera_uuid == cam_uuid))
        await db.flush()

    async def upsert_camera_from_channel_config(
        self,
        db: AsyncSession,
        *,
        dto: CameraUpsertDTO,
    ) -> Tuple[Camera, Dict[str, Any], Optional[str]]:
        """Create or update a Camera plus its ChannelConfiguration and pipeline
        membership, from a `CameraUpsertDTO`.

        Values come from the DTO first and fall back to the raw channel-config
        blob, which is what the edit path merges its patch into.
        """
        await self._ensure_pipeline_exists(db, dto.pipeline_id)

        d = self._to_dict(dto.channel_config)

        cam_uuid = self._optional_uuid(
            dto.cam_uuid or d.get("camera_uuid") or d.get("camera_id")
        )
        device_uuid = self._optional_uuid(
            dto.device_uuid if dto.device_uuid is not None else d.get("device_uuid")
        )

        source_url = dto.source_url or d.get("source_url")
        if not source_url:
            raise ValueError("channel_config.source_url is required")

        name = self._clean_str(dto.name if dto.name is not None else d.get("name"))
        location = self._clean_str(
            dto.location if dto.location is not None else d.get("location")
        )

        # The config blob may use either the runtime or the ORM field name.
        enabled = d.get("enabled", d.get("is_enabled", True))
        detection_enabled = d.get("detection_enabled", d.get("is_detection_enabled", True))
        notification_enabled = d.get(
            "notification_enabled", d.get("is_notification_enabled", True)
        )
        use_site_schedule = d.get("use_site_schedule", True)

        # These three are patch-sensitive: only overwrite when the key is
        # present, so an edit that omits them leaves the stored value alone.
        has_roi = "roi" in d
        has_trigger_mode = "notification_trigger_mode" in d
        has_playback = "camera_playback_enabled" in d
        trigger_mode = coerce_trigger_mode(d.get("notification_trigger_mode"))
        playback_mode = coerce_playback_mode(d.get("camera_playback_enabled"))

        cam = await self._get_camera_by_uuid(db, cam_uuid) if cam_uuid else None

        if cam:
            cam.source_url = source_url
            cam.is_enabled = bool(enabled)
            cam.is_detection_enabled = bool(detection_enabled)
            cam.is_notification_enabled = bool(notification_enabled)
            cam.use_site_schedule = bool(use_site_schedule)

            if has_roi:
                cam.roi = d.get("roi")
            if has_trigger_mode:
                cam.notification_trigger_mode = trigger_mode
            if has_playback:
                cam.camera_playback_enabled = playback_mode

            if dto.webrtc_url is not None:
                if cam.webrtc_url is None:
                    cam.webrtc_url = dto.webrtc_url
                elif cam.webrtc_url != dto.webrtc_url:
                    logger.warning(
                        "Ignoring webrtc_url change for camera=%s (immutable). old=%s new=%s",
                        cam.camera_uuid,
                        cam.webrtc_url,
                        dto.webrtc_url,
                    )

            if dto.site_uuid is not None:
                if cam.site_uuid is None:
                    cam.site_uuid = dto.site_uuid
                elif cam.site_uuid != dto.site_uuid:
                    raise ValueError(
                        "Changing site_uuid for an existing camera is not supported"
                    )

            if dto.camera_code is not None:
                cam.camera_code = dto.camera_code
            if name is not None:
                cam.name = name
            if location is not None:
                cam.location = location

            await db.flush()
        else:
            cam = await self._create_camera(
                db,
                dto=dto,
                cam_uuid=cam_uuid,
                device_uuid=device_uuid,
                source_url=source_url,
                name=name,
                location=location,
                enabled=enabled,
                detection_enabled=detection_enabled,
                notification_enabled=notification_enabled,
                use_site_schedule=use_site_schedule,
                roi=d.get("roi"),
                trigger_mode=trigger_mode if has_trigger_mode else "inherit",
                playback_mode=playback_mode if has_playback else "inherit",
            )

        if device_uuid is not None:
            await self.set_camera_device(
                db, camera_uuid=cam.camera_uuid, device_uuid=device_uuid
            )
        elif cam.is_detection_enabled:
            await self._ensure_camera_has_exactly_one_device(
                db, camera_uuid=cam.camera_uuid
            )

        cfg_json = self._build_channel_configuration_json(d)
        tz = d.get("timezone") or dto.timezone

        schedule = self._resolve_schedule(
            raw_schedule=d.get("schedule"),
            day_of_week=dto.day_of_week,
            start_time=dto.start_time,
            end_time=dto.end_time,
            is_enabled=dto.is_enabled,
        )

        cfg_json["schedule"] = schedule
        cfg_json["use_site_schedule"] = bool(use_site_schedule)
        if tz is not None:
            cfg_json["timezone"] = tz

        # Keep legacy scalar schedule columns populated with a valid same-day segment.
        scalar_day, scalar_start, scalar_end = self._scalar_schedule_window(schedule)

        await self._upsert_channel_configuration(
            db,
            camera_uuid=cam.camera_uuid,
            configuration=cfg_json,
            timezone=tz,
            day_of_week=scalar_day,
            start_time=scalar_start,
            end_time=scalar_end,
        )

        await self._set_pipeline_membership(
            db, camera_uuid=cam.camera_uuid, pipeline_id=dto.pipeline_id
        )

        return cam, cfg_json, tz

    @staticmethod
    def _optional_uuid(value: Any) -> Optional[uuid.UUID]:
        if not value:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))

    @staticmethod
    def _clean_str(value: Any) -> Optional[str]:
        """Trim a string, mapping blank to None; non-strings pass through."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    async def _create_camera(
        self,
        db: AsyncSession,
        *,
        dto: CameraUpsertDTO,
        cam_uuid: Optional[uuid.UUID],
        device_uuid: Optional[uuid.UUID],
        source_url: str,
        name: Optional[str],
        location: Optional[str],
        enabled: Any,
        detection_enabled: Any,
        notification_enabled: Any,
        use_site_schedule: Any,
        roi: Any,
        trigger_mode: str,
        playback_mode: str,
    ) -> Camera:
        """Insert a brand-new Camera row. Flush only; the caller commits."""
        if dto.user_id is None:
            raise ValueError("user_id is required to create a new camera")
        if not dto.camera_code:
            raise ValueError("camera_code is required to create a new camera")
        if dto.site_uuid is None:
            raise ValueError("site_uuid is required to create a new camera")
        if device_uuid is None:
            raise ValueError(
                "device_uuid is required to create a new camera "
                "(each camera must have a device)."
            )

        site_org_id = await self._ensure_site_exists(db, dto.site_uuid)
        cam = Camera(
            user_id=dto.user_id,
            created_by=dto.created_by if dto.created_by is not None else dto.user_id,
            # Org ownership always follows the site the camera lives on.
            org_id=site_org_id if site_org_id is not None else dto.org_id,
            site_uuid=dto.site_uuid,
            camera_code=dto.camera_code,
            source_url=source_url,
            webrtc_url=dto.webrtc_url,
            name=name,
            location=location,
            is_enabled=bool(enabled),
            is_detection_enabled=bool(detection_enabled),
            is_notification_enabled=bool(notification_enabled),
            use_site_schedule=bool(use_site_schedule),
            roi=roi,
            notification_trigger_mode=trigger_mode,
            camera_playback_enabled=playback_mode,
        )
        if cam_uuid is not None:
            cam.camera_uuid = cam_uuid

        db.add(cam)
        try:
            await db.flush()
        except IntegrityError as exc:
            raise ValueError(
                f"Camera already exists for user_id={dto.user_id} "
                f"camera_code={dto.camera_code}"
            ) from exc
        return cam

    async def _ensure_camera_has_exactly_one_device(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> None:
        assigned = (
            await db.execute(
                select(Camera.device_uuid).where(Camera.camera_uuid == camera_uuid)
            )
        ).scalar_one_or_none()
        if assigned is None:
            raise ValueError(f"Camera {camera_uuid} must have exactly 1 device assigned, found 0")

    async def _ensure_pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> None:
        exists = (await db.execute(select(Pipeline.id).where(Pipeline.id == pipeline_id))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

    async def _ensure_site_exists(self, db: AsyncSession, site_uuid: uuid.UUID) -> Optional[int]:
        """Validate the site exists and return its owning `org_id` (or None)."""
        row = (
            await db.execute(
                select(Site.org_id).where(Site.site_uuid == site_uuid, Site.is_deleted == False)
            )
        ).first()
        if row is None:
            raise ValueError(f"Site not found: {site_uuid}")
        return int(row[0]) if row[0] is not None else None

    async def _get_camera_by_uuid(self, db: AsyncSession, camera_uuid: uuid.UUID) -> Optional[Camera]:
        stmt = (
            select(Camera)
            .where(Camera.camera_uuid == camera_uuid)
            .options(selectinload(Camera.channel_configuration))
        )
        return (await db.execute(stmt)).scalar_one_or_none()

    def _to_dict(self, obj: ChannelConfigLike) -> Dict[str, Any]:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json", exclude_none=True)
        if isinstance(obj, dict):
            return jsonable_encoder(dict(obj), exclude_none=True)
        return jsonable_encoder({k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}, exclude_none=True)

    def _build_channel_configuration_json(self, d: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(d)
        for k in (
            "camera_uuid", "camera_id", "channel_id",
            "source_url", "webrtc_url",
            "device_url",
            "enabled", "detection_enabled", "notification_enabled",
            "site_uuid", "device_uuid", "user_id",
            "camera_code", "name", "location", "timezone",
            "is_enabled", "is_detection_enabled", "is_notification_enabled",
            "roi",
            "day_of_week", "start_time", "end_time",
        ):
            cfg.pop(k, None)
        return cfg

    async def _upsert_channel_configuration(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        configuration: Dict[str, Any],
        timezone: Optional[str],
        day_of_week: int,
        start_time: time,
        end_time: time,
    ) -> ChannelConfiguration:
        encoded = jsonable_encoder(configuration, exclude_none=True)

        row = (
            await db.execute(select(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid))
        ).scalar_one_or_none()

        if row is None:
            row = ChannelConfiguration(
                camera_uuid=camera_uuid,
                configuration=encoded,
                timezone=timezone,
                day_of_week=day_of_week,
                start_time=start_time,
                end_time=end_time,
            )
            db.add(row)
            await db.flush()
            return row

        merged = dict(row.configuration or {})
        merged.update(encoded)
        merged["schedule"] = encoded.get("schedule", merged.get("schedule", self._default_weekly_schedule()))

        row.configuration = merged
        if timezone is not None:
            row.timezone = timezone
        row.day_of_week = day_of_week
        row.start_time = start_time
        row.end_time = end_time

        await db.flush()
        return row

    async def _set_pipeline_membership(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        pipeline_id: uuid.UUID,
    ) -> None:
        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == camera_uuid))
        db.add(PipelineCamera(pipeline_id=pipeline_id, camera_uuid=camera_uuid))
        await db.flush()

    def _default_weekly_schedule(self) -> List[Dict[str, Any]]:
        return [
            {
                "day_of_week": day,
                "day_name": DAY_NAME_BY_VALUE[day],
                "start_time": DEFAULT_START_TIME.strftime("%H:%M:%S"),
                "end_time": DEFAULT_END_TIME.strftime("%H:%M:%S"),
                "is_enabled": True,
            }
            for day in SUNDAY_TO_SATURDAY
        ]

    def _resolve_schedule(
        self,
        *,
        raw_schedule: Optional[List[Dict[str, Any]]],
        day_of_week: Optional[List[int]],
        start_time: Optional[time],
        end_time: Optional[time],
        is_enabled: bool,
    ) -> List[Dict[str, Any]]:
        if raw_schedule:
            normalized = VideoChannelConfig.normalize_schedule(raw_schedule)
            return normalized or self._default_weekly_schedule()

        selected_days = SUNDAY_TO_SATURDAY if day_of_week is None else day_of_week
        st = self._coerce_time(start_time, DEFAULT_START_TIME)
        et = self._coerce_time(end_time, DEFAULT_END_TIME)

        normalized = [
            {
                "day_of_week": int(day),
                "day_name": DAY_NAME_BY_VALUE[int(day)],
                "start_time": st.strftime("%H:%M:%S"),
                "end_time": et.strftime("%H:%M:%S"),
                "is_enabled": bool(is_enabled),
            }
            for day in selected_days
        ]
        return self._sort_schedule_sunday_first(normalized)

    def _sort_schedule_sunday_first(self, schedule: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        order_index = {day: idx for idx, day in enumerate(SUNDAY_TO_SATURDAY)}
        return sorted(schedule, key=lambda x: order_index.get(int(x["day_of_week"]), 999))

    def _scalar_schedule_window(self, schedule: List[Dict[str, Any]]) -> Tuple[int, time, time]:
        """Pick one representative (day, start, end) for the legacy scalar columns.

        Those columns carry a CHECK(start < end), so an all-overnight schedule
        has no window that fits directly — the fallbacks below clamp one to the
        day boundary rather than violate the constraint.
        """
        for window in schedule or []:
            start = self._coerce_time(window.get("start_time"), DEFAULT_START_TIME)
            end = self._coerce_time(window.get("end_time"), DEFAULT_END_TIME)
            if start < end:
                return int(window["day_of_week"]), start, end

        if not schedule:
            return 6, DEFAULT_START_TIME, DEFAULT_END_TIME

        first = schedule[0]
        day = int(first["day_of_week"])
        start = self._coerce_time(first.get("start_time"), DEFAULT_START_TIME)
        end = self._coerce_time(first.get("end_time"), DEFAULT_END_TIME)

        # Overnight window: keep the evening half on this day, else roll the
        # morning half onto the next day.
        if start < DEFAULT_END_TIME:
            return day, start, DEFAULT_END_TIME
        if end > DEFAULT_START_TIME:
            return (day + 1) % 7, DEFAULT_START_TIME, end
        return day, DEFAULT_START_TIME, DEFAULT_END_TIME

    def _coerce_time(self, value: Any, default: time) -> time:
        if value is None:
            return default
        if isinstance(value, time):
            return value
        if isinstance(value, str):
            return time.fromisoformat(value)
        raise ValueError(f"Unsupported time value: {value!r}")
