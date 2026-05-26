# application/repositories/channel_repository.py
"""
Camera + ChannelConfiguration persistence.

Owns the `camera` and `channel_configurations` tables, and the camera side of
`pipeline_cameras` membership. Device rows belong to DeviceRepository; site
rows belong to SiteRepository.

Transaction policy: never commits, only flushes. Caller owns the transaction.
"""

import uuid
import logging
from datetime import time
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from application.channels.channel_config import VideoChannelConfig
from application.dtos import CameraUpdateDTO, CameraUpsertDTO
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


def _as_uuid(value: Any) -> Optional[uuid.UUID]:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


class ChannelRepository:
    """
    Camera + ChannelConfiguration + PipelineCamera-membership consistency.

    Each camera MUST have exactly 1 device assigned.
    """

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    async def get_camera(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        include_config: bool = False,
        include_device: bool = False,
    ) -> Optional[Camera]:
        """Return a single Camera by uuid, optionally eager-loading relations."""
        stmt = select(Camera).where(Camera.camera_uuid == _as_uuid(camera_uuid))
        opts = []
        if include_config:
            opts.append(selectinload(Camera.channel_configuration))
        if include_device:
            opts.append(selectinload(Camera.device))
        if opts:
            stmt = stmt.options(*opts)
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
        site_uuid: Optional[uuid.UUID] = None,
        device_uuid: Optional[uuid.UUID] = None,
        device_uuids: Optional[List[uuid.UUID]] = None,
        pipeline_id: Optional[uuid.UUID] = None,
        only_enabled: Optional[bool] = None,
        only_detection_enabled: Optional[bool] = None,
        include_config: bool = False,
        include_device: bool = False,
        order_by_created: bool = True,
    ) -> List[Camera]:
        """
        Return cameras matching any combination of the given filters.

        only_enabled / only_detection_enabled:
          - None  -> no filter
          - True  -> column is True
          - False -> column is False
        """
        stmt = select(Camera)

        if pipeline_id is not None:
            stmt = stmt.join(
                PipelineCamera, PipelineCamera.camera_uuid == Camera.camera_uuid
            ).where(PipelineCamera.pipeline_id == _as_uuid(pipeline_id))

        if camera_uuids is not None:
            clean = [_as_uuid(c) for c in (camera_uuids or []) if c is not None]
            if not clean:
                return []
            stmt = stmt.where(Camera.camera_uuid.in_(clean))

        if user_id is not None:
            stmt = stmt.where(Camera.user_id == int(user_id))

        if site_uuid is not None:
            stmt = stmt.where(Camera.site_uuid == _as_uuid(site_uuid))

        if device_uuid is not None:
            stmt = stmt.where(Camera.device_uuid == _as_uuid(device_uuid))

        if device_uuids is not None:
            clean_d = [_as_uuid(d) for d in device_uuids if d is not None]
            if not clean_d:
                return []
            stmt = stmt.where(Camera.device_uuid.in_(clean_d))

        if only_enabled is True:
            stmt = stmt.where(Camera.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(Camera.is_enabled.is_(False))

        if only_detection_enabled is True:
            stmt = stmt.where(Camera.is_detection_enabled.is_(True))
        elif only_detection_enabled is False:
            stmt = stmt.where(Camera.is_detection_enabled.is_(False))

        opts = []
        if include_config:
            opts.append(selectinload(Camera.channel_configuration))
        if include_device:
            opts.append(selectinload(Camera.device))
        if opts:
            stmt = stmt.options(*opts)

        if order_by_created:
            stmt = stmt.order_by(Camera.created_at.asc())

        return (await db.execute(stmt)).scalars().all()

    async def list_cameras_with_device_details(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: int,
        only_enabled: Optional[bool] = None,
    ) -> List[dict]:
        """Return camera rows joined with their device, as flat dicts."""
        stmt = (
            select(Camera, Device)
            .join(Device, Device.device_uuid == Camera.device_uuid, isouter=True)
            .where(
                Camera.site_uuid == _as_uuid(site_uuid),
                Camera.user_id == int(user_id),
            )
            .order_by(Camera.created_at.asc())
        )

        if only_enabled is True:
            stmt = stmt.where(Camera.is_enabled.is_(True))
        elif only_enabled is False:
            stmt = stmt.where(Camera.is_enabled.is_(False))

        rows = (await db.execute(stmt)).all()

        return [
            {
                "camera_uuid": cam.camera_uuid,
                "camera_code": cam.camera_code,
                "site_uuid": cam.site_uuid,
                "rtsp_url": cam.rtsp_url,
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

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def update_camera(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        dto: CameraUpdateDTO,
    ) -> int:
        """Update the Camera columns set on `dto`. Returns affected row count."""
        values: Dict[str, Any] = dto.model_dump(exclude_unset=True)
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
        await self._ensure_device_exists(db, clean)
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
        """
        Create or update a Camera (+ ChannelConfiguration + pipeline membership)
        from a `CameraUpsertDTO`.
        """
        # Unpack the DTO into the local names the body works with.
        pipeline_id = dto.pipeline_id
        channel_config = dto.channel_config
        user_id = dto.user_id
        cam_uuid = dto.cam_uuid
        camera_code = dto.camera_code
        site_uuid = dto.site_uuid
        webrtc_url = dto.webrtc_url
        rtsp_url = dto.rtsp_url
        device_uuid = dto.device_uuid
        name = dto.name
        location = dto.location
        timezone = dto.timezone
        day_of_week = dto.day_of_week
        start_time = dto.start_time
        end_time = dto.end_time
        is_enabled = dto.is_enabled

        await self._ensure_pipeline_exists(db, pipeline_id)

        d = self._to_dict(channel_config)

        raw_cam_uuid = cam_uuid or d.get("camera_uuid") or d.get("camera_id")
        cam_uuid = None
        if raw_cam_uuid:
            cam_uuid = raw_cam_uuid if isinstance(raw_cam_uuid, uuid.UUID) else uuid.UUID(str(raw_cam_uuid))

        if device_uuid is None:
            device_uuid = d.get("device_uuid")
        if device_uuid is not None:
            device_uuid = device_uuid if isinstance(device_uuid, uuid.UUID) else uuid.UUID(str(device_uuid))

        rtsp_url = rtsp_url or d.get("rtsp_url")
        if name is None:
            name = d.get("name")
        if location is None:
            location = d.get("location")

        if isinstance(name, str):
            name = name.strip() or None
        if isinstance(location, str):
            location = location.strip() or None

        enabled = d.get("enabled", d.get("is_enabled", True))
        detection_enabled = d.get("detection_enabled", d.get("is_detection_enabled", True))
        notification_enabled = d.get("notification_enabled", d.get("is_notification_enabled", True))
        use_site_schedule = d.get("use_site_schedule", True)

        has_roi = "roi" in d
        roi = d.get("roi")

        has_notification_trigger_mode = "notification_trigger_mode" in d
        notification_trigger_mode_val = d.get("notification_trigger_mode")
        if isinstance(notification_trigger_mode_val, str):
            notification_trigger_mode_val = notification_trigger_mode_val.strip() or "inherit"
        elif notification_trigger_mode_val is None:
            notification_trigger_mode_val = "inherit"
        else:
            notification_trigger_mode_val = str(notification_trigger_mode_val)
        if notification_trigger_mode_val not in ("inherit", "roi_enter", "any_detection"):
            notification_trigger_mode_val = "inherit"

        has_camera_playback_enabled = "camera_playback_enabled" in d
        camera_playback_enabled_val = d.get("camera_playback_enabled")
        if camera_playback_enabled_val is True:
            camera_playback_enabled_val = "always"
        elif camera_playback_enabled_val is False:
            camera_playback_enabled_val = "never"
        elif camera_playback_enabled_val is None:
            camera_playback_enabled_val = "inherit"
        else:
            camera_playback_enabled_val = str(camera_playback_enabled_val).strip() or "inherit"
        if camera_playback_enabled_val not in ("inherit", "always", "never"):
            camera_playback_enabled_val = "inherit"

        if not rtsp_url:
            raise ValueError("channel_config.rtsp_url is required")

        cam: Optional[Camera] = None
        if cam_uuid:
            cam = await self._get_camera_by_uuid(db, cam_uuid)

        if cam:
            cam.rtsp_url = rtsp_url
            cam.is_enabled = bool(enabled)
            cam.is_detection_enabled = bool(detection_enabled)
            cam.is_notification_enabled = bool(notification_enabled)
            cam.use_site_schedule = bool(use_site_schedule)

            if has_roi:
                cam.roi = roi

            if has_notification_trigger_mode:
                cam.notification_trigger_mode = notification_trigger_mode_val
            if has_camera_playback_enabled:
                cam.camera_playback_enabled = camera_playback_enabled_val

            if webrtc_url is not None:
                if cam.webrtc_url is None:
                    cam.webrtc_url = webrtc_url
                elif cam.webrtc_url != webrtc_url:
                    logger.warning(
                        "Ignoring webrtc_url change for camera=%s (immutable). old=%s new=%s",
                        str(cam.camera_uuid),
                        cam.webrtc_url,
                        webrtc_url,
                    )

            if site_uuid is not None:
                if cam.site_uuid is None:
                    cam.site_uuid = site_uuid
                elif cam.site_uuid != site_uuid:
                    raise ValueError("Changing site_uuid for an existing camera is not supported")

            if camera_code is not None:
                cam.camera_code = camera_code
            if name is not None:
                cam.name = name
            if location is not None:
                cam.location = location

            await db.flush()


        else:
            if user_id is None:
                raise ValueError("user_id is required to create a new camera")
            if not camera_code:
                raise ValueError("camera_code is required to create a new camera")
            if site_uuid is None:
                raise ValueError("site_uuid is required to create a new camera")
            if device_uuid is None:
                raise ValueError("device_uuid is required to create a new camera (each camera must have a device).")

            await self._ensure_site_exists(db, site_uuid)
            cam = Camera(
                user_id=user_id,
                site_uuid=site_uuid,
                camera_code=camera_code,
                rtsp_url=rtsp_url,
                webrtc_url=webrtc_url,
                name=name,
                location=location,
                is_enabled=bool(enabled),
                is_detection_enabled=bool(detection_enabled),
                is_notification_enabled=bool(notification_enabled),
                use_site_schedule=bool(use_site_schedule),
                roi=roi,
                notification_trigger_mode=notification_trigger_mode_val if has_notification_trigger_mode else "inherit",
                camera_playback_enabled=camera_playback_enabled_val if has_camera_playback_enabled else "inherit",
            )
            if cam_uuid is not None:
                cam.camera_uuid = cam_uuid

            db.add(cam)
            try:
                await db.flush()
            except IntegrityError as e:
                raise ValueError(
                    f"Camera already exists for user_id={user_id} camera_code={camera_code}"
                ) from e
                
        if device_uuid is not None:
            await self.set_camera_device(db, camera_uuid=cam.camera_uuid, device_uuid=device_uuid)
        else:
            if cam.is_detection_enabled:
                await self._ensure_camera_has_exactly_one_device(db, camera_uuid=cam.camera_uuid)

        cfg_json = self._build_channel_configuration_json(d)
        tz = d.get("timezone") or timezone

        schedule = self._resolve_schedule(
            raw_schedule=d.get("schedule"),
            day_of_week=day_of_week,
            start_time=start_time,
            end_time=end_time,
            is_enabled=is_enabled,
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

        await self._set_pipeline_membership(db, camera_uuid=cam.camera_uuid, pipeline_id=pipeline_id)

        return cam, cfg_json, tz

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _ensure_device_exists(self, db: AsyncSession, device_uuid: uuid.UUID) -> None:
        exists = (await db.execute(select(Device.device_uuid).where(Device.device_uuid == device_uuid))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Device not found: {device_uuid}")

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

    async def _ensure_site_exists(self, db: AsyncSession, site_uuid: uuid.UUID) -> None:
        exists = (await db.execute(select(Site).where(Site.site_uuid == site_uuid, Site.is_deleted == False))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Site not found: {site_uuid}")

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
            "rtsp_url", "webrtc_url",
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
        for window in schedule or []:
            day = int(window["day_of_week"])
            start_time = self._coerce_time(window.get("start_time"), DEFAULT_START_TIME)
            end_time = self._coerce_time(window.get("end_time"), DEFAULT_END_TIME)
            if start_time < end_time:
                return day, start_time, end_time

        if not schedule:
            return 6, DEFAULT_START_TIME, DEFAULT_END_TIME

        first_window = schedule[0]
        day = int(first_window["day_of_week"])
        start_time = self._coerce_time(first_window.get("start_time"), DEFAULT_START_TIME)
        end_time = self._coerce_time(first_window.get("end_time"), DEFAULT_END_TIME)

        if start_time < DEFAULT_END_TIME:
            return day, start_time, DEFAULT_END_TIME

        if end_time > DEFAULT_START_TIME:
            return (day + 1) % 7, DEFAULT_START_TIME, end_time

        return day, DEFAULT_START_TIME, DEFAULT_END_TIME

    def _coerce_time(self, value: Any, default: time) -> time:
        if value is None:
            return default
        if isinstance(value, time):
            return value
        if isinstance(value, str):
            return time.fromisoformat(value)
        raise ValueError(f"Unsupported time value: {value!r}")
