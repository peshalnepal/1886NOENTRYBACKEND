import uuid
import logging
from datetime import time
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi.encoders import jsonable_encoder
from sqlalchemy import delete, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from application.channels.channel_config import VideoChannelConfig
from core.database_orm import (
    Camera,
    CameraDevice,
    ChannelConfiguration,
    Device,
    Pipeline,
    PipelineCamera,
    Site,
    SiteSettings,
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
    """
    Camera + ChannelConfiguration + PipelineCamera consistency.

    each camera MUST have exactly 1 device assigned
    """

    async def upsert_camera_from_channel_config(
        self,
        db: AsyncSession,
        *,
        pipeline_id: uuid.UUID,
        channel_config: ChannelConfigLike,
        user_id: Optional[int] = None,
        cam_uuid: Optional[uuid.UUID] = None,
        camera_code: Optional[str] = None,
        site_uuid: Optional[uuid.UUID] = None,
        webrtc_url: Optional[str] = None,
        rtsp_url: Optional[str] = None,
        device_uuid: Optional[uuid.UUID] = None,
        name: Optional[str] = None,
        location: Optional[str] = None,
        timezone: Optional[str] = None,
        day_of_week: Optional[List[int]] = None,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
        is_enabled: bool = True,
    ) -> Tuple[Camera, Dict[str, Any], Optional[str]]:

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

            if device_uuid is not None:
                await self._set_camera_device(db, camera_uuid=cam.camera_uuid, device_uuid=device_uuid)
            else:
                if cam.is_detection_enabled:
                    await self._ensure_camera_has_exactly_one_device(db, camera_uuid=cam.camera_uuid)

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
            await self._ensure_device_exists(db, device_uuid)

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

            await self._set_camera_device(db, camera_uuid=cam.camera_uuid, device_uuid=device_uuid)

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

    async def upsert_site_settings(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
        config: Optional[Dict[str, Any]] = None,
        day_of_week: Optional[List[int]] = None,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
        is_enabled: bool = True,
    ) -> SiteSettings:
        await self._ensure_site_exists(db, site_uuid)

        incoming_config = jsonable_encoder(config or {}, exclude_none=True)

        schedule = self._resolve_schedule(
            raw_schedule=incoming_config.get("schedule"),
            day_of_week=day_of_week,
            start_time=start_time,
            end_time=end_time,
            is_enabled=is_enabled,
        )
        incoming_config["schedule"] = schedule

        scalar_day, scalar_start, scalar_end = self._scalar_schedule_window(schedule)

        row = (
            await db.execute(select(SiteSettings).where(SiteSettings.site_uuid == site_uuid))
        ).scalar_one_or_none()

        if row is None:
            row = SiteSettings(
                user_id=user_id,
                site_uuid=site_uuid,
                config=incoming_config,
                day_of_week=scalar_day,
                start_time=scalar_start,
                end_time=scalar_end,
                is_enabled=bool(is_enabled),
            )
            db.add(row)
            await db.flush()
            return row

        merged = dict(row.config or {})
        merged.update(incoming_config)
        merged["schedule"] = incoming_config.get("schedule", merged.get("schedule", self._default_weekly_schedule()))


        row.config = merged
        row.day_of_week = scalar_day
        row.start_time = scalar_start
        row.end_time = scalar_end
        row.is_enabled = bool(is_enabled)

        await db.flush()
        return row

    async def get_camera_full(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> Optional[Tuple[Camera, Optional[ChannelConfiguration], Optional[uuid.UUID]]]:
        cam = (
            await db.execute(
                select(Camera)
                .where(Camera.camera_uuid == camera_uuid)
                .options(
                    selectinload(Camera.channel_configuration),
                    selectinload(Camera.devices),
                )
            )
        ).scalar_one_or_none()

        if cam is None:
            return None

        pipeline_id = (
            await db.execute(select(PipelineCamera.pipeline_id).where(PipelineCamera.camera_uuid == cam.camera_uuid))
        ).scalar_one_or_none()

        return cam, cam.channel_configuration, pipeline_id

    async def get_device(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        required: bool = False,
        relaxed: bool = False,
    ) -> Optional[Device]:
        devices = await self.list_devices(db, camera_uuid=camera_uuid)

        if len(devices) == 1:
            return devices[0]
        if len(devices) == 0 and not required:
            return None
        if relaxed and devices:
            chosen = devices[0]
            logger.warning(
                "Camera %s has %s linked devices; using most recent device %s for legacy compatibility",
                camera_uuid,
                len(devices),
                getattr(chosen, "device_uuid", None),
            )
            return chosen

        raise ValueError(f"Camera {camera_uuid} must have exactly 1 device, found {len(devices)}")

    async def list_devices(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
    ) -> List[Device]:
        q = (
            select(Device)
            .join(CameraDevice, CameraDevice.device_uuid == Device.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
            .order_by(CameraDevice.created_at.desc(), CameraDevice.id.desc())
        )
        return (await db.execute(q)).scalars().all()

    async def delete_camera(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> None:
        await db.execute(delete(PipelineCamera).where(PipelineCamera.camera_uuid == camera_uuid))
        await db.execute(delete(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid))
        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
        await db.execute(delete(Camera).where(Camera.camera_uuid == camera_uuid))
        await db.flush()

    async def _ensure_device_exists(self, db: AsyncSession, device_uuid: uuid.UUID) -> None:
        exists = (await db.execute(select(Device.device_uuid).where(Device.device_uuid == device_uuid))).scalar_one_or_none()
        if exists is None:
            raise ValueError(f"Device not found: {device_uuid}")

    async def _ensure_camera_has_exactly_one_device(self, db: AsyncSession, *, camera_uuid: uuid.UUID) -> None:
        cnt = (
            await db.execute(
                select(func.count(CameraDevice.id)).where(CameraDevice.camera_uuid == camera_uuid)
            )
        ).scalar_one()
        if int(cnt) != 1:
            raise ValueError(f"Camera {camera_uuid} must have exactly 1 device assigned, found {cnt}")

    async def _set_camera_device(
        self,
        db: AsyncSession,
        *,
        camera_uuid: uuid.UUID,
        device_uuid: uuid.UUID,
    ) -> None:
        clean = device_uuid if isinstance(device_uuid, uuid.UUID) else uuid.UUID(str(device_uuid))
        await self._ensure_device_exists(db, clean)

        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
        db.add(CameraDevice(camera_uuid=camera_uuid, device_uuid=clean))
        await db.flush()

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
