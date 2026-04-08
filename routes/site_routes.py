# routes/sites.py
import asyncio
import logging
import uuid
from datetime import datetime, time as dt_time, timezone
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Site, Device, SiteDevice, Camera, CameraDevice, SiteSettings, Notification, VideoRecord
from dependencies import get_async_db, get_current_user, get_manager, get_manager_optional
from application.channels.channel_config import VideoChannelConfig
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from domain.events import ChannelCreateEvent
from application.services.manager import Manager
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import EventClipService
from core.schemas import CameraWithConfigSchema
from routes.device_routes import DeviceOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sites", tags=["sites"])
SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER = "roi_enter"
SITE_PRERECORD_TRIGGER_MODE_ANY_DETECTION = "any_detection"
SITE_PRERECORD_TRIGGER_MODES = {
    SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER,
    SITE_PRERECORD_TRIGGER_MODE_ANY_DETECTION,
}
SUNDAY_TO_SATURDAY = [6, 0, 1, 2, 3, 4, 5]
SCHEDULE_TIME_PATTERN = r"^\d{2}:\d{2}(:\d{2})?$"


# -----------------------
# Helpers
# -----------------------
def _is_blank(s: Optional[str]) -> bool:
    return s is None or (isinstance(s, str) and s.strip() == "")

def _gen_code(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def _normalize_trigger_mode(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in SITE_PRERECORD_TRIGGER_MODES:
        return normalized
    return SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


def _normalize_camera_schedule_inputs(model: BaseModel) -> BaseModel:
    for attr in ("timezone", "start_time", "end_time"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            cleaned = value.strip()
            setattr(model, attr, cleaned or None)

    raw_schedule = getattr(model, "schedule", None)
    if raw_schedule is not None:
        normalized_schedule = VideoChannelConfig.normalize_schedule(raw_schedule)
        if raw_schedule and not normalized_schedule:
            raise ValueError("schedule must contain at least one valid day/time window.")
        setattr(model, "schedule", normalized_schedule)
        if normalized_schedule:
            return model

    raw_days = getattr(model, "day_of_week", None)
    if raw_days is not None:
        normalized_days: List[int] = []
        seen_days = set()
        for value in raw_days:
            try:
                day = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("day_of_week values must be integers from 0 to 6.") from exc
            if day < 0 or day > 6:
                raise ValueError("day_of_week values must be between 0 and 6.")
            if day in seen_days:
                continue
            seen_days.add(day)
            normalized_days.append(day)
        setattr(model, "day_of_week", normalized_days)

    day_of_week = getattr(model, "day_of_week", None)
    start_time = getattr(model, "start_time", None)
    end_time = getattr(model, "end_time", None)
    if any(value is not None for value in (day_of_week, start_time, end_time)):
        if not day_of_week:
            raise ValueError("Select at least one day when providing a schedule.")
        if not start_time or not end_time:
            raise ValueError("start_time and end_time are required when providing a schedule.")
        try:
            start_obj = dt_time.fromisoformat(str(start_time))
            end_obj = dt_time.fromisoformat(str(end_time))
        except ValueError as exc:
            raise ValueError("start_time and end_time must use HH:MM or HH:MM:SS format.") from exc
        if start_obj == end_obj:
            raise ValueError("start_time and end_time must be different.")

    return model


def _sort_days_sunday_first(values: List[int]) -> List[int]:
    order_index = {day: idx for idx, day in enumerate(SUNDAY_TO_SATURDAY)}
    return sorted(values, key=lambda day: order_index.get(int(day), 999))


def _time_to_schedule_str(value: Any, default: dt_time) -> str:
    if value is None:
        return default.strftime("%H:%M:%S")
    if isinstance(value, dt_time):
        return value.strftime("%H:%M:%S")
    return dt_time.fromisoformat(str(value)).strftime("%H:%M:%S")


def _default_site_schedule_payload(timezone_name: Optional[str]) -> Dict[str, Any]:
    return {
        "timezone": str(timezone_name or "UTC"),
        "day_of_week": list(SUNDAY_TO_SATURDAY),
        "start_time": "00:00:00",
        "end_time": "23:59:59",
        "schedule": VideoChannelConfig.default_schedule(),
    }


def _primary_schedule_window_payload(schedule: List[Dict[str, Any]]) -> Dict[str, Any]:
    windows = VideoChannelConfig.schedule_windows(schedule)
    if not windows:
        return {
            "day_of_week": list(SUNDAY_TO_SATURDAY),
            "start_time": "00:00:00",
            "end_time": "23:59:59",
        }

    template = windows[0]
    return {
        "day_of_week": _sort_days_sunday_first(
            [int(value) for value in template.get("day_of_week") or []]
        )
        or list(SUNDAY_TO_SATURDAY),
        "start_time": _time_to_schedule_str(template.get("start_time"), dt_time(0, 0, 0)),
        "end_time": _time_to_schedule_str(template.get("end_time"), dt_time(23, 59, 59)),
    }


def _site_schedule_payload_from_row(
    row: Optional[SiteSettings],
    *,
    fallback_timezone: Optional[str],
) -> Dict[str, Any]:
    config = dict(row.config or {}) if row and isinstance(getattr(row, "config", None), dict) else {}
    schedule = VideoChannelConfig.normalize_schedule(config.get("schedule"))

    if not schedule and row is not None:
        row_day = getattr(row, "day_of_week", None)
        row_start = getattr(row, "start_time", None)
        row_end = getattr(row, "end_time", None)
        if row_day is not None or row_start is not None or row_end is not None:
            schedule = [
                {
                    "day_of_week": int(row_day if row_day is not None else 6),
                    "start_time": _time_to_schedule_str(row_start, dt_time(0, 0, 0)),
                    "end_time": _time_to_schedule_str(row_end, dt_time(23, 59, 59)),
                    "is_enabled": bool(getattr(row, "is_enabled", True)),
                }
            ]

    if not schedule:
        return _default_site_schedule_payload(config.get("timezone") or fallback_timezone)

    primary = _primary_schedule_window_payload(schedule)

    return {
        "timezone": str(config.get("timezone") or fallback_timezone or "UTC"),
        "day_of_week": list(primary["day_of_week"]),
        "start_time": str(primary["start_time"]),
        "end_time": str(primary["end_time"]),
        "schedule": schedule,
    }


def _sync_site_settings_timezone_row(
    row: Optional[SiteSettings],
    *,
    timezone_name: Optional[str],
) -> None:
    if row is None:
        return

    config = dict(row.config or {}) if isinstance(getattr(row, "config", None), dict) else {}
    config["timezone"] = str(timezone_name or "UTC")
    row.config = config


def _build_schedule_windows(
    *,
    day_of_week: List[int],
    start_time: str,
    end_time: str,
    schedule: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if schedule is not None:
        normalized = VideoChannelConfig.normalize_schedule(schedule)
        return normalized or VideoChannelConfig.default_schedule()

    normalized = VideoChannelConfig.normalize_schedule(
        [
            {
                "day_of_week": int(day),
                "start_time": _time_to_schedule_str(start_time, dt_time(0, 0, 0)),
                "end_time": _time_to_schedule_str(end_time, dt_time(23, 59, 59)),
                "is_enabled": True,
            }
            for day in _sort_days_sunday_first(day_of_week)
        ]
    )
    return normalized or VideoChannelConfig.default_schedule()


def _dedupe_uuid_list(values: Optional[List[uuid.UUID]]) -> List[uuid.UUID]:
    seen = set()
    out: List[uuid.UUID] = []
    for value in values or []:
        parsed = value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        key = str(parsed)
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)
    return out


async def _delete_blobs_background(keys: List[str], *, service_cls: type, label: str) -> None:
    """Delete a deduplicated list of blob storage keys. Errors per-blob are logged, never raised."""
    unique = list(dict.fromkeys(k for k in keys if k))
    if not unique:
        return
    svc = service_cls()
    try:
        for key in unique:
            try:
                await svc.delete_blob(blob_name=key)
            except Exception:
                logger.warning("Failed to delete %s blob %r", label, key, exc_info=True)
    finally:
        try:
            await svc.close()
        except Exception:
            pass


async def _cleanup_cameras_background(
    manager: "Manager",
    cam_snapshot: List[dict],
    active_pipeline: "Optional[Any]",
) -> None:
    """Fire-and-forget: stop edge inference streams, remove WebRTC publish keys,
    and evict channels from the in-memory pipeline.

    Each camera is handled independently — a timeout or error on one device
    never prevents cleanup of the remaining cameras.

    This intentionally runs AFTER the DB commit so the HTTP response is never
    blocked by slow or unreachable edge devices (which can take ~30 s per camera
    when the device is offline: 3 retries × 10 s connect-timeout each).
    """
    for cam in cam_snapshot:
        cam_uuid = cam["camera_uuid"]
        cam_code = cam.get("camera_code")
        device_urls = cam.get("device_urls") or []

        for dev_url in device_urls:
            try:
                await manager._edge.delete_camera(
                    device_url=dev_url, camera_uuid=str(cam_uuid)
                )
            except Exception:
                logger.warning(
                    "site cleanup [bg]: edge delete failed cam=%s url=%s",
                    cam_uuid, dev_url, exc_info=True,
                )

        if cam_code:
            try:
                await manager._webrtc.delete_stream(stream_key=str(cam_code))
            except Exception:
                logger.warning(
                    "site cleanup [bg]: WebRTC delete failed cam=%s code=%s",
                    cam_uuid, cam_code, exc_info=True,
                )

        if active_pipeline is not None:
            try:
                await active_pipeline.remove_channel(cam_uuid)
            except Exception:
                logger.warning(
                    "site cleanup [bg]: pipeline remove_channel failed cam=%s",
                    cam_uuid, exc_info=True,
                )


async def _invalidate_site_camera_mode_cache(
    *,
    db: AsyncSession,
    site_uuid: uuid.UUID,
) -> None:
    from routes.notifications_routes import invalidate_camera_mode_cache

    camera_uuids = (
        await db.execute(select(Camera.camera_uuid).where(Camera.site_uuid == site_uuid))
    ).scalars().all()
    for camera_uuid in camera_uuids:
        await invalidate_camera_mode_cache(camera_uuid)


async def _refresh_site_schedule_runtime(
    *,
    manager: Manager,
    user_id: int,
    site_uuid: uuid.UUID,
) -> None:
    sync_summary = await manager.sync_site_schedule_runtime(
        user_id=int(user_id),
        site_uuid=site_uuid,
    )
    if sync_summary.get("device_uuids"):
        asyncio.create_task(
            manager.reconcile_devices_best_effort(
                user_id=int(user_id),
                device_uuids=sync_summary["device_uuids"],
            )
        )


# -----------------------
# Schemas
# -----------------------
class SiteCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    site_code: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=255)
    timezone: Optional[str] = Field(default="UTC", max_length=50)


class SiteUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    site_code: Optional[str] = Field(default=None, max_length=64)
    address: Optional[str] = Field(default=None, max_length=255)
    timezone: Optional[str] = Field(default=None, max_length=50)


class SiteOut(BaseModel):
    site_uuid: uuid.UUID
    user_id: int
    name: str
    site_code: Optional[str] = None
    address: Optional[str] = None
    timezone: Optional[str] = "UTC"

    class Config:
        from_attributes = True


class LinkDeviceRequest(BaseModel):
    device_uuid: uuid.UUID


class SiteCameraCreate(BaseModel):
    device_uuid: uuid.UUID
    rtsp_url: str = Field(..., min_length=1, max_length=2048)
    name: Optional[str] = Field(default=None, max_length=255)
    location: Optional[str] = Field(default=None, max_length=255)
    is_enabled: bool = True
    is_detection_enabled: bool = True
    is_notification_enabled: bool = True
    sample_fps: float = Field(default=5.0, ge=0.1)
    timezone: Optional[str] = None
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None
    use_site_schedule: Optional[bool] = None

    @model_validator(mode="after")
    def _validate_schedule(self):
        return _normalize_camera_schedule_inputs(self)


class SiteMultiCameraPrerecordRule(BaseModel):
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteMultiCameraPrerecordRuleUpdate(BaseModel):
    enabled: bool = False
    camera_uuids: List[uuid.UUID] = Field(default_factory=list)
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteScheduleRule(BaseModel):
    timezone: str = Field(default="UTC", max_length=50)
    day_of_week: List[int] = Field(default_factory=lambda: list(SUNDAY_TO_SATURDAY))
    start_time: str = Field(default="00:00:00", pattern=SCHEDULE_TIME_PATTERN)
    end_time: str = Field(default="23:59:59", pattern=SCHEDULE_TIME_PATTERN)
    schedule: List[Dict[str, Any]] = Field(default_factory=VideoChannelConfig.default_schedule)


class SiteScheduleRuleUpdate(BaseModel):
    timezone: Optional[str] = Field(default=None, max_length=50)
    day_of_week: Optional[List[int]] = None
    start_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    end_time: Optional[str] = Field(default=None, pattern=SCHEDULE_TIME_PATTERN)
    schedule: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def _validate_schedule(self):
        return _normalize_camera_schedule_inputs(self)


class SiteSettingsOut(BaseModel):
    site_uuid: uuid.UUID
    schedule: SiteScheduleRule = Field(default_factory=SiteScheduleRule)
    multi_camera_prerecord: SiteMultiCameraPrerecordRule = Field(
        default_factory=SiteMultiCameraPrerecordRule
    )


class SiteSettingsUpdate(BaseModel):
    schedule: Optional[SiteScheduleRuleUpdate] = None
    multi_camera_prerecord: Optional[SiteMultiCameraPrerecordRuleUpdate] = None


async def _validate_site_prerecord_camera_uuids(
    db: AsyncSession,
    *,
    user_id: int,
    site_uuid: uuid.UUID,
    camera_uuids: List[uuid.UUID],
) -> List[uuid.UUID]:
    normalized = _dedupe_uuid_list(camera_uuids)
    if not normalized:
        return []

    stmt = select(Camera.camera_uuid).where(
        Camera.user_id == int(user_id),
        Camera.site_uuid == site_uuid,
        Camera.camera_uuid.in_(normalized),
    )
    rows = (await db.execute(stmt)).scalars().all()
    found = {str(value) for value in rows}
    missing = [str(value) for value in normalized if str(value) not in found]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Some selected cameras do not belong to this site.",
                "camera_uuids": missing,
            },
        )

    return normalized


def _serialize_site_settings(
    site_uuid: uuid.UUID,
    row: Optional[SiteSettings],
    *,
    fallback_timezone: Optional[str],
) -> SiteSettingsOut:
    config: Dict[str, Any] = row.config if isinstance(getattr(row, "config", None), dict) else {}
    prerecord = config.get("multi_camera_prerecord")
    if not isinstance(prerecord, dict):
        prerecord = {}

    camera_uuids: List[uuid.UUID] = []
    for value in prerecord.get("camera_uuids") or []:
        try:
            camera_uuids.append(uuid.UUID(str(value)))
        except Exception:
            continue
    camera_uuids = _dedupe_uuid_list(camera_uuids)
    schedule = _site_schedule_payload_from_row(row, fallback_timezone=fallback_timezone)

    return SiteSettingsOut(
        site_uuid=site_uuid,
        schedule=SiteScheduleRule(
            timezone=str(schedule.get("timezone") or fallback_timezone or "UTC"),
            day_of_week=[int(value) for value in schedule.get("day_of_week") or list(SUNDAY_TO_SATURDAY)],
            start_time=str(schedule.get("start_time") or "00:00:00"),
            end_time=str(schedule.get("end_time") or "23:59:59"),
            schedule=VideoChannelConfig.normalize_schedule(schedule.get("schedule"))
            or VideoChannelConfig.default_schedule(),
        ),
        multi_camera_prerecord=SiteMultiCameraPrerecordRule(
            enabled=bool(prerecord.get("enabled")),
            camera_uuids=camera_uuids,
            trigger_mode=_normalize_trigger_mode(prerecord.get("trigger_mode")),
        ),
    )


def _camera_out_to_response(cam_out) -> CameraWithConfigSchema:
    now = datetime.now(timezone.utc)
    return CameraWithConfigSchema(
        camera_uuid=cam_out.camera_uuid,
        camera_code=cam_out.camera_code,
        name=getattr(cam_out, "name", None),
        location=getattr(cam_out, "location", None),
        site_uuid=cam_out.site_uuid,
        device_uuid=cam_out.device_uuid,
        rtsp_url=cam_out.rtsp_url,
        webrtc_url=cam_out.webrtc_url,
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        use_site_schedule=bool(getattr(cam_out, "use_site_schedule", True)),
        roi=cam_out.roi,
        configuration=cam_out.configuration,
        timezone=cam_out.timezone,
        created_at=getattr(cam_out, "created_at", None) or now,
        updated_at=getattr(cam_out, "updated_at", None) or now,
    )


# -----------------------
# Routes
# -----------------------
@router.get("", response_model=List[SiteOut])
async def list_sites(
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_repo=SiteRepository()
    sites = await site_repo.get_sites(db, user_id=user.id)
    return sites

@router.get("/{site_uuid}/devices", response_model=List[DeviceOut])
async def list_site_devices(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db,user_id=user.id,site_uuid= site_uuid)
    q = (
        select(Device)
        .join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid)
        .where(SiteDevice.site_uuid == site.site_uuid, Device.user_id == user.id)
        .order_by(Device.created_at.desc())
    )
    return (await db.execute(q)).scalars().all()


@router.get("/{site_uuid}/settings", response_model=SiteSettingsOut)
async def get_site_settings(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, user_id=user.id,site_uuid= site_uuid)
    row = await site_repo.get_site_settings(db, user_id=int(user.id), site_uuid=site.site_uuid)
    return _serialize_site_settings(site.site_uuid, row, fallback_timezone=site.timezone)


@router.post("/{site_uuid}/cameras", response_model=CameraWithConfigSchema, status_code=status.HTTP_201_CREATED)
async def create_site_camera(
    site_uuid: uuid.UUID,
    payload: SiteCameraCreate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db,user_id=user.id,site_uuid= site_uuid)

    device = (
        await db.execute(
            select(Device)
            .join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid)
            .where(
                Device.user_id == int(user.id),
                Device.device_uuid == payload.device_uuid,
                SiteDevice.site_uuid == site.site_uuid,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if device is None:
        raise HTTPException(
            status_code=422,
            detail="Select a device already linked to this site before adding a camera.",
        )
    if not str(getattr(device, "device_url", "")).strip():
        raise HTTPException(status_code=422, detail="Selected device is missing device_url.")

    data = payload.model_dump(exclude_none=True)
    data["site_uuid"] = site.site_uuid
    data["device_url"] = device.device_url
    data["user_id"] = int(user.id)

    try:
        pipeline = await manager.get_activepipeline(user_id=user.id)
        ev = ChannelCreateEvent(
            channel_id=None,
            configs=data,
            created_at=datetime.now(timezone.utc),
        )

        result = await manager.update_pipeline(
            pipeline.pipeline_id,
            [ev],
            user_id=user.id,
            camera_code_prefix="cam",
        )
        if not result or not result.cameras:
            raise HTTPException(status_code=500, detail="Operation failed to create camera record")
        return _camera_out_to_response(result.cameras[0])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create camera: {str(exc)}") from exc


@router.patch("/{site_uuid}/settings", response_model=SiteSettingsOut)
async def update_site_settings(
    site_uuid: uuid.UUID,
    payload: SiteSettingsUpdate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, user_id=user.id,site_uuid=site_uuid)
    row = await site_repo.get_site_settings(db, user_id=int(user.id), site_uuid=site.site_uuid)
    config = dict(row.config or {}) if row and isinstance(row.config, dict) else {}
    schedule_payload = _site_schedule_payload_from_row(row, fallback_timezone=site.timezone)

    if payload.multi_camera_prerecord is not None:
        rule = payload.multi_camera_prerecord
        trigger_mode = _normalize_trigger_mode(rule.trigger_mode)
        camera_uuids = await _validate_site_prerecord_camera_uuids(
            db,
            user_id=int(user.id),
            site_uuid=site.site_uuid,
            camera_uuids=rule.camera_uuids,
        )

        if rule.enabled and not camera_uuids:
            raise HTTPException(
                status_code=422,
                detail="Select at least one site camera before enabling multi-camera prerecord.",
            )

        config["multi_camera_prerecord"] = {
            "enabled": bool(rule.enabled),
            "camera_uuids": [str(value) for value in camera_uuids],
            "trigger_mode": trigger_mode,
        }

    if payload.schedule is not None:
        schedule_windows = _build_schedule_windows(
            day_of_week=list(payload.schedule.day_of_week or schedule_payload.get("day_of_week") or list(SUNDAY_TO_SATURDAY)),
            start_time=str(payload.schedule.start_time or schedule_payload.get("start_time") or "00:00:00"),
            end_time=str(payload.schedule.end_time or schedule_payload.get("end_time") or "23:59:59"),
            schedule=payload.schedule.schedule,
        )
        primary = _primary_schedule_window_payload(schedule_windows)
        schedule_payload = {
            "timezone": str(payload.schedule.timezone or schedule_payload.get("timezone") or site.timezone or "UTC"),
            "day_of_week": list(primary["day_of_week"]),
            "start_time": str(primary["start_time"]),
            "end_time": str(primary["end_time"]),
            "schedule": schedule_windows,
        }
        config["timezone"] = schedule_payload["timezone"]
        config["schedule"] = schedule_windows
        site.timezone = schedule_payload["timezone"]

    if payload.schedule is None and payload.multi_camera_prerecord is None:
        return _serialize_site_settings(site.site_uuid, row, fallback_timezone=site.timezone)

    row = await site_repo.upsert_site_settings(
        db,
        user_id=int(user.id),
        site_uuid=site.site_uuid,
        config=config,
        day_of_week=list(schedule_payload.get("day_of_week") or list(SUNDAY_TO_SATURDAY)),
        start_time=dt_time.fromisoformat(str(schedule_payload.get("start_time") or "00:00:00")),
        end_time=dt_time.fromisoformat(str(schedule_payload.get("end_time") or "23:59:59")),
        is_enabled=True,
    )
    await db.commit()
    if payload.schedule is not None:
        await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    await db.refresh(row)
    if payload.schedule is not None:
        try:
            await _refresh_site_schedule_runtime(
                manager=manager,
                user_id=int(user.id),
                site_uuid=site.site_uuid,
            )
        except Exception:
            logger.warning(
                "Failed to refresh site schedule runtime for site=%s; settings were saved successfully.",
                site.site_uuid,
                exc_info=True,
            )
    return _serialize_site_settings(site.site_uuid, row, fallback_timezone=site.timezone)


@router.post("", response_model=SiteOut, status_code=status.HTTP_201_CREATED)
async def create_site(
    payload: SiteCreate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_code = payload.site_code
    if _is_blank(site_code):
        site_code = _gen_code("site")

    site = Site(
        user_id=user.id,
        name=payload.name,
        site_code=site_code,
        address=payload.address,
        timezone=payload.timezone or "UTC",
    )

    db.add(site)
    await db.commit()
    await db.refresh(site)
    return site


@router.get("/{site_uuid}", response_model=SiteOut)
async def get_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db,user_id=user.id,site_uuid=site_uuid)
    return site

@router.patch("/{site_uuid}", response_model=SiteOut)
async def update_site(
    site_uuid: uuid.UUID,
    payload: SiteUpdate,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db,user_id=user.id,site_uuid=site_uuid)


    data = payload.model_dump(exclude_unset=True)

    # If they included site_code but it’s blank -> regenerate
    if "site_code" in data and _is_blank(data.get("site_code")):
        data["site_code"] = _gen_code("site")

    # Apply patch
    for k, v in data.items():
        if v is not None:
            setattr(site, k, v)

    if data.get("timezone") is not None:
        row = await site_repo.get_site_settings(
            db,
            user_id=int(user.id),
            site_uuid=site.site_uuid,
        )
        _sync_site_settings_timezone_row(row, timezone_name=site.timezone)

    await db.commit()
    if data.get("timezone") is not None:
        await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    await db.refresh(site)
    if data.get("timezone") is not None:
        await _refresh_site_schedule_runtime(
            manager=manager,
            user_id=int(user.id),
            site_uuid=site.site_uuid,
        )
    return site


@router.delete("/{site_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    # get_manager_optional never raises 503 — DB deletion must succeed even
    # when the manager (pipeline / edge layer) has not fully started up or is
    # temporarily unavailable (Azure Container App cold-start, scale-out, etc.)
    manager: Optional[Manager] = Depends(get_manager_optional),
):
    from routes.notifications_routes import invalidate_camera_mode_cache

    site_repo = SiteRepository()
    site = await site_repo.get_site(db, user_id=user.id, site_uuid=site_uuid)

    # ── 1. Snapshot everything we need BEFORE the cascade commit ──────────
    #
    # After db.commit() the Camera, VideoRecord, Notification rows are gone.
    # Build plain-Python dicts so the data can be safely used in background
    # tasks after this request's DB session is closed.

    # Load cameras with their linked devices in one query.
    cameras_orm = (
        await db.execute(
            select(Camera)
            .where(Camera.site_uuid == site.site_uuid)
            .options(selectinload(Camera.devices))
        )
    ).scalars().all()

    # Serialise into plain dicts — ORM objects must NOT be used after commit.
    cam_snapshot = [
        {
            "camera_uuid": cam.camera_uuid,
            "camera_code": getattr(cam, "camera_code", None),
            "device_urls": list({
                str(getattr(dev, "device_url", "") or "").strip()
                for dev in (getattr(cam, "devices", None) or [])
                if str(getattr(dev, "device_url", "") or "").strip()
            }),
        }
        for cam in cameras_orm
    ]
    camera_uuids = [c["camera_uuid"] for c in cam_snapshot]

    # Alert image blobs — stored in Azure, rows CASCADE-deleted with site.
    alert_blob_keys: List[str] = []
    for payload in (
        await db.execute(select(Notification.payload).where(Notification.site_uuid == site.site_uuid))
    ).scalars().all():
        key = extract_image_storage_key(payload)
        if key:
            alert_blob_keys.append(key)

    # Video clip blobs — stored in Azure, rows CASCADE-deleted via Camera.
    clip_blob_keys: List[str] = []
    if camera_uuids:
        clip_blob_keys = [
            str(k).strip()
            for k in (
                await db.execute(
                    select(VideoRecord.storage_key).where(
                        VideoRecord.camera_uuid.in_(camera_uuids),
                        VideoRecord.storage_key.isnot(None),
                    )
                )
            ).scalars().all()
            if str(k or "").strip()
        ]

    # Resolve active pipeline now (in-memory lookup, fast) so the background
    # task can evict channels without needing the DB.
    active_pipeline = None
    if manager is not None:
        try:
            active_pipeline = await manager.get_activepipeline(user_id=user.id)
        except Exception:
            logger.warning(
                "Could not load active pipeline for site=%s; pipeline eviction skipped.",
                site_uuid, exc_info=True,
            )

    # ── 2. Delete the site row — DB CASCADE removes everything ────────────
    # Camera, ChannelConfiguration, CameraDevice, PipelineCamera,
    # VideoRecord, SiteSettings, SiteDevice, Notification, NotificationEmail.
    await db.delete(site)
    await db.commit()
    # The 204 is logically ready here. Everything below is cleanup.

    # ── 3. Background: edge inference + WebRTC + pipeline channel cleanup ──
    # _edge.delete_camera() retries 3× with a 10 s connect-timeout each
    # (~30 s total per unreachable device).  Running this synchronously was
    # the root cause of 503s: the Azure Container App proxy timed out while
    # waiting for the edge device.  It is now fire-and-forget.
    if manager is not None and cam_snapshot:
        asyncio.create_task(
            _cleanup_cameras_background(manager, cam_snapshot, active_pipeline)
        )
    elif manager is None:
        logger.warning(
            "Manager unavailable while deleting site=%s — "
            "edge/WebRTC/pipeline cleanup skipped.",
            site_uuid,
        )

    # ── 4. Background: Azure blob cleanup ─────────────────────────────────
    if alert_blob_keys:
        asyncio.create_task(
            _delete_blobs_background(alert_blob_keys, service_cls=AlertImageStorageService, label="alert image")
        )
    if clip_blob_keys:
        asyncio.create_task(
            _delete_blobs_background(clip_blob_keys, service_cls=EventClipService, label="clip")
        )

    # ── 5. Sync: flush notification-service in-memory caches ──────────────
    # These are pure in-memory dict operations — microseconds, safe to do
    # synchronously before returning.
    if manager is not None:
        notif_svc = getattr(manager, "_notification_service", None)
        if notif_svc is not None:
            try:
                notif_svc.invalidate_recipient_cache(user_id=int(user.id), site_uuid=site_uuid)
            except Exception:
                logger.warning("Failed to invalidate recipient cache site=%s", site_uuid, exc_info=True)
            for camera_uuid in camera_uuids:
                try:
                    notif_svc.invalidate_camera_roi_state(str(camera_uuid))
                except Exception:
                    logger.warning("Failed to invalidate ROI state camera=%s", camera_uuid, exc_info=True)

    # ── 6. Sync: evict per-camera notification-mode route cache ───────────
    for camera_uuid in camera_uuids:
        await invalidate_camera_mode_cache(camera_uuid)

    return None


# -----------------------------------------
# Optional: Site <-> Device linking endpoints
# -----------------------------------------
@router.post("/{site_uuid}/devices", status_code=status.HTTP_201_CREATED)
async def link_device_to_site(
    site_uuid: uuid.UUID,
    payload: LinkDeviceRequest,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db,user_id=user.id,site_uuid=site_uuid)


    qd = select(Device).where(Device.device_uuid == payload.device_uuid, Device.user_id == user.id)
    device = (await db.execute(qd)).scalar_one_or_none()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    qlink = select(SiteDevice).where(
        SiteDevice.site_uuid == site.site_uuid,
        SiteDevice.device_uuid == device.device_uuid,
    )
    exists = (await db.execute(qlink)).scalar_one_or_none()
    if exists:
        return {"linked": True, "already": True}

    db.add(SiteDevice(site_uuid=site.site_uuid, device_uuid=device.device_uuid))
    await db.commit()
    return {"linked": True, "already": False}


@router.delete("/{site_uuid}/devices/{device_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def unlink_device_from_site(
    site_uuid: uuid.UUID,
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db,user_id=user.id,site_uuid=site_uuid)


    camera_using_device = (
        await db.execute(
            select(Camera.camera_uuid)
            .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
            .where(
                Camera.site_uuid == site.site_uuid,
                CameraDevice.device_uuid == device_uuid,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if camera_using_device is not None:
        raise HTTPException(
            status_code=409,
            detail="Cannot unlink device while cameras in this site are assigned to it. Move or delete those cameras first.",
        )

    stmt = delete(SiteDevice).where(
        SiteDevice.site_uuid == site.site_uuid,
        SiteDevice.device_uuid == device_uuid,
    )
    await db.execute(stmt)
    await db.commit()
    return None
