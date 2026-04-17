# routes/sites.py
import asyncio
import logging
import uuid
from datetime import datetime, time as dt_time, timezone
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select, delete, update
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Site, Device, SiteDevice, Camera, CameraDevice, SiteSettings, Notification, VideoRecord
from dependencies import get_async_db, get_current_user, get_manager
from application.channels.channel_config import VideoChannelConfig
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from domain.events import ChannelCreateEvent
from application.services.manager import Manager
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import EventClipService
from application.services.webrtcgateway import resolve_camera_webrtc_url
from core.schemas import CameraWithConfigSchema
from routes.device_routes import DeviceOut
from core.database import AsyncSessionLocal

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
    """
    OPTIMIZED: Delete blobs in parallel batches instead of sequentially.
    
    Deletes up to 10 blobs concurrently, then moves to batch of 10.
    This is 10x faster than sequential deletion for large blob sets.
    """
    unique = list(dict.fromkeys(k for k in keys if k))
    if not unique:
        return
    
    logger.info(f"[Blob Cleanup] Starting deletion of {len(unique)} {label} blobs (parallel, batch size=10)")
    svc = service_cls()
    deleted = 0
    failed = 0
    batch_size = 10

    try:
        for i in range(0, len(unique), batch_size):
            batch = unique[i : i + batch_size]
            # Delete up to 10 blobs concurrently
            tasks = [svc.delete_blob(blob_name=k) for k in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for key, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.warning(
                        f"[Blob Cleanup] Failed to delete {label} blob {key}: {result}"
                    )
                else:
                    deleted += 1
            
            logger.info(
                f"[Blob Cleanup] Batch {i // batch_size + 1}: "
                f"deleted {sum(1 for r in results if not isinstance(r, Exception))}/{len(batch)} {label} blobs"
            )
    finally:
        try:
            await svc.close()
        except Exception:
            pass
    
    logger.info(f"[Blob Cleanup] COMPLETE: deleted {deleted} {label} blobs, {failed} failed")


def _extract_notification_clip_storage_keys(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []

    keys: List[str] = []

    def _append_from_clip_dict(raw_clip: Any) -> None:
        if not isinstance(raw_clip, dict):
            return
        key = str(raw_clip.get("storage_key") or "").strip()
        if key:
            keys.append(key)

    msg = payload.get("msg")
    if isinstance(msg, dict):
        _append_from_clip_dict(msg)
        _append_from_clip_dict(msg.get("clip"))

    extra = payload.get("extra")
    if isinstance(extra, dict):
        _append_from_clip_dict(extra)
        _append_from_clip_dict(extra.get("clip"))
        for raw_clip in list(extra.get("multi_camera_prerecordings") or []):
            _append_from_clip_dict(raw_clip)

    _append_from_clip_dict(payload.get("clip"))

    return list(dict.fromkeys(keys))


async def _cleanup_cameras_background(
    manager: Manager,
    cam_snapshot: List[dict],
    *,
    user_id: int,
    site_uuid: uuid.UUID,
) -> None:
    """
    Best-effort runtime cleanup after the site DB rows are gone.
    
    Cleanup order:
    1. Delete from edge devices (Jetson)
    2. Delete WebRTC streams (MediaMTX/go2rtc)
    3. Remove from in-memory pipeline

    Important:
    - Do NOT call manager.get_activepipeline() here. After the DB delete that
      can recreate a brand-new empty pipeline just to remove channels.
    - Only evict from an already-loaded in-memory pipeline if one exists.
    """
    logger.info(f"[Cleanup] Starting background cleanup of {len(cam_snapshot)} cameras for user={user_id}")
    
    cleanup_targets: Dict[str, dict] = {
        str(cam["camera_uuid"]): {
            "camera_uuid": cam["camera_uuid"],
            "camera_code": cam.get("camera_code"),
            "device_urls": list(dict.fromkeys(cam.get("device_urls") or [])),
        }
        for cam in cam_snapshot
    }

    active_pipeline = None
    try:
        active_pipeline = manager.get_loaded_pipeline(user_id=user_id)
        if active_pipeline is not None:
            logger.info(f"[Cleanup] Using already-loaded pipeline for user={user_id}")
        else:
            logger.info(f"[Cleanup] No in-memory pipeline loaded for user={user_id}; channel eviction limited to DB snapshot")
    except Exception as e:
        logger.warning(
            f"[Cleanup] Could not inspect loaded pipeline user={user_id}: {e}; "
            "channel eviction limited to DB snapshot",
            exc_info=True,
        )
        active_pipeline = None

    if active_pipeline is not None:
        try:
            for channel_id in active_pipeline.list_channel_ids():
                cfg = await active_pipeline.get_channel_config(channel_id)
                if cfg is None or getattr(cfg, "site_uuid", None) != site_uuid:
                    continue

                runtime_cam_uuid = getattr(cfg, "camera_uuid", None) or channel_id
                runtime_key = str(runtime_cam_uuid)
                entry = cleanup_targets.get(runtime_key)
                if entry is None:
                    device_url = str(getattr(cfg, "device_url", "") or "").strip()
                    cleanup_targets[runtime_key] = {
                        "camera_uuid": runtime_cam_uuid,
                        "camera_code": None,
                        "device_urls": [device_url] if device_url else [],
                    }
                    logger.warning(
                        f"[Cleanup] Found runtime-only site camera={runtime_key} in loaded pipeline; adding it to cleanup set"
                    )
                else:
                    device_url = str(getattr(cfg, "device_url", "") or "").strip()
                    if device_url:
                        entry["device_urls"] = list(
                            dict.fromkeys(list(entry.get("device_urls") or []) + [device_url])
                        )
        except Exception as e:
            logger.warning(
                f"[Cleanup] Failed scanning loaded pipeline for site={site_uuid}: {e}",
                exc_info=True,
            )

    for cam in cleanup_targets.values():
        cam_uuid = cam["camera_uuid"]
        cam_code = cam.get("camera_code")
        device_urls = list(dict.fromkeys(cam.get("device_urls") or []))

        # Cleanup 1: Remove from edge devices
        for dev_url in device_urls:
            try:
                logger.info(f"[Cleanup] Deleting camera={cam_uuid} from edge device url={dev_url}")
                await manager._edge.delete_camera(
                    device_url=dev_url,
                    camera_uuid=str(cam_uuid),
                )
                logger.info(f"[Cleanup] Successfully deleted camera={cam_uuid} from edge device")
            except Exception as e:
                logger.warning(
                    f"[Cleanup] Edge delete failed cam={cam_uuid} url={dev_url}: {e}",
                    exc_info=True,
                )

        # Cleanup 2: Remove from WebRTC gateway
        if cam_code:
            try:
                logger.info(f"[Cleanup] Deleting WebRTC stream for camera={cam_uuid} code={cam_code}")
                deleted = await manager._webrtc.delete_stream(stream_key=str(cam_code))
                if deleted:
                    logger.info(f"[Cleanup] Successfully deleted WebRTC stream for camera={cam_uuid}")
                else:
                    logger.info(f"[Cleanup] WebRTC stream already absent for camera={cam_uuid}")
            except Exception as e:
                logger.warning(
                    f"[Cleanup] WebRTC delete failed cam={cam_uuid} code={cam_code}: {e}",
                    exc_info=True,
                )

        # Cleanup 3: Remove from in-memory pipeline
        if active_pipeline is not None:
            try:
                logger.info(f"[Cleanup] Removing camera={cam_uuid} from pipeline")
                await active_pipeline.remove_channel(cam_uuid)
                logger.info(f"[Cleanup] Successfully removed camera={cam_uuid} from pipeline")
            except Exception as e:
                logger.warning(
                    f"[Cleanup] Pipeline remove_channel failed cam={cam_uuid}: {e}",
                    exc_info=True,
                )

    logger.info(f"[Cleanup] COMPLETE: Background cleanup finished for {len(cleanup_targets)} cameras")


def _spawn_bg_task(coro, *, name: str) -> None:
    """Spawn a background task with proper error handling and completion logging."""
    task = asyncio.create_task(coro, name=name)

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
            logger.info(f"[Background Task] {name}: SUCCESS")
        except asyncio.CancelledError:
            logger.info(f"[Background Task] {name}: CANCELLED")
        except Exception as e:
            logger.exception(f"[Background Task] {name}: FAILED with error: {e}")

    task.add_done_callback(_on_done)


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
    notification_trigger_mode: Optional[Literal["roi_enter", "any_detection"]] = Field(
        default=None,
        description="Per-camera notification trigger mode. null = inherit site-level setting.",
    )
    camera_playback_enabled: Optional[bool] = Field(
        default=None,
        description="Per-camera clip recording override. null = inherit site default (prerecord list).",
    )
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


class SiteNotificationRule(BaseModel):
    trigger_mode: Literal["roi_enter", "any_detection"] = SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER


class SiteNotificationRuleUpdate(BaseModel):
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
    notification: SiteNotificationRule = Field(default_factory=SiteNotificationRule)


class SiteSettingsUpdate(BaseModel):
    schedule: Optional[SiteScheduleRuleUpdate] = None
    multi_camera_prerecord: Optional[SiteMultiCameraPrerecordRuleUpdate] = None
    notification: Optional[SiteNotificationRuleUpdate] = None


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

    notification_block = config.get("notification")
    if isinstance(notification_block, dict) and notification_block.get("trigger_mode") is not None:
        notification_trigger_mode = _normalize_trigger_mode(notification_block.get("trigger_mode"))
    else:
        notification_trigger_mode = _normalize_trigger_mode(prerecord.get("trigger_mode"))
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
        notification=SiteNotificationRule(
            trigger_mode=notification_trigger_mode,
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
        webrtc_url=resolve_camera_webrtc_url(
            camera_code=getattr(cam_out, "camera_code", None),
            stored_url=getattr(cam_out, "webrtc_url", None),
        ),
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        notification_trigger_mode=getattr(cam_out, "notification_trigger_mode", None),
        camera_playback_enabled=getattr(cam_out, "camera_playback_enabled", None),
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
    # Single query: join Site for ownership check + fetch devices in one round-trip.
    q = (
        select(Device)
        .join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid)
        .join(Site, Site.site_uuid == SiteDevice.site_uuid)
        .where(
            SiteDevice.site_uuid == site_uuid,
            Site.user_id == int(user.id),
            Device.user_id == int(user.id),
        )
        .order_by(Device.created_at.desc())
    )
    return (await db.execute(q)).scalars().all()


@router.get("/{site_uuid}/settings", response_model=SiteSettingsOut)
async def get_site_settings(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
):
    # Single query: fetch site + settings together, ownership check via Site.user_id.
    result = await db.execute(
        select(Site, SiteSettings)
        .outerjoin(SiteSettings, SiteSettings.site_uuid == Site.site_uuid)
        .where(Site.site_uuid == site_uuid, Site.user_id == int(user.id), Site.is_deleted == False)
    )
    row = result.one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Site not found")
    site, settings_row = row
    return _serialize_site_settings(site.site_uuid, settings_row, fallback_timezone=site.timezone)


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

    if payload.notification is not None:
        config["notification"] = {
            "trigger_mode": _normalize_trigger_mode(payload.notification.trigger_mode),
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

    if (
        payload.schedule is None
        and payload.multi_camera_prerecord is None
        and payload.notification is None
    ):
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
    if payload.multi_camera_prerecord is not None:
        # Invalidate prerecord eligibility cache so cameras pick up the
        # change without waiting for the TTL to expire.
        try:
            svc = getattr(manager, "_notification_service", None) if manager else None
            if svc is not None and hasattr(svc, "invalidate_prerecord_eligible_cache"):
                svc.invalidate_prerecord_eligible_cache()
        except Exception:
            pass
        await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    if payload.notification is not None:
        # Invalidate cached notification trigger_mode on pipeline and notification
        # service so the new setting takes effect without waiting for TTL.
        try:
            svc = getattr(manager, "_notification_service", None) if manager else None
            if svc is not None and hasattr(svc, "invalidate_site_trigger_mode_cache"):
                svc.invalidate_site_trigger_mode_cache(str(site.site_uuid))
            pipeline = manager.get_loaded_pipeline(user_id=int(user.id)) if manager else None
            if pipeline is not None and hasattr(pipeline, "invalidate_site_trigger_mode_cache"):
                pipeline.invalidate_site_trigger_mode_cache(str(site.site_uuid))
        except Exception:
            pass
        await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    await db.refresh(row)
    if payload.schedule is not None and manager is not None:
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
    manager:Manager = Depends(get_manager),
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
    if data.get("timezone") is not None and manager is not None:
        try:
            await _refresh_site_schedule_runtime(
                manager=manager,
                user_id=int(user.id),
                site_uuid=site.site_uuid,
            )
        except Exception:
            logger.warning("Best-effort schedule runtime refresh failed site=%s", site_uuid, exc_info=True)
    return site


@router.delete("/{site_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user=Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    """
    OPTIMIZED site deletion with batching for bulk cleanup.
    
    Deletion order:
    1. Snapshot camera/device info from DB
    2. STOP CAMERAS FIRST (synchronous):
       a. Purge notification service in-memory state (no new alerts buffered)
       b. Remove cameras from edge devices (Jetson stops detecting)
       c. Delete WebRTC streams (no new clips recorded)
       d. Evict from in-memory pipeline
    3. Extract blob keys from the now-stable DB
    4. Batch-delete DB rows (no new data arriving)
    5. Schedule async blob deletion
    6. Invalidate caches
    """
    from routes.notifications_routes import invalidate_camera_mode_cache

    logger.info(f"[Site Delete] Starting deletion of site={site_uuid}")

    site_repo = SiteRepository()
    site = await site_repo.get_site(db, user_id=user.id, site_uuid=site_uuid)

    # ========================================
    # PHASE 1: Snapshot camera info BEFORE any changes
    # ========================================
    logger.info(f"[Site Delete] Phase 1: Gathering camera info")
    camera_rows = (
        await db.execute(
            select(Camera.camera_uuid, Camera.camera_code).where(
                Camera.site_uuid == site.site_uuid,
                Camera.user_id == int(user.id),
            )
        )
    ).all()

    camera_uuids = [row[0] for row in camera_rows]
    cam_to_urls: Dict[uuid.UUID, set[str]] = {row[0]: set() for row in camera_rows}
    cam_to_code: Dict[uuid.UUID, Optional[str]] = {row[0]: row[1] for row in camera_rows}

    if camera_uuids:
        device_url_rows = (
            await db.execute(
                select(CameraDevice.camera_uuid, Device.device_url)
                .join(Device, Device.device_uuid == CameraDevice.device_uuid)
                .where(CameraDevice.camera_uuid.in_(camera_uuids))
            )
        ).all()

        for cam_uuid_key, dev_url in device_url_rows:
            url = str(dev_url or "").strip()
            if url:
                cam_to_urls.setdefault(cam_uuid_key, set()).add(url)

    cam_snapshot = [
        {
            "camera_uuid": cam_uuid_key,
            "camera_code": cam_to_code.get(cam_uuid_key),
            "device_urls": list(cam_to_urls.get(cam_uuid_key) or ()),
        }
        for cam_uuid_key in camera_uuids
    ]
    logger.info(f"[Site Delete] Snapshotted {len(cam_snapshot)} cameras")

    # ========================================
    # PHASE 1b: Disable cameras in DB BEFORE edge/MediaMTX cleanup.
    #
    # The reconcile endpoint reads `is_enabled` and `is_detection_enabled`
    # from the DB to decide which cameras to provision on edge devices and
    # MediaMTX.  If a reconcile fires between our edge cleanup (Phase 2)
    # and the DB deletion (Phase 4), it re-adds every camera we just removed.
    # Setting both flags to False first closes this race window.
    # ========================================
    if camera_uuids:
        logger.info(f"[Site Delete] Phase 1b: Disabling {len(camera_uuids)} cameras in DB to prevent reconcile re-adds")
        await db.execute(
            update(Camera)
            .where(Camera.site_uuid == site.site_uuid, Camera.user_id == int(user.id))
            .values(is_enabled=False, is_detection_enabled=False)
            .execution_options(synchronize_session=False)
        )
        await db.commit()
        logger.info(f"[Site Delete] Cameras disabled in DB")

    # ========================================
    # PHASE 2: STOP CAMERAS (synchronous, BEFORE any DB deletion)
    #
    # This is the critical gate. New images/detections/clips are uploaded
    # to blob storage every second the cameras run. If we delete the DB rows
    # first, the pipeline just re-creates them. We MUST stop the hardware
    # and in-process pipeline before touching the DB.
    # ========================================
    logger.info(f"[Site Delete] Phase 2: Stopping cameras before deletion")

    # 2a: Purge notification service in-memory state immediately.
    #     This prevents buffered detections for these cameras from being
    #     written to the DB after we start deleting.
    notif_svc = getattr(manager, "_notification_service", None) if manager is not None else None
    if notif_svc is not None:
        try:
            purge_fn = getattr(notif_svc, "purge_deleted_site_runtime_state", None)
            if callable(purge_fn):
                await purge_fn(
                    user_id=int(user.id),
                    site_uuid=site.site_uuid,
                    camera_uuids=camera_uuids,
                )
            else:
                notif_svc.invalidate_recipient_cache(user_id=int(user.id), site_uuid=site.site_uuid)
                for cam_uuid in camera_uuids:
                    notif_svc.invalidate_camera_roi_state(str(cam_uuid))
            logger.info(f"[Site Delete] Notification service state purged")
        except Exception as exc:
            logger.warning(f"[Site Delete] Failed to purge notification service state: {exc}", exc_info=True)

    # 2b: Remove cameras from edge devices, WebRTC, and pipeline (synchronous with timeout).
    #     After this call completes (or times out), no new detections/clips arrive.
    if manager is not None:
        logger.info(
            f"[Site Delete] Stopping {len(cam_snapshot)} cameras on edge/WebRTC/pipeline"
        )
        try:
            await asyncio.wait_for(
                _cleanup_cameras_background(
                    manager,
                    cam_snapshot,
                    user_id=int(user.id),
                    site_uuid=site.site_uuid,
                ),
                timeout=90.0,
            )
            logger.info(f"[Site Delete] Cameras stopped successfully")
        except asyncio.TimeoutError:
            logger.warning(
                f"[Site Delete] Camera stop timed out after 90s — proceeding with deletion. "
                f"Some cameras on edge devices may still be running briefly."
            )
        except Exception as exc:
            logger.warning(
                f"[Site Delete] Camera stop encountered errors — proceeding: {exc}",
                exc_info=True,
            )
    elif manager is None:
        logger.warning(
            f"[Site Delete] Manager unavailable — edge/WebRTC/pipeline stop skipped. site={site_uuid}"
        )

    # ========================================
    # PHASE 3a: Extract video record blob keys BEFORE camera deletion.
    # Camera deletion CASCADE-deletes VideoRecords, so we must snapshot
    # storage_key values now.  10K videos = small SELECT, fast.
    # ========================================
    video_clip_keys: List[str] = []
    if camera_uuids:
        async with AsyncSessionLocal() as vr_session:
            vr_rows = (
                await vr_session.execute(
                    select(VideoRecord.storage_key)
                    .where(VideoRecord.camera_uuid.in_(camera_uuids))
                )
            ).scalars().all()
            video_clip_keys = [k.strip() for k in vr_rows if k and k.strip()]
        logger.info(f"[Site Delete] Phase 3a: Extracted {len(video_clip_keys)} video record blob keys")

    # ========================================
    # PHASE 3b: Fast Foreground DB Cleanup (site graph minus heavy tables)
    # Delete settings, relationships, cameras, and notification emails
    # so the UI reflects the deletion immediately.
    # keep_site_row=True keeps the site row alive so FK CASCADE on
    # Notification.site_uuid does NOT wipe notifications before the
    # background task can extract their blob storage keys.
    # Camera deletion CASCADE-deletes VideoRecords (keys already saved
    # above) and SET NULLs Notification.camera_uuid (notifications
    # survive because the site row is kept).
    # ========================================
    logger.info(f"[Site Delete] Phase 3b: Starting foreground database cleanup (site graph)")

    try:
        await site_repo.delete_site_graph_batched(
            AsyncSessionLocal,
            site_uuid=site.site_uuid,
            camera_uuids=camera_uuids,
            batch_size=2000,
            keep_site_row=True,
        )
        # Mark site as soft-deleted so it disappears from all queries immediately
        async with AsyncSessionLocal() as sd_session:
            await sd_session.execute(
                update(Site)
                .where(Site.site_uuid == site.site_uuid)
                .values(is_deleted=True)
                .execution_options(synchronize_session=False)
            )
            await sd_session.commit()
        logger.info(f"[Site Delete] Foreground database cleanup complete (site row soft-deleted)")
    except Exception as e:
        logger.error(f"[Site Delete] Database cleanup FAILED: {e}", exc_info=True)
        raise

    # ========================================
    # PHASE 4: Invalidate caches
    # ========================================
    logger.info(f"[Site Delete] Phase 4: Invalidating camera mode caches")
    for camera_uuid in camera_uuids:
        try:
            await invalidate_camera_mode_cache(camera_uuid)
        except Exception as e:
            logger.warning(f"[Site Delete] Failed to invalidate cache for camera={camera_uuid}: {e}")

    # ========================================
    # PHASE 5: Background Heavy Table Cleanup (Notifications, Blobs, Site Row)
    # The site row is still alive (keep_site_row=True) so Notification
    # rows with site_uuid FK have NOT been cascade-deleted.
    # VideoRecords WERE cascade-deleted when cameras were removed in
    # Phase 3b, but their blob keys were captured in Phase 3a.
    # After heavy cleanup, the background task deletes the site row.
    # ========================================
    logger.info(f"[Site Delete] Phase 5: Spawning background task to clean up heavy tables (Notifications/Videos)")

    async def _heavy_table_cleanup_task(
        s_uuid: uuid.UUID,
        preextracted_video_keys: List[str],
    ):
        logger.info(f"[Site Cleanup Task] Starting background heavy cleanup for site={s_uuid}")
        alert_blob_keys: List[str] = []
        clip_blob_keys: List[str] = list(preextracted_video_keys)

        try:
            # Batch delete notifications by site_uuid (site row still exists)
            await site_repo._batch_delete(
                AsyncSessionLocal,
                table=Notification,
                where_clause=Notification.site_uuid == s_uuid,
                batch_size=2000,
                label="site_notifications",
                extract_col=Notification.payload,
                extract_alert_fn=extract_image_storage_key,
                extract_clip_fn=_extract_notification_clip_storage_keys,
                alert_keys_out=alert_blob_keys,
                clip_keys_out=clip_blob_keys,
            )

            # Schedule the blob deletions
            if alert_blob_keys:
                _spawn_bg_task(
                    _delete_blobs_background(
                        alert_blob_keys,
                        service_cls=AlertImageStorageService,
                        label="alert image",
                    ),
                    name=f"delete_site_alert_blobs:{s_uuid}",
                )

            if clip_blob_keys:
                _spawn_bg_task(
                    _delete_blobs_background(
                        clip_blob_keys,
                        service_cls=EventClipService,
                        label="clip",
                    ),
                    name=f"delete_site_clip_blobs:{s_uuid}",
                )

            # Finally delete the site row (CASCADE cleans up any stragglers)
            await site_repo._fast_delete(
                AsyncSessionLocal, Site, Site.site_uuid == s_uuid
            )
            logger.info(f"[Site Cleanup Task] Background heavy cleanup COMPLETE for site={s_uuid}")

        except Exception as e:
            logger.error(f"[Site Cleanup Task] Failed heavy cleanup for site={s_uuid}: {e}", exc_info=True)

    _spawn_bg_task(
        _heavy_table_cleanup_task(site.site_uuid, video_clip_keys),
        name=f"delete_site_heavy_tables:{site.site_uuid}",
    )

    logger.info(f"[Site Delete] COMPLETE: site={site_uuid} has been successfully deleted")
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
