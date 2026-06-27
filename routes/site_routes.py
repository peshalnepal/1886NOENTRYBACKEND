# routes/sites.py
import asyncio
import logging
import uuid
from datetime import datetime, time as dt_time, timezone
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import select as _sa_select

from core.database_orm import AccessGrant, Role, Site, SiteSettings, Notification
from core.security.roles import Permission, RoleScope
from dependencies import (
    get_async_db,
    get_current_user,
    get_manager,
    RequirePermission,
    OrgContext,
)
from application.services.authz_service import AuthzService
from application.channels.channel_config import VideoChannelConfig
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from application.dtos import SiteCreateDTO, SiteSettingsUpsertDTO, SiteUpdateDTO
from application.repositories.device_repository import DeviceRepository
from application.repositories.video_repository import VideoRepository
from domain.events import ChannelCreateEvent
from application.services.manager import Manager
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import (
    EventClipService,
    extract_notification_clip_storage_keys as _extract_notification_clip_storage_keys,
)
from application.services.webrtcgateway import resolve_camera_webrtc_url
from core.schemas import (
    CameraWithConfigSchema,
    DeviceOut,
    LinkDeviceRequest,
    SITE_PRERECORD_TRIGGER_MODES,
    SITE_PRERECORD_TRIGGER_MODE_ROI_ENTER,
    SUNDAY_TO_SATURDAY,
    SiteCameraCreate,
    SiteCreate,
    SiteMultiCameraPrerecordRule,
    SiteNotificationRule,
    SiteOut,
    SiteScheduleRule,
    SiteSettingsOut,
    SiteSettingsUpdate,
    SiteUpdate,
)
from core.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sites", tags=["sites"])


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


from routes._background import _delete_blobs_background, _spawn_bg_task  # noqa: E402


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


async def _invalidate_site_camera_mode_cache(
    *,
    db: AsyncSession,
    site_uuid: uuid.UUID,
) -> None:
    from routes.notifications_routes import invalidate_camera_mode_cache

    channel_repo = ChannelRepository()
    cameras = await channel_repo.list_cameras(db, site_uuid=site_uuid)
    for cam in cameras:
        await invalidate_camera_mode_cache(cam.camera_uuid)


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


async def _validate_site_prerecord_camera_uuids(
    db: AsyncSession,
    *,
    org_id: int,
    site_uuid: uuid.UUID,
    camera_uuids: List[uuid.UUID],
) -> List[uuid.UUID]:
    normalized = _dedupe_uuid_list(camera_uuids)
    if not normalized:
        return []

    channel_repo = ChannelRepository()
    site_cameras = await channel_repo.list_cameras(
        db,
        site_uuid=site_uuid,
        org_id=int(org_id),
        camera_uuids=normalized,
    )
    found = {str(cam.camera_uuid) for cam in site_cameras}
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
        source_url=cam_out.source_url,
        webrtc_url=resolve_camera_webrtc_url(
            camera_code=getattr(cam_out, "camera_code", None),
            stored_url=getattr(cam_out, "webrtc_url", None),
        ),
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        notification_trigger_mode=str(getattr(cam_out, "notification_trigger_mode", "inherit") or "inherit"),
        camera_playback_enabled=str(getattr(cam_out, "camera_playback_enabled", "inherit") or "inherit"),
        use_site_schedule=bool(getattr(cam_out, "use_site_schedule", True)),
        roi=cam_out.roi,
        configuration=cam_out.configuration,
        timezone=cam_out.timezone,
        created_at=getattr(cam_out, "created_at", None) or now,
        updated_at=getattr(cam_out, "updated_at", None) or now,
    )


# -----------------------
# Org-scope helpers
# -----------------------
async def _site_scope(db: AsyncSession, ctx: OrgContext) -> Optional[List[uuid.UUID]]:
    """Allow-list of site UUIDs the caller may touch (None = every org site).

    Admins/operators get None (all org sites); plain members get the set of
    sites they hold a site-scoped grant for. Platform admins in super context
    (`ctx.org_id is None`) also get None — no scoping at all.
    """
    if ctx.org_id is None:
        return None
    accessible = await AuthzService.accessible_site_uuids(
        db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
    )
    return None if accessible is None else list(accessible)


async def _viewer_roles_for(
    db: AsyncSession, ctx: OrgContext, site_uuids: List[uuid.UUID]
) -> Dict[uuid.UUID, str]:
    """Resolve `viewer_role` for each site uuid for the calling user.

    Org admins (and platform admins) get "admin" for every site. Plain
    members get the role from their site-scoped access grant, defaulting to
    "read_only" if none is found (defence in depth; the access layer
    should already have excluded such sites).
    """
    if not site_uuids:
        return {}
    if ctx.is_admin or ctx.is_platform_admin:
        return {s: "admin" for s in site_uuids}
    res = await db.execute(
        _sa_select(AccessGrant.site_uuid, Role.name)
        .join(Role, Role.id == AccessGrant.role_id)
        .where(
            AccessGrant.user_id == int(ctx.user.id),
            AccessGrant.site_uuid.in_(site_uuids),
            Role.scope == RoleScope.SITE.value,
        )
    )
    rows = {row[0]: row[1] for row in res.all()}
    return {s: rows.get(s, "read_only") for s in site_uuids}


def _site_out(site: Site, viewer_role: Optional[str]) -> SiteOut:
    return SiteOut(
        site_uuid=site.site_uuid,
        user_id=site.user_id,
        name=site.name,
        site_code=getattr(site, "site_code", None),
        address=getattr(site, "address", None),
        timezone=getattr(site, "timezone", None) or "UTC",
        viewer_role=viewer_role,
    )


def _owner_id(site: Site, ctx: OrgContext) -> int:
    """Per-user runtime pipelines key off a user id; prefer the site creator,
    falling back to the acting user."""
    return int(site.user_id) if getattr(site, "user_id", None) is not None else int(ctx.user.id)


# -----------------------
# Routes
# -----------------------
@router.get("", response_model=List[SiteOut])
async def list_sites(
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    site_repo=SiteRepository()
    sites = await site_repo.get_sites(
        db,
        org_id=ctx.org_id,
        site_uuids=await _site_scope(db, ctx),
        allow_unscoped=ctx.org_id is None,
    )
    roles = await _viewer_roles_for(db, ctx, [s.site_uuid for s in sites])
    return [_site_out(s, roles.get(s.site_uuid)) for s in sites]

@router.get("/{site_uuid}/devices", response_model=List[DeviceOut])
async def list_site_devices(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    # Access check via Site, then fetch devices linked to the site.
    site_repo = SiteRepository()
    await site_repo.get_site(
        db, org_id=ctx.org_id, site_uuids=await _site_scope(db, ctx), site_uuid=site_uuid
    )

    device_repo = DeviceRepository()
    return await device_repo.list_devices(
        db,
        site_uuid=site_uuid,
        org_id=ctx.org_id,
        order_by_recent=True,
    )


@router.get("/{site_uuid}/settings", response_model=SiteSettingsOut)
async def get_site_settings(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    # Fetch site (access check) + settings.
    site_repo = SiteRepository()
    site = await site_repo.get_site(
        db, org_id=ctx.org_id, site_uuids=await _site_scope(db, ctx), site_uuid=site_uuid
    )
    settings_row = await site_repo.get_site_settings(db, site_uuid=site.site_uuid)
    return _serialize_site_settings(site.site_uuid, settings_row, fallback_timezone=site.timezone)


@router.post("/{site_uuid}/cameras", response_model=CameraWithConfigSchema, status_code=status.HTTP_201_CREATED)
async def create_site_camera(
    site_uuid: uuid.UUID,
    payload: SiteCameraCreate,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_CAMERAS)),
    manager: Manager = Depends(get_manager),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, org_id=ctx.org_id, site_uuid=site_uuid)
    owner_id = _owner_id(site, ctx)

    device_repo = DeviceRepository()
    device = await device_repo.get_device(
        db,
        device_uuid=payload.device_uuid,
        site_uuid=site.site_uuid,
        org_id=ctx.org_id,
    )
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
    data["user_id"] = owner_id

    try:
        pipeline = await manager.get_activepipeline(user_id=owner_id)
        ev = ChannelCreateEvent(
            channel_id=None,
            configs=data,
            created_at=datetime.now(timezone.utc),
        )

        result = await manager.update_pipeline(
            pipeline.pipeline_id,
            [ev],
            user_id=owner_id,
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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SITES)),
    manager: Manager = Depends(get_manager),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, org_id=ctx.org_id, site_uuid=site_uuid)
    owner_id = _owner_id(site, ctx)
    row = await site_repo.get_site_settings(db, site_uuid=site.site_uuid)
    config = dict(row.config or {}) if row and isinstance(row.config, dict) else {}
    schedule_payload = _site_schedule_payload_from_row(row, fallback_timezone=site.timezone)

    if payload.multi_camera_prerecord is not None:
        rule = payload.multi_camera_prerecord
        trigger_mode = _normalize_trigger_mode(rule.trigger_mode)
        camera_uuids = await _validate_site_prerecord_camera_uuids(
            db,
            org_id=ctx.org_id,
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
        await site_repo.update_site(
            db,
            site_uuid=site.site_uuid,
            dto=SiteUpdateDTO(timezone=schedule_payload["timezone"]),
        )

    if (
        payload.schedule is None
        and payload.multi_camera_prerecord is None
        and payload.notification is None
    ):
        return _serialize_site_settings(site.site_uuid, row, fallback_timezone=site.timezone)

    row = await site_repo.upsert_site_settings(
        db,
        dto=SiteSettingsUpsertDTO(
            user_id=owner_id,
            site_uuid=site.site_uuid,
            config=config,
            day_of_week=list(schedule_payload.get("day_of_week") or list(SUNDAY_TO_SATURDAY)),
            start_time=dt_time.fromisoformat(str(schedule_payload.get("start_time") or "00:00:00")),
            end_time=dt_time.fromisoformat(str(schedule_payload.get("end_time") or "23:59:59")),
            is_enabled=True,
        ),
    )
    await db.commit()
    if payload.multi_camera_prerecord is not None:
        try:
            svc = getattr(manager, "_notification_service", None) if manager else None
            if svc is not None and hasattr(svc, "invalidate_prerecord_eligible_cache"):
                svc.invalidate_prerecord_eligible_cache()
        except Exception:
            pass
        await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    if payload.notification is not None:
        try:
            svc = getattr(manager, "_notification_service", None) if manager else None
            if svc is not None and hasattr(svc, "invalidate_site_trigger_mode_cache"):
                svc.invalidate_site_trigger_mode_cache(str(site.site_uuid))
            pipeline = manager.get_loaded_pipeline(user_id=owner_id) if manager else None
            if pipeline is not None and hasattr(pipeline, "invalidate_site_trigger_mode_cache"):
                pipeline.invalidate_site_trigger_mode_cache(str(site.site_uuid))
        except Exception:
            pass
    await db.refresh(row)
    if payload.schedule is not None and manager is not None:
        try:
            await _refresh_site_schedule_runtime(
                manager=manager,
                user_id=owner_id,
                site_uuid=site.site_uuid,
            )
        except Exception:
            logger.warning(
                "Failed to refresh site schedule runtime for site=%s; settings were saved successfully.",
                site.site_uuid,
                exc_info=True,
            )
    await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    return _serialize_site_settings(site.site_uuid, row, fallback_timezone=site.timezone)


@router.post("", response_model=SiteOut, status_code=status.HTTP_201_CREATED)
async def create_site(
    payload: SiteCreate,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SITES)),
):
    site_code = payload.site_code
    if _is_blank(site_code):
        site_code = _gen_code("site")

    # Resolve target org: per-tenant callers use ctx.org_id; platform-admin
    # super context (org_id is None) must pass org_id in the payload.
    target_org_id = ctx.org_id if ctx.org_id is not None else payload.org_id
    if target_org_id is None:
        raise HTTPException(
            status_code=422,
            detail="org_id is required when creating a site as a platform admin.",
        )

    site_repo = SiteRepository()
    site = await site_repo.create_site(
        db,
        dto=SiteCreateDTO(
            org_id=int(target_org_id),
            user_id=ctx.user.id,
            created_by=ctx.user.id,
            name=payload.name,
            address=payload.address,
            timezone=payload.timezone or "UTC",
            site_code=site_code,
        ),
    )
    await db.commit()
    await db.refresh(site)
    return _site_out(site, "admin")


@router.get("/{site_uuid}", response_model=SiteOut)
async def get_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(
        db, org_id=ctx.org_id, site_uuids=await _site_scope(db, ctx), site_uuid=site_uuid
    )
    roles = await _viewer_roles_for(db, ctx, [site.site_uuid])
    return _site_out(site, roles.get(site.site_uuid))

@router.patch("/{site_uuid}", response_model=SiteOut)
async def update_site(
    site_uuid: uuid.UUID,
    payload: SiteUpdate,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SITES)),
    manager:Manager = Depends(get_manager),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, org_id=ctx.org_id, site_uuid=site_uuid)
    owner_id = _owner_id(site, ctx)


    data = payload.model_dump(exclude_unset=True)

    # If they included site_code but it’s blank -> regenerate
    if "site_code" in data and _is_blank(data.get("site_code")):
        data["site_code"] = _gen_code("site")

    # Apply patch
    update_fields = {k: v for k, v in data.items() if v is not None}
    if update_fields:
        await site_repo.update_site(
            db, site_uuid=site.site_uuid, dto=SiteUpdateDTO(**update_fields)
        )
        for k, v in update_fields.items():
            setattr(site, k, v)

    if data.get("timezone") is not None:
        row = await site_repo.get_site_settings(
            db,
            site_uuid=site.site_uuid,
        )
        if row is not None:
            config = dict(row.config or {}) if isinstance(getattr(row, "config", None), dict) else {}
            config["timezone"] = str(site.timezone or "UTC")
            await site_repo.upsert_site_settings(
                db,
                dto=SiteSettingsUpsertDTO(
                    user_id=owner_id,
                    site_uuid=site.site_uuid,
                    config=config,
                    day_of_week=[int(getattr(row, "day_of_week", 6))],
                    start_time=getattr(row, "start_time", None),
                    end_time=getattr(row, "end_time", None),
                    is_enabled=bool(getattr(row, "is_enabled", True)),
                ),
            )

    await db.commit()
    if data.get("timezone") is not None:
        await _invalidate_site_camera_mode_cache(db=db, site_uuid=site.site_uuid)
    await db.refresh(site)
    if data.get("timezone") is not None and manager is not None:
        try:
            await _refresh_site_schedule_runtime(
                manager=manager,
                user_id=owner_id,
                site_uuid=site.site_uuid,
            )
        except Exception:
            logger.warning("Best-effort schedule runtime refresh failed site=%s", site_uuid, exc_info=True)
    return _site_out(site, "admin")


@router.delete("/{site_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SITES)),
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
    site = await site_repo.get_site(db, org_id=ctx.org_id, site_uuid=site_uuid)
    owner_id = _owner_id(site, ctx)

    # ========================================
    # PHASE 1: Snapshot camera info BEFORE any changes
    # ========================================
    logger.info(f"[Site Delete] Phase 1: Gathering camera info")
    channel_repo = ChannelRepository()
    camera_rows = await channel_repo.list_cameras_with_device_details(
        db,
        site_uuid=site.site_uuid,
        org_id=ctx.org_id,
    )

    camera_uuids = [row["camera_uuid"] for row in camera_rows]
    cam_to_urls: Dict[uuid.UUID, set[str]] = {}
    cam_to_code: Dict[uuid.UUID, Optional[str]] = {}
    for row in camera_rows:
        cam_uuid_key = row["camera_uuid"]
        cam_to_code[cam_uuid_key] = row.get("camera_code")
        url = str(row.get("device_url") or "").strip()
        bucket = cam_to_urls.setdefault(cam_uuid_key, set())
        if url:
            bucket.add(url)

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
        await channel_repo.disable_cameras(
            db, site_uuid=site.site_uuid, user_id=owner_id
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
                    user_id=owner_id,
                    site_uuid=site.site_uuid,
                    camera_uuids=camera_uuids,
                )
            else:
                notif_svc.invalidate_recipient_cache(user_id=owner_id, site_uuid=site.site_uuid)
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
                    user_id=owner_id,
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
        video_repo = VideoRepository()
        async with AsyncSessionLocal() as vr_session:
            vr_rows = await video_repo.list_storage_keys(
                vr_session, camera_uuids=camera_uuids
            )
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
            await site_repo.soft_delete_site(sd_session, site_uuid=site.site_uuid)
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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SITES)),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, org_id=ctx.org_id, site_uuid=site_uuid)

    device_repo = DeviceRepository()
    device = await device_repo.get_device(
        db, device_uuid=payload.device_uuid, org_id=ctx.org_id
    )
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    if await site_repo.site_device_exists(
        db, site_uuid=site.site_uuid, device_uuid=device.device_uuid
    ):
        return {"linked": True, "already": True}

    await site_repo.add_device_to_site(
        db, site_uuid=site.site_uuid, device_uuid=device.device_uuid
    )
    await db.commit()
    return {"linked": True, "already": False}


@router.delete("/{site_uuid}/devices/{device_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def unlink_device_from_site(
    site_uuid: uuid.UUID,
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SITES)),
):
    site_repo=SiteRepository()
    site = await site_repo.get_site(db, org_id=ctx.org_id, site_uuid=site_uuid)

    channel_repo = ChannelRepository()
    cameras_using_device = await channel_repo.list_cameras(
        db, site_uuid=site.site_uuid, device_uuid=device_uuid
    )
    if cameras_using_device:
        raise HTTPException(
            status_code=409,
            detail="Cannot unlink device while cameras in this site are assigned to it. Move or delete those cameras first.",
        )

    await site_repo.remove_device_from_site(
        db, site_uuid=site.site_uuid, device_uuid=device_uuid
    )
    await db.commit()
    return None


# =====================================================================
# Arm / disarm
# =====================================================================
class ArmStateOut(BaseModel):
    site_uuid: uuid.UUID
    is_armed: bool
    # Whether a temporary user override is currently in force (vs. following the
    # schedule), and when it auto-clears (next schedule boundary; null = no
    # boundary, so the override is permanent until changed).
    override_active: bool = False
    override_until: Optional[datetime] = None


class SetArmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_armed: bool


async def _resolve_site_schedule_for_arm(
    db: AsyncSession, site: Site
) -> tuple[List[Dict[str, Any]], str]:
    """Return the (schedule, timezone) that governs a site's armed state.

    Mirrors how the manager resolves the site schedule so the override boundary
    computed here matches the one the reconcile loop / pipeline evaluate against.
    """
    settings_row = await SiteRepository().get_site_settings(db, site_uuid=site.site_uuid)
    payload = _site_schedule_payload_from_row(settings_row, fallback_timezone=site.timezone)
    return (payload.get("schedule") or [], str(payload.get("timezone") or "UTC"))


def _effective_arm_state_out(
    site: Site, schedule: List[Dict[str, Any]], tz: str, *, now: datetime
) -> ArmStateOut:
    override = VideoChannelConfig.effective_arm_override(
        site.arm_override, site.arm_override_until, now_utc=now
    )
    if override is not None:
        armed = bool(override)
        override_active = True
    else:
        armed = VideoChannelConfig.schedule_is_active(schedule, tz, now_utc=now)
        override_active = False
    return ArmStateOut(
        site_uuid=site.site_uuid,
        is_armed=armed,
        override_active=override_active,
        override_until=site.arm_override_until if override_active else None,
    )


@router.get(
    "/{site_uuid}/arm",
    response_model=ArmStateOut,
    dependencies=[Depends(RequirePermission(Permission.SITE_READ))],
)
async def get_site_arm_state(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
):
    """Read the current armed state of a site.

    Read-only site members can hit this endpoint to see the flag but
    cannot mutate it; see `PATCH` below for that.
    """
    site = (
        await db.execute(
            _sa_select(Site).where(Site.site_uuid == site_uuid).limit(1)
        )
    ).scalar_one_or_none()
    if site is None or bool(site.is_deleted):
        raise HTTPException(status_code=404, detail="Site not found")
    schedule, tz = await _resolve_site_schedule_for_arm(db, site)
    return _effective_arm_state_out(site, schedule, tz, now=datetime.now(timezone.utc))


@router.patch(
    "/{site_uuid}/arm",
    response_model=ArmStateOut,
)
async def set_site_arm_state(
    site_uuid: uuid.UUID,
    payload: SetArmRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.SITE_ARM_DISARM)),
    manager: Manager = Depends(get_manager),
):
    """Arm or disarm a site, schedule-aware.

    Permission: requires `SiteRole.ADMIN` or `SiteRole.ARM_DISARM`
    (Org Admins satisfy this implicitly through `AuthzService`).
    Read-only members will get a 403 here.

    The site's schedule is the default source of truth for whether it is armed.
    Arming/disarming records a *temporary override* that lasts only until the
    next schedule boundary, after which the schedule resumes control:

      * Disarming during an active window keeps the site disarmed until that
        window ends (the next time the schedule flips off).
      * Arming while off-schedule keeps the site armed until the next window
        begins (the next time the schedule flips on).

    If the requested state already matches the schedule, any existing override is
    simply cleared. The effective armed state gates BOTH edge detection (the
    reconcile loop tears cameras down on the Jetson when disarmed) and
    notifications (the pipeline suppresses alerts when disarmed).
    """
    site = (
        await db.execute(
            _sa_select(Site).where(Site.site_uuid == site_uuid).limit(1)
        )
    ).scalar_one_or_none()
    if site is None or bool(site.is_deleted):
        raise HTTPException(status_code=404, detail="Site not found")

    want_armed = bool(payload.is_armed)
    schedule, tz = await _resolve_site_schedule_for_arm(db, site)

    now = datetime.now(timezone.utc)
    scheduled_now = VideoChannelConfig.schedule_is_active(schedule, tz, now_utc=now)

    if want_armed == scheduled_now:
        # Desired state already equals what the schedule dictates: drop any
        # override so the schedule drives the site again.
        site.arm_override = None
        site.arm_override_until = None
    else:
        # Override the schedule until its next boundary. A None boundary (no
        # schedule configured) means there is nothing to clear it, so it stays
        # until changed.
        site.arm_override = want_armed
        site.arm_override_until = VideoChannelConfig.next_schedule_transition(
            schedule, tz, now_utc=now
        )

    await db.commit()
    await db.refresh(site)

    owner_id = _owner_id(site, ctx)

    # Collect every device backing the site's cameras: the override applies to
    # all cameras regardless of per-camera schedules, so every device must be
    # reconciled.
    cams = await ChannelRepository().list_cameras(db, site_uuid=site.site_uuid)
    device_uuids = sorted(
        {
            cam.device_uuid
            for cam in cams
            if getattr(cam, "device_uuid", None) is not None
        },
        key=str,
    )

    # Best-effort: nudge the runtime so notifications and edge detection react
    # now instead of waiting for the next periodic reconcile / cache lapse.
    try:
        await _refresh_site_arm_runtime(
            manager=manager,
            user_id=owner_id,
            site_uuid=site.site_uuid,
            device_uuids=device_uuids,
        )
    except Exception:
        logger.warning(
            "Best-effort arm/disarm runtime refresh failed for site=%s",
            site_uuid, exc_info=True,
        )

    logger.info(
        "Site=%s want_armed=%s override=%s until=%s by user=%s",
        site_uuid,
        want_armed,
        site.arm_override,
        site.arm_override_until,
        int(ctx.user.id),
    )
    return _effective_arm_state_out(site, schedule, tz, now=now)


async def _refresh_site_arm_runtime(
    *,
    manager: Manager,
    user_id: int,
    site_uuid: uuid.UUID,
    device_uuids: List[uuid.UUID],
) -> None:
    """Push a fresh arm/disarm decision to the live runtime.

    1. Invalidate the pipeline's per-site arm cache so the notification gate
       flips on the next frame instead of after the TTL.
    2. Fire a best-effort edge reconcile for the site's devices so detection is
       torn down (disarm) or brought back up (arm) on the Jetson immediately.

    The DB (`arm_override` + schedule) remains authoritative; the periodic edge
    reconcile loop will converge anyway, so any failure here is non-fatal.
    """
    try:
        pipeline = manager.get_loaded_pipeline(user_id=int(user_id))
        if pipeline is not None and hasattr(pipeline, "invalidate_site_arm_state"):
            pipeline.invalidate_site_arm_state(str(site_uuid))
    except Exception:
        logger.warning(
            "Failed to invalidate arm-state cache site=%s", site_uuid, exc_info=True
        )

    if device_uuids:
        asyncio.create_task(
            manager.reconcile_devices_best_effort(
                user_id=int(user_id),
                device_uuids=list(device_uuids),
            )
        )
