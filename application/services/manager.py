# agents/application/services/agent_manager.py

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from sqlalchemy import delete, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from fastapi import HTTPException

from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository
from core.database_orm import Site, SiteDevice, SiteSettings, Device, Camera, CameraDevice
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent, VideoChannelEvent
from domain.model_pipeline import ModelPipeline
from application.channels.channel_config import VideoChannelConfig
from application.channels.channel import VideoChannel
from application.services.edgeinference import EdgeCameraInventoryError, EdgeInferenceClient
from application.services.webrtcgateway import WebRTCGatewayClient

logger = logging.getLogger(__name__)

# -------------------------
# API DTOs
# -------------------------

    
class CameraOut(BaseModel):
    camera_uuid: uuid.UUID
    camera_code: Optional[str] = None
    name: Optional[str] = None
    location: Optional[str] = None
    site_uuid: uuid.UUID

    rtsp_url: str
    webrtc_url: Optional[str] = None

    enabled: bool
    detection_enabled: bool
    notification_enabled: bool
    roi: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)
    timezone: Optional[str] = None
    notification_trigger_mode: str = "inherit"
    camera_playback_enabled: str = "inherit"
    use_site_schedule: Optional[bool] = None

    device_uuid: uuid.UUID
    device_url: str

    sample_fps: float = Field(default=5.0, ge=0.1)
    decode_backend: str = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = None
    emit_format: str = Field(default="raw")
    jpeg_quality: int = Field(default=80, ge=1, le=100)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PipelineUpdateResult(BaseModel):
    pipeline_id: uuid.UUID
    active_in_memory: bool = False
    cameras: List[CameraOut] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class EdgeDeviceUnavailableError(RuntimeError):
    def __init__(self, device_url: str, cause: Exception):
        self.device_url = device_url
        self.cause = cause
        cause_name = type(cause).__name__
        cause_msg = str(cause).strip()
        detail = f"{cause_name}: {cause_msg}" if cause_msg else cause_name
        super().__init__(f"Edge device unreachable ({device_url}): {detail}")


def _edge_health_ready(health: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(health, dict):
        return False
    if health.get("ok") is False:
        return False
    if health.get("pipeline_ready") is False:
        return False
    return True


# -------------------------
# External clients
# -------------------------


# -------------------------
# Manager
# -------------------------

ChannelConfigLike = Union[dict, Any]
ModelConfigLike = Union[dict, Any]

HARD_PATCH_KEYS = {
    "rtsp_url",
    "webrtc_url",
    "device_id",
    "device_uuid",
    "device_url",
    "decode_backend",
    "resize",
    "emit_format",
    "jpeg_quality",
    "reconnect_base_ms",
    "reconnect_max_ms",
}

JETSON_PATCH_KEYS = {
    "rtsp_url",
    "enabled",
    "detection_enabled",
    "notification_enabled",
    "sample_fps",
    "decode_backend",
    "resize",
    "emit_format",
    "jpeg_quality",
    "reconnect_base_ms",
    "reconnect_max_ms",
    "camera_uuid",
    "channel_id",
}
def _only_jetson_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (patch or {}).items() if k in JETSON_PATCH_KEYS and v is not None}

RUNTIME_CONFIG_FORBIDDEN_KEYS = {
    "camera_uuid",
    "channel_id",
    "site_uuid",
    "device_uuid",
    "device_url",
    "rtsp_url",
    "webrtc_url",
    "enabled",
    "detection_enabled",
    "notification_enabled",
    "is_enabled",
    "is_detection_enabled",
    "is_notification_enabled",
    "user_id",
    "roi",
}

RUNTIME_CONFIG_ALLOWED_KEYS = set(VideoChannelConfig.model_fields.keys())

def _coerce_tri_trigger_mode(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ("roi_enter", "any_detection", "inherit"):
        return s
    return "inherit"


def _coerce_tri_playback_mode(v: Any) -> str:
    if v is True:
        return "always"
    if v is False:
        return "never"
    s = str(v or "").strip().lower()
    if s in ("always", "never", "inherit"):
        return s
    return "inherit"


def _runtime_config_overrides(cfg: Dict[str, Any], *, extra_forbidden: Optional[set] = None) -> Dict[str, Any]:
    forbidden = set(RUNTIME_CONFIG_FORBIDDEN_KEYS)
    if extra_forbidden:
        forbidden.update(extra_forbidden)

    out: Dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        if k not in RUNTIME_CONFIG_ALLOWED_KEYS:
            continue
        if k in forbidden:
            continue
        if v is None:
            continue
        if k == "notification_trigger_mode":
            out[k] = _coerce_tri_trigger_mode(v)
        elif k == "camera_playback_enabled":
            out[k] = _coerce_tri_playback_mode(v)
        else:
            out[k] = v
    return out


def _edge_runtime_enabled(*, detection_enabled: bool) -> bool:
    """
    Jetson's `enabled` flag currently gates whether the camera remains active in
    the inference runtime. Playback provisioning is handled separately by
    Azure-side `Camera.is_enabled`.
    """
    return bool(detection_enabled)

class Manager:
    """
    Azure Manager:
      - DB is source of truth
      - WebRTC gateway provides playback URL
      - Jetson device provides detections
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._locks_by_user: Dict[int, asyncio.Lock] = {}  # Per-user locks instead of global

        self._repo = PipelineRepository()
        self.channel_repo = ChannelRepository()

        self._webrtc = WebRTCGatewayClient()
        self._edge = EdgeInferenceClient()

        self._pipelines_by_user: Dict[int, ModelPipeline] = {}
        self._pipeline_id_by_user: Dict[int, uuid.UUID] = {}
        self._notification_service: Optional[Any] = None

        self._default_user_id = int(os.getenv("DEFAULT_USER_ID", "1"))
        self._default_request_timeout_s = float(os.getenv("REQUEST_TIMEOUT_S", "3.0"))
        self._external_timeout_s = float(os.getenv("EXTERNAL_SERVICE_TIMEOUT_S", "10.0"))
        self._edge_retry_max_attempts = max(1, int(os.getenv("EDGE_RETRY_MAX_ATTEMPTS", "3")))
        self._edge_retry_base_ms = max(100, int(os.getenv("EDGE_RETRY_BASE_MS", "500")))

    def _get_user_lock(self, uid: int) -> asyncio.Lock:
        """Get or create a per-user lock to prevent concurrent pipeline operations for same user."""
        if uid not in self._locks_by_user:
            self._locks_by_user[uid] = asyncio.Lock()
        return self._locks_by_user[uid]

    async def _call_with_timeout(self, coro, timeout_s: Optional[float] = None):
        """Wrap an async call with timeout handling."""
        timeout = timeout_s or self._external_timeout_s
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"External service call timed out after {timeout}s") from e

    def _wire_pipeline(self, mp: ModelPipeline) -> None:
        mp.set_session_factory(self._session_factory)
        if self._notification_service is None:
            return
        mp.set_notification_service(self._notification_service)
        roi_provider = getattr(self._notification_service, "_get_rois", None)
        if callable(roi_provider):
            mp.set_roi_provider(roi_provider)

    def set_notification_service(self, notification_service: Optional[Any]) -> None:
        self._notification_service = notification_service
        for mp in list(self._pipelines_by_user.values()):
            self._wire_pipeline(mp)

    def _event_includes_roi_patch(self, ev: VideoChannelEvent) -> bool:
        configs = getattr(ev, "configs", None)
        if isinstance(configs, dict):
            return "roi" in configs

        fields_set = getattr(configs, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(configs, "__fields_set__", None)
        if fields_set is not None:
            return "roi" in fields_set

        return False

    def _invalidate_camera_roi_state(self, camera_uuid: uuid.UUID) -> None:
        cam = str(camera_uuid)

        svc = self._notification_service
        invalidate = getattr(svc, "invalidate_camera_roi_state", None) if svc is not None else None
        if callable(invalidate):
            try:
                invalidate(cam)
            except Exception:
                logger.exception("Failed invalidating notification ROI state camera=%s", cam)

        for mp in list(self._pipelines_by_user.values()):
            if mp is None:
                continue
            invalidate_mp = getattr(mp, "invalidate_camera_roi_state", None)
            if not callable(invalidate_mp):
                continue
            try:
                invalidate_mp(cam)
            except Exception:
                logger.exception("Failed invalidating pipeline ROI state camera=%s", cam)

    async def shutdown(self) -> None:
        # Collect all user locks and pipelines
        all_locks = list(self._locks_by_user.values())
        self._locks_by_user.clear()
        pipelines = list(self._pipelines_by_user.values())
        self._pipelines_by_user.clear()
        self._pipeline_id_by_user.clear()

        for mp in pipelines:
            if mp is None:
                continue
            try:
                await mp.shutdown()
            except Exception:
                logger.exception(
                    "Pipeline shutdown failed pipeline_id=%s",
                    getattr(mp, "pipeline_id", None),
                )
        await self._webrtc.close()
        await self._edge.close()

    async def start_background_pipelines(self) -> Dict[str, Any]:
        """
        Start polling pipelines for every user that has at least one
        detection-enabled camera so notifications continue even when nobody is
        logged in.
        """
        async with self._session_factory() as db:
            rows = await db.execute(
                select(Camera.user_id)
                .where(Camera.is_detection_enabled.is_(True))
                .distinct()
                .order_by(Camera.user_id.asc())
            )
            user_ids = sorted(
                {
                    int(row[0])
                    for row in rows.all()
                    if row and row[0] is not None
                }
            )

        summary: Dict[str, Any] = {
            "user_ids": user_ids,
            "started": [],
            "errors": [],
        }

        for uid in user_ids:
            try:
                mp = await self.get_activepipeline(user_id=uid)
                summary["started"].append(
                    {
                        "user_id": uid,
                        "pipeline_id": str(getattr(mp, "pipeline_id", "")),
                        "channel_count": len(mp.list_channel_ids()),
                    }
                )
            except Exception as exc:
                logger.exception("Failed to start background pipeline user=%s", uid)
                summary["errors"].append(
                    {
                        "user_id": uid,
                        "error": str(exc),
                    }
                )

        summary["started_count"] = len(summary["started"])
        summary["error_count"] = len(summary["errors"])
        return summary

    def _patch_to_dict(self, obj: Any) -> Dict[str, Any]:
        nullable_keep = {"roi"}
        if obj is None:
            out: Dict[str, Any] = {}
        elif hasattr(obj, "model_dump"):
            out = obj.model_dump(exclude_unset=True)
            fields_set = getattr(obj, "model_fields_set", None)
            if fields_set is None:
                fields_set = getattr(obj, "__fields_set__", None)
            if fields_set is not None and "roi" in fields_set and "roi" not in out:
                out["roi"] = None
            out = {k: v for k, v in out.items() if v is not None or k in nullable_keep}
        elif isinstance(obj, dict):
            out = {k: v for k, v in obj.items() if v is not None or k in nullable_keep}
        else:
            out = {}

        # normalize names
        if "enabled" not in out and "is_enabled" in out:
            out["enabled"] = out["is_enabled"]
        if "detection_enabled" not in out and "is_detection_enabled" in out:
            out["detection_enabled"] = out["is_detection_enabled"]
        if "notification_enabled" not in out and "is_notification_enabled" in out:
            out["notification_enabled"] = out["is_notification_enabled"]

        return out

    def _as_uuid(self, v: Any, name: str) -> uuid.UUID:
        if isinstance(v, uuid.UUID):
            return v
        try:
            return uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid {name}: {v}") from e

    async def _get_device(
        self,
        db: AsyncSession,
        device_uuid: uuid.UUID,
        *,
        user_id: Optional[int] = None,
    ) -> Device:
        q = select(Device).where(Device.device_uuid == device_uuid)
        if user_id is not None:
            q = q.where(Device.user_id == int(user_id))

        dev = (await db.execute(q)).scalar_one_or_none()
        if dev is None:
            raise ValueError(f"Device not found: {device_uuid}")
        if not getattr(dev, "device_url", None):
            raise ValueError(f"Device missing device_url: {device_uuid}")
        return dev

    async def _get_site_devices(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: int,
    ) -> List[Device]:
        q = (
            select(Device)
            .join(SiteDevice, SiteDevice.device_uuid == Device.device_uuid)
            .where(
                SiteDevice.site_uuid == site_uuid,
                Device.user_id == int(user_id),
            )
            .order_by(
                Device.is_enabled.desc(),
                SiteDevice.created_at.desc(),
                Device.created_at.desc(),
            )
        )
        return (await db.execute(q)).scalars().all()

    async def _pick_site_device_uuid(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: int,
    ) -> Optional[uuid.UUID]:
        """Auto-select device if exactly one is available. Returns None if 0 or 2+ devices."""
        devices = await self._get_site_devices(db, site_uuid=site_uuid, user_id=user_id)
        if len(devices) == 1:
            return devices[0].device_uuid
        return None

    async def _resolve_site_device(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        user_id: int,
        requested_device_uuid: Optional[uuid.UUID] = None,
        required: bool = True,
    ) -> Optional[Device]:
        """
        Resolve the device a camera should use for a site.

        Rules:
        - if the client explicitly asks for a device, validate that it exists
          and, when the site already has linked devices, ensure it belongs to the site
        - if no device is requested, auto-pick only when the site resolves to exactly
          one usable device; ambiguity stays explicit instead of silently choosing
          the wrong Jetson
        """
        site_devices = await self._get_site_devices(db, site_uuid=site_uuid, user_id=user_id)

        if requested_device_uuid is not None:
            dev = await self._get_device(db, requested_device_uuid, user_id=user_id)
            if site_devices:
                site_device_ids = {getattr(item, "device_uuid", None) for item in site_devices}
                if dev.device_uuid not in site_device_ids:
                    raise ValueError(
                        f"Device {requested_device_uuid} is not linked to site {site_uuid}."
                    )
            return dev

        if not site_devices:
            if required:
                raise ValueError(
                    f"Site {site_uuid} has no linked device. Link a device to the site first."
                )
            return None

        enabled_devices = [dev for dev in site_devices if bool(getattr(dev, "is_enabled", True))]
        candidates = enabled_devices or site_devices
        if len(candidates) == 1:
            return candidates[0]

        if required:
            raise ValueError(
                f"Site {site_uuid} has multiple linked devices. Specify device_uuid explicitly."
            )
        return None

    async def _ensure_site_owned_by_user(self, db: AsyncSession, *, site_uuid: uuid.UUID, user_id: int) -> None:
        site = (
            await db.execute(
                select(Site.site_uuid).where(
                    Site.site_uuid == site_uuid,
                    Site.user_id == int(user_id),
                )
            )
        ).scalar_one_or_none()
        if site is None:
            raise ValueError(f"Site not found for user: {site_uuid}")

    async def _get_single_camera_device(self, db: AsyncSession, camera_uuid: uuid.UUID, *, required: bool = True) -> Optional[Device]:
        """
        Returns the primary Device assigned to this camera (most recent).
        Now supports flexible device management.
        """
        devices = await self._get_camera_devices(db, camera_uuid)

        if len(devices) > 0:
            dev = devices[0]
            if not getattr(dev, "device_url", None):
                raise ValueError(f"Assigned device has no device_url for camera {camera_uuid}")
            return dev

        if required:
            raise ValueError(f"Camera {camera_uuid} has no device assigned. Assign a device first.")
        return None

    async def _get_camera_devices(self, db: AsyncSession, camera_uuid: uuid.UUID) -> List[Device]:
        q = (
            select(Device)
            .join(CameraDevice, CameraDevice.device_uuid == Device.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
            .order_by(CameraDevice.created_at.desc(), CameraDevice.id.desc())
        )
        return (await db.execute(q)).scalars().all()

    def _normalize_device_url(self, device_url: Optional[str]) -> str:
        return str(device_url or "").strip().rstrip("/")
    
    async def _list_devices_for_physical_device(self, db: AsyncSession, *, device: Device) -> List[Device]:
        target_url = self._normalize_device_url(getattr(device, "device_url", None))
        root_key = str(getattr(device, "device_uuid", ""))

        if not target_url:
            return [device]

        rows = (
            await db.execute(
                select(Device).where(Device.device_url == target_url)
            )
        ).scalars().all()

        out: List[Device] = []
        seen: Set[str] = set()

        for row in rows:
            row_uuid = getattr(row, "device_uuid", None)
            row_key = str(row_uuid) if row_uuid is not None else ""
            if not row_key:
                continue
            if row_key != root_key and not bool(getattr(row, "is_enabled", True)):
                continue
            if row_key in seen:
                continue
            seen.add(row_key)
            out.append(row)

        if root_key and root_key not in seen:
            out.insert(0, device)

        return out or [device]
    
    async def _list_cameras_for_device_uuids(
        self,
        db: AsyncSession,
        *,
        device_uuids: List[uuid.UUID],
    ) -> List[Camera]:
        clean_device_uuids = [du for du in device_uuids if du is not None]
        if not clean_device_uuids:
            return []

        q = (
            select(Camera)
            .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
            .where(CameraDevice.device_uuid.in_(clean_device_uuids))
            .options(selectinload(Camera.channel_configuration))
        )
        return (await db.execute(q)).scalars().all()

    async def _list_enabled_reconcile_devices(
        self,
        db: AsyncSession,
        *,
        user_id: Optional[int] = None,
    ) -> List[Device]:
        stmt = select(Device).where(Device.is_enabled.is_(True))
        if user_id is not None:
            stmt = stmt.where(Device.user_id == int(user_id))
        return (await db.execute(stmt)).scalars().all()

    async def _set_single_camera_device(self, db: AsyncSession, camera_uuid: uuid.UUID, device_uuid: uuid.UUID) -> None:
        """
        Enforces exactly one device link row in camera_devices.
        Safe even if old rows exist.
        """
        await db.execute(delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
        db.add(CameraDevice(camera_uuid=camera_uuid, device_uuid=device_uuid))
        await db.flush()

    def _edge_payload_from_config(self, *, camera_uuid: str, rtsp_url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(config or {})
        payload.update({"camera_uuid": camera_uuid, "rtsp_url": rtsp_url})
        return payload

    async def _load_site_schedule_state(
        self,
        db: AsyncSession,
        *,
        site_uuid: uuid.UUID,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cache_key = str(site_uuid)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        site_timezone = (
            await db.execute(select(Site.timezone).where(Site.site_uuid == site_uuid))
        ).scalar_one_or_none()
        settings_row = (
            await db.execute(select(SiteSettings).where(SiteSettings.site_uuid == site_uuid))
        ).scalar_one_or_none()

        config = (
            dict(settings_row.config or {})
            if settings_row is not None and isinstance(getattr(settings_row, "config", None), dict)
            else {}
        )
        state = {
            "schedule": VideoChannelConfig.normalize_schedule(config.get("schedule")),
            "timezone": str(config.get("timezone") or site_timezone or "UTC"),
        }
        if cache is not None:
            cache[cache_key] = state
        return state

    def _get_loaded_channel_configuration(self, cam: Camera) -> Optional[Any]:
        try:
            state = inspect(cam)
        except Exception:
            return None

        if "channel_configuration" in getattr(state, "unloaded", set()):
            return None

        try:
            return getattr(cam, "channel_configuration", None)
        except Exception:
            return None

    async def _resolve_runtime_schedule(
        self,
        db: AsyncSession,
        *,
        cam: Camera,
        cfg_json: Optional[Dict[str, Any]] = None,
        cfg_timezone: Optional[str] = None,
        site_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        cfg = dict(cfg_json or {})
        use_site_schedule = bool(cfg.get("use_site_schedule", getattr(cam, "use_site_schedule", True)))
        loaded_channel_cfg = self._get_loaded_channel_configuration(cam)

        schedule = VideoChannelConfig.normalize_schedule(cfg.get("schedule"))
        timezone_name = str(
            cfg.get("timezone")
            or cfg_timezone
            or getattr(loaded_channel_cfg, "timezone", None)
            or "UTC"
        )

        if use_site_schedule:
            site_state = await self._load_site_schedule_state(
                db,
                site_uuid=cam.site_uuid,
                cache=site_cache,
            )
            if site_state.get("schedule"):
                schedule = site_state["schedule"]
            timezone_name = str(site_state.get("timezone") or timezone_name or "UTC")

        if not schedule:
            schedule = VideoChannelConfig.default_schedule()

        return {
            "schedule": schedule,
            "timezone": timezone_name or "UTC",
            "use_site_schedule": use_site_schedule,
            "active": VideoChannelConfig.schedule_is_active(
                schedule,
                timezone_name or "UTC",
                now_utc=datetime.now(timezone.utc),
            ),
        }

    async def sync_site_schedule_runtime(
        self,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Rebuild in-memory runtime configs for site cameras that inherit the site schedule.

        Returns the affected device UUIDs so callers can trigger best-effort edge reconcile
        without coupling route code to camera/device lookup details.
        """
        uid = int(user_id)
        active = self._pipelines_by_user.get(uid)
        refreshed = 0
        skipped = 0
        device_uuids: Set[uuid.UUID] = set()

        async with self._session_factory() as db:
            stmt = (
                select(Camera)
                .where(
                    Camera.user_id == uid,
                    Camera.site_uuid == site_uuid,
                )
                .options(
                    selectinload(Camera.channel_configuration),
                    selectinload(Camera.devices),
                )
            )
            cams = (await db.execute(stmt)).scalars().all()
            site_schedule_cache: Dict[str, Dict[str, Any]] = {}

            for cam in cams:
                cfg_json = {}
                if getattr(cam, "channel_configuration", None) and getattr(cam.channel_configuration, "configuration", None):
                    cfg_json = cam.channel_configuration.configuration or {}

                use_site_schedule = bool(
                    cfg_json.get("use_site_schedule", getattr(cam, "use_site_schedule", True))
                )
                if not use_site_schedule:
                    skipped += 1
                    continue

                devices = list(getattr(cam, "devices", None) or [])
                if len(devices) == 0:
                    logger.warning(
                        "Skipping site schedule refresh for camera %s because device count=%s",
                        cam.camera_uuid,
                        len(devices),
                    )
                    skipped += 1
                    continue
                if len(devices) > 1:
                    logger.warning(
                        "Camera %s has %s linked devices during site schedule refresh; using most recent device %s",
                        cam.camera_uuid,
                        len(devices),
                        getattr(devices[0], "device_uuid", None),
                    )

                device = devices[0]
                if getattr(device, "device_uuid", None) is not None:
                    device_uuids.add(device.device_uuid)

                if active is None:
                    continue

                schedule_state = await self._resolve_runtime_schedule(
                    db,
                    cam=cam,
                    cfg_json=cfg_json,
                    cfg_timezone=getattr(getattr(cam, "channel_configuration", None), "timezone", None),
                    site_cache=site_schedule_cache,
                )
                runtime_overrides = _runtime_config_overrides(
                    cfg_json,
                    extra_forbidden={
                        "sample_fps",
                        "decode_backend",
                        "request_timeout_s",
                        "schedule",
                        "timezone",
                        "use_site_schedule",
                    },
                )
                vcc = VideoChannelConfig(
                    camera_uuid=cam.camera_uuid,
                    rtsp_url=cam.rtsp_url,
                    webrtc_url=cam.webrtc_url or "",
                    site_uuid=cam.site_uuid,
                    device_uuid=device.device_uuid,
                    device_url=device.device_url,
                    enabled=bool(getattr(cam, "is_enabled", True)),
                    detection_enabled=bool(getattr(cam, "is_detection_enabled", True)),
                    notification_enabled=bool(getattr(cam, "is_notification_enabled", True)),
                    sample_fps=float(cfg_json.get("sample_fps", 5.0)),
                    decode_backend=str(cfg_json.get("decode_backend", "gstreamer")),
                    request_timeout_s=float(
                        cfg_json.get("request_timeout_s", self._default_request_timeout_s)
                    ),
                    timezone=schedule_state["timezone"],
                    schedule=schedule_state["schedule"],
                    use_site_schedule=schedule_state["use_site_schedule"],
                    **runtime_overrides,
                )
                await active.edit_channel(VideoChannel(config=vcc))
                refreshed += 1

        return {
            "refreshed": refreshed,
            "skipped": skipped,
            "device_uuids": sorted(device_uuids, key=str),
        }

    async def reconcile_devices_best_effort(
        self,
        *,
        user_id: int,
        device_uuids: List[Union[str, uuid.UUID]],
    ) -> None:
        """
        Best-effort targeted reconcile for devices affected by a site schedule change.

        We explicitly allow removal here so cameras that become inactive due to schedule
        changes are removed from edge/WebRTC immediately instead of waiting for the next
        full reconcile cycle.
        """
        uid = int(user_id)
        targets: List[uuid.UUID] = []
        seen_device_keys: Set[str] = set()
        seen_urls: Set[str] = set()

        async with self._session_factory() as db:
            for raw_device_uuid in device_uuids or []:
                try:
                    device_uuid = self._as_uuid(raw_device_uuid, "device_uuid")
                except Exception:
                    logger.warning("Skipping invalid device UUID during best-effort reconcile: %r", raw_device_uuid)
                    continue

                device_key = str(device_uuid)
                if device_key in seen_device_keys:
                    continue
                seen_device_keys.add(device_key)

                try:
                    dev = await self._get_device(db, device_uuid, user_id=uid)
                except Exception:
                    logger.warning(
                        "Skipping missing device during best-effort reconcile device=%s",
                        device_key,
                        exc_info=True,
                    )
                    continue

                target_key = self._normalize_device_url(getattr(dev, "device_url", None)) or device_key
                if target_key in seen_urls:
                    continue
                seen_urls.add(target_key)
                targets.append(device_uuid)

        for device_uuid in targets:
            key = str(device_uuid)
            try:
                await self.reconcile_device_edge_simple(
                    device_uuid=device_uuid,
                    user_id=uid,
                    dry_run=False,
                    delete_unknown=True,
                )
            except Exception:
                logger.warning(
                    "Best-effort reconcile failed after site schedule update device=%s",
                    key,
                    exc_info=True,
                )


    async def _create_pipeline_unlocked(self, uid: int) -> ModelPipeline:
        async with self._session_factory() as db:
            pipeline_row = await self._repo.upsert_pipeline(
                db,
                user_id=uid,
                pipeline_id=None,
                name="default",
                is_active=True,
            )
            pid = pipeline_row.id

            full_pl = await self._repo.get_full_pipeline(db, pid)

            mp = ModelPipeline(pipeline_id=pid)
            self._wire_pipeline(mp)

            if full_pl and getattr(full_pl, "cameras", None):
                site_schedule_cache: Dict[str, Dict[str, Any]] = {}
                for cam in full_pl.cameras:
                    enabled = bool(getattr(cam, "is_enabled", True))
                    det_enabled = bool(getattr(cam, "is_detection_enabled", True))
                    devices = list(getattr(cam, "devices", None) or [])
                    if len(devices) == 0:
                        repair_device = await self._resolve_site_device(
                            db,
                            site_uuid=cam.site_uuid,
                            user_id=uid,
                            requested_device_uuid=None,
                            required=False,
                        )
                        if repair_device is None:
                            logger.warning(
                                "Skipping camera %s (enabled=%s detection=%s) because device count=%s and no unique site device could be inferred",
                                cam.camera_uuid, enabled, det_enabled, len(devices)
                            )
                            continue
                        await self._set_single_camera_device(
                            db,
                            cam.camera_uuid,
                            repair_device.device_uuid,
                        )
                        devices = [repair_device]
                        logger.info(
                            "Auto-linked missing camera_devices row camera=%s site=%s device=%s during pipeline load",
                            cam.camera_uuid,
                            cam.site_uuid,
                            repair_device.device_uuid,
                        )
                    if len(devices) > 1:
                        logger.warning(
                            "Camera %s has %s linked devices; using first loaded device %s for runtime compatibility",
                            cam.camera_uuid,
                            len(devices),
                            getattr(devices[0], "device_uuid", None),
                        )

                    device = devices[0]
                    d_url = getattr(device, "device_url", None)
                    d_uuid = getattr(device, "device_uuid", None)
                    if not d_url or not d_uuid:
                        raise ValueError(f"Camera {cam.camera_uuid} has invalid device assignment.")

                    cfg_json = {}
                    if getattr(cam, "channel_configuration", None) and getattr(cam.channel_configuration, "configuration", None):
                        cfg_json = cam.channel_configuration.configuration or {}
                    schedule_state = await self._resolve_runtime_schedule(
                        db,
                        cam=cam,
                        cfg_json=cfg_json,
                        cfg_timezone=getattr(getattr(cam, "channel_configuration", None), "timezone", None),
                        site_cache=site_schedule_cache,
                    )

                    runtime_overrides = _runtime_config_overrides(
                        cfg_json,
                        extra_forbidden={
                            "sample_fps",
                            "decode_backend",
                            "request_timeout_s",
                            "schedule",
                            "timezone",
                            "use_site_schedule",
                        },
                    )

                    vcc = VideoChannelConfig(
                        camera_uuid=cam.camera_uuid,
                        rtsp_url=cam.rtsp_url,
                        webrtc_url=cam.webrtc_url or "",
                        site_uuid=cam.site_uuid,
                        device_uuid=d_uuid,
                        device_url=d_url,
                        enabled=enabled,
                        detection_enabled=det_enabled,
                        notification_enabled=bool(getattr(cam, "is_notification_enabled", True)),
                        sample_fps=float(cfg_json.get("sample_fps", 5.0)),
                        decode_backend=str(cfg_json.get("decode_backend", "gstreamer")),
                        request_timeout_s=float(
                            cfg_json.get("request_timeout_s", self._default_request_timeout_s)
                        ),
                        timezone=schedule_state["timezone"],
                        schedule=schedule_state["schedule"],
                        use_site_schedule=schedule_state["use_site_schedule"],
                        **runtime_overrides,
                    )
                    await mp.add_channel(VideoChannel(config=vcc))

            await db.commit()

        self._pipelines_by_user[uid] = mp
        self._pipeline_id_by_user[uid] = pid
        return mp

    async def create_pipeline(self, user_id: int | None = None) -> ModelPipeline:
        """
        Creates (loads) the user's default pipeline and builds a config-only ModelPipeline.
        """
        uid = int(user_id or self._default_user_id)

        user_lock = self._get_user_lock(uid)
        async with user_lock:
            return await self._create_pipeline_unlocked(uid)

    async def get_activepipeline(self, user_id: int | None = None) -> ModelPipeline:
        uid = int(user_id or self._default_user_id)
        mp = self._pipelines_by_user.get(uid)
        if mp is None:
            user_lock = self._get_user_lock(uid)
            async with user_lock:
                mp = self._pipelines_by_user.get(uid)
                if mp is None:
                    mp = await self._create_pipeline_unlocked(uid)
        await mp.start()
        return mp

    def get_loaded_pipeline(self, user_id: int | None = None) -> Optional[ModelPipeline]:
        """
        Return the already-loaded in-memory pipeline for a user without creating
        or starting a new one as a side effect.
        """
        uid = int(user_id or self._default_user_id)
        return self._pipelines_by_user.get(uid)
    
    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID, None],
        channel_events: List[VideoChannelEvent],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
    ) -> Optional[PipelineUpdateResult]:

        uid = int(user_id or self._default_user_id)

        # get active pipeline OUTSIDE any Manager lock
        active = await self.get_activepipeline(uid)

        active_pid = getattr(active, "pipeline_id", None) or self._pipeline_id_by_user.get(uid)
        if not active_pid:
            user_lock = self._get_user_lock(uid)
            async with user_lock:
                self._pipelines_by_user.pop(uid, None)
                self._pipeline_id_by_user.pop(uid, None)
            active = await self.get_activepipeline(uid)
            active_pid = getattr(active, "pipeline_id", None)

        pid = self._as_uuid(active_pid, "pipeline_id")

        if pipeline_id is not None:
            try:
                supplied_pid = self._as_uuid(pipeline_id, "pipeline_id")
                if supplied_pid != pid:
                    logger.warning(
                        "update_pipeline called with pipeline_id=%s but active pipeline_id=%s user=%s; using active",
                        supplied_pid, pid, uid
                    )
            except Exception:
                logger.warning("update_pipeline got non-uuid pipeline_id=%s; using active", pipeline_id)

        for attempt in (1, 2):
            async with self._session_factory() as db:
                exists = await self._repo.pipeline_exists(db, pid)
                if not exists:
                    logger.warning(
                        "Active pipeline %s missing in DB for user %s (attempt %s).",
                        pid, uid, attempt
                    )
                    user_lock = self._get_user_lock(uid)
                    async with user_lock:
                        self._pipelines_by_user.pop(uid, None)
                        self._pipeline_id_by_user.pop(uid, None)

                    if attempt == 1:
                        active = await self.get_activepipeline(uid)
                        active_pid = getattr(active, "pipeline_id", None) or self._pipeline_id_by_user.get(uid)
                        if not active_pid:
                            return None
                        pid = self._as_uuid(active_pid, "pipeline_id")
                        continue
                    return None

                cameras_out: List[CameraOut] = []
                events_out: List[Dict[str, Any]] = []
                roi_reset_camera_ids: Set[uuid.UUID] = set()

                for ev in (channel_events or []):
                    et = getattr(ev, "event_type", None)
                    if et is None and isinstance(ev, dict):
                        et = ev.get("event_type")
                    et_norm = str(et or "").lower()

                    if et_norm == "create_channel" or isinstance(ev, ChannelCreateEvent):
                        cams, evs = await self._add_channel(
                            db, pid=pid, ev=ev, user_id=uid,
                            camera_code_prefix=camera_code_prefix, active=active
                        )
                    elif et_norm == "edit_channel" or isinstance(ev, ChannelEditEvent):
                        if self._event_includes_roi_patch(ev):
                            cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
                            if cam_uuid is not None:
                                roi_reset_camera_ids.add(self._as_uuid(cam_uuid, "camera_uuid"))
                        cams, evs = await self._edit_channel(db, pid=pid, ev=ev, user_id=uid, active=active)
                    elif et_norm == "remove_channel" or isinstance(ev, ChannelRemoveEvent):
                        cams, evs = await self._remove_channel(db, pid=pid, ev=ev, user_id=uid, active=active)
                    else:
                        logger.warning("Event type not matched: %s", et)
                        continue

                    cameras_out.extend(cams)
                    events_out.extend(evs)

                await db.commit()
                for cam_uuid in roi_reset_camera_ids:
                    self._invalidate_camera_roi_state(cam_uuid)

                return PipelineUpdateResult(
                    pipeline_id=pid,
                    active_in_memory=True,
                    cameras=cameras_out,
                    events=events_out,
                )

        return None
    
    async def _reconcile_device_edge_with_retry(
        self,
        *,
        device_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Wrapper around reconcile_device_edge_simple with exponential backoff retry.
        """
        last_exception = None
        for attempt in range(1, self._edge_retry_max_attempts + 1):
            try:
                return await self.reconcile_device_edge_simple(
                    device_uuid=device_uuid,
                    user_id=user_id,
                    dry_run=dry_run,
                    delete_unknown=delete_unknown,
                )
            except Exception as e:
                last_exception = e
                if attempt < self._edge_retry_max_attempts:
                    # Exponential backoff: base * (2 ^ (attempt-1))
                    wait_ms = self._edge_retry_base_ms * (2 ** (attempt - 1))
                    logger.warning(
                        "Device reconcile attempt %d/%d failed for %s, retrying in %dms: %s",
                        attempt,
                        self._edge_retry_max_attempts,
                        device_uuid,
                        wait_ms,
                        str(e),
                    )
                    await asyncio.sleep(wait_ms / 1000.0)
                else:
                    logger.error(
                        "Device reconcile failed after %d attempts for %s: %s",
                        attempt,
                        device_uuid,
                        str(e),
                    )
        raise last_exception or RuntimeError("Device reconcile failed")

    async def reconcile_device_edge_simple(
        self,
        *,
        device_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = True,
    ) -> Dict[str, List[str]]:
        uid = int(user_id) if user_id is not None else None
        async with self._session_factory() as db:
            dev = await self._get_device(db, device_uuid, user_id=uid)
            peer_devices = await self._list_devices_for_physical_device(db, device=dev)
            reconcile_device_uuids: List[uuid.UUID] = []
            seen_reconcile_devices: Set[str] = set()
            for peer in peer_devices:
                peer_uuid = getattr(peer, "device_uuid", None)
                if peer_uuid is None:
                    continue
                peer_key = str(peer_uuid)
                if peer_key in seen_reconcile_devices:
                    continue
                seen_reconcile_devices.add(peer_key)
                reconcile_device_uuids.append(peer_uuid)

            if not reconcile_device_uuids:
                reconcile_device_uuids = [device_uuid]

            cams = await self._list_cameras_for_device_uuids(
                db,
                device_uuids=reconcile_device_uuids,
            )
            site_schedule_cache: Dict[str, Dict[str, Any]] = {}
            desired_set: Set[str] = set()
            active_streams: Set[str] = set()
            for cam in cams:
                cfg = (
                    cam.channel_configuration.configuration or {}
                    if getattr(cam, "channel_configuration", None) and getattr(cam.channel_configuration, "configuration", None)
                    else {}
                )
                schedule_state = await self._resolve_runtime_schedule(
                    db,
                    cam=cam,
                    cfg_json=cfg,
                    cfg_timezone=getattr(getattr(cam, "channel_configuration", None), "timezone", None),
                    site_cache=site_schedule_cache,
                )
                # Alert schedules should not tear down active detection runtimes.
                if bool(cam.is_detection_enabled):
                    desired_set.add(str(cam.camera_uuid))
                # Keep playback paths provisioned for enabled cameras regardless of
                # alert schedule state so live view remains stable.
                if bool(cam.is_enabled) and getattr(cam, "camera_code", None):
                    active_streams.add(str(cam.camera_code))

        # --- WebRTC stream provisioning (independent of edge device) ---
        # Always provision WHEP streams in MediaMTX so live view works even
        # when the Jetson edge device is unreachable.
        try:
            webrtc_list = await self._webrtc.list_webrtc_cameras()
        except Exception as e:
            logger.warning("Cannot reach WebRTC gateway during reconcile: %s", e)
            webrtc_list = []
        webrtc_set = {
            str(c.get("stream_key"))
            for c in webrtc_list
            if isinstance(c, dict) and c.get("stream_key")
        }
        known_streams = {str(c.camera_code) for c in cams if c.camera_code}
        to_add_stream = sorted(active_streams - webrtc_set)
        to_remove_stream = sorted((webrtc_set & known_streams) - active_streams)
        cams_by_code = {str(c.camera_code): c for c in cams if c.camera_code}

        webrtc_added: List[str] = []
        webrtc_errors: List[str] = []
        for cu in to_add_stream:
            cam = cams_by_code.get(cu)
            if cam is None:
                webrtc_errors.append(f"Camera not found in DB during WebRTC reconcile: {cu}")
                continue
            try:
                await self._webrtc.ensure_stream(stream_key=cu, rtsp_url=cam.rtsp_url)
                webrtc_added.append(str(cam.camera_uuid))
            except Exception as e:
                logger.warning("WebRTC ensure_stream failed during reconcile for %s: %s", cu, e)
                webrtc_errors.append(f"Failed to provision stream {cu}: {e}")

        webrtc_removed: List[str] = []
        if delete_unknown:
            for cu in to_remove_stream:
                try:
                    await self._webrtc.delete_stream(stream_key=cu)
                    removed_cam = cams_by_code.get(cu)
                    removed_uuid = str(removed_cam.camera_uuid) if removed_cam else cu
                    webrtc_removed.append(removed_uuid)
                except Exception as e:
                    logger.warning("WebRTC delete_stream failed during reconcile for %s: %s", cu, e)
                    webrtc_errors.append(f"Failed to remove stream {cu}: {e}")

        # --- Edge device reconcile ---
        device_url = dev.device_url
        edge_warnings: List[str] = []
        try:
            edge_set = await self._call_with_timeout(
                self._edge.list_cameras(device_url=device_url),
                timeout_s=self._external_timeout_s
            )
        except EdgeCameraInventoryError as e:
            if _edge_health_ready(getattr(e, "health", None)):
                logger.warning(
                    "Edge camera inventory unavailable for %s during reconcile; continuing with add-only sync: %s",
                    device_url,
                    e,
                )
                edge_set = set()
                edge_warnings.append(
                    "Edge camera inventory was unavailable, so sync continued in add-only mode. "
                    "Existing unknown cameras on the edge device were not removed."
                )
            else:
                logger.warning(
                    "Cannot reach edge device %s during reconcile (%s): %s",
                    device_url,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                # Even though edge is unreachable, return WebRTC results so they aren't lost
                raise EdgeDeviceUnavailableError(device_url, e) from e
        except Exception as e:
            logger.warning(
                "Cannot reach edge device %s during reconcile (%s): %s",
                device_url,
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise EdgeDeviceUnavailableError(device_url, e) from e
        to_add = sorted(desired_set - edge_set)
        to_remove = sorted(edge_set - desired_set)

        out: Dict[str, List[str]] = {
            "to_add": to_add,
            "to_remove": to_remove,
            "to_add_stream": to_add_stream,
            "to_remove_stream": to_remove_stream,
            "added": list(webrtc_added),
            "removed": list(webrtc_removed),
            "errors": list(webrtc_errors),
            "warnings": edge_warnings,
        }

        if dry_run:
            return out

        cams_by_uuid = {str(c.camera_uuid): c for c in cams}

        for cu in to_add:
            cam = cams_by_uuid.get(cu)
            if cam is None:
                out["errors"].append(f"Camera not found in DB during reconcile: {cu}")
                continue
            cfg = (cam.channel_configuration.configuration or {}) if cam.channel_configuration else {}
            payload = {
                **_only_jetson_config(cfg),
                # Explicit fields last so they always win over whatever is in channel config
                "camera_uuid": cu,
                "rtsp_url": cam.rtsp_url,
                "enabled": _edge_runtime_enabled(detection_enabled=True),
                "detection_enabled": True,
                "notification_enabled": bool(cam.is_notification_enabled),
            }
            try:
                await self._call_with_timeout(
                    self._edge.upsert_camera(device_url=device_url, payload=payload),
                    timeout_s=self._external_timeout_s
                )
                out["added"].append(cu)
            except Exception as e:
                logger.warning("Edge upsert failed during reconcile for camera %s", cu, exc_info=True)
                out["errors"].append(f"Failed to add {cu}: {e}")

        if delete_unknown:
            for cu in to_remove:
                try:
                    await self._call_with_timeout(
                        self._edge.delete_camera(device_url=device_url, camera_uuid=cu),
                        timeout_s=self._external_timeout_s
                    )
                    out["removed"].append(cu)
                except Exception as e:
                    logger.warning("Edge delete failed during reconcile for camera %s", cu, exc_info=True)
                    out["errors"].append(f"Failed to remove {cu}: {e}")

        return out

    async def reconcile_all_devices_edge(
        self,
        *,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = False,
    ) -> Dict[str, Any]:
        """
        Reconcile every device for the user.
        Useful at startup/after edge reboot so Jetson gets re-hydrated from DB state.
        """
        uid = int(user_id) if user_id is not None else None
        async with self._session_factory() as db:
            device_rows = await self._list_enabled_reconcile_devices(db, user_id=uid)

        device_uuids: List[uuid.UUID] = []
        seen_targets: Set[str] = set()
        for dev in device_rows:
            du = getattr(dev, "device_uuid", None)
            if du is None:
                continue
            key = self._normalize_device_url(getattr(dev, "device_url", None)) or str(du)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            device_uuids.append(du)

        summary: Dict[str, Any] = {
            "user_id": uid,
            "device_count": len(device_uuids),
            "device_row_count": len(device_rows),
            "devices": {},
            "errors": [],
            "warnings": [],
        }

        for du in device_uuids:
            key = str(du)
            try:
                result = await self._reconcile_device_edge_with_retry(
                    device_uuid=du,
                    user_id=uid,
                    dry_run=dry_run,
                    delete_unknown=delete_unknown,
                )
                summary["devices"][key] = result
                for warning in result.get("warnings") or []:
                    summary["warnings"].append("{}: {}".format(key, warning))
            except Exception as e:
                logger.warning("Startup reconcile failed for device=%s", key, exc_info=True)
                summary["errors"].append("{}: {}".format(key, e))

        return summary

    # ------------------------------------------------------------------
    # Background edge helpers — fire-and-forget so the main lock is
    # never held during slow Jetson HTTP calls.
    # ------------------------------------------------------------------
    async def _bg_edge_upsert(self, *, device_url: str, payload: dict) -> None:
        cam_uuid = payload.get("camera_uuid", "unknown")
        try:
            await self._edge.upsert_camera(device_url=device_url, payload=payload)
            logger.debug("Background edge upsert succeeded camera_uuid=%s", cam_uuid)
        except Exception:
            logger.warning(
                "Background edge upsert failed camera_uuid=%s — run Sync to fix",
                cam_uuid,
                exc_info=True,
            )

    async def _bg_edge_delete(self, *, device_url: str, camera_uuid: str) -> None:
        try:
            await self._edge.delete_camera(device_url=device_url, camera_uuid=camera_uuid)
        except Exception:
            logger.warning(
                "Background edge delete failed camera_uuid=%s",
                camera_uuid,
                exc_info=True,
            )

    async def _bg_edge_patch(self, *, device_url: str, camera_uuid: str, patch: dict) -> None:
        try:
            await self._edge.patch_camera(device_url=device_url, camera_uuid=camera_uuid, patch=patch)
            logger.debug("Background edge patch succeeded camera_uuid=%s", camera_uuid)
        except Exception:
            logger.warning(
                "Background edge patch failed camera_uuid=%s — run Sync to fix",
                camera_uuid,
                exc_info=True,
            )

    async def _bg_webrtc_update_stream(self, *, stream_key: str, rtsp_url: str) -> None:
        try:
            await self._webrtc.update_stream(stream_key=stream_key, rtsp_url=rtsp_url)
            logger.debug("Background WebRTC update_stream succeeded stream_key=%s", stream_key)
        except Exception:
            logger.warning(
                "Background WebRTC update_stream failed stream_key=%s — run Sync to fix",
                stream_key,
                exc_info=True,
            )

    async def _add_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: int,
        camera_code_prefix: str,
        active: Optional[ModelPipeline],
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        patch = self._patch_to_dict(getattr(ev, "configs", None))

        rtsp_url = patch.get("rtsp_url")
        if not rtsp_url:
            raise ValueError("Create_Channel requires rtsp_url")

        site_uuid = patch.get("site_uuid")
        if not site_uuid:
            raise ValueError("Create_Channel requires site_uuid")
        site_uuid = self._as_uuid(site_uuid, "site_uuid")

        cam_uuid = patch.get("camera_uuid") or uuid.uuid4()
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")
        patch["camera_uuid"] = cam_uuid
        patch["channel_id"] = cam_uuid  # keep compatibility
        camera_code = f"{camera_code_prefix}-{cam_uuid.hex[:8]}"
        webrtc_url = await self._webrtc.ensure_stream(stream_key=str(camera_code), rtsp_url=str(rtsp_url))

        # Now validate ownership (first DB query — connection checked out here).
        await self._ensure_site_owned_by_user(db, site_uuid=site_uuid, user_id=user_id)
        raw_device_uuid = patch.get("device_uuid")
        device_uuid = (
            self._as_uuid(raw_device_uuid, "device_uuid")
            if raw_device_uuid is not None else None
        )
        dev = await self._resolve_site_device(
            db,
            site_uuid=site_uuid,
            user_id=user_id,
            requested_device_uuid=device_uuid,
            required=True,
        )
        device_uuid = dev.device_uuid
        patch["device_uuid"] = device_uuid

        cam, cfg_json, tz = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config=patch,
            user_id=user_id,
            cam_uuid=cam_uuid,
            camera_code=camera_code,
            site_uuid=site_uuid,
            webrtc_url=webrtc_url,
            device_uuid=device_uuid,  
        )

        await self._set_single_camera_device(db, cam.camera_uuid, device_uuid)

        enabled = bool(getattr(cam, "is_enabled", True))
        det_enabled = bool(getattr(cam, "is_detection_enabled", True))
        schedule_state = await self._resolve_runtime_schedule(
            db,
            cam=cam,
            cfg_json=cfg_json,
            cfg_timezone=tz,
        )

        edge_payload = self._edge_payload_from_config(
            camera_uuid=str(cam.camera_uuid),
            rtsp_url=cam.rtsp_url,
            config={
                **_only_jetson_config(patch),
                "enabled": _edge_runtime_enabled(detection_enabled=det_enabled),
                "detection_enabled": det_enabled,
                "notification_enabled": bool(getattr(cam, "is_notification_enabled", True)),
            },
        )

        # Fire-and-forget: do NOT await the Jetson call inside the lock.
        # If Jetson is unreachable the 3-retry × 15s timeout would hold
        # self._lock for up to 45s, blocking all other operations.
        # The background reconcile loop (every 90s) will catch any failure.
        if det_enabled:
            asyncio.create_task(
                self._bg_edge_upsert(device_url=dev.device_url, payload=edge_payload)
            )
        else:
            asyncio.create_task(
                self._bg_edge_delete(device_url=dev.device_url, camera_uuid=str(cam.camera_uuid))
            )

        if active:
            runtime_overrides = _runtime_config_overrides(
                cfg_json or {},
                extra_forbidden={
                    "sample_fps",
                    "decode_backend",
                    "request_timeout_s",
                    "schedule",
                    "timezone",
                    "use_site_schedule",
                },
            )
            vcc = VideoChannelConfig(
                camera_uuid=cam.camera_uuid,
                rtsp_url=cam.rtsp_url,
                webrtc_url=cam.webrtc_url or "",
                site_uuid=cam.site_uuid,
                device_uuid=device_uuid,
                device_url=dev.device_url,
                enabled=enabled,
                detection_enabled=det_enabled,
                notification_enabled=bool(getattr(cam, "is_notification_enabled", True)),
                sample_fps=float((cfg_json or {}).get("sample_fps", patch.get("sample_fps", 5.0))),
                decode_backend=str((cfg_json or {}).get("decode_backend", patch.get("decode_backend", "gstreamer"))),
                request_timeout_s=float(
                    (cfg_json or {}).get(
                        "request_timeout_s",
                        patch.get("request_timeout_s", self._default_request_timeout_s),
                    )
                ),
                timezone=schedule_state["timezone"],
                schedule=schedule_state["schedule"],
                use_site_schedule=schedule_state["use_site_schedule"],
                **runtime_overrides,
            )
            await active.add_channel(VideoChannel(config=vcc))

        cameras_out = [
            CameraOut(
                camera_uuid=cam.camera_uuid,
                camera_code=getattr(cam, "camera_code", None),
                name=getattr(cam, "name", None),
                location=getattr(cam, "location", None),
                site_uuid=cam.site_uuid,
                rtsp_url=cam.rtsp_url,
                webrtc_url=cam.webrtc_url,
                enabled=bool(cam.is_enabled),
                detection_enabled=bool(cam.is_detection_enabled),
                notification_enabled=bool(cam.is_notification_enabled),
                device_uuid=device_uuid,
                device_url=dev.device_url,
                sample_fps=float(patch.get("sample_fps", 5.0)),
                decode_backend=str(patch.get("decode_backend", "gstreamer")),
                resize=patch.get("resize"),
                emit_format=str(patch.get("emit_format", "raw")),
                jpeg_quality=int(patch.get("jpeg_quality", 80)),
                roi=cam.roi,
                configuration=cfg_json or {},
                timezone=tz,
                notification_trigger_mode=str(getattr(cam, "notification_trigger_mode", "inherit") or "inherit"),
                camera_playback_enabled=str(getattr(cam, "camera_playback_enabled", "inherit") or "inherit"),
                use_site_schedule=schedule_state.get("use_site_schedule"),
                created_at=getattr(cam, "created_at", None),
                updated_at=getattr(cam, "updated_at", None),
            )
        ]

        events_out = [
            {
                "event_type": "Create_Channel",
                "camera_uuid": str(cam.camera_uuid),
                "site_uuid": str(cam.site_uuid),
                "device_uuid": str(device_uuid),
                "rtsp_url": cam.rtsp_url,
                "webrtc_url": cam.webrtc_url,
            }
        ]
        return cameras_out, events_out

    async def _edit_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: int,
        active: Optional[ModelPipeline],
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
        if cam_uuid is None:
            raise ValueError("Edit_Channel missing channel_id (camera_uuid).")
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")

        full = await self.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if not full:
            raise ValueError(f"Camera not found: {cam_uuid}")

        cam_db, chan_cfg_db, existing_pid = full
        if int(getattr(cam_db, "user_id", -1)) != int(user_id):
            raise ValueError(f"Camera does not belong to user: {cam_uuid}")
        if existing_pid is not None and existing_pid != pid:
            raise ValueError("Camera does not belong to provided pipeline_id")

        old_rtsp = cam_db.rtsp_url
        old_webrtc = cam_db.webrtc_url
        patch = self._patch_to_dict(getattr(ev, "configs", None))
        patch.pop("webrtc_url", None)

        old_devices = await self._get_camera_devices(db, cam_uuid)
        if len(old_devices) > 1:
            logger.warning(
                "Camera %s has %s linked devices during edit; using most recent device %s",
                cam_uuid,
                len(old_devices),
                getattr(old_devices[0], "device_uuid", None),
            )
        old_dev = old_devices[0] if old_devices else None
        if old_dev is not None and not getattr(old_dev, "device_url", None):
            raise ValueError(f"Assigned device has no device_url for camera {cam_uuid}")

        # Determine new device: requested > preserve old > auto-pick
        requested_device_uuid = patch.get("device_uuid")
        if requested_device_uuid is not None:
            requested_device_uuid = self._as_uuid(requested_device_uuid, "device_uuid")
            new_dev = await self._resolve_site_device(
                db,
                site_uuid=cam_db.site_uuid,
                user_id=user_id,
                requested_device_uuid=requested_device_uuid,
                required=True,
            )
        elif old_dev is not None:
            # PRESERVE: Keep existing device if not changing
            new_dev = old_dev
        else:
            # AUTO-PICK: Try if exactly 1 device in site
            picked_uuid = await self._pick_site_device_uuid(
                db,
                site_uuid=cam_db.site_uuid,
                user_id=user_id,
            )
            if picked_uuid is None:
                raise ValueError(
                    f"Camera has no device assigned. Link a Device to this site or assign explicitly."
                )
            new_dev = await self._get_device(db, picked_uuid, user_id=user_id)
            logger.info(
                "Auto-linked camera %s to device %s",
                cam_uuid,
                new_dev.device_uuid,
            )
        new_device_uuid = new_dev.device_uuid
        patch["device_uuid"] = new_device_uuid

        # merge config json
        merged_cfg: Dict[str, Any] = {}
        if chan_cfg_db and getattr(chan_cfg_db, "configuration", None):
            merged_cfg.update(chan_cfg_db.configuration or {})
        merged_cfg.update(patch)

        cam2, cfg_json, tz = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config={
                **merged_cfg,
                "camera_uuid": cam_uuid,
                "webrtc_url": old_webrtc,
                "rtsp_url": merged_cfg.get("rtsp_url", old_rtsp),
                "enabled": merged_cfg.get("enabled", cam_db.is_enabled),
                "detection_enabled": merged_cfg.get("detection_enabled", cam_db.is_detection_enabled),
                "notification_enabled": merged_cfg.get("notification_enabled", cam_db.is_notification_enabled),
            },
            site_uuid=cam_db.site_uuid,
            webrtc_url=old_webrtc,
            device_uuid=new_device_uuid,  # ok if repo uses it
        )

        stale_old_devices = [
            dev for dev in old_devices
            if getattr(dev, "device_uuid", None) != new_device_uuid and getattr(dev, "device_url", None)]
        for dev in stale_old_devices:
            asyncio.create_task(
                self._bg_edge_delete(device_url=dev.device_url, camera_uuid=str(cam_uuid))
            )

        device_changed = old_dev is None or old_dev.device_uuid != new_device_uuid
        if len(old_devices) != 1 or device_changed:
            await self._set_single_camera_device(db, cam_uuid, new_device_uuid)

        # Fire-and-forget: do NOT await WebRTC call while X lock is held.
        if cam2.rtsp_url != old_rtsp and cam2.camera_code:
            asyncio.create_task(
                self._bg_webrtc_update_stream(stream_key=str(cam2.camera_code), rtsp_url=cam2.rtsp_url)
            )

        enabled = bool(cam2.is_enabled)
        det_enabled = bool(cam2.is_detection_enabled)
        schedule_state = await self._resolve_runtime_schedule(
            db,
            cam=cam2,
            cfg_json=cfg_json,
            cfg_timezone=tz,
        )

        # Fire-and-forget edge sync — do NOT await while X lock is held on camera row.
        if det_enabled:
            if device_changed:
                edge_payload = self._edge_payload_from_config(
                    camera_uuid=str(cam_uuid),
                    rtsp_url=cam2.rtsp_url,
                    config={
                        **_only_jetson_config(merged_cfg),
                        "enabled": _edge_runtime_enabled(detection_enabled=det_enabled),
                        "detection_enabled": det_enabled,
                        "notification_enabled": bool(cam2.is_notification_enabled),
                    },
                )
                asyncio.create_task(
                    self._bg_edge_upsert(device_url=new_dev.device_url, payload=edge_payload)
                )
            else:
                edge_patch = _only_jetson_config(patch)
                edge_patch.setdefault("rtsp_url", cam2.rtsp_url)
                edge_patch["enabled"] = _edge_runtime_enabled(detection_enabled=det_enabled)
                edge_patch["detection_enabled"] = det_enabled
                edge_patch["notification_enabled"] = bool(cam2.is_notification_enabled)
                asyncio.create_task(
                    self._bg_edge_patch(device_url=new_dev.device_url, camera_uuid=str(cam_uuid), patch=edge_patch)
                )
        else:
            asyncio.create_task(
                self._bg_edge_delete(device_url=new_dev.device_url, camera_uuid=str(cam_uuid))
            )

        if active:
            runtime_overrides = _runtime_config_overrides(
                merged_cfg,
                extra_forbidden={"schedule", "timezone", "use_site_schedule"},
            )
            vcc = VideoChannelConfig(
                camera_uuid=cam2.camera_uuid,
                rtsp_url=cam2.rtsp_url,
                webrtc_url=cam2.webrtc_url or "",
                site_uuid=cam2.site_uuid,
                device_uuid=new_device_uuid,
                device_url=new_dev.device_url,
                enabled=enabled,
                detection_enabled=det_enabled,
                notification_enabled=bool(cam2.is_notification_enabled),
                timezone=schedule_state["timezone"],
                schedule=schedule_state["schedule"],
                use_site_schedule=schedule_state["use_site_schedule"],
                **runtime_overrides,
            )
            await active.edit_channel(VideoChannel(config=vcc))

        cameras_out = [
            CameraOut(
                camera_uuid=cam2.camera_uuid,
                camera_code=getattr(cam2, "camera_code", None),
                name=getattr(cam2, "name", None),
                location=getattr(cam2, "location", None),
                site_uuid=cam2.site_uuid,
                rtsp_url=cam2.rtsp_url,
                webrtc_url=cam2.webrtc_url,
                enabled=bool(cam2.is_enabled),
                detection_enabled=bool(cam2.is_detection_enabled),
                notification_enabled=bool(cam2.is_notification_enabled),
                device_uuid=new_device_uuid,
                device_url=new_dev.device_url,
                sample_fps=float(merged_cfg.get("sample_fps", 5.0)),
                decode_backend=str(merged_cfg.get("decode_backend", "gstreamer")),
                resize=merged_cfg.get("resize"),
                emit_format=str(merged_cfg.get("emit_format", "raw")),
                jpeg_quality=int(merged_cfg.get("jpeg_quality", 80)),
                roi=cam2.roi,
                configuration=cfg_json or {},
                timezone=tz,
                notification_trigger_mode=str(getattr(cam2, "notification_trigger_mode", "inherit") or "inherit"),
                camera_playback_enabled=str(getattr(cam2, "camera_playback_enabled", "inherit") or "inherit"),
                use_site_schedule=schedule_state.get("use_site_schedule"),
                created_at=getattr(cam2, "created_at", None),
                updated_at=getattr(cam2, "updated_at", None),
            )
        ]

        events_out = [{
            "event_type": "Edit_Channel",
            "camera_uuid": str(cam2.camera_uuid),
            "rtsp_url": cam2.rtsp_url,
            "webrtc_url": cam2.webrtc_url,
            "device_uuid": str(new_device_uuid),
            "patch": patch,
        }]

        return cameras_out, events_out

    async def _remove_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: int,
        active: Optional[ModelPipeline],
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
        if cam_uuid is None:
            raise ValueError("Remove_Channel missing channel_id (camera_uuid).")
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")

        full = await self.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if full:
            cam_db, _cfg, existing_pid = full
            if int(getattr(cam_db, "user_id", -1)) != int(user_id):
                raise ValueError(f"Camera does not belong to user: {cam_uuid}")
            if existing_pid is not None and existing_pid != pid:
                raise ValueError("Camera does not belong to provided pipeline_id")

            try:
                devices = await self._get_camera_devices(db, cam_uuid)
                seen_urls: Set[str] = set()
                for dev in devices:
                    dev_url = str(getattr(dev, "device_url", "") or "").strip()
                    if not dev_url or dev_url in seen_urls:
                        continue
                    seen_urls.add(dev_url)
                    await self._edge.delete_camera(device_url=dev_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.warning("Edge delete failed during camera removal", exc_info=True)
            try:
                if cam_db.camera_code:
                    await self._webrtc.delete_stream(stream_key=str(cam_db.camera_code))
            except Exception:
                logger.warning("WebRTC delete failed during camera removal", exc_info=True)

            await self.channel_repo.delete_camera(db, camera_uuid=cam_db.camera_uuid)

        if active:
            try:
                await active.remove_channel(cam_uuid)
            except Exception:
                logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

        events_out = [{"event_type": "Remove_Channel", "camera_uuid": str(cam_uuid)}]
        return [], events_out

    async def cleanup_user_resources(self, db: AsyncSession, *, user_id: int) -> Dict[str, Any]:
        """
        Clean up external/runtime resources for a user before deleting the DB user row.
        This does not delete DB rows directly; caller should delete the User after this succeeds.
        """
        uid = int(user_id)
        errors: List[str] = []
        edge_deleted: List[str] = []
        streams_deleted: List[str] = []

        q = (
            select(Camera)
            .where(Camera.user_id == uid)
            .options(selectinload(Camera.devices))
        )
        user_cameras = (await db.execute(q)).scalars().all()

        edge_seen: Set[Tuple[str, str]] = set()
        stream_seen: Set[str] = set()

        # Best-effort external cleanup while camera/device metadata still exists.
        for cam in user_cameras:
            cam_uuid_str = str(cam.camera_uuid)
            for dev in list(getattr(cam, "devices", None) or []):
                dev_url = str(getattr(dev, "device_url", "") or "").strip()
                if not dev_url:
                    continue
                key = (dev_url, cam_uuid_str)
                if key in edge_seen:
                    continue
                edge_seen.add(key)
                try:
                    await self._edge.delete_camera(device_url=dev_url, camera_uuid=cam_uuid_str)
                    edge_deleted.append(cam_uuid_str)
                except Exception as e:
                    logger.warning(
                        "Failed deleting edge camera during user cleanup user=%s camera=%s device=%s",
                        uid,
                        cam_uuid_str,
                        getattr(dev, "device_uuid", None),
                        exc_info=True,
                    )
                    errors.append(f"edge:{cam_uuid_str}:{e}")

            stream_key = str(getattr(cam, "camera_code", "") or "").strip()
            if stream_key and stream_key not in stream_seen:
                stream_seen.add(stream_key)
                try:
                    await self._webrtc.delete_stream(stream_key=stream_key)
                    streams_deleted.append(stream_key)
                except Exception as e:
                    logger.warning(
                        "Failed deleting WebRTC stream during user cleanup user=%s stream=%s",
                        uid,
                        stream_key,
                        exc_info=True,
                    )
                    errors.append(f"webrtc:{stream_key}:{e}")

        pipeline_to_shutdown: Optional[ModelPipeline] = None
        user_lock = self._get_user_lock(uid)
        async with user_lock:
            pipeline_to_shutdown = self._pipelines_by_user.pop(uid, None)
            self._pipeline_id_by_user.pop(uid, None)

        if pipeline_to_shutdown is not None:
            try:
                await pipeline_to_shutdown.shutdown()
            except Exception as e:
                logger.warning("Failed shutting down in-memory pipeline for deleted user=%s", uid, exc_info=True)
                errors.append(f"pipeline_shutdown:{e}")

        return {
            "user_id": uid,
            "camera_count": len(user_cameras),
            "edge_deleted_count": len(edge_deleted),
            "stream_deleted_count": len(streams_deleted),
            "errors": errors,
        }

    async def cleanup_device_resources(self, db: AsyncSession, *, device_uuid: uuid.UUID,active: Optional[ModelPipeline]) -> None:
        """
        Called when a Device is about to be deleted.
        Finds all cameras on this device and sends DELETE to the edge service.
        """
        try:
            dev = await self._get_device(db, device_uuid)
            if not dev.device_url:
                return

            q = (
                select(Camera)
                .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
                .where(CameraDevice.device_uuid == device_uuid)
            )
            cameras_on_device = (await db.execute(q)).scalars().all()
            
            for cam in cameras_on_device:
                try:
                    await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam.camera_uuid))
                    if cam.camera_code:
                        await self._webrtc.delete_stream(stream_key=str(cam.camera_code))
                    await self.channel_repo.delete_camera(db, camera_uuid=cam.camera_uuid)
                    if active:
                        try:
                            await active.remove_channel(cam.camera_uuid)
                        except Exception:
                            logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

                except Exception:
                    logger.warning("Failed cleanup camera %s on device %s", cam.camera_uuid, dev.device_uuid, exc_info=True)
        except Exception:
            logger.exception("Error during device cleanup for %s", device_uuid)
            
    async def _get_site(self, db: AsyncSession, user_id: int, site_uuid: uuid.UUID) -> Site:
        q = select(Site).where(Site.site_uuid == site_uuid, Site.user_id == user_id, Site.is_deleted == False)
        site = (await db.execute(q)).scalar_one_or_none()
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        return site


    async def cleanup_site_resources(self, db: AsyncSession, *,user_id: int, site_uuid: uuid.UUID,active: Optional[ModelPipeline]) -> None:
        """
        Called when a Site is about to be deleted.
        Finds all cameras on this site and sends DELETE to the edge service.
        """
        try:
            site = await self._get_site(db,user_id=user_id, site_uuid=site_uuid)
            if not site.site_uuid:
                return
            q = (
                select(Camera)
                .where(Camera.site_uuid == site_uuid)
                .options(selectinload(Camera.devices))
            )
            cameras_on_site = (await db.execute(q)).scalars().all()

            for cam in cameras_on_site:
                seen_urls: Set[str] = set()
                for dev in list(getattr(cam, "devices", None) or []):
                    dev_url = str(getattr(dev, "device_url", "") or "").strip()
                    if not dev_url or dev_url in seen_urls:
                        continue
                    seen_urls.add(dev_url)
                    await self._edge.delete_camera(device_url=dev_url, camera_uuid=str(cam.camera_uuid))
                if cam.camera_code:
                    await self._webrtc.delete_stream(stream_key=str(cam.camera_code))
                await self.channel_repo.delete_camera(db, camera_uuid=cam.camera_uuid)
                if active:
                    try:
                        await active.remove_channel(cam.camera_uuid)
                    except Exception:
                        logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

        except Exception:
            logger.exception("Error during device cleanup for %s", site_uuid)
