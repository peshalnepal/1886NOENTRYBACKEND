# agents/application/services/agent_manager.py

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from fastapi import HTTPException

from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository
from core.database_orm import Site,Device, Camera, CameraDevice
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent, VideoChannelEvent
from domain.model_pipeline import ModelPipeline
from application.channels.channel_config import VideoChannelConfig
from application.channels.channel import VideoChannel
from application.services.edgeinference import EdgeInferenceClient
from application.services.webrtcgateway import WebRTCGatewayClient

logger = logging.getLogger(__name__)

# -------------------------
# API DTOs
# -------------------------

    
class CameraOut(BaseModel):
    camera_uuid: uuid.UUID
    camera_code: Optional[str] = None
    site_uuid: uuid.UUID

    rtsp_url: str
    webrtc_url: Optional[str] = None

    enabled: bool
    detection_enabled: bool
    notification_enabled: bool
    roi: Optional[Dict[str, Any]] = None
    configuration: Dict[str, Any] = Field(default_factory=dict)
    timezone: Optional[str] = None

    device_uuid: uuid.UUID
    device_url: str

    sample_fps: float = Field(default=5.0, ge=0.1)
    decode_backend: str = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = None
    emit_format: str = Field(default="raw")
    jpeg_quality: int = Field(default=80, ge=1, le=100)


class PipelineUpdateResult(BaseModel):
    pipeline_id: uuid.UUID
    active_in_memory: bool = False
    cameras: List[CameraOut] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)


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

def _runtime_config_overrides(cfg: Dict[str, Any], *, extra_forbidden: Optional[set] = None) -> Dict[str, Any]:
    forbidden = set(RUNTIME_CONFIG_FORBIDDEN_KEYS)
    if extra_forbidden:
        forbidden.update(extra_forbidden)

    out: Dict[str, Any] = {}
    for k, v in (cfg or {}).items():
        if v is None:
            continue
        if k not in RUNTIME_CONFIG_ALLOWED_KEYS:
            continue
        if k in forbidden:
            continue
        out[k] = v
    return out

class Manager:
    """
    Azure Manager:
      - DB is source of truth
      - WebRTC gateway provides playback URL
      - Jetson device provides detections
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._lock = asyncio.Lock()

        self._repo = PipelineRepository()
        self.channel_repo = ChannelRepository()

        self._webrtc = WebRTCGatewayClient()
        self._edge = EdgeInferenceClient()

        self._pipelines_by_user: Dict[int, ModelPipeline] = {}
        self._pipeline_id_by_user: Dict[int, uuid.UUID] = {}

        self._default_user_id = int(os.getenv("DEFAULT_USER_ID", "1"))
        self._default_request_timeout_s = 3.0

    async def shutdown(self) -> None:
        async with self._lock:
            self._pipelines_by_user.clear()
            self._pipeline_id_by_user.clear()
        await self._webrtc.close()
        await self._edge.close()

    def _patch_to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            out: Dict[str, Any] = {}
        elif hasattr(obj, "model_dump"):
            out = obj.model_dump(exclude_unset=True, exclude_none=True)
        elif isinstance(obj, dict):
            out = {k: v for k, v in obj.items() if v is not None}
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
        Returns the single Device assigned to this camera (enforces exactly one).
        """
        q = (
            select(Device)
            .join(CameraDevice, CameraDevice.device_uuid == Device.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
        )
        devices = (await db.execute(q)).scalars().all()

        if len(devices) == 1:
            dev = devices[0]
            if not getattr(dev, "device_url", None):
                raise ValueError(f"Assigned device has no device_url for camera {camera_uuid}")
            return dev

        if not required and len(devices) == 0:
            return None

        raise ValueError(f"Camera {camera_uuid} must have exactly 1 device assigned, found {len(devices)}")

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


    async def create_pipeline(self, user_id: int | None = None) -> ModelPipeline:
        """
        Creates (loads) the user's default pipeline and builds a config-only ModelPipeline.
        """
        uid = int(user_id or self._default_user_id)

        async with self._lock:
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

                if full_pl and getattr(full_pl, "cameras", None):
                    for cam in full_pl.cameras:
                        enabled = bool(getattr(cam, "is_enabled", True))
                        det_enabled = bool(getattr(cam, "is_detection_enabled", True))

                        devices = list(getattr(cam, "devices", None) or [])
                        if len(devices) != 1:
                            if enabled and det_enabled:
                                raise ValueError(
                                    f"Camera {cam.camera_uuid} must have exactly 1 device assigned, found {len(devices)}"
                                )
                            logger.warning(
                                "Skipping camera %s (enabled=%s detection=%s) because device count=%s",
                                cam.camera_uuid, enabled, det_enabled, len(devices)
                            )
                            continue

                        device = devices[0]
                        d_url = getattr(device, "device_url", None)
                        d_uuid = getattr(device, "device_uuid", None)
                        if not d_url or not d_uuid:
                            raise ValueError(f"Camera {cam.camera_uuid} has invalid device assignment.")

                        cfg_json = {}
                        if getattr(cam, "channel_configuration", None) and getattr(cam.channel_configuration, "configuration", None):
                            cfg_json = cam.channel_configuration.configuration or {}

                        runtime_overrides = _runtime_config_overrides(
                            cfg_json,
                            extra_forbidden={"sample_fps", "decode_backend", "request_timeout_s"},
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
                            **runtime_overrides,
                        )
                        await mp.add_channel(VideoChannel(config=vcc))

                await db.commit()

                self._pipelines_by_user[uid] = mp
                self._pipeline_id_by_user[uid] = pid
                return mp

    async def get_activepipeline(self, user_id: int | None = None) -> ModelPipeline:
        uid = int(user_id or self._default_user_id)
        mp = self._pipelines_by_user.get(uid)
        if mp is None:
            mp = await self.create_pipeline(uid)
        await mp.start()
        return mp

    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID],
        channel_events: List[VideoChannelEvent],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
    ) -> Optional[PipelineUpdateResult]:

        pid = self._as_uuid(pipeline_id, "pipeline_id")
        uid = int(user_id or self._default_user_id)

        # do not call create_pipeline while holding lock
        active = await self.get_activepipeline(uid)

        for attempt in [1, 2]:
            async with self._lock:
                async with self._session_factory() as db:
                    exists = await self._repo.pipeline_exists(db, pid)
                    if not exists:
                        logger.warning("Pipeline %s does not exist in DB for user %s. Invalidating cache (attempt %s).", pid, uid, attempt)
                        self._pipelines_by_user.pop(uid, None)
                        self._pipeline_id_by_user.pop(uid, None)
                        if attempt == 1:
                             # refresh 'active' and 'pid' for retry
                             active = await self.get_activepipeline(uid)
                             pid = self._pipeline_id_by_user.get(uid)
                             if not pid:
                                 return None
                             continue
                        return None

                    cameras_out: List[CameraOut] = []
                    events_out: List[Dict[str, Any]] = []

                    for ev in (channel_events or []):
                        et = getattr(ev, "event_type", None)
                        if et is None and isinstance(ev, dict):
                            et = ev.get("event_type")
                        et_norm = str(et or "").lower()

                        if et_norm == "create_channel" or isinstance(ev, ChannelCreateEvent):
                            cams, evs = await self._add_channel(
                                db, pid=pid, ev=ev, user_id=uid, camera_code_prefix=camera_code_prefix, active=active
                            )
                        elif et_norm == "edit_channel" or isinstance(ev, ChannelEditEvent):
                            cams, evs = await self._edit_channel(db, pid=pid, ev=ev, user_id=uid, active=active)
                        elif et_norm == "remove_channel" or isinstance(ev, ChannelRemoveEvent):
                            cams, evs = await self._remove_channel(db, pid=pid, ev=ev, user_id=uid, active=active)
                        else:
                            logger.warning("Event type not matched: %s", et)
                            continue

                        cameras_out.extend(cams)
                        events_out.extend(evs)

                    await db.commit()

                return PipelineUpdateResult(
                    pipeline_id=pid,
                    active_in_memory=False,
                    cameras=cameras_out,
                    events=events_out,
                )
        return None

    async def reconcile_device_edge_simple(
        self,
        *,
        device_uuid: uuid.UUID,
        user_id: Optional[int] = None,
        dry_run: bool = False,
        delete_unknown: bool = True,
    ) -> Dict[str, List[str]]:
        uid = int(user_id) if user_id is not None else None

        # No self._lock here: reconcile only reads DB state and calls external
        # HTTP APIs — it never touches self._pipelines_by_user.  Holding the
        # lock across 3×15s Jetson retries was blocking manual Sync whenever
        # the background reconcile loop was already running.
        async with self._session_factory() as db:
            dev = await self._get_device(db, device_uuid, user_id=uid)
            edge_set = await self._edge.list_cameras(device_url=dev.device_url)
            webrtc_list = await self._webrtc.list_webrtc_cameras()
            webrtc_set = {
                str(c.get("stream_key"))
                for c in webrtc_list
                if isinstance(c, dict) and c.get("stream_key")
            }
            q = (
                select(Camera)
                .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
                .where(CameraDevice.device_uuid == device_uuid)
                .options(selectinload(Camera.channel_configuration))
            )
            cams = (await db.execute(q)).scalars().all()

            desired_set = {str(c.camera_uuid) for c in cams if c.is_enabled and c.is_detection_enabled}
            known_streams = {str(c.camera_code) for c in cams if c.camera_code}
            active_streams = {str(c.camera_code) for c in cams if c.is_enabled and c.camera_code}
            to_add = sorted(desired_set - edge_set)
            to_remove = sorted(edge_set - desired_set)
            to_add_stream = sorted(active_streams - webrtc_set)
            to_remove_stream = sorted((webrtc_set & known_streams) - active_streams)

            out: Dict[str, List[str]] = {
                "to_add": to_add,
                "to_remove": to_remove,
                "to_add_stream": to_add_stream,
                "to_remove_stream": to_remove_stream,
                "added": [],
                "removed": [],
                "errors": [],
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
                    "camera_uuid": cu,
                    "rtsp_url": cam.rtsp_url,
                    "enabled": True,
                    "detection_enabled": True,
                    "notification_enabled": bool(cam.is_notification_enabled),
                    **_only_jetson_config(cfg),
                }
                try:
                    await self._edge.upsert_camera(device_url=dev.device_url, payload=payload)
                    out["added"].append(cu)
                except Exception as e:
                    logger.warning("Edge upsert failed during reconcile for camera %s", cu, exc_info=True)
                    out["errors"].append(f"Failed to add {cu}: {e}")

            if delete_unknown:
                for cu in to_remove:
                    try:
                        await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=cu)
                        out["removed"].append(cu)
                    except Exception as e:
                        logger.warning("Edge delete failed during reconcile for camera %s", cu, exc_info=True)
                        out["errors"].append(f"Failed to remove {cu}: {e}")
            cams_by_code = {str(c.camera_code): c for c in cams if c.camera_code}

            for cu in to_add_stream:
                cam = cams_by_code.get(cu)
                if cam is None:
                    out["errors"].append(f"Camera not found in DB during reconcile: {cu}")
                    continue
                try:
                    await self._webrtc.ensure_stream(stream_key=cu, rtsp_url=cam.rtsp_url)
                    cam_uuid = str(cam.camera_uuid)
                    if cam_uuid not in out["added"]:
                        out["added"].append(cam_uuid)
                except Exception as e:
                    logger.warning("Edge upsert failed during reconcile for camera %s", cu, exc_info=True)
                    out["errors"].append(f"Failed to add {cu}: {e}")

            if delete_unknown:
                for cu in to_remove_stream:
                    try:
                        await self._webrtc.delete_stream(stream_key=cu)
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
            stmt = select(Device.device_uuid).where(Device.is_enabled.is_(True))
            if uid is not None:
                stmt = stmt.where(Device.user_id == uid)
            rows = await db.execute(stmt)
            device_uuids = [r[0] for r in rows.all()]

        summary: Dict[str, Any] = {
            "user_id": uid,
            "device_count": len(device_uuids),
            "devices": {},
            "errors": [],
        }

        for du in device_uuids:
            key = str(du)
            try:
                result = await self.reconcile_device_edge_simple(
                    device_uuid=du,
                    user_id=uid,
                    dry_run=dry_run,
                    delete_unknown=delete_unknown,
                )
                summary["devices"][key] = result
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
        await self._ensure_site_owned_by_user(db, site_uuid=site_uuid, user_id=user_id)

        device_uuid = patch.get("device_uuid")
        if not device_uuid:
            raise ValueError("Create_Channel requires device_uuid (each camera must have exactly 1 device).")
        device_uuid = self._as_uuid(device_uuid, "device_uuid")
        dev = await self._get_device(db, device_uuid, user_id=user_id)

        cam_uuid = patch.get("camera_uuid") or uuid.uuid4()
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")
        patch["camera_uuid"] = cam_uuid
        patch["channel_id"] = cam_uuid  # keep compatibility

        camera_code = f"{camera_code_prefix}-{cam_uuid.hex[:8]}"
        webrtc_url = await self._webrtc.ensure_stream(stream_key=str(camera_code), rtsp_url=str(rtsp_url))
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

        edge_payload = self._edge_payload_from_config(
            camera_uuid=str(cam.camera_uuid),
            rtsp_url=cam.rtsp_url,
            config={
                **_only_jetson_config(patch),
                "enabled": enabled,
                "detection_enabled": det_enabled,
                "notification_enabled": bool(getattr(cam, "is_notification_enabled", True)),
            },
        )

        # Fire-and-forget: do NOT await the Jetson call inside the lock.
        # If Jetson is unreachable the 3-retry × 15s timeout would hold
        # self._lock for up to 45s, blocking all other operations.
        # The background reconcile loop (every 90s) will catch any failure.
        if enabled and det_enabled:
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
                extra_forbidden={"sample_fps", "decode_backend", "request_timeout_s"},
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
                **runtime_overrides,
            )
            await active.add_channel(VideoChannel(config=vcc))

        cameras_out = [
            CameraOut(
                camera_uuid=cam.camera_uuid,
                camera_code=getattr(cam, "camera_code", None),
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

        old_dev = await self._get_single_camera_device(db, cam_uuid, required=True)

        patch = self._patch_to_dict(getattr(ev, "configs", None))
        patch.pop("webrtc_url", None)
        new_device_uuid = patch.get("device_uuid")
        if new_device_uuid is None:
            new_device_uuid = old_dev.device_uuid
        new_device_uuid = self._as_uuid(new_device_uuid, "device_uuid")

        new_dev = await self._get_device(db, new_device_uuid, user_id=user_id)

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

        if old_dev.device_uuid != new_device_uuid:
            try:
                await self._edge.delete_camera(device_url=old_dev.device_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.warning("Failed removing camera from old device during reassignment", exc_info=True)

            await self._set_single_camera_device(db, cam_uuid, new_device_uuid)

        if cam2.rtsp_url != old_rtsp and cam2.camera_code:
            await self._webrtc.update_stream(stream_key=str(cam2.camera_code), rtsp_url=cam2.rtsp_url)

        enabled = bool(cam2.is_enabled)
        det_enabled = bool(cam2.is_detection_enabled)

        try:
            if enabled and det_enabled:
                if old_dev.device_uuid != new_device_uuid:
                    edge_payload = self._edge_payload_from_config(
                        camera_uuid=str(cam_uuid),
                        rtsp_url=cam2.rtsp_url,
                        config={
                            **_only_jetson_config(merged_cfg),
                            "enabled": enabled,
                            "detection_enabled": det_enabled,
                            "notification_enabled": bool(cam2.is_notification_enabled),
                        },
                    )
                    await self._edge.upsert_camera(device_url=new_dev.device_url, payload=edge_payload)
                else:
                    edge_patch = _only_jetson_config(patch)
                    edge_patch.setdefault("rtsp_url", cam2.rtsp_url)
                    edge_patch["enabled"] = enabled
                    edge_patch["detection_enabled"] = det_enabled
                    edge_patch["notification_enabled"] = bool(cam2.is_notification_enabled)
                    await self._edge.patch_camera(device_url=new_dev.device_url, camera_uuid=str(cam_uuid), patch=edge_patch)
            else:
                await self._edge.delete_camera(device_url=new_dev.device_url, camera_uuid=str(cam_uuid))
        except Exception:
            logger.warning("Edge sync failed during camera edit", exc_info=True)

        if active:
            runtime_overrides = _runtime_config_overrides(merged_cfg)
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
                **runtime_overrides,
            )
            await active.edit_channel(VideoChannel(config=vcc))

        cameras_out = [
            CameraOut(
                camera_uuid=cam2.camera_uuid,
                camera_code=getattr(cam2, "camera_code", None),
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
                dev = await self._get_single_camera_device(db, cam_uuid, required=False)
                if dev and dev.device_url:
                    await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam_uuid))
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
        async with self._lock:
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
        q = select(Site).where(Site.site_uuid == site_uuid, Site.user_id == user_id)
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

                if cam.devices and cam.devices[0].device_url:
                    await self._edge.delete_camera(device_url=cam.devices[0].device_url, camera_uuid=str(cam.camera_uuid))    
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
