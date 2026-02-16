# agents/application/services/agent_manager.py

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import quote
from datetime import datetime, date
import httpx
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository
from core.database_orm import Device, Camera, CameraDevice  # ✅ removed SiteDevice
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent, VideoChannelEvent
from domain.model_pipeline import ModelPipeline
from application.channels.channel_config import VideoChannelConfig
from application.channels.channel import VideoChannel
logger = logging.getLogger(__name__)

# -------------------------
# API DTOs
# -------------------------
def to_jsonable(obj):
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[str(k)] = to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj
    
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

class WebRTCGatewayClient:
    """
    Provisions (or updates) RTSP->WebRTC streams on an Azure-hosted gateway (MediaMTX/go2rtc/etc).

    We support two modes:
      1) Admin API available -> call it to upsert streams.
      2) No admin API -> derive a stable public webrtc_url from WEBRTC_PUBLIC_BASE_URL + stream_key.

    Environment:
      - WEBRTC_ADMIN_API_URL (optional)
      - WEBRTC_ADMIN_UPSERT_PATH (default: /streams)
      - WEBRTC_ADMIN_UPDATE_PATH (default: /streams/{stream_key})
      - WEBRTC_ADMIN_DELETE_PATH (default: /streams/{stream_key})
      - WEBRTC_PUBLIC_BASE_URL (required for derivation if admin doesn't return a url)
      - WEBRTC_ADMIN_API_KEY (optional header: x-api-key)
    """

    def __init__(self):
        # Default to MediaMTX API localhost if not set
        self.admin_api_url = (os.getenv("WEBRTC_ADMIN_API_URL") or "https://noentrymtxfdxidm.centralus.azurecontainer.io:9997").rstrip("/")
        
        # Public URL for the frontend to consume (e.g. port 8889 for WebRTC)
        pub_host = os.getenv("PUBLIC_HOST", "localhost")
        pub_scheme = os.getenv("PUBLIC_SCHEME", "http")
        pub_port = os.getenv("WEBRTC_HTTP_PORT", "8889")
        self.public_base = (os.getenv("WEBRTC_PUBLIC_BASE_URL") or f"{pub_scheme}://{pub_host}:{pub_port}").rstrip("/")

        self.api_user = os.getenv("MTX_API_USER", "api")
        self.api_pass = os.getenv("MTX_API_PASS", "api_pass_123")

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()
    
    def _auth(self) -> Tuple[str, str]:
        return (self.api_user, self.api_pass)

    def _derive_public_webrtc_url(self, stream_key: str) -> str:
        # This opens MediaMTX’s built-in WebRTC player page
        return f"{self.public_base}/{stream_key}"


    async def ensure_stream(self, *, stream_key: str, rtsp_url: str) -> Optional[str]:
        """
        Ensure stream exists in MediaMTX. Returns webrtc_url (stable).
        """
        if not self.admin_api_url:
            return self._derive_public_webrtc_url(stream_key)

        # MediaMTX v3: /v3/config/paths/add/{name}
        safe_name = quote(stream_key, safe="")
        add_url = f"{self.admin_api_url}/v3/config/paths/add/{safe_name}"
        payload = {"source": rtsp_url, "rtspTransport": "tcp"}

        # 1. Try Add
        try:
            r = await self._client.post(add_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                return self._derive_public_webrtc_url(stream_key)
        except Exception:
            logger.warning("MediaMTX add request failed, trying patch or ignoring", exc_info=True)

        # 2. If add failed (likely 400 exists), try Patch
        patch_url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        try:
            r = await self._client.patch(patch_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                return self._derive_public_webrtc_url(stream_key)
        except Exception:
            logger.error("MediaMTX patch request failed", exc_info=True)
            
        # Fallback: return derived URL anyway, optimization
        return self._derive_public_webrtc_url(stream_key)

    async def update_stream(self, *, stream_key: str, rtsp_url: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        payload = {"source": rtsp_url}
        
        try:
            await self._client.patch(url, json=payload, auth=self._auth())
        except Exception:
             logger.error(f"MediaMTX update failed for {stream_key}", exc_info=True)

    async def delete_stream(self, *, stream_key: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/delete/{safe_name}"
        
        try:
            await self._client.delete(url, auth=self._auth())
        except Exception:
             logger.warning(f"MediaMTX delete failed for {stream_key}", exc_info=True)


class EdgeInferenceClient:
    """
    Talks to the Jetson TensorRT (inference) service.

    Environment (defaults are guesses; set them to match your Jetson routes):
      - EDGE_ADD_PATH (default: /api/cameras)
      - EDGE_PATCH_PATH (default: /api/cameras/{camera_uuid})
      - EDGE_DELETE_PATH (default: /api/cameras/{camera_uuid})
      - EDGE_API_KEY (optional header: x-api-key)

    Expected semantics on Jetson:
      - POST   add/upsert a camera by camera_uuid
      - PATCH  update config for camera_uuid
      - DELETE remove camera_uuid
    """

    def __init__(self):
        self.add_path = os.getenv("EDGE_ADD_PATH", "/cameras")
        self.patch_path = os.getenv("EDGE_PATCH_PATH", "/cameras/{camera_uuid}")
        self.delete_path = os.getenv("EDGE_DELETE_PATH", "/cameras/{camera_uuid}")
        self.api_key = os.getenv("EDGE_API_KEY")

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()



    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    async def get_health(self, *, device_url: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort health probe.
        Tries /health then /api/health and returns parsed JSON payload.
        Accepts 503 responses too if they include structured readiness details.
        """
        base = device_url.rstrip("/")
        urls = [
            "{}/health".format(base),
            "{}/api/health".format(base),
        ]
        for url in urls:
            try:
                r = await self._client.get(url, headers=self._headers())
                data = r.json()
                if isinstance(data, dict):
                    if any(k in data for k in ("ok", "pipeline_ready", "startup_error")):
                        return data
                    if r.status_code < 400:
                        return data
            except Exception:
                continue
        return None

    async def ensure_pipeline_ready(self, *, device_url: str) -> None:
        """
        Raise a clear error if edge reports startup failure / not-ready state.
        If health endpoint is unavailable, this is a no-op (backward compatible).
        """
        h = await self.get_health(device_url=device_url)
        if not isinstance(h, dict):
            return

        # Support both older {"ok": true} and richer payloads.
        pipeline_ready = h.get("pipeline_ready")
        ok = h.get("ok")
        if pipeline_ready is False or ok is False:
            startup_error = h.get("startup_error")
            if startup_error:
                raise RuntimeError(
                    "Edge pipeline not ready at {} (startup_error: {})".format(device_url, startup_error)
                )
            raise RuntimeError("Edge pipeline not ready at {}".format(device_url))

    async def upsert_camera(self, *, device_url: str, payload: dict) -> None:
        url = f"{device_url.rstrip('/')}{self.add_path}"
        await self._request("POST", url, json=payload)

    async def patch_camera(self, *, device_url: str, camera_uuid: str, patch: dict) -> None:
        """
        Best-effort patch. If PATCH is not supported by the Jetson service, fallback to POST upsert.
        """
        url = f"{device_url.rstrip('/')}{self.patch_path.format(camera_uuid=camera_uuid)}"
        try:
            await self._request("PATCH", url, json=patch)
        except Exception:
            # fallback to upsert
            upsert_url = f"{device_url.rstrip('/')}{self.add_path}"
            await self._request("POST", upsert_url, json=patch)

    async def delete_camera(self, *, device_url: str, camera_uuid: str) -> None:
        url = f"{device_url.rstrip('/')}{self.delete_path.format(camera_uuid=camera_uuid)}"
        await self._request("DELETE", url)

    async def _request(self, method: str, url: str, *, json: Optional[dict] = None) -> None:
        last_exc: Optional[Exception] = None
        json_payload = to_jsonable(json) if json is not None else None
        for attempt in range(3):
            try:
                r = await self._client.request(method, url, headers=self._headers(), json=json_payload)
                # treat 404 delete as ok (idempotent)
                if method == "DELETE" and r.status_code == 404:
                    return
                if r.status_code >= 400:
                    raise RuntimeError(f"Edge service error {r.status_code}: {r.text[:300]}")
                return
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.2 * (2 ** attempt))
        raise last_exc or RuntimeError("Edge service request failed")


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

    async def _get_device(self, db: AsyncSession, device_uuid: uuid.UUID) -> Device:
        dev = (await db.execute(select(Device).where(Device.device_uuid == device_uuid))).scalar_one_or_none()
        if dev is None:
            raise ValueError(f"Device not found: {device_uuid}")
        if not getattr(dev, "device_url", None):
            raise ValueError(f"Device missing device_url: {device_uuid}")
        return dev

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
                            request_timeout_s=float(cfg_json.get("request_timeout_s", 2.0)),
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
        if mp is not None:
            return mp
        return await self.create_pipeline(uid)

    # -------------------------
    # Update pipeline from events
    # -------------------------
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
                            cams, evs = await self._edit_channel(db, pid=pid, ev=ev, active=active)
                        elif et_norm == "remove_channel" or isinstance(ev, ChannelRemoveEvent):
                            cams, evs = await self._remove_channel(db, pid=pid, ev=ev, active=active)
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

    # -------------------------
    # Event handlers (UPDATED)
    # -------------------------
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

        device_uuid = patch.get("device_uuid")
        if not device_uuid:
            raise ValueError("Create_Channel requires device_uuid (each camera must have exactly 1 device).")
        device_uuid = self._as_uuid(device_uuid, "device_uuid")
        dev = await self._get_device(db, device_uuid)

        cam_uuid = patch.get("camera_uuid") or uuid.uuid4()
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")
        patch["camera_uuid"] = cam_uuid
        patch["channel_id"] = cam_uuid  # keep compatibility

        camera_code = f"{camera_code_prefix}-{uuid.uuid4().hex[:8]}"
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

        try:
            if enabled and det_enabled:
                await self._edge.upsert_camera(device_url=dev.device_url, payload=edge_payload)
            else:
                await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam.camera_uuid))
        except Exception:
            logger.warning("Edge sync failed during camera creation", exc_info=True)

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
                request_timeout_s=float((cfg_json or {}).get("request_timeout_s", patch.get("request_timeout_s", 2.0))),
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
        if existing_pid is not None and existing_pid != pid:
            raise ValueError("Camera does not belong to provided pipeline_id")

        old_rtsp = cam_db.rtsp_url
        old_webrtc = cam_db.webrtc_url

        # ✅ must have exactly one device already
        old_dev = await self._get_single_camera_device(db, cam_uuid, required=True)

        patch = self._patch_to_dict(getattr(ev, "configs", None))
        patch.pop("webrtc_url", None)  # we keep existing / stable mapping

        # device_uuid is optional on edit; if omitted, keep existing
        new_device_uuid = patch.get("device_uuid")
        if new_device_uuid is None:
            new_device_uuid = old_dev.device_uuid
        new_device_uuid = self._as_uuid(new_device_uuid, "device_uuid")

        new_dev = await self._get_device(db, new_device_uuid)

        # merge config json
        merged_cfg: Dict[str, Any] = {}
        if chan_cfg_db and getattr(chan_cfg_db, "configuration", None):
            merged_cfg.update(chan_cfg_db.configuration or {})
        merged_cfg.update(patch)

        # write camera + channel_config
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
            # remove from old device first (best-effort)
            try:
                await self._edge.delete_camera(device_url=old_dev.device_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.warning("Failed removing camera from old device during reassignment", exc_info=True)

            await self._set_single_camera_device(db, cam_uuid, new_device_uuid)

        # update WebRTC source if rtsp changed
        if cam2.rtsp_url != old_rtsp and cam2.camera_code:
            await self._webrtc.update_stream(stream_key=str(cam2.camera_code), rtsp_url=cam2.rtsp_url)

        enabled = bool(cam2.is_enabled)
        det_enabled = bool(cam2.is_detection_enabled)

        # send to edge
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

        # update cached model pipeline
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
        active: Optional[ModelPipeline],
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
        if cam_uuid is None:
            raise ValueError("Remove_Channel missing channel_id (camera_uuid).")
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")

        full = await self.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if full:
            cam_db, _cfg, existing_pid = full
            if existing_pid is not None and existing_pid != pid:
                raise ValueError("Camera does not belong to provided pipeline_id")

            # delete from edge (best-effort)
            try:
                dev = await self._get_single_camera_device(db, cam_uuid, required=False)
                if dev and dev.device_url:
                    await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.warning("Edge delete failed during camera removal", exc_info=True)

            # delete WebRTC stream mapping
            try:
                if cam_db.camera_code:
                    await self._webrtc.delete_stream(stream_key=str(cam_db.camera_code))
            except Exception:
                logger.warning("WebRTC delete failed during camera removal", exc_info=True)

            await self.channel_repo.delete_camera(db, camera_uuid=cam_db.camera_uuid)

        # update cached model pipeline
        if active:
            try:
                await active.remove_channel(cam_uuid)
            except Exception:
                logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

        events_out = [{"event_type": "Remove_Channel", "camera_uuid": str(cam_uuid)}]
        return [], events_out

    async def cleanup_device_resources(self, db: AsyncSession, *, device_uuid: uuid.UUID) -> None:
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
                except Exception:
                    logger.warning("Failed cleanup camera %s on device %s", cam.camera_uuid, dev.device_uuid, exc_info=True)
        except Exception:
            logger.exception("Error during device cleanup for %s", device_uuid)
