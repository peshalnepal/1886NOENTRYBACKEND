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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository
from core.database_orm import Device, SiteDevice  # used to auto-pick a device
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent, VideoChannelEvent
from domain.model_pipeline import ModelPipeline
from application.channels.channel_config import VideoChannelConfig
from application.channels.channel import VideoChannel
logger = logging.getLogger(__name__)

# -------------------------
# API DTOs
# -------------------------
def to_jsonable(obj):
    # UUID -> str
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # datetime/date -> ISO
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # pydantic model -> dict
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)

    # dict -> recurse
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                continue
            out[str(k)] = to_jsonable(v)
        return out

    # list/tuple/set -> list
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

    # assignment (one primary device for inference)
    primary_device_uuid: Optional[uuid.UUID] = None
    primary_device_url: Optional[str] = None

    # stored tuning knobs (optional)
    sample_fps: float = Field(default=5.0, ge=0.1, description="Frames/sec to publish as RTSPEvent on Jetson")
    decode_backend: str = Field(default="gstreamer")
    resize: Optional[Tuple[int, int]] = None
    emit_format: str = Field(default="raw")
    jpeg_quality: int = Field(default=80, ge=1, le=100)


class PipelineUpdateResult(BaseModel):
    pipeline_id: uuid.UUID
    active_in_memory: bool = False  # Azure no longer runs the RTSP ingest pipeline in-memory
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
    "primary_device_uuid",
    "device_uuids",
    "user_id",
    "roi",
}

RUNTIME_CONFIG_ALLOWED_KEYS = set(VideoChannelConfig.model_fields.keys())


def _runtime_config_overrides(
    cfg: Dict[str, Any],
    *,
    extra_forbidden: Optional[set] = None,
) -> Dict[str, Any]:
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
      - DB is source of truth (Azure SQL/Postgres)
      - WebRTC gateway provides playback URL (Azure service)
      - Jetson device provides detections (TensorRT service)

    This manager no longer runs a local RTSP ingest pipeline.
    """

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._lock = asyncio.Lock()

        # DB repos
        self._repo = PipelineRepository()
        self.channel_repo = ChannelRepository()

        # External provisioning
        self._webrtc = WebRTCGatewayClient()
        self._edge = EdgeInferenceClient()

        self._active_user_id: Optional[int] = None
        self._active_id: Optional[uuid.UUID] = None
        self._active_pipeline: Optional[ModelPipeline] = None
        self._default_user_id = int(os.getenv("DEFAULT_USER_ID", "1"))

    async def shutdown(self) -> None:
        async with self._lock:
            if self._active_pipeline:
                await self._active_pipeline.shutdown()
            self._active_pipeline = None
            self._active_id = None
        await self._webrtc.close()
        await self._edge.close()

    async def create_pipeline(self, user_id: int | None = None) -> ModelPipeline:
        """
        Creates (loads) the single active pipeline for this user and populates channels from DB.
        One user -> one persistent pipeline row (name='default').
        """
        uid = int(user_id or self._default_user_id)

        # Fast path (no lock)
        if self._active_pipeline is not None and self._active_id is not None and self._active_user_id == uid:
            return self._active_pipeline

        async with self._lock:
            # Double-check inside lock
            if self._active_pipeline is not None and self._active_id is not None and self._active_user_id == uid:
                return self._active_pipeline

            # If something exists but for another user (future-proofing)
            if self._active_pipeline is not None and self._active_user_id is not None and self._active_user_id != uid:
                try:
                    await self._active_pipeline.shutdown()
                except Exception:
                    logger.warning("Failed to shutdown previous active pipeline", exc_info=True)
                self._active_pipeline = None
                self._active_id = None
                self._active_user_id = None

            async with self._session_factory() as db:
                # 1) Ensure (user_id, name='default') pipeline exists (stable)
                pipeline_row = await self._repo.upsert_pipeline(
                    db,
                    user_id=uid,
                    pipeline_id=None,
                    name="default",
                    is_active=True,
                )
                pid = pipeline_row.id
                logger.info("Loaded persistent pipeline user_id=%s pipeline_id=%s", uid, pid)

                # 2) Load full pipeline (cameras + devices + configs)
                full_pl = await self._repo.get_full_pipeline(db, pid)

                # 3) Build in-memory handle (Azure side does not ingest RTSP; this is config/runtime helper)
                mp = ModelPipeline(pipeline_id=pid)

                if full_pl and getattr(full_pl, "cameras", None):
                    for cam in full_pl.cameras:
                        primary_dev = None
                        if getattr(cam, "camera_devices", None):
                            for cd in cam.camera_devices:
                                if getattr(cd, "is_primary", False):
                                    primary_dev = cd.device
                                    break
                            if primary_dev is None and cam.camera_devices:
                                primary_dev = cam.camera_devices[0].device

                        d_url = getattr(primary_dev, "device_url", None) if primary_dev else None
                        d_uuid = getattr(primary_dev, "device_uuid", None) if primary_dev else None

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
                            enabled=bool(getattr(cam, "is_enabled", True)),
                            sample_fps=float(cfg_json.get("sample_fps", 5.0)),
                            decode_backend=str(cfg_json.get("decode_backend", "gstreamer")),
                            request_timeout_s=float(cfg_json.get("request_timeout_s", 2.0)),
                            **runtime_overrides,
                        )

                        await mp.add_channel(VideoChannel(config=vcc))

                # Commit only after we’ve built the in-memory view (avoids expire-on-commit surprises)
                await db.commit()

            self._active_user_id = uid
            self._active_id = pid
            self._active_pipeline = mp
            return mp


    async def get_activepipeline(self) -> ModelPipeline:
        return await self.create_pipeline()

    def _patch_to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            out: Dict[str, Any] = {}
        elif hasattr(obj, "model_dump"):
            out = obj.model_dump(exclude_unset=True, exclude_none=True)
        elif isinstance(obj, dict):
            out = {k: v for k, v in obj.items() if v is not None}
        else:
            out = {}

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

    async def update_pipeline(
        self,
        pipeline_id: Union[str, uuid.UUID],
        channel_events: List[VideoChannelEvent],
        *,
        user_id: Optional[int] = None,
        camera_code_prefix: str = "cam",
    ) -> Optional[PipelineUpdateResult]:

        pid = self._as_uuid(pipeline_id, "pipeline_id")
        
        # Ensure active pipeline is loaded so we can update in-memory state too
        active = await self.get_activepipeline()
        
        async with self._lock:
            async with self._session_factory() as db:
                exists = await self._repo.pipeline_exists(db, pid)
                logger.info(f"Pipeline exists check: {pid} -> {exists}")
                if not exists:
                    logger.warning(f"Pipeline {pid} does not exist in DB.")
                    return None
               
                cameras_out: List[CameraOut] = []
                events_out: List[Dict[str, Any]] = []

                logger.info(f"Processing {len(channel_events or [])} events")

                for ev in (channel_events or []):
                    try:
                        et = getattr(ev, "event_type", None)
                        if et is None and isinstance(ev, dict):
                            et = ev.get("event_type")
                        
                        logger.info(f"Processing event: {ev}, et={et}, type={type(ev)}, dir={dir(ev)}")
                        logger.info(f"Event dict: {ev.__dict__ if hasattr(ev, '__dict__') else 'no __dict__'}")

                        if et == "Create_Channel" or str(et).lower() == "create_channel" or isinstance(ev, ChannelCreateEvent):
                            logger.info("Matched Create_Channel event")
                            cams, evs = await self._add_channel(db, pid=pid, ev=ev, user_id=user_id, camera_code_prefix=camera_code_prefix)
                            
                        elif et == "Edit_Channel" or str(et).lower() == "edit_channel" or isinstance(ev, ChannelEditEvent):
                            cams, evs = await self._edit_channel(db, pid=pid, ev=ev)
                        elif et == "Remove_Channel" or str(et).lower() == "remove_channel" or isinstance(ev, ChannelRemoveEvent):
                            cams, evs = await self._remove_channel(db, pid=pid, ev=ev)
                            if active:
                                 for e in evs:
                                     await active.remove_channel(e["camera_uuid"])

                        else:
                            logger.warning(f"Event type not matched: et={et}, ev={ev}")
                            continue

                        cameras_out.extend(cams)
                        events_out.extend(evs)
                        logger.info(f"Added {len(cams)} cameras to output, total now: {len(cameras_out)}")
                    except Exception as e:
                        logger.exception(f"Error processing individual event {ev}: {e}")
                        # Depending on policy, we might want to continue or raise.
                        # For creation, failing the whole batch is safer.
                        raise

                await db.commit()
                logger.info(f"DB committed. Returning {len(cameras_out)} cameras.")
                
            return PipelineUpdateResult(
                pipeline_id=pid,
                active_in_memory=False,
                cameras=cameras_out,
                events=events_out,
            )


    async def _pick_site_device_uuid(self, db: AsyncSession, site_uuid: uuid.UUID) -> Optional[uuid.UUID]:
        """
        Pick one device linked to the site (first match).
        You can refine ordering/policy later (prefer enabled, lowest load, etc.).
        """
        q = select(SiteDevice.device_uuid).where(SiteDevice.site_uuid == site_uuid)
        return (await db.execute(q)).scalars().first()

    async def _get_device(self, db: AsyncSession, device_uuid: uuid.UUID) -> Device:
        dev = (await db.execute(select(Device).where(Device.device_uuid == device_uuid))).scalar_one_or_none()
        if dev is None:
            raise ValueError(f"Device not found: {device_uuid}")
        if not getattr(dev, "device_url", None):
            raise ValueError(f"Device missing device_url: {device_uuid}")
        return dev

    def _edge_payload_from_config(self, *, camera_uuid: str, rtsp_url: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the payload you send to Jetson.

        Keep it generous; Jetson can ignore unknown fields.
        """
        payload = dict(config)
        payload.update(
            {
                "camera_uuid": camera_uuid,
                "rtsp_url": rtsp_url,
            }
        )
        return payload

    async def _add_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: Optional[int],
        camera_code_prefix: str,
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:
        logger.info(f"_add_channel called with pid={pid}, user_id={user_id}")
        patch = self._patch_to_dict(getattr(ev, "configs", None))
        logger.info(f"Extracted patch: {patch}")

        if user_id is None:
            raise ValueError("user_id is required when creating a new camera.")

        rtsp_url = patch.get("rtsp_url")
        if not rtsp_url:
            raise ValueError("Create_Channel requires rtsp_url")

        site_uuid = patch.get("site_uuid")
        if not site_uuid:
            raise ValueError("Create_Channel requires site_uuid")
        site_uuid = self._as_uuid(site_uuid, "site_uuid")

        # choose / validate device assignment
        primary_device_uuid = patch.get("primary_device_uuid") or patch.get("device_uuid")
        if primary_device_uuid:
            primary_device_uuid = self._as_uuid(primary_device_uuid, "primary_device_uuid")
        else:
            primary_device_uuid = await self._pick_site_device_uuid(db, site_uuid)
            if primary_device_uuid is None:
                raise ValueError("No device linked to this site. Link a Device to the Site first.")

        # pick camera_uuid deterministically (so we can provision WebRTC with the same key)
        cam_uuid = patch.get("camera_uuid") or uuid.uuid4()
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")
        patch["camera_uuid"] = cam_uuid
        patch["channel_id"]=cam_uuid
        camera_code = f"{camera_code_prefix}-{uuid.uuid4().hex[:8]}"

        webrtc_url = await self._webrtc.ensure_stream(stream_key=str(camera_code), rtsp_url=str(rtsp_url))

        # 2) persist DB (webrtc_url stored once; immutable on edit)
        cam, cfg_json, tz = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config=patch,
            user_id=user_id,
            cam_uuid=cam_uuid,
            camera_code=camera_code,
            site_uuid=site_uuid,
            webrtc_url=webrtc_url,
            device_uuids=[primary_device_uuid],
            primary_device_uuid=primary_device_uuid,
        )

        # 3) provision Jetson edge inference
        dev = await self._get_device(db, primary_device_uuid)

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
        logger.info("######################### Edge Payload ################################")
        logger.info(edge_payload)
        logger.info("#########################################################")
        if enabled and det_enabled:
            # Fail early with a clear edge startup error, if exposed by edge health.
            await self._edge.upsert_camera(device_url=dev.device_url, payload=edge_payload)
        else:
            await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam.camera_uuid))

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
                primary_device_uuid=primary_device_uuid,
                primary_device_url=dev.device_url,
                sample_fps=float(patch.get("sample_fps", 5.0)),
                decode_backend=str(patch.get("decode_backend", "gstreamer")),
                resize=patch.get("resize"),
                emit_format=str(patch.get("emit_format", "raw")),
                jpeg_quality=int(patch.get("jpeg_quality", 80)),
                roi=cam.roi,
                configuration=cfg_json,
                timezone=tz,
            )
        ]

        # Sync to in-memory active pipeline
        active = self._active_pipeline
        if active:
            try:
                # VideoChannelConfig has 'enabled', 'detection_enabled', etc.
                # but 'patch' might have 'is_enabled', etc. from the client schema.
                vcc_data = {
                    "camera_uuid": cam.camera_uuid,
                    "rtsp_url": cam.rtsp_url,
                    "webrtc_url": cam.webrtc_url or "",
                    "site_uuid": cam.site_uuid,
                    "device_uuid": primary_device_uuid,
                    "device_url": dev.device_url,
                    "enabled": enabled,
                    "detection_enabled": det_enabled,
                    "notification_enabled": bool(getattr(cam, "is_notification_enabled", True)),
                }
                
                # Add other fields from patch, making sure we don't duplicate or use wrong names
                for k, v in patch.items():
                    # Map is_enabled -> enabled etc if they are in the patch
                    if k == "is_enabled" and "enabled" not in vcc_data: vcc_data["enabled"] = v
                    elif k == "is_detection_enabled" and "detection_enabled" not in vcc_data: vcc_data["detection_enabled"] = v
                    elif k == "is_notification_enabled" and "notification_enabled" not in vcc_data: vcc_data["notification_enabled"] = v

                
                vcc = VideoChannelConfig(**vcc_data)
                await active.add_channel(VideoChannel(config=vcc))
                logger.info(f"Added camera {cam.camera_uuid} to in-memory pipeline")
            except Exception as e:
                logger.exception(f"Failed to add camera to in-memory pipeline: {e}")
                # We don't want to fail the whole API call if just the in-memory update fails
                # since it's already in the DB.

        events_out = [
            {
                "event_type": "Create_Channel",
                "camera_uuid": str(cam.camera_uuid),
                "site_uuid": str(cam.site_uuid),
                "primary_device_uuid": str(primary_device_uuid),
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
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None)
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
        old_primary_dev = await self.channel_repo.get_primary_device(db, camera_uuid=cam_uuid)
        patch = self._patch_to_dict(getattr(ev, "configs", None))
        patch.pop("webrtc_url", None)
        new_primary_device_uuid = patch.get("primary_device_uuid") or patch.get("device_uuid")
        if new_primary_device_uuid:
            new_primary_device_uuid = self._as_uuid(new_primary_device_uuid, "primary_device_uuid")
        else:
            new_primary_device_uuid = old_primary_dev.device_uuid if old_primary_dev else None
        merged_cfg: Dict[str, Any] = {}
        if chan_cfg_db and getattr(chan_cfg_db, "configuration", None):
            merged_cfg.update(chan_cfg_db.configuration)
        merged_cfg.update(patch)

        cam2, cfg_json, tz = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config={
                **merged_cfg,
                "camera_uuid": cam_uuid,
                "webrtc_url":old_webrtc,
                "rtsp_url": merged_cfg.get("rtsp_url", old_rtsp),
                "enabled": merged_cfg.get("enabled", cam_db.is_enabled),
                "detection_enabled": merged_cfg.get("detection_enabled", cam_db.is_detection_enabled),
                "notification_enabled": merged_cfg.get("notification_enabled", cam_db.is_notification_enabled),
            },
            site_uuid=cam_db.site_uuid,
            webrtc_url=old_webrtc,
            device_uuids=[d for d in ([new_primary_device_uuid] if new_primary_device_uuid else [])] or None,
            primary_device_uuid=new_primary_device_uuid,
        )

        # WebRTC gateway: update mapping if rtsp_url changed, but keep the same webrtc_url
        if cam2.rtsp_url != old_rtsp:
            await self._webrtc.update_stream(stream_key=str(cam2.camera_code), rtsp_url=cam2.rtsp_url)

        enabled = bool(cam2.is_enabled)
        det_enabled = bool(cam2.is_detection_enabled)
        
        dev = None
        if old_primary_dev and new_primary_device_uuid and old_primary_dev.device_uuid != new_primary_device_uuid:
            try:
                await self._edge.delete_camera(device_url=old_primary_dev.device_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.exception("Failed removing camera from old device during reassignment")

            new_dev = await self._get_device(db, new_primary_device_uuid)
            edge_payload = self._edge_payload_from_config(
                camera_uuid=str(cam_uuid),
                rtsp_url=cam2.rtsp_url,
                config={
                    **merged_cfg,
                    "enabled": enabled,
                    "detection_enabled": det_enabled,
                    "notification_enabled": bool(cam2.is_notification_enabled),
                },
            )
            if enabled and det_enabled:
                await self._edge.upsert_camera(device_url=new_dev.device_url, payload=edge_payload)
            else:
                await self._edge.delete_camera(device_url=new_dev.device_url, camera_uuid=str(cam_uuid))
            primary_device_url = new_dev.device_url
            dev = new_dev
        else:
            # same device
            dev = old_primary_dev
            if dev is None and new_primary_device_uuid is not None:
                dev = await self._get_device(db, new_primary_device_uuid)

            if dev is None:
                raise ValueError("Camera has no assigned device. Assign a Device to this camera.")

            edge_patch = _only_jetson_config(patch)
            if "rtsp_url" not in edge_patch:
                edge_patch["rtsp_url"] = cam2.rtsp_url
            edge_patch["enabled"] = enabled
            edge_patch["detection_enabled"] = det_enabled
            edge_patch["notification_enabled"] = bool(cam2.is_notification_enabled)

            if enabled and det_enabled:
                await self._edge.patch_camera(device_url=dev.device_url, camera_uuid=str(cam_uuid), patch=edge_patch)
            else:
                await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam_uuid))

            primary_device_url = dev.device_url
            new_primary_device_uuid = dev.device_uuid

        # Sync in-memory
        active = self._active_pipeline
        ch_in_mem = await active.get_channel_config(cam_uuid) if active else None

        if active:
             # Just replace it
            runtime_overrides = _runtime_config_overrides(merged_cfg)
            vcc = VideoChannelConfig(
                camera_uuid=cam2.camera_uuid,
                rtsp_url=cam2.rtsp_url,
                webrtc_url=cam2.webrtc_url or "",
                site_uuid=cam2.site_uuid,
                device_uuid=new_primary_device_uuid,
                device_url=dev.device_url,
                enabled=enabled,
                detection_enabled=det_enabled,
                notification_enabled=bool(cam2.is_notification_enabled),
                **runtime_overrides
            )
            # edit_channel in ModelPipeline replaces it
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
                primary_device_uuid=new_primary_device_uuid,
                primary_device_url=primary_device_url,
                sample_fps=float(merged_cfg.get("sample_fps", 5.0)),
                decode_backend=str(merged_cfg.get("decode_backend", "gstreamer")),
                resize=merged_cfg.get("resize"),
                emit_format=str(merged_cfg.get("emit_format", "raw")),
                jpeg_quality=int(merged_cfg.get("jpeg_quality", 80)),
                roi=cam2.roi,
                configuration=cfg_json,
                timezone=tz,
            )
        ]

        events_out = [{
            "event_type": "Edit_Channel",
            "camera_uuid": str(cam2.camera_uuid),
            "rtsp_url": cam2.rtsp_url,
            "webrtc_url": cam2.webrtc_url,
            "primary_device_uuid": str(new_primary_device_uuid) if new_primary_device_uuid else None,
            "patch": patch,
        }]

        return cameras_out, events_out

    async def cleanup_device_resources(self, db: AsyncSession, *, device_uuid: uuid.UUID) -> None:
        """
        Called when a Device is about to be deleted.
        Finds all cameras on this device and sends DELETE to the edge service.
        """
        try:
            # We need the device itself to get the URL
            dev = await self._get_device(db, device_uuid)
            if not dev.device_url:
                return

            # Find all cameras linked to this device
            q = (
                select(Camera)
                .join(CameraDevice, CameraDevice.camera_uuid == Camera.camera_uuid)
                .where(CameraDevice.device_uuid == device_uuid)
            )
            cameras_on_device = (await db.execute(q)).scalars().all()

            for cam in cameras_on_device:
                try:
                    logger.info(f"Cleaning up camera {cam.camera_uuid} from device {device_uuid} before deletion")
                    await self._edge.delete_camera(
                        device_url=dev.device_url,
                        camera_uuid=str(cam.camera_uuid),
                    )
                except Exception:
                    logger.warning(
                        f"Failed to cleanup camera {cam.camera_uuid} on device {dev.device_uuid}",
                        exc_info=True,
                    )
        except Exception:
            logger.exception(f"Error during device cleanup for {device_uuid}")

    async def _remove_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None)
        if cam_uuid is None:
            raise ValueError("Remove_Channel missing channel_id (camera_uuid).")
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")

        full = await self.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if full:
            cam_db, _cfg, existing_pid = full
            if existing_pid is not None and existing_pid != pid:
                raise ValueError("Camera does not belong to provided pipeline_id")

            # attempt external cleanup (best-effort)
            try:
                # NEW: iterate all associated devices, not just primary
                devices = await self.channel_repo.get_associated_devices(db, camera_uuid=cam_uuid)
                for dev in devices:
                    if dev.device_url:
                        try:
                            # We don't want one failure to stop others
                            await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam_uuid))
                        except Exception:
                            logger.warning(f"Failed removing camera {cam_uuid} from device {dev.device_url}", exc_info=True)
            except Exception:
                logger.exception("Edge delete failed during camera removal")

            try:
                # Use camera_code as stream key for WebRTC cleanup
                if cam_db.camera_code:
                    await self._webrtc.delete_stream(stream_key=str(cam_db.camera_code))
            except Exception:
                logger.exception("WebRTC delete failed during camera removal")

            await self.channel_repo.delete_camera(db, camera_uuid=cam_db.camera_uuid)

        events_out = [{"event_type": "Remove_Channel", "camera_uuid": str(cam_uuid)}]
        return [], events_out
