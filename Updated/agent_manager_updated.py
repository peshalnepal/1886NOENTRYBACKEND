# agents/application/services/agent_manager.py

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.pipeline_repository import PipelineRepository
from application.repositories.channel_repository import ChannelRepository
from core.database_orm import Device, SiteDevice  # used to auto-pick a device
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent, VideoChannelEvent
from domain.model_pipeline import ModelPipeline

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
        self.admin_api_url = (os.getenv("WEBRTC_ADMIN_API_URL") or "").rstrip("/")
        self.public_base = (os.getenv("WEBRTC_PUBLIC_BASE_URL") or "").rstrip("/")
        self.api_key = os.getenv("WEBRTC_ADMIN_API_KEY")

        self.upsert_path = os.getenv("WEBRTC_ADMIN_UPSERT_PATH", "/streams")
        self.update_path = os.getenv("WEBRTC_ADMIN_UPDATE_PATH", "/streams/{stream_key}")
        self.delete_path = os.getenv("WEBRTC_ADMIN_DELETE_PATH", "/streams/{stream_key}")

        if not self.admin_api_url and not self.public_base:
            logger.warning(
                "WebRTC gateway not configured: set WEBRTC_ADMIN_API_URL or WEBRTC_PUBLIC_BASE_URL"
            )

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _derive_public_webrtc_url(self, stream_key: str) -> Optional[str]:
        if not self.public_base:
            return None
        return f"{self.public_base}/{stream_key}"

    async def ensure_stream(self, *, stream_key: str, rtsp_url: str) -> Optional[str]:
        """
        Ensure stream exists. Returns webrtc_url (stable).
        """
        if not self.admin_api_url:
            return self._derive_public_webrtc_url(stream_key)

        url = f"{self.admin_api_url}{self.upsert_path}"
        payload = {"stream_key": stream_key, "rtsp_url": rtsp_url}

        data = await self._request_json("POST", url, json=payload)
        webrtc_url = None
        if isinstance(data, dict):
            webrtc_url = data.get("webrtc_url") or data.get("url")
        return webrtc_url or self._derive_public_webrtc_url(stream_key)

    async def update_stream(self, *, stream_key: str, rtsp_url: str) -> None:
        if not self.admin_api_url:
            return
        url = f"{self.admin_api_url}{self.update_path.format(stream_key=stream_key)}"
        await self._request_json("PATCH", url, json={"rtsp_url": rtsp_url})

    async def delete_stream(self, *, stream_key: str) -> None:
        if not self.admin_api_url:
            return
        url = f"{self.admin_api_url}{self.delete_path.format(stream_key=stream_key)}"
        await self._request_json("DELETE", url)

    async def _request_json(self, method: str, url: str, *, json: Optional[dict] = None) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                r = await self._client.request(method, url, headers=self._headers(), json=json)
                if r.status_code >= 400:
                    raise RuntimeError(f"WebRTC gateway error {r.status_code}: {r.text[:300]}")
                if r.content:
                    return r.json()
                return None
            except Exception as e:
                last_exc = e
                await asyncio.sleep(0.2 * (2 ** attempt))
        raise last_exc or RuntimeError("WebRTC gateway request failed")


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
        self.add_path = os.getenv("EDGE_ADD_PATH", "/api/cameras")
        self.patch_path = os.getenv("EDGE_PATCH_PATH", "/api/cameras/{camera_uuid}")
        self.delete_path = os.getenv("EDGE_DELETE_PATH", "/api/cameras/{camera_uuid}")
        self.api_key = os.getenv("EDGE_API_KEY")

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

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
        for attempt in range(3):
            try:
                r = await self._client.request(method, url, headers=self._headers(), json=json)
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

        # Optional lightweight registry to keep compatibility with older code paths
        self._active_id: Optional[uuid.UUID] = None
        self._active_pipeline: Optional[ModelPipeline] = None

    async def shutdown(self) -> None:
        async with self._lock:
            self._active_pipeline = None
            self._active_id = None
        await self._webrtc.close()
        await self._edge.close()

    # ------------------------------------------------------------------
    # Pipeline handle (DB-backed)
    # ------------------------------------------------------------------

    async def get_activepipeline(self) -> ModelPipeline:
        """
        Ensure there is at least one pipeline row and return a lightweight handle.

        (We keep ModelPipeline only for backwards compatibility with routes that read pipeline_id.)
        """
        async with self._lock:
            if self._active_pipeline is not None:
                return self._active_pipeline

            async with self._session_factory() as db:
                pipeline = await self._repo.upsert_pipeline(db, pipeline_id=None)
                await db.commit()

                self._active_id = pipeline.id
                self._active_pipeline = ModelPipeline(pipeline_id=pipeline.id)
                return self._active_pipeline

    # ------------------------------------------------------------------
    # Update pipeline = camera CRUD + external provisioning
    # ------------------------------------------------------------------

    def _patch_to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_unset=True, exclude_none=True)
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if v is not None}
        return {}

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
        async with self._lock:
            async with self._session_factory() as db:
                if not await self._repo.pipeline_exists(db, pid):
                    return None

                cameras_out: List[CameraOut] = []
                events_out: List[Dict[str, Any]] = []

                for ev in (channel_events or []):
                    et = getattr(ev, "event_type", None)

                    if et == "Create_Channel" or isinstance(ev, ChannelCreateEvent):
                        cams, evs = await self._add_channel(db, pid=pid, ev=ev, user_id=user_id, camera_code_prefix=camera_code_prefix)
                    elif et == "Edit_Channel" or isinstance(ev, ChannelEditEvent):
                        cams, evs = await self._edit_channel(db, pid=pid, ev=ev)
                    elif et == "Remove_Channel" or isinstance(ev, ChannelRemoveEvent):
                        cams, evs = await self._remove_channel(db, pid=pid, ev=ev)
                    else:
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

    # ------------------------------------------------------------------
    # Create / Edit / Remove
    # ------------------------------------------------------------------

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
        patch = self._patch_to_dict(getattr(ev, "configs", None))

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

        camera_code = f"{camera_code_prefix}-{uuid.uuid4().hex[:8]}"

        webrtc_url = await self._webrtc.ensure_stream(stream_key=str(camera_code), rtsp_url=str(rtsp_url))

        # 2) persist DB (webrtc_url stored once; immutable on edit)
        cam = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config=patch,
            user_id=user_id,
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

        # send full knobs to edge (it can ignore unknown keys)
        edge_payload = self._edge_payload_from_config(
            camera_uuid=str(cam.camera_uuid),
            rtsp_url=cam.rtsp_url,
            config={
                **patch,
                "enabled": enabled,
                "detection_enabled": det_enabled,
                "notification_enabled": bool(getattr(cam, "is_notification_enabled", True)),
            },
        )

        if enabled and det_enabled:
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
            )
        ]

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

        # current state
        old_rtsp = cam_db.rtsp_url
        old_webrtc = cam_db.webrtc_url
        old_primary_dev = await self.channel_repo.get_primary_device(db, camera_uuid=cam_uuid)

        # merge patches (webrtc_url is immutable)
        patch = self._patch_to_dict(getattr(ev, "configs", None))
        patch.pop("webrtc_url", None)
        patch.pop("webrtcUrl", None)

        # allow reassigning primary device via patch
        new_primary_device_uuid = patch.get("primary_device_uuid") or patch.get("device_uuid")
        if new_primary_device_uuid:
            new_primary_device_uuid = self._as_uuid(new_primary_device_uuid, "primary_device_uuid")
        else:
            new_primary_device_uuid = old_primary_dev.device_uuid if old_primary_dev else None

        # merge config knobs from ChannelConfiguration + patch
        merged_cfg: Dict[str, Any] = {}
        if chan_cfg_db and getattr(chan_cfg_db, "configuration", None):
            merged_cfg.update(chan_cfg_db.configuration)
        merged_cfg.update(patch)

        # persist DB update (webrtc_url stays unchanged)
        cam2 = await self.channel_repo.upsert_camera_from_channel_config(
            db,
            pipeline_id=pid,
            channel_config={
                **merged_cfg,
                "camera_uuid": cam_uuid,
                "rtsp_url": merged_cfg.get("rtsp_url", old_rtsp),
                "enabled": merged_cfg.get("enabled", cam_db.is_enabled),
                "detection_enabled": merged_cfg.get("detection_enabled", cam_db.is_detection_enabled),
                "notification_enabled": merged_cfg.get("notification_enabled", cam_db.is_notification_enabled),
            },
            site_uuid=cam_db.site_uuid,
            # keep stored webrtc_url, never overwrite
            webrtc_url=old_webrtc,
            device_uuids=[d for d in ([new_primary_device_uuid] if new_primary_device_uuid else [])] or None,
            primary_device_uuid=new_primary_device_uuid,
        )

        # WebRTC gateway: update mapping if rtsp_url changed, but keep the same webrtc_url
        if cam2.rtsp_url != old_rtsp:
            await self._webrtc.update_stream(stream_key=str(cam_uuid), rtsp_url=cam2.rtsp_url)

        # Edge: update on Jetson
        enabled = bool(cam2.is_enabled)
        det_enabled = bool(cam2.is_detection_enabled)

        # if device changed: remove from old + add to new
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
        else:
            # same device
            dev = old_primary_dev
            if dev is None and new_primary_device_uuid is not None:
                dev = await self._get_device(db, new_primary_device_uuid)

            if dev is None:
                raise ValueError("Camera has no assigned device. Assign a Device to this camera.")

            edge_patch = dict(patch)
            # ensure core fields are present
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
                dev = await self.channel_repo.get_primary_device(db, camera_uuid=cam_uuid)
                if dev is not None and dev.device_url:
                    await self._edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.exception("Edge delete failed during camera removal")

            try:
                await self._webrtc.delete_stream(stream_key=str(cam_uuid))
            except Exception:
                logger.exception("WebRTC delete failed during camera removal")

            await self.channel_repo.delete_camera(db, camera_uuid=cam_db.camera_uuid)

        events_out = [{"event_type": "Remove_Channel", "camera_uuid": str(cam_uuid)}]
        return [], events_out
