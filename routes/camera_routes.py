# agents/api/routes/camera_routes.py

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from application.services.user_snapshot_cache import (
    CachedUserSnapshot,
    UserSnapshotCache,
    UserSnapshotLookupError,
)
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete as sql_delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_async_db, get_current_user, get_manager
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from domain.events import ChannelCreateEvent, ChannelEditEvent
from core.database_orm import (
    Camera,
    CameraDevice,
    ChannelConfiguration,
    Device,
    Notification,
    PipelineCamera,
    User,
    VideoRecord,
)
from core.database import AsyncSessionLocal, db_manager
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import EventClipService
from application.services.webrtcgateway import resolve_camera_webrtc_url, WebRTCGatewayClient
from core.security.tokens import decode_access_token
from core.schemas import (
    CameraSchema,
    CameraCreateSchema,
    CameraEditSchema,
    CameraWithConfigSchema,
)  # type: ignore
from application.services.manager import Manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])
_SNAPSHOT_HTTP = httpx.AsyncClient(
    timeout=httpx.Timeout(8.0, connect=3.0, read=8.0, write=5.0, pool=5.0),
    follow_redirects=True,
)

def _ensure_user_owns_camera(cam: Any, user_id: int) -> None:
    if int(getattr(cam, "user_id", -1)) != int(user_id):
        raise HTTPException(status_code=404, detail="Camera not found")


def _get_user_snapshot_cache(request: Request) -> UserSnapshotCache:
    cache = getattr(request.app.state, "user_snapshot_cache", None)
    if cache is None:
        cache = UserSnapshotCache()
        request.app.state.user_snapshot_cache = cache
    return cache


def _camera_webrtc_url(cam: Any) -> Optional[str]:
    return resolve_camera_webrtc_url(
        camera_code=getattr(cam, "camera_code", None),
        stored_url=getattr(cam, "webrtc_url", None),
    )

async def _resolve_stream_user(
    *,
    request: Request,
    access_token: Optional[str],
) -> CachedUserSnapshot:
    auth_header = request.headers.get("authorization", "")
    token = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = str(access_token or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    raw_user_id = payload.get("user_id") or payload.get("sub")
    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    cache = _get_user_snapshot_cache(request)
    try:
        user = await cache.get(session_factory=sf, user_id=user_id)
    except UserSnapshotLookupError:
        raise HTTPException(status_code=503, detail="Database not available")
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def _session_factory_from_app(request: Request):
    sf = getattr(request.app.state, "session_factory", None)
    if sf is not None:
        return sf
    return getattr(db_manager, "AsyncSessionLocal", None)


def _device_snapshot_urls(device_url: str, camera_uuid: uuid.UUID) -> List[str]:
    base = str(device_url or "").rstrip("/")
    camera_id = str(camera_uuid)

    urls = [f"{base}/cameras/{camera_id}/snapshot.jpg"]
    if not base.endswith("/api"):
        urls.insert(0, f"{base}/api/cameras/{camera_id}/snapshot.jpg")
    return urls

async def _fetch_device_snapshot(*, device_url: str, camera_uuid: uuid.UUID) -> Response:
    urls = _device_snapshot_urls(device_url, camera_uuid)
    last_error: Optional[str] = None
    saw_not_found = False
    saw_transport_error = False
    saw_upstream_error = False

    for url in urls:
        try:
            upstream = await _SNAPSHOT_HTTP.get(
                url,
                headers={"Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.1"},
            )
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as exc:
            saw_transport_error = True
            last_error = f"Jetson snapshot unavailable: {type(exc).__name__}"
            continue
        except httpx.HTTPError as exc:
            saw_upstream_error = True
            last_error = f"Jetson snapshot request failed: {type(exc).__name__}"
            continue

        if upstream.status_code == 404:
            saw_not_found = True
            last_error = "No snapshot available yet on Jetson."
            continue

        if upstream.status_code >= 400:
            saw_upstream_error = True
            last_error = f"Jetson snapshot request failed with status {upstream.status_code}."
            continue

        content_type = str(upstream.headers.get("content-type") or "image/jpeg").split(";", 1)[0].strip() or "image/jpeg"
        if not content_type.startswith("image/"):
            content_type = "image/jpeg"

        return Response(
            content=upstream.content,
            media_type=content_type,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    if saw_transport_error:
        raise HTTPException(status_code=503, detail=last_error or "Jetson snapshot unavailable.")

    raise HTTPException(
        status_code=404 if saw_not_found and not saw_upstream_error else 502,
        detail=last_error or "Jetson snapshot request failed.",
    )

# -------------------------
# Detection Schemas
# -------------------------
class BoxPx(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class BoxNorm(BaseModel):
    x: float  # 0..1 left
    y: float  # 0..1 top
    w: float  # 0..1 width
    h: float  # 0..1 height


class DetectionItemOut(BaseModel):
    box: BoxPx
    cls_name: str
    conf: float
    box_norm: Optional[BoxNorm] = None


class DetectionOut(BaseModel):
    camera_uuid: str
    frame_ts_ms: int
    frame_seq: int
    inference_ms: Optional[int] = None
    model_id: Optional[str] = None

    frame_w: Optional[int] = None
    frame_h: Optional[int] = None

    detections: List[DetectionItemOut] = Field(default_factory=list)
    pose: Optional[Any] = None


def _normalize_box_px(box: Dict[str, Any], frame_w: Optional[int], frame_h: Optional[int]) -> Optional[BoxNorm]:
    if not frame_w or not frame_h:
        return None
    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return BoxNorm(
        x=max(0.0, min(1.0, x1 / frame_w)),
        y=max(0.0, min(1.0, y1 / frame_h)),
        w=max(0.0, min(1.0, w / frame_w)),
        h=max(0.0, min(1.0, h / frame_h)),
    )


def _resp_to_detection_out(resp: Any, *, normalize: bool) -> DetectionOut:
    fw = getattr(resp, "frame_w", None)
    fh = getattr(resp, "frame_h", None)

    items: List[DetectionItemOut] = []
    for d in (list(resp.detections) if getattr(resp, "detections", None) else []):
        box = d.get("box") or {}
        if not all(k in box for k in ("x1", "y1", "x2", "y2")):
            continue

        box_px = BoxPx(x1=box["x1"], y1=box["y1"], x2=box["x2"], y2=box["y2"])
        items.append(
            DetectionItemOut(
                box=box_px,
                cls_name=str(d.get("cls_name") or ""),
                conf=float(d.get("conf") or 0.0),
                box_norm=_normalize_box_px(box, fw, fh) if normalize else None,
            )
        )

    return DetectionOut(
        camera_uuid=str(resp.camera_uuid),
        frame_ts_ms=int(resp.frame_ts_ms),
        frame_seq=int(resp.frame_seq),
        inference_ms=int(resp.inference_ms) if getattr(resp, "inference_ms", None) is not None else None,
        model_id=str(getattr(resp, "model_id", None)) if getattr(resp, "model_id", None) is not None else None,
        frame_w=int(fw) if fw is not None else None,
        frame_h=int(fh) if fh is not None else None,
        detections=items,
        pose=getattr(resp, "pose", None),
    )


# -------------------------
# Cameras
# -------------------------
@router.get("", response_model=List[CameraSchema])
async def list_cameras(
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
    site_uuid: uuid.UUID = None,  # keep query param; if None -> 422 below
):
    """
    List cameras from Azure DB for a site.
    Uses camera.webrtc_url for playback in the frontend.
    """
    if site_uuid is None:
        raise HTTPException(status_code=422, detail="site_uuid query param is required")

    repo = ChannelRepository()
    site_repo=SiteRepository()
    if site_uuid is None:
        raise HTTPException(status_code=400, detail="site_uuid is required")

    cams = await site_repo.list_cameras_by_site(db, site_uuid=site_uuid,user_id=int(user.id))

    out: List[CameraSchema] = []
    for cam in cams:
        loaded_devices = list(getattr(cam, "devices", None) or [])
        dev = loaded_devices[0] if loaded_devices else None

        out.append(
            CameraSchema(
                camera_uuid=cam.camera_uuid,
                camera_code=cam.camera_code,
                name=getattr(cam, "name", None),
                location=getattr(cam, "location", None),
                site_uuid=cam.site_uuid,
                device_uuid=(dev.device_uuid if dev else None),
                rtsp_url=cam.rtsp_url,
                webrtc_url=_camera_webrtc_url(cam),
                is_enabled=cam.is_enabled,
                is_detection_enabled=cam.is_detection_enabled,
                is_notification_enabled=cam.is_notification_enabled,
                use_site_schedule=bool(getattr(cam, "use_site_schedule", True)),
                roi=getattr(cam, "roi", None),
                created_at=cam.created_at,
                updated_at=cam.updated_at,
            )
        )
    return out

@router.get("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def get_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)

    dev = await repo.get_device(db, camera_uuid=cam.camera_uuid, required=False, relaxed=True)

    return CameraWithConfigSchema(
        camera_uuid=cam.camera_uuid,
        camera_code=cam.camera_code,
        name=getattr(cam, "name", None),
        location=getattr(cam, "location", None),
        site_uuid=cam.site_uuid,
        device_uuid=(dev.device_uuid if dev else None),
        rtsp_url=cam.rtsp_url,
        webrtc_url=_camera_webrtc_url(cam),
        is_enabled=cam.is_enabled,
        is_detection_enabled=cam.is_detection_enabled,
        is_notification_enabled=cam.is_notification_enabled,
        use_site_schedule=bool(getattr(cam, "use_site_schedule", True)),
        roi=getattr(cam, "roi", None),
        configuration=(cfg.configuration if cfg else {}),
        timezone=(cfg.timezone if cfg else None),
        created_at=cam.created_at,
        updated_at=cam.updated_at,
    )


@router.get("/{camera_uuid}/playback")
async def get_camera_playback(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    """
    Returns the WebRTC playback URL for this camera.
    Ensures the stream is provisioned in MediaMTX before returning.
    """
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)
    
    # Validate camera has required fields
    rtsp_url = getattr(cam, "rtsp_url", None)
    camera_code = getattr(cam, "camera_code", None)
    if not camera_code:
        raise HTTPException(status_code=409, detail="Camera code not set")
    if not rtsp_url:
        raise HTTPException(status_code=409, detail="RTSP URL not configured")
    
    # Provision stream in MediaMTX if not already done
    try:
        webrtc_client = WebRTCGatewayClient()
        webrtc_url = await webrtc_client.ensure_stream(
            stream_key=camera_code,
            rtsp_url=rtsp_url
        )
        await webrtc_client.close()
        
        if not webrtc_url:
            raise HTTPException(
                status_code=503,
                detail="Failed to provision WebRTC stream: no URL returned from gateway"
            )
        
        logger.info(
            "Stream provisioned for playback: camera_uuid=%s, camera_code=%s, webrtc_url=%s",
            camera_uuid,
            camera_code,
            webrtc_url
        )
        return {"camera_uuid": str(cam.camera_uuid), "webrtc_url": webrtc_url}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to provision WebRTC stream for playback: camera_uuid=%s, camera_code=%s, error=%s",
            camera_uuid,
            camera_code,
            str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail=f"Failed to provision WebRTC stream: {str(e)}"
        )


# -------------------------
# Detection endpoints
# -------------------------
@router.get("/{camera_uuid}/detections/latest", response_model=DetectionOut)
async def get_latest_detection(
    camera_uuid: uuid.UUID,
    refresh: bool = False,
    normalize: bool = False,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
    user: User = Depends(get_current_user),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)

    pipeline = await manager.get_activepipeline(user_id=user.id)

    resp = await pipeline.get_latest_detection(str(camera_uuid))
    if resp is None and refresh:
        resp = await pipeline.refresh_detection_once(str(camera_uuid))

    if resp is None:
        raise HTTPException(status_code=404, detail="No detection available yet")

    return _resp_to_detection_out(resp, normalize=normalize)

@router.get("/{camera_uuid}/detections/stream")
async def stream_detections_sse(
    request: Request,
    camera_uuid: uuid.UUID,
    after_ts_ms: int = 0,
    after_seq: int = 0,
    timeout_ms: int = 30000,
    normalize: bool = False,
    access_token: Optional[str] = None,
    manager: Manager = Depends(get_manager),
):
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user = await _resolve_stream_user(request=request, access_token=access_token)
    async with sf() as db:
        repo = ChannelRepository()
        full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
        if not full:
            raise HTTPException(status_code=404, detail="Camera not found")
        cam, _cfg, _pid = full
        _ensure_user_owns_camera(cam, user.id)

    pipeline = await manager.get_activepipeline(user_id=user.id)
    cam_key = str(camera_uuid)

    async def gen():
        last_ts = int(after_ts_ms)
        last_seq = int(after_seq)

        initial = await pipeline.get_latest_detection(cam_key)
        if initial is not None and last_ts == 0 and last_seq > 0:
            last_ts = int(initial.frame_ts_ms)
        if initial is not None:
            its, isq = int(initial.frame_ts_ms), int(initial.frame_seq)
            if its > last_ts or (its == last_ts and isq > last_seq):
                last_ts, last_seq = its, isq
                payload = _resp_to_detection_out(initial, normalize=normalize).model_dump()
                yield f"event: detection\ndata: {json.dumps(payload)}\n\n"

        while True:
            if await request.is_disconnected():
                return

            resp = await pipeline.detect_store.wait_new(
                cam_key,
                after_ts_ms=last_ts,
                after_seq=last_seq,
                timeout_ms=int(timeout_ms),
            )

            if resp is None:
                yield "event: heartbeat\ndata: {}\n\n"
                continue

            last_ts = int(resp.frame_ts_ms)
            last_seq = int(resp.frame_seq)
            payload = _resp_to_detection_out(resp, normalize=normalize).model_dump()
            yield f"event: detection\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/detections/stream")
async def stream_all_detections_sse(
    request: Request,
    timeout_ms: int = 30000,
    normalize: bool = False,
    access_token: Optional[str] = None,
    manager: Manager = Depends(get_manager),
):
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user = await _resolve_stream_user(request=request, access_token=access_token)
    pipeline = await manager.get_activepipeline(user_id=user.id)
    hub = pipeline.detection_hub

    async def gen():
        q = await hub.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    resp = await asyncio.wait_for(q.get(), timeout=float(timeout_ms) / 1000.0)
                    payload = _resp_to_detection_out(resp, normalize=normalize).model_dump()
                    yield f"event: detection\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            await hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/detections/latest")
async def latest_detections_for_site(
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
    user: User = Depends(get_current_user),
):
    repo = ChannelRepository()
    site_repo=SiteRepository()
    if site_uuid is None:
        raise HTTPException(status_code=400, detail="site_uuid is required")

    cams = await site_repo.list_cameras_by_site(db, site_uuid=site_uuid,user_id=int(user.id))

    pipeline = await manager.get_activepipeline(user_id=user.id)

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for cam in cams:
        cam_id = str(cam.camera_uuid)
        resp = await pipeline.get_latest_detection(cam_id)
        out[cam_id] = _resp_to_detection_out(resp, normalize=False).model_dump() if resp is not None else None

    return {"site_uuid": str(site_uuid), "latest": out}


@router.post("", response_model=CameraWithConfigSchema)
async def create_camera(
    payload: CameraCreateSchema,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
    user: User = Depends(get_current_user),
):
    try:
        data = payload.model_dump(exclude_none=True)

        # Trust JWT context, not client-provided user_id.
        data["user_id"] = int(user.id)
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

        cam_out = result.cameras[0]
        
        # Provision WebRTC stream in MediaMTX after camera creation
        camera_code = getattr(cam_out, "camera_code", None)
        rtsp_url = getattr(cam_out, "rtsp_url", None)
        webrtc_url = None
        
        if camera_code and rtsp_url:
            try:
                webrtc_client = WebRTCGatewayClient()
                webrtc_url = await webrtc_client.ensure_stream(
                    stream_key=camera_code,
                    rtsp_url=rtsp_url
                )
                await webrtc_client.close()
                logger.info(
                    "Stream provisioned on camera creation: camera_uuid=%s, camera_code=%s, webrtc_url=%s",
                    cam_out.camera_uuid,
                    camera_code,
                    webrtc_url
                )
            except Exception as e:
                logger.warning(
                    "Failed to provision stream on camera creation (will retry on playback): "
                    "camera_uuid=%s, camera_code=%s, error=%s",
                    cam_out.camera_uuid,
                    camera_code,
                    str(e),
                    exc_info=True
                )
                # Don't fail camera creation if provisioning fails - will retry on playback
        
        if not webrtc_url:
            webrtc_url = resolve_camera_webrtc_url(
                camera_code=camera_code,
                stored_url=getattr(cam_out, "webrtc_url", None),
            )
        
        from routes.notifications_routes import invalidate_camera_mode_cache

        await invalidate_camera_mode_cache(cam_out.camera_uuid)
        return CameraWithConfigSchema(
            camera_uuid=cam_out.camera_uuid,
            camera_code=cam_out.camera_code,
            name=getattr(cam_out, "name", None),
            location=getattr(cam_out, "location", None),
            site_uuid=cam_out.site_uuid,
            device_uuid=cam_out.device_uuid,
            rtsp_url=cam_out.rtsp_url,
            webrtc_url=webrtc_url,
            is_enabled=cam_out.enabled,
            is_detection_enabled=cam_out.detection_enabled,
            is_notification_enabled=cam_out.notification_enabled,
            use_site_schedule=bool(getattr(cam_out, "use_site_schedule", True)),
            roi=cam_out.roi,
            configuration=cam_out.configuration,
            timezone=cam_out.timezone,
            created_at=getattr(cam_out, "created_at", None) or datetime.now(timezone.utc),
            updated_at=getattr(cam_out, "updated_at", None) or datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error creating camera")
        raise HTTPException(status_code=500, detail=f"Failed to create camera: {str(e)}")


@router.patch("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def edit_camera(
    camera_uuid: uuid.UUID,
    payload: CameraEditSchema,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
    user: User = Depends(get_current_user),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)

    pipeline = await manager.get_activepipeline(user_id=user.id)
    patch_payload = payload.model_dump(exclude_unset=True)
    patch_payload.pop("user_id", None)

    ev = ChannelEditEvent(
        channel_id=camera_uuid,
        configs=patch_payload,
        created_at=datetime.now(timezone.utc),
    )

    try:
        result = await manager.update_pipeline(pipeline.pipeline_id, [ev], user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not result or not result.cameras:
        raise HTTPException(status_code=500, detail="Failed to edit camera")

    cam_out = result.cameras[0]
    from routes.notifications_routes import invalidate_camera_mode_cache

    await invalidate_camera_mode_cache(cam_out.camera_uuid)
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
        use_site_schedule=bool(getattr(cam_out, "use_site_schedule", True)),
        roi=cam_out.roi,
        configuration=cam_out.configuration,
        timezone=cam_out.timezone,
        created_at=getattr(cam_out, "created_at", None) or datetime.now(timezone.utc),
        updated_at=getattr(cam_out, "updated_at", None) or datetime.now(timezone.utc),
    )


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


async def _delete_blobs_background(keys: List[str], *, service_cls: type, label: str) -> None:
    """Delete blobs in parallel batches of 10."""
    unique = list(dict.fromkeys(k for k in keys if k))
    if not unique:
        return

    logger.info(f"[Blob Cleanup] Starting deletion of {len(unique)} {label} blobs")
    svc = service_cls()
    deleted = 0
    failed = 0
    batch_size = 10
    try:
        for i in range(0, len(unique), batch_size):
            batch = unique[i: i + batch_size]
            results = await asyncio.gather(
                *[svc.delete_blob(blob_name=k) for k in batch],
                return_exceptions=True,
            )
            for key, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.warning(f"[Blob Cleanup] Failed to delete {label} blob {key}: {result}")
                else:
                    deleted += 1
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



@router.delete("/{camera_uuid}")
async def delete_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
    user: User = Depends(get_current_user),
):
    """
    Full camera deletion:
    1. Snapshot camera info
    2. Stop camera on edge/WebRTC/pipeline (before any DB changes)
    3. Extract blob keys from notifications + video records
    4. Delete all linked DB rows (notifications, video records, relationships, camera)
    5. Async blob deletion (alert images + clips)
    6. Invalidate caches
    """
    from routes.notifications_routes import invalidate_camera_mode_cache

    logger.info(f"[Camera Delete] Starting deletion of camera={camera_uuid}")

    # ========================================
    # PHASE 1: Validate + snapshot info BEFORE any changes
    # ========================================
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)

    cam_code: Optional[str] = getattr(cam, "camera_code", None)
    site_uuid = cam.site_uuid

    device_url_rows = (
        await db.execute(
            select(CameraDevice.camera_uuid, Device.device_url)
            .join(Device, Device.device_uuid == CameraDevice.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
        )
    ).all()
    device_urls = [str(row[1]).strip() for row in device_url_rows if str(row[1] or "").strip()]

    # ========================================
    # PHASE 2: Stop camera BEFORE any DB changes
    # Prevents new detections/clips being written while we delete.
    # ========================================
    logger.info(f"[Camera Delete] Phase 2: Stopping camera on edge/WebRTC/pipeline")

    # 2a: Purge notification service in-memory state
    notif_svc = getattr(manager, "_notification_service", None) if manager is not None else None
    if notif_svc is not None:
        try:
            purge_fn = getattr(notif_svc, "purge_deleted_site_runtime_state", None)
            if callable(purge_fn):
                await purge_fn(
                    user_id=int(user.id),
                    site_uuid=site_uuid,
                    camera_uuids=[camera_uuid],
                )
            else:
                notif_svc.invalidate_camera_roi_state(str(camera_uuid))
        except Exception as exc:
            logger.warning(f"[Camera Delete] Notification service purge failed: {exc}", exc_info=True)

    # 2b: Stop on edge device, WebRTC, and evict from pipeline
    if manager is not None:
        try:
            active_pipeline = await asyncio.wait_for(
                manager.get_activepipeline(user_id=int(user.id)), timeout=10.0
            )
        except Exception:
            active_pipeline = None
            logger.warning(f"[Camera Delete] Could not get active pipeline for cam={camera_uuid}")

        for dev_url in device_urls:
            try:
                await manager._edge.delete_camera(device_url=dev_url, camera_uuid=str(camera_uuid))
            except Exception as exc:
                logger.warning(f"[Camera Delete] Edge delete failed cam={camera_uuid} url={dev_url}: {exc}")

        if cam_code:
            try:
                await manager._webrtc.delete_stream(stream_key=str(cam_code))
            except Exception as exc:
                logger.warning(f"[Camera Delete] WebRTC delete failed cam={camera_uuid} code={cam_code}: {exc}")

        if active_pipeline is not None:
            try:
                await active_pipeline.remove_channel(camera_uuid)
            except Exception as exc:
                logger.warning(f"[Camera Delete] Pipeline evict failed cam={camera_uuid}: {exc}")

    # ========================================
    # PHASE 3: Extract blob storage keys from stable DB
    # Camera is stopped — no new rows being created.
    # ========================================
    logger.info(f"[Camera Delete] Phase 3: Extracting blob storage keys")
    alert_blob_keys: List[str] = []
    clip_blob_keys: List[str] = []

    offset = 0
    batch_size = 5000
    while True:
        async with AsyncSessionLocal() as blob_db:
            batch = (
                await blob_db.execute(
                    select(Notification.id, Notification.payload)
                    .where(Notification.camera_uuid == camera_uuid)
                    .order_by(Notification.id)
                    .offset(offset)
                    .limit(batch_size)
                )
            ).all()

        if not batch:
            break

        for _nid, payload in batch:
            key = extract_image_storage_key(payload)
            if key:
                alert_blob_keys.append(key)
            clip_blob_keys.extend(_extract_notification_clip_storage_keys(payload))

        offset += batch_size
        if len(batch) < batch_size:
            break

    video_clip_keys = (
        await db.execute(
            select(VideoRecord.storage_key).where(
                VideoRecord.camera_uuid == camera_uuid,
                VideoRecord.storage_key.isnot(None),
            )
        )
    ).scalars().all()
    clip_blob_keys.extend([str(k).strip() for k in video_clip_keys if str(k or "").strip()])

    logger.info(
        f"[Camera Delete] Found {len(alert_blob_keys)} alert image blobs, "
        f"{len(clip_blob_keys)} clip blobs"
    )

    # ========================================
    # PHASE 4: Delete all linked DB rows
    # Order: notifications first (SET NULL FK — must be explicit),
    # then remaining children, then camera itself.
    # ========================================
    logger.info(f"[Camera Delete] Phase 4: Deleting database rows")
    async with AsyncSessionLocal() as del_db:
        # Notifications have ondelete="SET NULL" — must delete explicitly
        await del_db.execute(sql_delete(Notification).where(Notification.camera_uuid == camera_uuid))
        # Children with CASCADE FKs (explicit for safety)
        await del_db.execute(sql_delete(VideoRecord).where(VideoRecord.camera_uuid == camera_uuid))
        await del_db.execute(sql_delete(PipelineCamera).where(PipelineCamera.camera_uuid == camera_uuid))
        await del_db.execute(sql_delete(CameraDevice).where(CameraDevice.camera_uuid == camera_uuid))
        await del_db.execute(sql_delete(ChannelConfiguration).where(ChannelConfiguration.camera_uuid == camera_uuid))
        # Finally delete the camera row itself
        await del_db.execute(sql_delete(Camera).where(Camera.camera_uuid == camera_uuid))
        await del_db.commit()
    logger.info(f"[Camera Delete] Database deletion complete")

    # ========================================
    # PHASE 5: Async blob deletion
    # ========================================
    if alert_blob_keys:
        logger.info(f"[Camera Delete] Phase 5a: Scheduling deletion of {len(alert_blob_keys)} alert image blobs")
        _spawn_bg_task(
            _delete_blobs_background(alert_blob_keys, service_cls=AlertImageStorageService, label="alert image"),
            name=f"delete_camera_alert_blobs:{camera_uuid}",
        )

    if clip_blob_keys:
        logger.info(f"[Camera Delete] Phase 5b: Scheduling deletion of {len(clip_blob_keys)} clip blobs")
        _spawn_bg_task(
            _delete_blobs_background(clip_blob_keys, service_cls=EventClipService, label="clip"),
            name=f"delete_camera_clip_blobs:{camera_uuid}",
        )

    # ========================================
    # PHASE 6: Invalidate caches
    # ========================================
    await invalidate_camera_mode_cache(camera_uuid)

    logger.info(f"[Camera Delete] COMPLETE: camera={camera_uuid} has been successfully deleted")
    return {"ok": True}

@router.get("/{camera_uuid}/snapshot.jpg")
async def snapshot_jpg(
    request: Request,
    camera_uuid: uuid.UUID,
    access_token: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
):
    try:
        user = await _resolve_stream_user(request=request, access_token=access_token)
    except HTTPException as e:
        if e.status_code == 401:
            logger.warning(f"Snapshot access denied for camera {camera_uuid}: {e.detail}")
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please ensure you are logged in and have a valid token."
            )
        raise
    
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        logger.warning(f"Camera {camera_uuid} not found for user {user.id}")
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)

    dev = await repo.get_device(db, camera_uuid=cam.camera_uuid, required=False, relaxed=True)
    device_url = str(getattr(dev, "device_url", "") or "").strip() if dev is not None else ""
    if not device_url:
        logger.warning(f"Camera {camera_uuid} has no device assigned or device_url missing")
        raise HTTPException(
            status_code=409, 
            detail="Camera is not properly configured. Device URL is missing. Please contact administrator."
        )

    return await _fetch_device_snapshot(device_url=device_url, camera_uuid=cam.camera_uuid)


@router.get("/{camera_uuid}/stream.mjpg")
async def stream_mjpeg(camera_uuid: uuid.UUID):
    raise HTTPException(
        status_code=410,
        detail="MJPEG streaming disabled. Use WebRTC playback URL for video.",
    )
