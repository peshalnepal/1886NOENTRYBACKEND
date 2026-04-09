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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_async_db, get_current_user, get_manager, get_manager_optional
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from domain.events import ChannelCreateEvent, ChannelEditEvent, ChannelRemoveEvent
from core.database_orm import CameraDevice, Device, User
from core.database import db_manager
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
                webrtc_url=getattr(cam, "webrtc_url", None),
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
        webrtc_url=getattr(cam, "webrtc_url", None),
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
    """Returns the WebRTC playback URL for this camera."""
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)
    if not getattr(cam, "webrtc_url", None):
        raise HTTPException(status_code=409, detail="WebRTC URL not provisioned yet")

    return {"camera_uuid": str(cam.camera_uuid), "webrtc_url": cam.webrtc_url}


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

        # New rule: device_uuid is required
        if not data.get("device_uuid"):
            raise HTTPException(status_code=422, detail="device_uuid is required (each camera must have a device).")

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
            webrtc_url=cam_out.webrtc_url,
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

    result = await manager.update_pipeline(pipeline.pipeline_id, [ev], user_id=user.id)
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
        webrtc_url=cam_out.webrtc_url,
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


async def _delete_camera_bg(
    *,
    manager: "Manager",
    camera_uuid: uuid.UUID,
    cam_code: Optional[str],
    device_urls: List[str],
    user_id: int,
) -> None:
    """Background: tell every edge device and the WebRTC server to stop this
    camera, then evict it from the in-memory pipeline.

    Runs AFTER the HTTP 200 is sent so slow / unreachable edge devices
    (3 retries × 10 s connect-timeout = 30 s per device) never cause a 502.
    """
    for dev_url in device_urls:
        try:
            await manager._edge.delete_camera(device_url=dev_url, camera_uuid=str(camera_uuid))
        except Exception:
            logger.warning(
                "cam delete [bg]: edge delete failed cam=%s url=%s", camera_uuid, dev_url, exc_info=True
            )

    if cam_code:
        try:
            await manager._webrtc.delete_stream(stream_key=str(cam_code))
        except Exception:
            logger.warning(
                "cam delete [bg]: WebRTC delete failed cam=%s code=%s", camera_uuid, cam_code, exc_info=True
            )

    try:
        active_pipeline = await asyncio.wait_for(
            manager.get_activepipeline(user_id=user_id), timeout=10.0
        )
        await active_pipeline.remove_channel(camera_uuid)
    except Exception:
        logger.warning("cam delete [bg]: pipeline evict failed cam=%s", camera_uuid, exc_info=True)


@router.delete("/{camera_uuid}")
async def delete_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    # get_manager_optional never raises 503 — DB deletion must succeed even if
    # the manager (pipeline / edge layer) has not fully started up.
    manager: Optional[Manager] = Depends(get_manager_optional),
    user: User = Depends(get_current_user),
):
    from routes.notifications_routes import invalidate_camera_mode_cache

    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam, _cfg, _pid = full
    _ensure_user_owns_camera(cam, user.id)

    # Snapshot data needed for background edge/WebRTC cleanup BEFORE any DB
    # changes — these columns are gone after commit.
    cam_code: Optional[str] = getattr(cam, "camera_code", None)

    # Scalar join — no ORM Camera objects added to session identity map.
    device_url_rows = (
        await db.execute(
            select(CameraDevice.camera_uuid, Device.device_url)
            .join(Device, Device.device_uuid == CameraDevice.device_uuid)
            .where(CameraDevice.camera_uuid == camera_uuid)
        )
    ).all()
    device_urls = [
        str(row[1]).strip() for row in device_url_rows if str(row[1] or "").strip()
    ]

    # ── Attempt pipeline-managed delete (DB + edge + WebRTC + in-memory) ──
    # update_pipeline acquires the user-lock, does the DB commit, then calls
    # _remove_channel which fires edge/WebRTC HTTP calls while STILL HOLDING
    # the lock.  Those calls can take up to 30 s (3 retries × 10 s).
    # We allow 20 s max so the Azure proxy never times us out at ~240 s,
    # then fall back to a direct DB-only delete + background cleanup.
    if manager is not None:
        try:
            pipeline = await asyncio.wait_for(
                manager.get_activepipeline(user_id=user.id), timeout=5.0
            )
            ev = ChannelRemoveEvent(
                channel_id=camera_uuid,
                configs={},
                created_at=datetime.now(timezone.utc),
            )
            await asyncio.wait_for(
                manager.update_pipeline(pipeline.pipeline_id, [ev], user_id=user.id),
                timeout=20.0,
            )
            # Manager handled everything (DB + edge + WebRTC) within the timeout.
            await invalidate_camera_mode_cache(camera_uuid)
            return {"ok": True}
        except asyncio.TimeoutError:
            logger.warning(
                "Camera delete via manager timed out (edge/WebRTC slow) for cam=%s; "
                "falling back to direct DB delete + background cleanup",
                camera_uuid,
            )
        except HTTPException:
            raise
        except Exception:
            logger.warning(
                "Camera delete via manager failed for cam=%s; "
                "falling back to direct DB delete + background cleanup",
                camera_uuid, exc_info=True,
            )

    # ── Fallback: direct DB delete + background edge/WebRTC cleanup ───────
    # The manager's update_pipeline may have rolled back (timeout during edge
    # calls → the async-with-session exits via CancelledError → rollback).
    # Re-fetch the camera to confirm it still needs deleting.
    full2 = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if full2:
        cam2, _, _ = full2
        await db.delete(cam2)
        await db.commit()

    if device_urls or cam_code:
        asyncio.create_task(
            _delete_camera_bg(
                manager=manager,
                camera_uuid=camera_uuid,
                cam_code=cam_code,
                device_urls=device_urls,
                user_id=int(user.id),
            ),
            name="camera_cleanup",
        )

    await invalidate_camera_mode_cache(camera_uuid)
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
