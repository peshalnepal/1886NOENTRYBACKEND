# agents/api/routes/camera_routes.py

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from application.services.user_snapshot_cache import (
    CachedUserSnapshot,
    UserSnapshotLookupError,
)
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import (
    get_async_db,
    get_current_user,
    get_manager,
    get_user_snapshot_cache,
    RequirePermission,
    OrgContext,
)
from core.security.roles import Permission
from application.services.authz_service import AuthzService
from application.repositories.channel_repository import ChannelRepository
from application.repositories.site_repository import SiteRepository
from application.repositories.video_repository import VideoRepository
from application.repositories.notification_repository import NotificationRepository
from domain.events import ChannelCreateEvent, ChannelEditEvent
from core.database_orm import Notification, User
from core.database import AsyncSessionLocal, db_manager
from application.services.alert_image_storage import AlertImageStorageService, extract_image_storage_key
from application.services.clip_storage import (
    EventClipService,
    extract_notification_clip_storage_keys as _extract_notification_clip_storage_keys,
)
from application.services.webrtcgateway import resolve_camera_webrtc_url, WebRTCGatewayClient
from application.services.detection_stream import (
    normalize_box_px,
    resp_to_detection_out,
    stream_camera_detections,
)
from routes._stream_auth import DB_UNAVAILABLE, resolve_stream_user
from core.schemas import (
    CameraCreateSchema,
    CameraEditSchema,
    CameraSchema,
    CameraWithConfigSchema,
    DetectionOut,
)
from application.services.manager import Manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])
_SNAPSHOT_HTTP = httpx.AsyncClient(
    timeout=httpx.Timeout(8.0, connect=3.0, read=8.0, write=5.0, pool=5.0),
    follow_redirects=True,
)

def _camera_owner_id(cam: Any, fallback_user_id: int) -> int:
    """Detection pipelines are keyed by a user id; use the camera's creator
    so viewers (members) read the owner's pipeline, not their own."""
    owner = getattr(cam, "user_id", None)
    return int(owner) if owner is not None else int(fallback_user_id)


def _owner_id_for_site(site: Any, ctx: OrgContext) -> int:
    """Pipeline owner for a site: its creator, falling back to the actor."""
    owner = getattr(site, "user_id", None)
    return int(owner) if owner is not None else int(ctx.user.id)


async def _ensure_stream_camera_access(db: AsyncSession, cam: Any, user: User) -> int:
    """Access check for SSE handlers that resolve the user from a token
    (no `OrgContext` dependency available). Returns the owner user id."""
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if bool(getattr(user, "is_platform_admin", False)):
        return _camera_owner_id(cam, int(user.id))
    resolved = await AuthzService.resolve_user_org(db, user=user)
    if resolved is None:
        raise HTTPException(status_code=403, detail="You do not belong to any organization.")
    org_id, role = resolved
    cam_org = getattr(cam, "org_id", None)
    if cam_org is not None and int(cam_org) != int(org_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    accessible = await AuthzService.accessible_site_uuids(
        db, user=user, org_id=org_id, role=role
    )
    if accessible is not None and getattr(cam, "site_uuid", None) not in accessible:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _camera_owner_id(cam, int(user.id))


async def _ensure_camera_access(db: AsyncSession, cam: Any, ctx: OrgContext) -> None:
    """404 unless the camera is in the caller's org and (for plain members)
    on a site they have been granted access to.

    Platform admins in super context (ctx.org_id is None) skip both checks.
    """
    if cam is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    if ctx.org_id is None:
        return
    cam_org = getattr(cam, "org_id", None)
    if cam_org is not None and int(cam_org) != int(ctx.org_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    accessible = await AuthzService.accessible_site_uuids(
        db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
    )
    if accessible is not None and getattr(cam, "site_uuid", None) not in accessible:
        raise HTTPException(status_code=404, detail="Camera not found")


def _camera_webrtc_url(cam: Any) -> Optional[str]:
    return resolve_camera_webrtc_url(
        camera_code=getattr(cam, "camera_code", None),
        stored_url=getattr(cam, "webrtc_url", None),
    )


def _tri_state(cam: Any, attr: str) -> str:
    """Read a tri-state camera column, defaulting to "inherit"."""
    return str(getattr(cam, attr, "inherit") or "inherit")


def _camera_with_config_out(
    cam_out: Any, *, webrtc_url: Optional[str]
) -> CameraWithConfigSchema:
    """Build the camera response from a `CameraOut` (the manager's DTO).

    `CameraOut` names the flags `enabled` / `detection_enabled` / …, while the
    HTTP contract uses the `is_` prefix, so the mapping cannot be automatic.
    """
    now = datetime.now(timezone.utc)
    return CameraWithConfigSchema(
        camera_uuid=cam_out.camera_uuid,
        camera_code=cam_out.camera_code,
        name=getattr(cam_out, "name", None),
        location=getattr(cam_out, "location", None),
        site_uuid=cam_out.site_uuid,
        device_uuid=cam_out.device_uuid,
        source_url=cam_out.source_url,
        webrtc_url=webrtc_url,
        is_enabled=cam_out.enabled,
        is_detection_enabled=cam_out.detection_enabled,
        is_notification_enabled=cam_out.notification_enabled,
        notification_trigger_mode=_tri_state(cam_out, "notification_trigger_mode"),
        camera_playback_enabled=_tri_state(cam_out, "camera_playback_enabled"),
        use_site_schedule=bool(getattr(cam_out, "use_site_schedule", True)),
        roi=cam_out.roi,
        configuration=cam_out.configuration,
        timezone=cam_out.timezone,
        created_at=getattr(cam_out, "created_at", None) or now,
        updated_at=getattr(cam_out, "updated_at", None) or now,
    )

async def _resolve_stream_user(
    *,
    request: Request,
    access_token: Optional[str],
) -> CachedUserSnapshot:
    """Request-shaped wrapper over the shared stream-auth resolver."""
    sf = _session_factory_from_app(request)
    if sf is None:
        raise HTTPException(status_code=503, detail=DB_UNAVAILABLE)
    return await resolve_stream_user(
        auth_header=request.headers.get("authorization", ""),
        access_token=access_token,
        cache=get_user_snapshot_cache(request),
        session_factory=sf,
    )


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

# Serialization lives in application/services/detection_stream.py so the public
# wall stream reuses exactly this shape. Kept under the old private names here to
# avoid churning the call sites below.
_normalize_box_px = normalize_box_px
_resp_to_detection_out = resp_to_detection_out


# -------------------------
# Cameras
# -------------------------
@router.get("/source-schemes")
async def get_source_schemes(_user: User = Depends(get_current_user)):
    """Camera source-URL schemes the platform accepts.

    The single source of truth is ``core.source_url`` (same data drives
    server-side validation), so the frontend can render exactly what users may
    enter without hardcoding/drifting. Declared before ``/{camera_uuid}`` so the
    static path matches first.
    """
    from core.source_url import supported_source_schemes
    return supported_source_schemes()


@router.get("", response_model=List[CameraSchema])
async def list_cameras(
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
    site_uuid: uuid.UUID = None,  # keep query param; if None -> 422 below
):
    """
    List cameras from Azure DB for a site.
    Uses camera.webrtc_url for playback in the frontend.
    """
    if site_uuid is None:
        raise HTTPException(status_code=422, detail="site_uuid query param is required")

    # Enforce org + (for members) site-level access. Platform admins in super
    # context (org_id is None) bypass the org scoping check entirely.
    if ctx.org_id is not None:
        accessible = await AuthzService.accessible_site_uuids(
            db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
        )
        if accessible is not None and site_uuid not in accessible:
            raise HTTPException(status_code=404, detail="Site not found")

    repo = ChannelRepository()

    cams = await repo.list_cameras(
        db,
        site_uuid=site_uuid,
        org_id=ctx.org_id,
        include_device=True,
    )

    return [
        CameraSchema(
            camera_uuid=cam.camera_uuid,
            camera_code=cam.camera_code,
            name=getattr(cam, "name", None),
            location=getattr(cam, "location", None),
            site_uuid=cam.site_uuid,
            device_uuid=getattr(getattr(cam, "device", None), "device_uuid", None),
            source_url=cam.source_url,
            webrtc_url=_camera_webrtc_url(cam),
            is_enabled=cam.is_enabled,
            is_detection_enabled=cam.is_detection_enabled,
            is_notification_enabled=cam.is_notification_enabled,
            notification_trigger_mode=_tri_state(cam, "notification_trigger_mode"),
            camera_playback_enabled=_tri_state(cam, "camera_playback_enabled"),
            use_site_schedule=bool(getattr(cam, "use_site_schedule", True)),
            roi=getattr(cam, "roi", None),
            created_at=cam.created_at,
            updated_at=cam.updated_at,
        )
        for cam in cams
    ]

@router.get("/{camera_uuid}", response_model=CameraWithConfigSchema)
async def get_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, cfg, _pid = full
    await _ensure_camera_access(db, cam, ctx)

    return CameraWithConfigSchema(
        camera_uuid=cam.camera_uuid,
        camera_code=cam.camera_code,
        name=getattr(cam, "name", None),
        location=getattr(cam, "location", None),
        site_uuid=cam.site_uuid,
        device_uuid=getattr(getattr(cam, "device", None), "device_uuid", None),
        source_url=cam.source_url,
        webrtc_url=_camera_webrtc_url(cam),
        is_enabled=cam.is_enabled,
        is_detection_enabled=cam.is_detection_enabled,
        is_notification_enabled=cam.is_notification_enabled,
        notification_trigger_mode=_tri_state(cam, "notification_trigger_mode"),
        camera_playback_enabled=_tri_state(cam, "camera_playback_enabled"),
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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
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
    await _ensure_camera_access(db, cam, ctx)

    # Validate camera has required fields
    source_url = getattr(cam, "source_url", None)
    camera_code = getattr(cam, "camera_code", None)
    if not camera_code:
        raise HTTPException(status_code=409, detail="Camera code not set")
    if not source_url:
        raise HTTPException(status_code=409, detail="RTSP URL not configured")
    
    # Provision stream in MediaMTX if not already done
    try:
        webrtc_client = WebRTCGatewayClient()
        webrtc_url = await webrtc_client.ensure_stream(
            stream_key=camera_code,
            source_url=source_url
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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    await _ensure_camera_access(db, cam, ctx)

    pipeline = await manager.get_activepipeline(user_id=_camera_owner_id(cam, ctx.user.id))

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
        raise HTTPException(status_code=503, detail=DB_UNAVAILABLE)

    user = await _resolve_stream_user(request=request, access_token=access_token)
    async with sf() as db:
        repo = ChannelRepository()
        full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
        if not full:
            raise HTTPException(status_code=404, detail="Camera not found")
        cam, _cfg, _pid = full
        owner_id = await _ensure_stream_camera_access(db, cam, user)

    pipeline = await manager.get_activepipeline(user_id=owner_id)

    gen = stream_camera_detections(
        pipeline=pipeline,
        camera_uuid=str(camera_uuid),
        after_ts_ms=after_ts_ms,
        after_seq=after_seq,
        timeout_ms=timeout_ms,
        normalize=normalize,
        is_disconnected=request.is_disconnected,
    )

    return StreamingResponse(gen, media_type="text/event-stream")


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
        raise HTTPException(status_code=503, detail=DB_UNAVAILABLE)

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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    repo = ChannelRepository()
    if site_uuid is None:
        raise HTTPException(status_code=400, detail="site_uuid is required")

    if ctx.org_id is not None:
        accessible = await AuthzService.accessible_site_uuids(
            db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
        )
        if accessible is not None and site_uuid not in accessible:
            raise HTTPException(status_code=404, detail="Site not found")

    cams = await repo.list_cameras(db, site_uuid=site_uuid, org_id=ctx.org_id)

    owner_id = _camera_owner_id(cams[0], ctx.user.id) if cams else int(ctx.user.id)
    pipeline = await manager.get_activepipeline(user_id=owner_id)

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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_CAMERAS)),
):
    try:
        data = payload.model_dump(exclude_none=True)

        # The target site must belong to the caller's org; the camera's
        # org ownership is then derived from the site in the repository.
        site_uuid = data.get("site_uuid")
        if site_uuid is None:
            raise HTTPException(status_code=422, detail="site_uuid is required")
        site = await SiteRepository().get_site(
            db, org_id=ctx.org_id, site_uuid=site_uuid
        )
        owner_id = _owner_id_for_site(site, ctx)

        data["user_id"] = owner_id
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

        cam_out = result.cameras[0]
        
        # Provision WebRTC stream in MediaMTX after camera creation
        camera_code = getattr(cam_out, "camera_code", None)
        source_url = getattr(cam_out, "source_url", None)
        webrtc_url = None
        
        if camera_code and source_url:
            try:
                webrtc_client = WebRTCGatewayClient()
                webrtc_url = await webrtc_client.ensure_stream(
                    stream_key=camera_code,
                    source_url=source_url
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
        return _camera_with_config_out(cam_out, webrtc_url=webrtc_url)
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
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_CAMERAS)),
):
    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam, _cfg, _pid = full
    await _ensure_camera_access(db, cam, ctx)
    owner_id = _camera_owner_id(cam, ctx.user.id)
    prev_device_uuid = getattr(cam, "device_uuid", None)

    pipeline = await manager.get_activepipeline(user_id=owner_id)
    patch_payload = payload.model_dump(exclude_unset=True)
    patch_payload.pop("user_id", None)

    ev = ChannelEditEvent(
        channel_id=camera_uuid,
        configs=patch_payload,
        created_at=datetime.now(timezone.utc),
    )

    try:
        result = await manager.update_pipeline(pipeline.pipeline_id, [ev], user_id=owner_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not result or not result.cameras:
        raise HTTPException(status_code=500, detail="Failed to edit camera")

    cam_out = result.cameras[0]
    new_device_uuid = getattr(cam_out, "device_uuid", None)
    if prev_device_uuid is not None and str(prev_device_uuid) != str(new_device_uuid):
        affected = [d for d in (prev_device_uuid, new_device_uuid) if d is not None]
        try:
            await manager.reconcile_devices_best_effort(
                user_id=owner_id,
                device_uuids=affected,
                org_id=ctx.org_id,
            )
        except Exception:
            logger.warning(
                "Best-effort reconcile after device re-assignment failed for camera %s "
                "(old_device=%s new_device=%s); edge may be stale until next sync.",
                cam_out.camera_uuid,
                prev_device_uuid,
                new_device_uuid,
                exc_info=True,
            )

    from routes.notifications_routes import invalidate_camera_mode_cache

    await invalidate_camera_mode_cache(cam_out.camera_uuid)
    return _camera_with_config_out(cam_out, webrtc_url=_camera_webrtc_url(cam_out))


from routes._background import _delete_blobs_background, _spawn_bg_task  # noqa: E402


async def _cleanup_camera_runtime(
    manager: Manager,
    *,
    user_id: int,
    camera_uuid: uuid.UUID,
    camera_code: Optional[str],
    device_urls: List[str],
) -> None:
    """Best-effort teardown of one camera's runtime presence.

    Deliberately uses `get_loaded_pipeline`, never `get_activepipeline`: the
    latter *builds and starts* a pipeline, so asking for one during a delete
    would spin up the very thing being torn down.
    """
    device_url_targets = list(
        dict.fromkeys(url.strip() for url in device_urls if url and url.strip())
    )

    try:
        active_pipeline = manager.get_loaded_pipeline(user_id=user_id)
    except Exception:
        logger.warning(
            "[Camera Delete] Could not inspect loaded pipeline for cam=%s",
            camera_uuid,
            exc_info=True,
        )
        active_pipeline = None

    # The running channel may point at a device the DB row no longer names.
    if active_pipeline is not None:
        try:
            cfg = await active_pipeline.get_channel_config(camera_uuid)
            runtime_device_url = str(getattr(cfg, "device_url", "") or "").strip()
            if runtime_device_url and runtime_device_url not in device_url_targets:
                device_url_targets.append(runtime_device_url)
        except Exception:
            logger.warning(
                "[Camera Delete] Failed reading pipeline config for cam=%s",
                camera_uuid,
                exc_info=True,
            )

    for dev_url in device_url_targets:
        try:
            await manager.edge.delete_camera(
                device_url=dev_url, camera_uuid=str(camera_uuid)
            )
        except Exception as exc:
            logger.warning(
                "[Camera Delete] Edge delete failed cam=%s url=%s: %s",
                camera_uuid,
                dev_url,
                exc,
            )

    if camera_code:
        try:
            await manager.webrtc.delete_stream(stream_key=str(camera_code))
        except Exception as exc:
            logger.warning(
                "[Camera Delete] WebRTC delete failed cam=%s code=%s: %s",
                camera_uuid,
                camera_code,
                exc,
            )

    if active_pipeline is not None:
        try:
            await active_pipeline.remove_channel(camera_uuid)
        except Exception as exc:
            logger.warning("[Camera Delete] Pipeline evict failed cam=%s: %s", camera_uuid, exc)



@router.delete("/{camera_uuid}")
async def delete_camera(
    camera_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    manager: Manager = Depends(get_manager),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_CAMERAS)),
):
    """Delete a camera and everything attached to it.

    Ordering matters throughout:

    1. Disable the camera in the DB first. Reconcile decides what to provision
       from `is_enabled`/`is_detection_enabled`, so a reconcile firing between
       the edge teardown and the row delete would re-add the camera.
    2. Stop it on the edge, WebRTC and the in-memory pipeline, so nothing new
       is written while the delete runs.
    3. Snapshot blob keys and notification ids *before* the row goes: deleting
       the camera CASCADEs its VideoRecords away and SET NULLs
       `Notification.camera_uuid`, which would strand both.
    4. Delete the camera row, then sweep the heavy tables and blobs in the
       background so the response is fast.
    """
    from routes.notifications_routes import invalidate_camera_mode_cache

    logger.info("[Camera Delete] Starting deletion of camera=%s", camera_uuid)

    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        raise HTTPException(status_code=404, detail="Camera not found")
    cam, _cfg, _pid = full
    await _ensure_camera_access(db, cam, ctx)
    owner_id = _camera_owner_id(cam, ctx.user.id)

    cam_code: Optional[str] = getattr(cam, "camera_code", None)
    site_uuid = cam.site_uuid
    device_url = str(getattr(getattr(cam, "device", None), "device_url", "") or "").strip()
    device_urls = [device_url] if device_url else []

    # 1. Close the reconcile race window before touching anything external.
    await repo.disable_cameras(db, camera_uuids=[camera_uuid], user_id=owner_id)
    await db.commit()

    # 2. Stop the camera everywhere it is running.
    notif_svc = getattr(manager, "notification_service", None) if manager else None
    if notif_svc is not None:
        try:
            purge_fn = getattr(notif_svc, "purge_deleted_site_runtime_state", None)
            if callable(purge_fn):
                await purge_fn(
                    user_id=owner_id, site_uuid=site_uuid, camera_uuids=[camera_uuid]
                )
            else:
                notif_svc.invalidate_camera_roi_state(str(camera_uuid))
        except Exception:
            logger.warning(
                "[Camera Delete] Notification service purge failed", exc_info=True
            )

    if manager is not None:
        try:
            await asyncio.wait_for(
                _cleanup_camera_runtime(
                    manager,
                    user_id=owner_id,
                    camera_uuid=camera_uuid,
                    camera_code=cam_code,
                    device_urls=device_urls,
                ),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[Camera Delete] Runtime cleanup timed out for cam=%s; proceeding with DB delete",
                camera_uuid,
            )
        except Exception:
            logger.warning(
                "[Camera Delete] Runtime cleanup failed for cam=%s", camera_uuid, exc_info=True
            )

    # 3. Snapshot what the delete is about to make unreachable.
    async with AsyncSessionLocal() as session:
        video_clip_keys = await VideoRepository().list_storage_keys(
            session, camera_uuid=camera_uuid
        )
        notification_ids = await NotificationRepository().list_notification_ids(
            session, camera_uuid=camera_uuid
        )
    logger.info(
        "[Camera Delete] Snapshotted %s clip blob keys and %s notification ids",
        len(video_clip_keys),
        len(notification_ids),
    )

    # 4. Drop the camera row, then hand the heavy tables to the background.
    async with AsyncSessionLocal() as del_db:
        await repo.delete_camera(del_db, camera_uuid=camera_uuid)
        await del_db.commit()

    await invalidate_camera_mode_cache(camera_uuid)

    async def _heavy_table_cleanup_task(
        cam_uuid: uuid.UUID,
        notif_ids: List[int],
        pre_video_keys: List[str],
    ):
        """Delete the camera's notifications and blobs after the response.

        Notifications are matched by id, not camera_uuid: the column was
        SET NULL when the camera row went away.
        """
        logger.info("[Camera Cleanup Task] Starting cleanup for camera=%s", cam_uuid)
        repo_for_delete = SiteRepository()
        alert_blob_keys: List[str] = []
        clip_blob_keys: List[str] = list(pre_video_keys)

        try:
            if notif_ids:
                for i in range(0, len(notif_ids), 2000):
                    batch_ids = notif_ids[i : i + 2000]
                    await repo_for_delete._batch_delete(
                        AsyncSessionLocal,
                        table=Notification,
                        where_clause=Notification.id.in_(batch_ids),
                        batch_size=2000,
                        label="camera_notifications",
                        extract_col=Notification.payload,
                        extract_alert_fn=extract_image_storage_key,
                        extract_clip_fn=_extract_notification_clip_storage_keys,
                        alert_keys_out=alert_blob_keys,
                        clip_keys_out=clip_blob_keys,
                    )

            if alert_blob_keys:
                _spawn_bg_task(
                    _delete_blobs_background(
                        alert_blob_keys,
                        service_cls=AlertImageStorageService,
                        label="alert image",
                    ),
                    name=f"delete_camera_alert_blobs:{cam_uuid}",
                )
            if clip_blob_keys:
                _spawn_bg_task(
                    _delete_blobs_background(
                        clip_blob_keys, service_cls=EventClipService, label="clip"
                    ),
                    name=f"delete_camera_clip_blobs:{cam_uuid}",
                )
            logger.info("[Camera Cleanup Task] Cleanup COMPLETE for camera=%s", cam_uuid)
        except Exception:
            logger.exception(
                "[Camera Cleanup Task] Failed cleanup for camera=%s", cam_uuid
            )

    _spawn_bg_task(
        _heavy_table_cleanup_task(camera_uuid, notification_ids, video_clip_keys),
        name=f"delete_camera_heavy_tables:{camera_uuid}",
    )

    logger.info("[Camera Delete] camera=%s deleted", camera_uuid)
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
            logger.warning(
                "Snapshot access denied for camera %s: %s", camera_uuid, e.detail
            )
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Please ensure you are logged in and have a valid token.",
            )
        raise

    repo = ChannelRepository()
    full = await repo.get_camera_full(db, camera_uuid=camera_uuid)
    if not full:
        logger.warning("Camera %s not found for user %s", camera_uuid, user.id)
        raise HTTPException(status_code=404, detail="Camera not found")

    cam, _cfg, _pid = full
    await _ensure_stream_camera_access(db, cam, user)

    device_url = str(getattr(getattr(cam, "device", None), "device_url", "") or "").strip()
    if not device_url:
        logger.warning("Camera %s has no device assigned or device_url missing", camera_uuid)
        raise HTTPException(
            status_code=409,
            detail="Camera is not properly configured. Device URL is missing. Please contact administrator.",
        )

    return await _fetch_device_snapshot(device_url=device_url, camera_uuid=cam.camera_uuid)


@router.get("/{camera_uuid}/stream.mjpg")
async def stream_mjpeg(camera_uuid: uuid.UUID):
    raise HTTPException(
        status_code=410,
        detail="MJPEG streaming disabled. Use WebRTC playback URL for video.",
    )
