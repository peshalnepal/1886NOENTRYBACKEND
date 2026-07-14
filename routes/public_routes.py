# routes/public_routes.py
"""
Public, UNAUTHENTICATED endpoints for published camera walls.

This is the only router in the app with no auth dependency, which is why it is a
separate module: nothing declared here can inherit a `RequirePermission` by
accident, and anything added here is obviously public by construction.

Security notes for anyone editing this file:

  * Never return a `CameraSchema`. It requires `source_url`, which can embed RTSP
    credentials. Public responses use `PublicWallCameraSchema` and nothing else.
  * Never call `manager.get_activepipeline()`. It *creates and starts* a pipeline,
    which would let an anonymous request spin up model inference on the GPU box.
    Only `get_loaded_pipeline()` (a plain registry lookup) is safe here.
  * Every failure path returns the same 404 with the same detail, so a caller
    cannot tell "expired" from "revoked" from "never existed".
  * The WHEP URLs handed out here are served anonymously by MediaMTX. Revoking a
    share token stops discovery of the wall, NOT playback for someone who already
    saved a stream URL. See mediamtx.yml.
"""

import logging
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from application.repositories.wall_repository import WallRepository, share_is_live
from application.services.detection_stream import stream_multi_camera_detections
from application.services.webrtcgateway import resolve_camera_webrtc_url
from core.database import db_manager
from core.schemas import PublicWallCameraSchema, PublicWallSchema
from dependencies import get_manager_optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public", tags=["public"])

# One detail string for every failure, so the response never distinguishes
# "no such token" from "revoked" from "expired".
_NOT_FOUND = "This link is not available."

REAUTHORIZE_EVERY_S = 60.0


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_NOT_FOUND)


def _session_factory(request: Request):
    sf = getattr(request.app.state, "session_factory", None)
    if sf is not None:
        return sf
    return getattr(db_manager, "AsyncSessionLocal", None)


async def _load_shared_wall(request: Request, share_token: str):
    """Resolve a live share token to (wall, [(camera, site, position), ...])."""
    sf = _session_factory(request)
    if sf is None:
        raise HTTPException(status_code=503, detail="Service unavailable")

    repo = WallRepository()
    async with sf() as db:
        wall = await repo.get_wall_by_share_token(db, share_token=share_token)
        if wall is None or not share_is_live(wall):
            raise _not_found()

        rows = await repo.list_wall_cameras(db, wall_uuid=wall.wall_uuid)
        return wall, rows


@router.get("/walls/{share_token}", response_model=PublicWallSchema)
async def get_public_wall(share_token: str, request: Request):
    wall, rows = await _load_shared_wall(request, share_token)

    cameras: List[PublicWallCameraSchema] = [
        PublicWallCameraSchema(
            camera_uuid=cam.camera_uuid,
            name=cam.name,
            location=cam.location,
            site_name=site.name,
            webrtc_url=resolve_camera_webrtc_url(
                camera_code=cam.camera_code, stored_url=cam.webrtc_url
            ),
            position=position,
        )
        for cam, site, position in rows
    ]

    return PublicWallSchema(name=wall.name, cameras=cameras)


@router.get("/walls/{share_token}/detections/stream")
async def stream_public_wall_detections(
    share_token: str,
    request: Request,
    timeout_ms: int = 30000,
    normalize: bool = True,
):
    """One SSE carrying detections for every camera on the wall.

    Multiplexed on purpose: a stream per camera would use up the browser's
    six-connections-per-origin budget on the wall alone and stall the page's own
    requests.
    """
    wall, rows = await _load_shared_wall(request, share_token)

    manager = get_manager_optional(request)

    cameras: List[Tuple[str, Any]] = []
    if manager is not None:
        for cam, _site, _position in rows:
            # Pipelines are keyed by the camera's owner. A NULL/0 owner must be
            # skipped, not passed through: both manager lookups do
            # `int(user_id or default_user_id)` and would silently attach this
            # camera to the default user's pipeline.
            owner_id = getattr(cam, "user_id", None)
            if not owner_id:
                continue

            pipeline = manager.get_loaded_pipeline(user_id=int(owner_id))
            if pipeline is None:
                # Nothing running for this owner. Never create one from an
                # anonymous request — just carry no detections for this camera.
                continue

            cameras.append((str(cam.camera_uuid), pipeline))

    sf = _session_factory(request)

    async def still_authorized() -> bool:
        """Re-checked periodically so revoking a link also drops live viewers."""
        if sf is None:
            return False
        try:
            async with sf() as db:
                current = await WallRepository().get_wall_by_share_token(
                    db, share_token=share_token
                )
                return current is not None and share_is_live(current)
        except Exception:
            logger.warning("Share re-validation failed for wall %s", wall.wall_uuid, exc_info=True)
            # Fail closed: drop the viewer rather than stream on a token we can
            # no longer verify.
            return False

    gen = stream_multi_camera_detections(
        cameras=cameras,
        timeout_ms=timeout_ms,
        normalize=normalize,
        is_disconnected=request.is_disconnected,
        still_authorized=still_authorized,
        reauthorize_every_s=REAUTHORIZE_EVERY_S,
    )

    return StreamingResponse(gen, media_type="text/event-stream")
