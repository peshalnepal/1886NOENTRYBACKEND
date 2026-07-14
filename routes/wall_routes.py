# routes/wall_routes.py
"""
Custom walls: org-owned, ordered camera sets that may span multiple sites, and
the share-link lifecycle that publishes one for public no-login viewing.

The public read side lives in `routes/public_routes.py` — deliberately a separate
router so that no auth dependency declared here can ever leak onto it.
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.wall_repository import WallRepository, share_is_live
from application.services.authz_service import AuthzService
from application.services.webrtcgateway import resolve_camera_webrtc_url
from core.database_orm import Wall
from core.schemas import (
    MAX_WALL_CAMERAS,
    WallCameraSchema,
    WallCreateSchema,
    WallEditSchema,
    WallSchema,
    WallShareRequest,
)
from core.security.roles import Permission
from dependencies import OrgContext, RequirePermission, get_async_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/walls", tags=["walls"])


def _require_org(ctx: OrgContext) -> int:
    """Walls are org-scoped. Platform admins act cross-tenant via /api/platform/*."""
    if ctx.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Walls are organization-scoped; select an organization.",
        )
    return int(ctx.org_id)


async def _wall_out(db: AsyncSession, ctx: OrgContext, wall: Wall) -> WallSchema:
    """Build the response, re-filtering cameras against the caller's site grants.

    Membership is validated when a wall is written, but walls are visible to every
    admin/operator in the org and site grants get revoked afterwards — so the read
    has to filter too, or a wall becomes a way around site-level access control.
    """
    repo = WallRepository()

    accessible = await AuthzService.accessible_site_uuids(
        db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
    )
    rows = await repo.list_wall_cameras(db, wall_uuid=wall.wall_uuid)
    total = await repo.count_wall_cameras(db, wall_uuid=wall.wall_uuid)

    cameras: List[WallCameraSchema] = []
    for cam, site, position in rows:
        if accessible is not None and cam.site_uuid not in accessible:
            continue

        cameras.append(
            WallCameraSchema(
                camera_uuid=cam.camera_uuid,
                name=cam.name,
                location=cam.location,
                site_uuid=cam.site_uuid,
                site_name=site.name,
                webrtc_url=resolve_camera_webrtc_url(
                    camera_code=cam.camera_code, stored_url=cam.webrtc_url
                ),
                position=position,
            )
        )

    live = share_is_live(wall)

    return WallSchema(
        wall_uuid=wall.wall_uuid,
        name=wall.name,
        cameras=cameras,
        total_camera_count=total,
        share_enabled=live,
        share_token=wall.share_token if live else None,
        share_expires_at=wall.share_expires_at,
        created_at=wall.created_at,
        updated_at=wall.updated_at,
    )


async def _load_wall(db: AsyncSession, *, wall_uuid: uuid.UUID, org_id: int) -> Wall:
    wall = await WallRepository().get_wall(db, wall_uuid=wall_uuid, org_id=org_id)
    if wall is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wall not found")
    return wall


async def _validate_camera_uuids(
    db: AsyncSession, ctx: OrgContext, org_id: int, camera_uuids: List[uuid.UUID]
) -> None:
    """Every camera must be in the caller's org AND in a site they can access."""
    if not camera_uuids:
        return

    if len(camera_uuids) > MAX_WALL_CAMERAS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"A wall may hold at most {MAX_WALL_CAMERAS} cameras.",
        )

    found = dict(
        await WallRepository().resolve_org_camera_uuids(
            db, org_id=org_id, camera_uuids=camera_uuids
        )
    )

    missing = [str(cu) for cu in camera_uuids if cu not in found]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more cameras were not found in this organization.",
        )

    accessible = await AuthzService.accessible_site_uuids(
        db, user=ctx.user, org_id=org_id, role=ctx.role
    )
    if accessible is None:
        return

    forbidden = [str(cu) for cu, site_uuid in found.items() if site_uuid not in accessible]
    if forbidden:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to one or more of the selected cameras.",
        )


@router.get("", response_model=List[WallSchema])
async def list_walls(
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    org_id = _require_org(ctx)
    walls = await WallRepository().list_walls(db, org_id=org_id)
    return [await _wall_out(db, ctx, wall) for wall in walls]


@router.post("", response_model=WallSchema, status_code=status.HTTP_201_CREATED)
async def create_wall(
    payload: WallCreateSchema,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    org_id = _require_org(ctx)
    await _validate_camera_uuids(db, ctx, org_id, payload.camera_uuids)

    repo = WallRepository()
    wall = await repo.create_wall(
        db, org_id=org_id, created_by=int(ctx.user.id), name=payload.name
    )
    await repo.set_wall_cameras(db, wall_uuid=wall.wall_uuid, camera_uuids=payload.camera_uuids)
    await db.commit()

    return await _wall_out(db, ctx, wall)


@router.get("/{wall_uuid}", response_model=WallSchema)
async def get_wall(
    wall_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    org_id = _require_org(ctx)
    wall = await _load_wall(db, wall_uuid=wall_uuid, org_id=org_id)
    return await _wall_out(db, ctx, wall)


@router.patch("/{wall_uuid}", response_model=WallSchema)
async def edit_wall(
    wall_uuid: uuid.UUID,
    payload: WallEditSchema,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    org_id = _require_org(ctx)
    wall = await _load_wall(db, wall_uuid=wall_uuid, org_id=org_id)

    repo = WallRepository()

    if payload.name is not None:
        wall.name = payload.name

    if payload.camera_uuids is not None:
        await _validate_camera_uuids(db, ctx, org_id, payload.camera_uuids)
        await repo.set_wall_cameras(
            db, wall_uuid=wall.wall_uuid, camera_uuids=payload.camera_uuids
        )

    await db.commit()
    return await _wall_out(db, ctx, wall)


@router.delete("/{wall_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_wall(
    wall_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    org_id = _require_org(ctx)
    wall = await _load_wall(db, wall_uuid=wall_uuid, org_id=org_id)

    await WallRepository().delete_wall(db, wall=wall)
    await db.commit()
    return None


@router.post("/{wall_uuid}/share", response_model=WallSchema)
async def publish_wall(
    wall_uuid: uuid.UUID,
    payload: WallShareRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    """Publish (or re-date) the wall's public link. Idempotent: keeps the token."""
    org_id = _require_org(ctx)
    wall = await _load_wall(db, wall_uuid=wall_uuid, org_id=org_id)

    await WallRepository().set_share(
        db, wall=wall, enabled=True, expires_at=payload.expires_at, rotate=False
    )
    await db.commit()

    logger.info("Wall %s published by user %s", wall_uuid, ctx.user.id)
    return await _wall_out(db, ctx, wall)


@router.post("/{wall_uuid}/share/rotate", response_model=WallSchema)
async def rotate_wall_share(
    wall_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    """Issue a new token, invalidating the old link."""
    org_id = _require_org(ctx)
    wall = await _load_wall(db, wall_uuid=wall_uuid, org_id=org_id)

    if not bool(wall.share_enabled):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This wall is not published; publish it first.",
        )

    await WallRepository().set_share(
        db, wall=wall, enabled=True, expires_at=wall.share_expires_at, rotate=True
    )
    await db.commit()

    logger.info("Wall %s share token rotated by user %s", wall_uuid, ctx.user.id)
    return await _wall_out(db, ctx, wall)


@router.delete("/{wall_uuid}/share", response_model=WallSchema)
async def revoke_wall_share(
    wall_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.WALL_MANAGE)),
):
    """Unpublish. Open viewers lose the wall within one re-validation cycle.

    This does NOT stop playback for anyone who already saved a raw WHEP URL —
    MediaMTX serves those anonymously. See mediamtx.yml.
    """
    org_id = _require_org(ctx)
    wall = await _load_wall(db, wall_uuid=wall_uuid, org_id=org_id)

    await WallRepository().set_share(db, wall=wall, enabled=False)
    await db.commit()

    logger.info("Wall %s share revoked by user %s", wall_uuid, ctx.user.id)
    return await _wall_out(db, ctx, wall)
