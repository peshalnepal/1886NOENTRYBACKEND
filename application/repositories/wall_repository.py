"""
Custom wall persistence.

Owns the `walls` and `wall_cameras` tables. A wall is an org-owned, ordered set
of cameras that may span any number of sites, optionally published as a public
no-login view via a share token.

Transaction policy: never commits — the caller owns the transaction.

Read policy (applies to every camera read here, authenticated or public):
  * skip cameras whose site was soft-deleted (`Site.is_deleted`) — the
    `wall_cameras` FK cascade never fires for a soft delete, so those rows survive;
  * skip disabled cameras with `is_enabled IS NOT FALSE`, because the column is
    nullable and `== True` would drop legacy NULL rows.
"""

import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Camera, Site, Wall, WallCamera, utc_now

logger = logging.getLogger(__name__)

SHARE_TOKEN_BYTES = 32  # -> 43-char urlsafe string, 256 bits of entropy


def generate_share_token() -> str:
    return secrets.token_urlsafe(SHARE_TOKEN_BYTES)


def share_is_live(wall: Wall, *, now: Optional[datetime] = None) -> bool:
    """True when a wall's share link should currently resolve."""
    if not bool(wall.share_enabled):
        return False
    if not str(wall.share_token or "").strip():
        return False

    expires_at = wall.share_expires_at
    if expires_at is None:
        return True

    # aiomysql strips tzinfo on the way back out, so re-attach UTC before compare.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    return expires_at > (now or utc_now())


class WallRepository:
    """Stateless; instantiate per call."""

    async def list_walls(self, db: AsyncSession, *, org_id: int) -> List[Wall]:
        result = await db.execute(
            select(Wall).where(Wall.org_id == org_id).order_by(Wall.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_wall(
        self, db: AsyncSession, *, wall_uuid: uuid.UUID, org_id: Optional[int] = None
    ) -> Optional[Wall]:
        stmt = select(Wall).where(Wall.wall_uuid == wall_uuid)
        if org_id is not None:
            stmt = stmt.where(Wall.org_id == org_id)

        result = await db.execute(stmt)
        return result.scalars().first()

    async def get_wall_by_share_token(self, db: AsyncSession, *, share_token: str) -> Optional[Wall]:
        token = str(share_token or "").strip()
        if not token:
            return None

        result = await db.execute(select(Wall).where(Wall.share_token == token))
        return result.scalars().first()

    async def list_wall_cameras(
        self, db: AsyncSession, *, wall_uuid: uuid.UUID
    ) -> List[Tuple[Camera, Site, int]]:
        """Ordered (camera, site, position) for a wall, already filtered."""
        result = await db.execute(
            select(Camera, Site, WallCamera.position)
            .join(WallCamera, WallCamera.camera_uuid == Camera.camera_uuid)
            .join(Site, Site.site_uuid == Camera.site_uuid)
            .where(
                WallCamera.wall_uuid == wall_uuid,
                Site.is_deleted.is_(False),
                Camera.is_enabled.isnot(False),
            )
            .order_by(WallCamera.position.asc())
        )
        return [(row[0], row[1], int(row[2])) for row in result.all()]

    async def count_wall_cameras(self, db: AsyncSession, *, wall_uuid: uuid.UUID) -> int:
        """Every membership row, including ones a caller may not be allowed to see.

        Paired with the filtered read above to render "showing N of M".
        """
        result = await db.execute(
            select(WallCamera.id).where(WallCamera.wall_uuid == wall_uuid)
        )
        return len(result.all())

    async def create_wall(
        self, db: AsyncSession, *, org_id: int, created_by: Optional[int], name: str
    ) -> Wall:
        wall = Wall(
            wall_uuid=uuid.uuid4(),
            org_id=org_id,
            created_by=created_by,
            name=name,
        )
        db.add(wall)
        await db.flush()
        return wall

    async def set_wall_cameras(
        self, db: AsyncSession, *, wall_uuid: uuid.UUID, camera_uuids: Sequence[uuid.UUID]
    ) -> None:
        """Replace the whole ordered membership. Position is the list index."""
        await db.execute(
            delete(WallCamera)
            .where(WallCamera.wall_uuid == wall_uuid)
            .execution_options(synchronize_session=False)
        )

        for position, camera_uuid in enumerate(camera_uuids):
            db.add(
                WallCamera(
                    wall_uuid=wall_uuid,
                    camera_uuid=camera_uuid,
                    position=position,
                )
            )

        await db.flush()

    async def delete_wall(self, db: AsyncSession, *, wall: Wall) -> None:
        # wall_cameras rows go with it via ON DELETE CASCADE.
        await db.delete(wall)
        await db.flush()

    async def resolve_org_camera_uuids(
        self, db: AsyncSession, *, org_id: int, camera_uuids: Sequence[uuid.UUID]
    ) -> List[Tuple[uuid.UUID, uuid.UUID]]:
        """(camera_uuid, site_uuid) for the given cameras that really are in this org.

        Anything missing from the result does not exist, belongs to another org, or
        hangs off a soft-deleted site — the caller rejects the write.
        """
        if not camera_uuids:
            return []

        result = await db.execute(
            select(Camera.camera_uuid, Camera.site_uuid)
            .join(Site, Site.site_uuid == Camera.site_uuid)
            .where(
                Camera.camera_uuid.in_(list(camera_uuids)),
                Camera.org_id == org_id,
                Site.is_deleted.is_(False),
            )
        )
        return [(row[0], row[1]) for row in result.all()]

    async def set_share(
        self,
        db: AsyncSession,
        *,
        wall: Wall,
        enabled: bool,
        expires_at: Optional[datetime] = None,
        rotate: bool = False,
    ) -> Wall:
        if enabled:
            if rotate or not str(wall.share_token or "").strip():
                wall.share_token = generate_share_token()
            wall.share_enabled = True
            wall.share_expires_at = expires_at
        else:
            # Drop the token outright rather than just flipping the flag, so a
            # revoked link can never be resurrected by re-enabling sharing.
            wall.share_token = None
            wall.share_enabled = False
            wall.share_expires_at = None

        wall.updated_at = utc_now()
        await db.flush()
        return wall
