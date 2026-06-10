"""
Authorization service for the multi-tenant RBAC system.

Centralises the read-side queries that decide whether the current
user has the privilege to act on a given organization or site.

Design notes:

  * Reads only. Writes go through `OrganizationRepository`.
  * Access is stored in the unified `access_grants` table: a user holds a
    `Role` (scoped to ``org`` or ``site``) on a context (an org or a site).
    Every query here uses explicit JOINs over `AccessGrant`/`Role`/
    `Permission`; the ORM relationships are declared ``lazy="raise"`` so
    nothing loads implicitly.
  * Authorization is **permission-first**: routes declare a `Permission` and
    `has_permission` / `require_permission` answer it by resolving the
    caller's role from their grant and consulting the `ROLE_PERMISSIONS`
    catalog (the single source of truth). The role-name helpers
    (`get_org_role`, `get_site_role`, `resolve_user_org`) exist to build the
    `OrgContext` used for query scoping, not to gate routes.
  * Platform Admins bypass every check.
  * Org Admins implicitly satisfy every site-level check inside their
    own organization, even without an explicit site grant.
"""
from __future__ import annotations

import uuid
from typing import Optional, Union

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import (
    AccessGrant,
    Permission as PermissionModel,
    Role,
    Site,
    User,
    role_permissions,
)
from core.security.roles import OrgRole, Permission, RoleScope, SiteRole


def _permission_name(permission: Union[Permission, str]) -> str:
    """Normalize a `Permission` enum (or raw string) to its string value."""
    return permission.value if isinstance(permission, Permission) else str(permission)


class AuthzService:
    # ------------------------------------------------------------------
    # Platform
    # ------------------------------------------------------------------
    @staticmethod
    def ensure_platform_admin(user: User) -> None:
        """Raise 403 unless the user holds the platform-admin flag."""
        if not bool(getattr(user, "is_platform_admin", False)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Platform admin privileges required.",
            )

    # ------------------------------------------------------------------
    # Organization
    # ------------------------------------------------------------------
    @staticmethod
    async def get_org_role(
        db: AsyncSession, *, user_id: int, org_id: int
    ) -> Optional[str]:
        """Return the user's org `Role` name for `org_id`, or None."""
        res = await db.execute(
            select(Role.name)
            .select_from(AccessGrant)
            .join(Role, Role.id == AccessGrant.role_id)
            .where(
                AccessGrant.user_id == user_id,
                AccessGrant.org_id == org_id,
                Role.scope == RoleScope.ORG.value,
            )
            .limit(1)
        )
        return res.scalar_one_or_none()

    @staticmethod
    async def resolve_user_org(
        db: AsyncSession, *, user: User
    ) -> Optional[tuple[int, str]]:
        """Return the `(org_id, role)` for the user's single organization.

        A non-platform user is expected to belong to exactly one
        organization (product invariant). Returns the first grant ordered
        by creation if more than one ever exists, or None when the user has
        no org grant at all.
        """
        res = await db.execute(
            select(AccessGrant.org_id, Role.name)
            .select_from(AccessGrant)
            .join(Role, Role.id == AccessGrant.role_id)
            .where(
                AccessGrant.user_id == int(user.id),
                AccessGrant.org_id.is_not(None),
                Role.scope == RoleScope.ORG.value,
            )
            .order_by(AccessGrant.created_at.asc())
            .limit(1)
        )
        row = res.first()
        if row is None:
            return None
        return int(row[0]), str(row[1])

    @classmethod
    async def is_effective_org_admin(
        cls, db: AsyncSession, *, user: User, org_id: int
    ) -> bool:
        """Whether the user wields full org-admin powers over `org_id`.

        True when:
          * the user is a platform admin, or
          * their org role is ADMIN, or
          * their org role is OPERATOR **and** they hold a site-admin grant
            on any site within `org_id` (operator-as-admin elevation).
        """
        if bool(getattr(user, "is_platform_admin", False)):
            return True

        role = await cls.get_org_role(db, user_id=int(user.id), org_id=int(org_id))
        if role == OrgRole.ADMIN.value:
            return True
        if role == OrgRole.OPERATOR.value:
            site_admin = (
                await db.execute(
                    select(AccessGrant.id)
                    .join(Role, Role.id == AccessGrant.role_id)
                    .join(Site, Site.site_uuid == AccessGrant.site_uuid)
                    .where(
                        AccessGrant.user_id == int(user.id),
                        Role.name == SiteRole.ADMIN.value,
                        Role.scope == RoleScope.SITE.value,
                        Site.org_id == int(org_id),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return site_admin is not None
        return False

    @classmethod
    async def accessible_site_uuids(
        cls, db: AsyncSession, *, user: User, org_id: Optional[int], role: Optional[str] = None
    ) -> Optional[set[uuid.UUID]]:
        """Set of site UUIDs the user may see in `org_id`.

        Returns `None` as a sentinel meaning "every site in scope":
        every site for platform admins (or `org_id is None` super
        context), every site in the org for effective admins/operators.
        Plain members only see sites they hold a site grant for.
        """
        if bool(getattr(user, "is_platform_admin", False)) or org_id is None:
            return None

        if role is None:
            role = await cls.get_org_role(db, user_id=int(user.id), org_id=int(org_id))
        if role in (OrgRole.ADMIN.value, OrgRole.OPERATOR.value):
            return None
        if await cls.is_effective_org_admin(db, user=user, org_id=int(org_id)):
            return None

        rows = (
            await db.execute(
                select(AccessGrant.site_uuid)
                .join(Site, Site.site_uuid == AccessGrant.site_uuid)
                .where(
                    AccessGrant.user_id == int(user.id),
                    AccessGrant.site_uuid.is_not(None),
                    Site.org_id == int(org_id),
                )
            )
        ).scalars().all()
        return {r if isinstance(r, uuid.UUID) else uuid.UUID(str(r)) for r in rows}

    # ------------------------------------------------------------------
    # Site
    # ------------------------------------------------------------------
    @staticmethod
    async def _get_site_org_id(
        db: AsyncSession, site_uuid: uuid.UUID
    ) -> Optional[int]:
        res = await db.execute(
            select(Site.org_id).where(Site.site_uuid == site_uuid).limit(1)
        )
        return res.scalar_one_or_none()

    @classmethod
    async def get_site_role(
        cls,
        db: AsyncSession,
        *,
        user_id: int,
        site_uuid: uuid.UUID,
    ) -> Optional[str]:
        """Return the user's site `Role` name for `site_uuid`, or None."""
        res = await db.execute(
            select(Role.name)
            .select_from(AccessGrant)
            .join(Role, Role.id == AccessGrant.role_id)
            .where(
                AccessGrant.user_id == user_id,
                AccessGrant.site_uuid == site_uuid,
                Role.scope == RoleScope.SITE.value,
            )
            .limit(1)
        )
        return res.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Permissions (fine-grained ACL surface)
    # ------------------------------------------------------------------
    @classmethod
    async def has_permission(
        cls,
        db: AsyncSession,
        *,
        user: User,
        permission: Union[Permission, str],
        org_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
    ) -> bool:
        """Whether `user` holds `permission` in the given context.

        A single explicit JOIN `AccessGrant -> Role -> role_permissions ->
        Permission` resolves the answer. The two product short-circuits are
        applied first: platform admins always pass, and an effective Org
        Admin satisfies any check inside their org (covering the implicit
        "org admin can do anything to its sites" rule).
        """
        if bool(getattr(user, "is_platform_admin", False)):
            return True

        perm_name = _permission_name(permission)

        # Resolve the org context (directly, or via the site's org) so the
        # org-admin short-circuit and org-grant matching both work.
        effective_org_id = org_id
        if effective_org_id is None and site_uuid is not None:
            effective_org_id = await cls._get_site_org_id(db, site_uuid)

        if effective_org_id is not None and await cls.is_effective_org_admin(
            db, user=user, org_id=int(effective_org_id)
        ):
            return True

        stmt = (
            select(AccessGrant.id)
            .select_from(AccessGrant)
            .join(Role, Role.id == AccessGrant.role_id)
            .join(role_permissions, role_permissions.c.role_id == Role.id)
            .join(PermissionModel, PermissionModel.id == role_permissions.c.permission_id)
            .where(
                AccessGrant.user_id == int(user.id),
                PermissionModel.name == perm_name,
            )
        )

        # Constrain to the requested context. A site permission is satisfied
        # either by a grant on that exact site or by an org grant (which the
        # org-admin role carries the site permissions for).
        if site_uuid is not None:
            conditions = [AccessGrant.site_uuid == site_uuid]
            if effective_org_id is not None:
                conditions.append(AccessGrant.org_id == int(effective_org_id))
            stmt = stmt.where(or_(*conditions))
        elif org_id is not None:
            stmt = stmt.where(AccessGrant.org_id == int(org_id))

        res = await db.execute(stmt.limit(1))
        return res.first() is not None

    @classmethod
    async def require_permission(
        cls,
        db: AsyncSession,
        *,
        user: User,
        permission: Union[Permission, str],
        org_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
    ) -> bool:
        """Assert `user` holds `permission` in context, else raise 403."""
        if await cls.has_permission(
            db, user=user, permission=permission, org_id=org_id, site_uuid=site_uuid
        ):
            return True
        perm_name = _permission_name(permission)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required permission: '{perm_name}' for this resource.",
        )
