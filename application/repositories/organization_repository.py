"""
Repository for organizations and the unified `access_grants` table that
implements the multi-tenant RBAC hierarchy.

This is the single write surface for:

  * `organizations`   - tenant containers
  * `access_grants`   - User <-> (Org | Site) with a Role

The former `org_memberships` / `site_memberships` tables are gone; an
"org membership" is now an org-scoped grant and a "site membership" a
site-scoped grant. The public method names are preserved so route
handlers and the signup flow keep working unchanged.

All public methods are async and operate on an `AsyncSession`. None of
them commit; the caller owns the transaction so that route handlers can
compose multiple repo calls inside a single commit.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import (
    OrganizationCreateDTO,
    OrganizationUpdateDTO,
    OrgMembershipUpsertDTO,
    SiteMembershipUpsertDTO,
)
from core.database_orm import (
    AccessGrant,
    Organization,
    Role,
    Site,
    User,
    utc_now,
)
from core.security.roles import OrgRole, RoleScope, SiteRole


class OrganizationRepository:
    # Cache of (role_name, scope) -> role_id. Role rows are seeded once at
    # startup and never change id, so a process-lifetime cache is safe.
    _role_id_cache: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Grant query building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _grants_in_scope(selectable, scope: str):
        """JOIN a grant query to `roles`, restricted to one role scope."""
        return selectable.select_from(AccessGrant).join(
            Role, (Role.id == AccessGrant.role_id) & (Role.scope == scope)
        )

    def _org_grants(self, selectable):
        return self._grants_in_scope(selectable, RoleScope.ORG.value)

    def _site_grants(self, selectable):
        return self._grants_in_scope(selectable, RoleScope.SITE.value)

    # ------------------------------------------------------------------
    # Role resolution
    # ------------------------------------------------------------------
    async def _role_id(self, db: AsyncSession, *, name: str, scope: str) -> int:
        """Resolve the `roles.id` for a (name, scope) pair, with caching."""
        key = (name, scope)
        cached = self._role_id_cache.get(key)
        if cached is not None:
            return cached
        role_id = (
            await db.execute(
                select(Role.id)
                .where(Role.name == name, Role.scope == scope)
                .limit(1)
            )
        ).scalar_one_or_none()
        if role_id is None:
            raise ValueError(f"Unknown role '{name}' in scope '{scope}' (catalog not seeded?)")
        self._role_id_cache[key] = int(role_id)
        return int(role_id)

    # ------------------------------------------------------------------
    # Organizations
    # ------------------------------------------------------------------
    async def create_organization(
        self, db: AsyncSession, dto: OrganizationCreateDTO
    ) -> Organization:
        """Insert an organization row. Caller commits."""
        org = Organization(
            name=dto.name,
            slug=dto.slug.strip().lower(),
            owner_user_id=dto.owner_user_id,
        )
        db.add(org)
        try:
            await db.flush()
        except IntegrityError as exc:
            raise ValueError("Organization slug already in use") from exc
        return org

    async def get_by_id(
        self, db: AsyncSession, org_id: int
    ) -> Optional[Organization]:
        res = await db.execute(
            select(Organization).where(Organization.id == org_id).limit(1)
        )
        return res.scalar_one_or_none()

    async def get_by_slug(
        self, db: AsyncSession, slug: str
    ) -> Optional[Organization]:
        res = await db.execute(
            select(Organization)
            .where(Organization.slug == slug.strip().lower())
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def list_organizations(self, db: AsyncSession) -> List[Organization]:
        res = await db.execute(
            select(Organization).order_by(Organization.created_at.desc())
        )
        return list(res.scalars().all())

    async def update_organization(
        self, db: AsyncSession, org_id: int, dto: OrganizationUpdateDTO
    ) -> None:
        values = {k: v for k, v in dto.model_dump(exclude_unset=True).items() if v is not None}
        if "slug" in values:
            values["slug"] = values["slug"].strip().lower()
        if not values:
            return
        await db.execute(
            update(Organization).where(Organization.id == org_id).values(**values)
        )

    async def delete_organization(self, db: AsyncSession, org_id: int) -> int:
        """Hard delete. CASCADE wipes access grants and sites."""
        res = await db.execute(
            delete(Organization)
            .where(Organization.id == org_id)
            .execution_options(synchronize_session=False)
        )
        return res.rowcount or 0

    # ------------------------------------------------------------------
    # Org memberships (org-scoped access grants)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_org_role(role: str) -> str:
        # Raises ValueError when the value is not a valid OrgRole.
        return OrgRole(role).value

    async def upsert_org_membership(
        self, db: AsyncSession, dto: OrgMembershipUpsertDTO
    ) -> AccessGrant:
        """Insert or update the user's org-scoped grant for an org.

        The role is validated against `OrgRole`. On conflict the existing
        grant's role is overwritten.
        """
        role = self._validate_org_role(dto.role)
        role_id = await self._role_id(db, name=role, scope=RoleScope.ORG.value)

        existing = await self.get_org_membership(
            db, user_id=dto.user_id, org_id=dto.org_id
        )
        if existing is not None:
            existing.role_id = role_id
            existing.updated_at = utc_now()
            await db.flush()
            return existing

        row = AccessGrant(user_id=dto.user_id, org_id=dto.org_id, role_id=role_id)
        db.add(row)
        await db.flush()
        return row

    async def get_org_membership(
        self, db: AsyncSession, *, user_id: int, org_id: int
    ) -> Optional[AccessGrant]:
        """The user's org-scoped grant for `org_id` (any org role), or None."""
        res = await db.execute(
            self._org_grants(select(AccessGrant))
            .where(
                AccessGrant.user_id == user_id,
                AccessGrant.org_id == org_id,
            )
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def list_org_members(
        self, db: AsyncSession, *, org_id: int
    ) -> List[Tuple[str, User]]:
        """`(role_name, user)` for every member of `org_id`, oldest grant first."""
        res = await db.execute(
            self._org_grants(select(Role.name, User))
            .join(User, User.id == AccessGrant.user_id)
            .where(AccessGrant.org_id == org_id)
            .order_by(AccessGrant.created_at.asc())
        )
        return list(res.all())

    async def list_user_orgs(
        self, db: AsyncSession, *, user_id: int
    ) -> List[Tuple[str, Organization]]:
        """`(role_name, organization)` for every org the user belongs to."""
        res = await db.execute(
            self._org_grants(select(Role.name, Organization))
            .join(Organization, Organization.id == AccessGrant.org_id)
            .where(AccessGrant.user_id == user_id)
        )
        return list(res.all())

    async def remove_org_membership(
        self, db: AsyncSession, *, user_id: int, org_id: int
    ) -> int:
        # org_id is non-null only for org-scoped grants, so this never
        # touches site grants.
        res = await db.execute(
            delete(AccessGrant)
            .where(
                AccessGrant.user_id == user_id,
                AccessGrant.org_id == org_id,
            )
            .execution_options(synchronize_session=False)
        )
        return res.rowcount or 0

    async def _count_org_role(self, db: AsyncSession, *, org_id: int, role: str) -> int:
        res = await db.execute(
            self._org_grants(select(func.count())).where(
                AccessGrant.org_id == int(org_id),
                Role.name == role,
            )
        )
        return int(res.scalar_one() or 0)

    async def count_admins(self, db: AsyncSession, *, org_id: int) -> int:
        """How many org Admin grants the organization currently has.

        Used by `remove_org_membership_safe` to decide whether removing the
        requested admin would leave the organization without any admin.
        """
        return await self._count_org_role(db, org_id=org_id, role=OrgRole.ADMIN.value)

    async def count_operators(self, db: AsyncSession, *, org_id: int) -> int:
        """How many org Operator grants the organization has.

        Drives the notification approval gate: when > 0, new alerts are
        held `pending` until an operator approves them.
        """
        return await self._count_org_role(db, org_id=org_id, role=OrgRole.OPERATOR.value)

    async def has_operator(self, db: AsyncSession, *, org_id: int) -> bool:
        return (await self.count_operators(db, org_id=int(org_id))) > 0

    async def list_operator_user_ids(self, db: AsyncSession, *, org_id: int) -> List[int]:
        """User-ids of the org's operators.

        Drives the realtime operator pipeline: held alerts are published live to
        these users (instead of the end user) until one of them approves.
        """
        res = await db.execute(
            self._org_grants(select(AccessGrant.user_id)).where(
                AccessGrant.org_id == int(org_id),
                Role.name == OrgRole.OPERATOR.value,
            )
        )
        return [int(uid) for uid in res.scalars().all()]

    async def remove_org_membership_safe(
        self, db: AsyncSession, *, user_id: int, org_id: int
    ) -> dict:
        """Remove an org grant, cascading the org delete if the org loses
        its last admin.

        Returns a dict describing what happened:

            {"removed": bool, "org_deleted": bool, "was_last_admin": bool}

        The cascade mirrors the product requirement: if the only Org Admin
        leaves, the entire tenant (sites + cameras + grants) is wiped. FK
        CASCADEs handle the deep cleanup.
        """
        # Resolve the role name without lazy-loading the relationship.
        role_name = await self.get_org_role_name(db, user_id=user_id, org_id=org_id)
        if role_name is None:
            return {"removed": False, "org_deleted": False, "was_last_admin": False}

        removing_an_admin = role_name == OrgRole.ADMIN.value

        await self.remove_org_membership(db, user_id=user_id, org_id=org_id)

        was_last_admin = False
        org_deleted = False
        if removing_an_admin:
            remaining = await self.count_admins(db, org_id=org_id)
            if remaining == 0:
                was_last_admin = True
                # No admin left -> drop the org. CASCADE wipes every
                # site/grant belonging to it.
                await self.delete_organization(db, org_id)
                org_deleted = True

        return {
            "removed": True,
            "org_deleted": org_deleted,
            "was_last_admin": was_last_admin,
        }

    async def count_org_members(
        self, db: AsyncSession, *, org_id: int, exclude_user_id: Optional[int] = None
    ) -> int:
        """How many distinct users hold an org-scoped grant in this org.

        `exclude_user_id` answers "is anyone else still here?" — used by the
        last-admin succession check on account deletion.
        """
        stmt = self._org_grants(
            select(func.count(func.distinct(AccessGrant.user_id)))
        ).where(AccessGrant.org_id == int(org_id))
        if exclude_user_id is not None:
            stmt = stmt.where(AccessGrant.user_id != int(exclude_user_id))
        return int((await db.execute(stmt)).scalar_one() or 0)

    async def get_org_role_name(
        self, db: AsyncSession, *, user_id: int, org_id: int
    ) -> Optional[str]:
        """The user's org-scoped role name, or None if they aren't a member."""
        res = await db.execute(
            self._org_grants(select(Role.name))
            .where(
                AccessGrant.user_id == int(user_id),
                AccessGrant.org_id == int(org_id),
            )
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def list_org_ids_for_user(
        self, db: AsyncSession, *, user_id: int
    ) -> List[int]:
        """Org ids the user holds an org-scoped grant in.

        Snapshotted *before* an account deletion, because the grants are
        CASCADE-wiped along with the user row and the orgs they pointed at
        can no longer be found afterwards.
        """
        res = await db.execute(
            select(AccessGrant.org_id)
            .where(
                AccessGrant.user_id == int(user_id),
                AccessGrant.org_id.is_not(None),
            )
            .distinct()
        )
        return [int(oid) for oid in res.scalars().all()]

    async def delete_if_abandoned(
        self, db: AsyncSession, *, org_ids: List[int]
    ) -> List[int]:
        """Drop any of `org_ids` that no longer has a single member.

        Deleting a user CASCADEs their `access_grants` away and only SET NULLs
        `organizations.owner_user_id`, so removing the sole member of an org
        leaves an owner-less, member-less tenant behind. This mirrors the
        cascade rule in `remove_org_membership_safe`: an org nobody belongs to
        must not survive.

        Returns the ids actually deleted. Does NOT commit.
        """
        if not org_ids:
            return []

        abandoned = (
            await db.execute(
                select(Organization.id).where(
                    Organization.id.in_([int(o) for o in org_ids]),
                    ~select(AccessGrant.id)
                    .where(AccessGrant.org_id == Organization.id)
                    .exists(),
                )
            )
        ).scalars().all()

        deleted: List[int] = []
        for org_id in abandoned:
            if await self.delete_organization(db, int(org_id)):
                deleted.append(int(org_id))
        return deleted

    # ------------------------------------------------------------------
    # Site memberships (site-scoped access grants)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_site_role(role: str) -> str:
        return SiteRole(role).value

    async def upsert_site_membership(
        self, db: AsyncSession, dto: SiteMembershipUpsertDTO
    ) -> AccessGrant:
        """Insert or update a user's site-scoped grant.

        Enforces the invariant that the user must already have an org grant
        for the organization that owns the site before they can be granted
        a site role.
        """
        role = self._validate_site_role(dto.role)
        role_id = await self._role_id(db, name=role, scope=RoleScope.SITE.value)

        # Look up the org owning the site so we can validate.
        site_org_id = (
            await db.execute(
                select(Site.org_id).where(Site.site_uuid == dto.site_uuid).limit(1)
            )
        ).scalar_one_or_none()
        if site_org_id is None:
            # Either the site does not exist, or it predates org-keying.
            site_exists = (
                await db.execute(
                    select(Site.site_uuid).where(Site.site_uuid == dto.site_uuid).limit(1)
                )
            ).scalar_one_or_none()
            if site_exists is None:
                raise ValueError("Site not found")
        else:
            # When the site has an org_id set, require an org grant first.
            org_mem = await self.get_org_membership(
                db, user_id=dto.user_id, org_id=int(site_org_id)
            )
            if org_mem is None:
                raise ValueError(
                    "User must be an organization member before being added to a site"
                )

        existing = await self.get_site_membership(
            db, user_id=dto.user_id, site_uuid=dto.site_uuid
        )
        if existing is not None:
            existing.role_id = role_id
            existing.updated_at = utc_now()
            await db.flush()
            return existing

        row = AccessGrant(
            user_id=dto.user_id, site_uuid=dto.site_uuid, role_id=role_id
        )
        db.add(row)
        await db.flush()
        return row

    async def get_site_membership(
        self, db: AsyncSession, *, user_id: int, site_uuid: uuid.UUID
    ) -> Optional[AccessGrant]:
        res = await db.execute(
            self._site_grants(select(AccessGrant))
            .where(
                AccessGrant.user_id == user_id,
                AccessGrant.site_uuid == site_uuid,
            )
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def list_site_members(
        self, db: AsyncSession, *, site_uuid: uuid.UUID
    ) -> List[Tuple[str, User]]:
        """`(role_name, user)` for every member of a site, oldest grant first."""
        res = await db.execute(
            self._site_grants(select(Role.name, User))
            .join(User, User.id == AccessGrant.user_id)
            .where(AccessGrant.site_uuid == site_uuid)
            .order_by(AccessGrant.created_at.asc())
        )
        return list(res.all())

    async def remove_site_membership(
        self, db: AsyncSession, *, user_id: int, site_uuid: uuid.UUID
    ) -> int:
        res = await db.execute(
            delete(AccessGrant)
            .where(
                AccessGrant.user_id == user_id,
                AccessGrant.site_uuid == site_uuid,
            )
            .execution_options(synchronize_session=False)
        )
        return res.rowcount or 0

    # ------------------------------------------------------------------
    # Convenience queries used by routes
    # ------------------------------------------------------------------
    async def list_org_sites(
        self, db: AsyncSession, *, org_id: int, include_deleted: bool = False
    ) -> List[Site]:
        """Every site that belongs to the given organization."""
        stmt = select(Site).where(Site.org_id == org_id)
        if not include_deleted:
            stmt = stmt.where(Site.is_deleted.is_(False))
        stmt = stmt.order_by(Site.created_at.desc())
        res = await db.execute(stmt)
        return list(res.scalars().all())
