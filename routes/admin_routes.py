"""
Organization-admin routes.

Endpoints used by an Org Admin (or a Platform Admin acting on the
org's behalf) to manage the users that live inside one organization
and to grant them site-level access:

    GET    /api/admin/orgs/{org_id}/members                  - list org members
    POST   /api/admin/orgs/{org_id}/members                  - create user + add to org
    POST   /api/admin/orgs/{org_id}/members/attach           - attach an existing user
    PATCH  /api/admin/orgs/{org_id}/members/{user_id}        - change OrgRole
    DELETE /api/admin/orgs/{org_id}/members/{user_id}        - delete member account

    GET    /api/admin/orgs/{org_id}/sites                    - sites in this org
    GET    /api/admin/orgs/{org_id}/sites/{site_uuid}/members
    POST   /api/admin/orgs/{org_id}/sites/{site_uuid}/members - grant site role
    DELETE /api/admin/orgs/{org_id}/sites/{site_uuid}/members/{user_id}

Permission model:
  * Each endpoint declares the permission it needs via
    `RequirePermission(...)` — `ORG_MANAGE_MEMBERS` / `ORG_MANAGE_SITES`
    for org-level actions, `SITE_MANAGE_MEMBERS` for site-role grants.
    Today only the Org Admin role carries these, and Platform Admins
    bypass the check inside `AuthzService.has_permission`.
  * Site-level mutations additionally verify that the target site
    actually belongs to `org_id`. This prevents a malicious admin
    from pivoting to another tenant by guessing site UUIDs.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import (
    OrgMembershipUpsertDTO,
    SiteMembershipUpsertDTO,
    UserCreateDTO,
)
from application.repositories.organization_repository import OrganizationRepository
from application.repositories.user_repository import UserRepository
from core.database_orm import Organization, Site, User
from core.security.hashing import get_password_hash
from core.security.roles import OrgRole, Permission, SiteRole
from application.services.manager import Manager
from dependencies import (
    get_async_db,
    get_current_user,
    get_manager,
    RequirePermission,
)
from routes._errors import ORGANIZATION_NOT_FOUND, SITE_NOT_FOUND, USER_NOT_FOUND

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# =====================================================================
# Schemas
# =====================================================================
class MemberOut(BaseModel):
    user_id: int
    user_name: str
    email: EmailStr
    role: str


class SiteMemberOut(BaseModel):
    user_id: int
    user_name: str
    email: EmailStr
    role: str  # SiteRole value


class AdminSiteOut(BaseModel):
    site_uuid: uuid.UUID
    name: str
    address: Optional[str]


class CreateMemberRequest(BaseModel):
    """Create a brand-new user and add them to the organization.

    The same payload covers Operators and plain Members; pick the
    role via the `role` field.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str = Field(..., min_length=1, max_length=255)
    user_email: EmailStr
    password: SecretStr
    role: OrgRole = OrgRole.MEMBER
    contact_phone: Optional[str] = Field(default=None, max_length=20)


class AttachExistingMemberRequest(BaseModel):
    """Attach an already-existing user to this organization."""

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[int] = None
    user_email: Optional[EmailStr] = None
    role: OrgRole = OrgRole.MEMBER


class UpdateMemberRoleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: OrgRole


class GrantSiteRoleRequest(BaseModel):
    """Grant a `SiteRole` to a user that already belongs to the org."""

    model_config = ConfigDict(extra="forbid")

    user_id: int
    role: SiteRole


# =====================================================================
# Helpers
# =====================================================================
async def _ensure_site_in_org(
    db: AsyncSession, *, org_id: int, site_uuid: uuid.UUID
) -> Site:
    """404 if the site does not exist or belongs to a different org."""
    res = await db.execute(
        select(Site).where(Site.site_uuid == site_uuid).limit(1)
    )
    site = res.scalar_one_or_none()
    if site is None:
        raise HTTPException(status_code=404, detail=SITE_NOT_FOUND)
    if site.org_id is not None and int(site.org_id) != int(org_id):
        # Treat cross-tenant probes as 404 to avoid leaking existence.
        raise HTTPException(status_code=404, detail=SITE_NOT_FOUND)
    return site


def _member_out(user: User, role: str) -> MemberOut:
    return MemberOut(
        user_id=int(user.id),
        user_name=user.user_name,
        email=user.email,
        role=role,
    )


# =====================================================================
# Organization members
# =====================================================================
@router.get(
    "/orgs/{org_id}/members",
    response_model=List[MemberOut],
    dependencies=[Depends(RequirePermission(Permission.ORG_MANAGE_MEMBERS))],
)
async def list_org_members(org_id: int, db: AsyncSession = Depends(get_async_db)):
    rows = await OrganizationRepository().list_org_members(db, org_id=org_id)
    return [_member_out(u, role_name) for (role_name, u) in rows]


@router.post(
    "/orgs/{org_id}/members",
    response_model=MemberOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(RequirePermission(Permission.ORG_MANAGE_MEMBERS))],
)
async def create_org_member(
    org_id: int,
    payload: CreateMemberRequest,
    db: AsyncSession = Depends(get_async_db),
    actor: User = Depends(get_current_user),
):
    """Create a fresh user and attach them to the organization.

    Used by an Org Admin to onboard a new Member or Operator. The
    new user is email-verified by default so they can log in
    immediately with the provided password.
    """
    user_repo = UserRepository()
    org_repo = OrganizationRepository()
    # Reject if the org does not exist (defence in depth, since the
    # ORG_MANAGE_MEMBERS gate already implies the caller belongs to it).
    if await org_repo.get_by_id(db, org_id) is None:
        raise HTTPException(status_code=404, detail=ORGANIZATION_NOT_FOUND)

    try:
        new_user = await user_repo.create_user(
            db,
            UserCreateDTO(
                user_name=payload.user_name,
                email=str(payload.user_email),
                password_hash=get_password_hash(payload.password.get_secret_value()),
                contact_phone=payload.contact_phone,
                email_verified=True,
            ),
        )
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await org_repo.upsert_org_membership(
        db,
        OrgMembershipUpsertDTO(
            user_id=int(new_user.id),
            org_id=int(org_id),
            role=payload.role.value,
        ),
    )
    await db.commit()
    await db.refresh(new_user)

    logger.info(
        "Org=%s admin=%s created member user_id=%s role=%s",
        org_id, int(actor.id), int(new_user.id), payload.role.value,
    )
    return _member_out(new_user, payload.role.value)


@router.post(
    "/orgs/{org_id}/members/attach",
    response_model=MemberOut,
    dependencies=[Depends(RequirePermission(Permission.ORG_MANAGE_MEMBERS))],
)
async def attach_existing_user(
    org_id: int,
    payload: AttachExistingMemberRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Add an existing user (by id or email) to the organization."""
    if payload.user_id is None and payload.user_email is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either user_id or user_email.",
        )

    user_repo = UserRepository()
    org_repo = OrganizationRepository()
    target: Optional[User] = None
    if payload.user_id is not None:
        target = await user_repo.get_by_id(db, int(payload.user_id))
    elif payload.user_email is not None:
        target = await user_repo.get_by_email(db, str(payload.user_email))

    if target is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)

    await org_repo.upsert_org_membership(
        db,
        OrgMembershipUpsertDTO(
            user_id=int(target.id),
            org_id=int(org_id),
            role=payload.role.value,
        ),
    )
    await db.commit()
    return _member_out(target, payload.role.value)


@router.patch(
    "/orgs/{org_id}/members/{user_id}",
    response_model=MemberOut,
    dependencies=[Depends(RequirePermission(Permission.ORG_MANAGE_MEMBERS))],
)
async def update_member_role(
    org_id: int,
    user_id: int,
    payload: UpdateMemberRoleRequest,
    db: AsyncSession = Depends(get_async_db),
):
    user = await UserRepository().get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)

    org_repo = OrganizationRepository()
    existing = await org_repo.get_org_membership(db, user_id=user_id, org_id=org_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="User is not a member of this org")

    await org_repo.upsert_org_membership(
        db,
        OrgMembershipUpsertDTO(
            user_id=user_id, org_id=org_id, role=payload.role.value
        ),
    )
    await db.commit()
    return _member_out(user, payload.role.value)


class RemoveMemberResponse(BaseModel):
    """Body returned by DELETE /orgs/{org_id}/members/{user_id}.

    `account_deleted=True` means the member's whole account was erased,
    not merely detached from the org. `org_deleted=True` means removing
    this user emptied the last admin seat and the organization itself
    was cascaded away.
    """

    removed: bool
    was_last_admin: bool
    org_deleted: bool
    account_deleted: bool = False


@router.delete(
    "/orgs/{org_id}/members/{user_id}",
    response_model=RemoveMemberResponse,
    dependencies=[Depends(RequirePermission(Permission.ORG_MANAGE_MEMBERS))],
)
async def remove_member(
    org_id: int,
    user_id: int,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    actor: User = Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    """Delete a member of this organization, account and all.

    Removing someone used to only drop their `access_grants` row, which
    left a live but org-less account behind: it could still log in, yet
    appeared on no admin screen. Members here belong to the org, so the
    removal runs the same full deletion pipeline as `/users/me` — sites,
    cameras, notifications, blobs, OTP rows and the user row itself.

    The actor's own account is refused: an admin leaving is
    `DELETE /users/me`, which enforces the last-admin succession rule.
    """
    if int(actor.id) == int(user_id):
        raise HTTPException(
            status_code=409,
            detail="Delete your own account via Settings → Delete Account.",
        )

    org_repo = OrganizationRepository()
    membership = await org_repo.get_org_membership(db, user_id=user_id, org_id=org_id)
    if membership is None:
        raise HTTPException(status_code=404, detail="Membership not found")

    was_last_admin = False
    role_name = await org_repo.get_org_role_name(db, user_id=user_id, org_id=org_id)
    if role_name == OrgRole.ADMIN.value:
        was_last_admin = (await org_repo.count_admins(db, org_id=org_id)) <= 1

    from routes.user_routes import perform_user_deletion

    await perform_user_deletion(
        user_id=int(user_id), request=request, db=db, manager=manager
    )

    org_deleted = (
        await db.execute(select(Organization.id).where(Organization.id == org_id).limit(1))
    ).scalar_one_or_none() is None

    logger.info(
        "Org=%s member=%s fully deleted by actor=%s (last_admin=%s, org_deleted=%s)",
        org_id, user_id, int(actor.id), was_last_admin, org_deleted,
    )
    return RemoveMemberResponse(
        removed=True,
        was_last_admin=was_last_admin,
        org_deleted=org_deleted,
        account_deleted=True,
    )


# =====================================================================
# Sites within the organization
# =====================================================================
@router.get(
    "/orgs/{org_id}/sites",
    response_model=List[AdminSiteOut],
    dependencies=[Depends(RequirePermission(Permission.ORG_MANAGE_SITES))],
)
async def list_org_sites(org_id: int, db: AsyncSession = Depends(get_async_db)):
    sites = await OrganizationRepository().list_org_sites(db, org_id=org_id)
    return [AdminSiteOut(site_uuid=s.site_uuid, name=s.name, address=s.address) for s in sites]


@router.get(
    "/orgs/{org_id}/sites/{site_uuid}/members",
    response_model=List[SiteMemberOut],
    dependencies=[Depends(RequirePermission(Permission.SITE_MANAGE_MEMBERS))],
)
async def list_site_members(
    org_id: int,
    site_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
):
    await _ensure_site_in_org(db, org_id=org_id, site_uuid=site_uuid)
    rows = await OrganizationRepository().list_site_members(
        db, site_uuid=site_uuid
    )
    return [
        SiteMemberOut(
            user_id=int(u.id),
            user_name=u.user_name,
            email=u.email,
            role=role_name,
        )
        for (role_name, u) in rows
    ]


@router.post(
    "/orgs/{org_id}/sites/{site_uuid}/members",
    response_model=SiteMemberOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(RequirePermission(Permission.SITE_MANAGE_MEMBERS))],
)
async def grant_site_role(
    org_id: int,
    site_uuid: uuid.UUID,
    payload: GrantSiteRoleRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Assign a `SiteRole` to a user that already belongs to the org."""
    await _ensure_site_in_org(db, org_id=org_id, site_uuid=site_uuid)

    user = await UserRepository().get_by_id(db, int(payload.user_id))
    if user is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)

    try:
        await OrganizationRepository().upsert_site_membership(
            db,
            SiteMembershipUpsertDTO(
                user_id=int(payload.user_id),
                site_uuid=site_uuid,
                role=payload.role.value,
            ),
        )
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    await db.commit()
    return SiteMemberOut(
        user_id=int(user.id),
        user_name=user.user_name,
        email=user.email,
        role=payload.role.value,
    )


@router.delete(
    "/orgs/{org_id}/sites/{site_uuid}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(RequirePermission(Permission.SITE_MANAGE_MEMBERS))],
)
async def revoke_site_role(
    org_id: int,
    site_uuid: uuid.UUID,
    user_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    await _ensure_site_in_org(db, org_id=org_id, site_uuid=site_uuid)
    deleted = await OrganizationRepository().remove_site_membership(
        db, user_id=user_id, site_uuid=site_uuid
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Site membership not found")
    await db.commit()
    return None
