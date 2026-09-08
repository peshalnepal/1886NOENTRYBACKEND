"""
Platform-admin routes.

Only callable by users with `users.is_platform_admin = True`. These
endpoints provide a god-mode view across every tenant and the
bootstrap path for new organizations:

    GET    /api/platform/users              - every user in the system
    GET    /api/platform/users/{user_id}    - user details + orgs + sites
    GET    /api/platform/organizations      - every organization
    POST   /api/platform/organizations      - create org + seed first Org Admin
    GET    /api/platform/organizations/{org_id}            - org details
    PATCH  /api/platform/organizations/{org_id}            - rename / deactivate
    DELETE /api/platform/organizations/{org_id}            - drop tenant
    GET    /api/platform/sites                              - every site
    POST   /api/platform/users/{user_id}/promote            - grant platform admin
    POST   /api/platform/users/{user_id}/demote             - revoke platform admin

Everything mutates inside a single transaction per request; the
`OrganizationRepository` does the heavy lifting and the route is the
permission gate.
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import (
    OrganizationCreateDTO,
    OrganizationUpdateDTO,
    OrgMembershipUpsertDTO,
    UserCreateDTO,
)
from application.repositories.organization_repository import OrganizationRepository
from application.repositories.user_repository import UserRepository
from core.database_orm import AccessGrant, Camera, Device, Organization, Site, User
from core.security.hashing import get_password_hash
from core.security.roles import OrgRole
from dependencies import get_async_db, get_current_user, get_manager, require_platform_admin
from application.services.manager import Manager
from routes._errors import ORGANIZATION_NOT_FOUND, USER_NOT_FOUND


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/platform",
    tags=["platform-admin"],
    dependencies=[Depends(require_platform_admin)],
)


class UserSummary(BaseModel):
    """Compact view of a user, used in cross-tenant listings."""

    id: int
    user_name: str
    email: EmailStr
    is_platform_admin: bool
    email_verified: bool


class OrgSummary(BaseModel):
    """Organization plus headline counts for the listing page."""

    id: int
    name: str
    slug: str
    owner_user_id: Optional[int]
    is_active: bool
    member_count: int = 0
    site_count: int = 0


class SiteSummary(BaseModel):
    """Site row enriched with its organization id."""

    site_uuid: uuid.UUID
    name: str
    address: Optional[str]
    org_id: Optional[int]
    user_id: Optional[int]


class DeviceSummary(BaseModel):
    """Device row enriched with its organization id."""

    device_uuid: uuid.UUID
    name: Optional[str]
    device_code: Optional[str]
    device_url: str
    is_enabled: bool
    org_id: Optional[int]
    user_id: Optional[int]
    created_by: Optional[int]


class CameraSummary(BaseModel):
    """Camera row enriched with its organization + site id."""

    camera_uuid: uuid.UUID
    name: Optional[str]
    camera_code: Optional[str]
    site_uuid: uuid.UUID
    org_id: Optional[int]
    user_id: Optional[int]
    created_by: Optional[int]
    is_enabled: bool


class OrgMembershipOut(BaseModel):
    user: UserSummary
    role: str


class UserDetail(BaseModel):
    """Everything a platform admin needs to know about a user.

    `owned_sites` are the sites the user created/owns directly. The
    `org_sites`/`org_devices`/`org_cameras` lists cover *every* resource
    in the organizations the user belongs to, so a platform admin can see
    the full picture for that user's tenant(s).
    """

    user: UserSummary
    organizations: List[OrgMembershipOut]
    owned_sites: List[SiteSummary]
    org_sites: List[SiteSummary] = []
    org_devices: List[DeviceSummary] = []
    org_cameras: List[CameraSummary] = []


class CreateOrgRequest(BaseModel):
    """Payload for `POST /platform/organizations`.

    Either `owner_user_id` (existing user) or the bundle
    (`owner_user_name`, `owner_email`, `owner_password`) must be
    provided. When the bundle is provided a fresh user is created and
    flagged as the first Org Admin.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=2, max_length=64)

    owner_user_id: Optional[int] = None
    owner_user_name: Optional[str] = Field(default=None, max_length=255)
    owner_email: Optional[EmailStr] = None
    owner_password: Optional[SecretStr] = None


class UpdateOrgRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: Optional[str] = Field(default=None, max_length=255)
    slug: Optional[str] = Field(default=None, max_length=64)
    is_active: Optional[bool] = None


def _to_user_summary(u: User) -> UserSummary:
    return UserSummary(
        id=int(u.id),
        user_name=u.user_name,
        email=u.email,
        is_platform_admin=bool(u.is_platform_admin),
        email_verified=bool(u.email_verified),
    )


def _to_site_summary(s: Site) -> SiteSummary:
    return SiteSummary(
        site_uuid=s.site_uuid,
        name=s.name,
        address=s.address,
        org_id=int(s.org_id) if s.org_id is not None else None,
        user_id=int(s.user_id) if s.user_id is not None else None,
    )


def _to_device_summary(d: Device) -> DeviceSummary:
    return DeviceSummary(
        device_uuid=d.device_uuid,
        name=d.name,
        device_code=d.device_code,
        device_url=d.device_url,
        is_enabled=bool(d.is_enabled),
        org_id=int(d.org_id) if d.org_id is not None else None,
        user_id=int(d.user_id) if d.user_id is not None else None,
        created_by=int(d.created_by) if getattr(d, "created_by", None) is not None else None,
    )


def _to_camera_summary(c: Camera) -> CameraSummary:
    return CameraSummary(
        camera_uuid=c.camera_uuid,
        name=c.name,
        camera_code=c.camera_code,
        site_uuid=c.site_uuid,
        org_id=int(c.org_id) if c.org_id is not None else None,
        user_id=int(c.user_id) if c.user_id is not None else None,
        created_by=int(c.created_by) if getattr(c, "created_by", None) is not None else None,
        is_enabled=bool(c.is_enabled),
    )


@router.get("/users", response_model=List[UserSummary])
async def list_all_users(db: AsyncSession = Depends(get_async_db)):
    """Every user account in the system."""
    res = await db.execute(select(User).order_by(User.id.asc()))
    return [_to_user_summary(u) for u in res.scalars().all()]


@router.get("/users/{user_id}", response_model=UserDetail)
async def get_user_detail(user_id: int, db: AsyncSession = Depends(get_async_db)):
    """User + every organization they belong to + every site they own."""
    user = await UserRepository().get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)

    org_repo = OrganizationRepository()
    org_rows = await org_repo.list_user_orgs(db, user_id=user_id)
    org_ids = [int(o.id) for (_role, o) in org_rows]

    # Sites the user owns/created directly (legacy single-tenant ownership).
    site_rows = (
        await db.execute(
            select(Site)
            .where(Site.user_id == user_id, Site.is_deleted.is_(False))
            .order_by(Site.created_at.desc())
        )
    ).scalars().all()

    # Everything inside the organization(s) the user belongs to.
    org_sites: List[Site] = []
    org_devices: List[Device] = []
    org_cameras: List[Camera] = []
    if org_ids:
        org_sites = list(
            (
                await db.execute(
                    select(Site)
                    .where(Site.org_id.in_(org_ids), Site.is_deleted.is_(False))
                    .order_by(Site.created_at.desc())
                )
            ).scalars().all()
        )
        org_devices = list(
            (
                await db.execute(
                    select(Device).where(Device.org_id.in_(org_ids))
                    .order_by(Device.created_at.desc())
                )
            ).scalars().all()
        )
        org_cameras = list(
            (
                await db.execute(
                    select(Camera).where(Camera.org_id.in_(org_ids))
                    .order_by(Camera.created_at.desc())
                )
            ).scalars().all()
        )

    return UserDetail(
        user=_to_user_summary(user),
        organizations=[
            OrgMembershipOut(user=_to_user_summary(user), role=role_name)
            for (role_name, _o) in org_rows
        ],
        owned_sites=[_to_site_summary(s) for s in site_rows],
        org_sites=[_to_site_summary(s) for s in org_sites],
        org_devices=[_to_device_summary(d) for d in org_devices],
        org_cameras=[_to_camera_summary(c) for c in org_cameras],
    )


@router.post("/users/{user_id}/promote", response_model=UserSummary)
async def promote_to_platform_admin(
    user_id: int, db: AsyncSession = Depends(get_async_db)
):
    """Set `is_platform_admin=True` on the given user."""
    user = await UserRepository().get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)
    user.is_platform_admin = True
    await db.commit()
    await db.refresh(user)
    return _to_user_summary(user)


@router.post("/users/{user_id}/demote", response_model=UserSummary)
async def demote_platform_admin(
    user_id: int, db: AsyncSession = Depends(get_async_db),
):
    """Clear `is_platform_admin` on the given user."""
    user = await UserRepository().get_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)
    user.is_platform_admin = False
    await db.commit()
    await db.refresh(user)
    return _to_user_summary(user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    actor: User = Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    """Hard-delete any user via the same pipeline as self-service delete.

    Refuses self-deletion (use `/users/me`) and refuses removing the
    last remaining platform admin to avoid bricking the install.
    """
    from routes.user_routes import perform_user_deletion

    if int(actor.id) == int(user_id):
        raise HTTPException(
            status_code=409,
            detail="Delete your own account via /users/me.",
        )

    target = await UserRepository().get_by_id(db, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail=USER_NOT_FOUND)

    if bool(getattr(target, "is_platform_admin", False)):
        remaining = (
            await db.execute(
                select(func.count())
                .select_from(User)
                .where(User.is_platform_admin.is_(True), User.id != int(user_id))
            )
        ).scalar_one()
        if int(remaining or 0) == 0:
            raise HTTPException(
                status_code=409,
                detail="Refusing to delete the last platform admin.",
            )

    await perform_user_deletion(
        user_id=int(user_id), request=request, db=db, manager=manager
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/organizations", response_model=List[OrgSummary])
async def list_organizations(db: AsyncSession = Depends(get_async_db)):
    """Every organization, enriched with member + site counts."""
    org_repo = OrganizationRepository()
    orgs = await org_repo.list_organizations(db)

    out: List[OrgSummary] = []
    for org in orgs:
        member_count = (
            await db.execute(
                select(func.count())
                .select_from(AccessGrant)
                .where(AccessGrant.org_id == org.id)
            )
        ).scalar_one()
        site_count = (
            await db.execute(
                select(func.count())
                .select_from(Site)
                .where(Site.org_id == org.id, Site.is_deleted.is_(False))
            )
        ).scalar_one()
        out.append(
            OrgSummary(
                id=int(org.id),
                name=org.name,
                slug=org.slug,
                owner_user_id=org.owner_user_id,
                is_active=bool(org.is_active),
                member_count=int(member_count or 0),
                site_count=int(site_count or 0),
            )
        )
    return out


@router.post(
    "/organizations",
    response_model=OrgSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_organization(
    payload: CreateOrgRequest, db: AsyncSession = Depends(get_async_db)):
    """Create an organization and seed its first Org Admin.

    Provide either:
      * `owner_user_id` of an existing user (granted Org Admin), or
      * `owner_user_name` + `owner_email` + `owner_password` to create
        a brand-new user inline.
    """
    user_repo = UserRepository()
    org_repo = OrganizationRepository()

    # Resolve the owner user --------------------------------------
    owner: Optional[User] = None
    if payload.owner_user_id is not None:
        owner = await user_repo.get_by_id(db, int(payload.owner_user_id))
        if owner is None:
            raise HTTPException(status_code=404, detail="Owner user not found")
    elif (
        payload.owner_user_name
        and payload.owner_email
        and payload.owner_password
    ):
        try:
            owner = await user_repo.create_user(
                db,
                UserCreateDTO(
                    user_name=payload.owner_user_name,
                    email=str(payload.owner_email),
                    password_hash=get_password_hash(
                        payload.owner_password.get_secret_value()
                    ),
                    email_verified=True,
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                "Provide either owner_user_id or "
                "owner_user_name + owner_email + owner_password."
            ),
        )

    # Create the org row + the Org Admin membership ---------------
    try:
        org = await org_repo.create_organization(
            db,
            OrganizationCreateDTO(
                name=payload.name, slug=payload.slug, owner_user_id=int(owner.id)
            ),
        )
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await org_repo.upsert_org_membership(
        db,
        OrgMembershipUpsertDTO(
            user_id=int(owner.id),
            org_id=int(org.id),
            role=OrgRole.ADMIN.value,
        ),
    )
    await db.commit()
    await db.refresh(org)

    return OrgSummary(
        id=int(org.id),
        name=org.name,
        slug=org.slug,
        owner_user_id=org.owner_user_id,
        is_active=bool(org.is_active),
        member_count=1,
        site_count=0,
    )


@router.get("/organizations/{org_id}", response_model=OrgSummary)
async def get_organization(
    org_id: int, db: AsyncSession = Depends(get_async_db)):
    org_repo = OrganizationRepository()
    org = await org_repo.get_by_id(db, org_id)
    if org is None:
        raise HTTPException(status_code=404, detail=ORGANIZATION_NOT_FOUND)
    members = await org_repo.list_org_members(db, org_id=org_id)
    sites = await org_repo.list_org_sites(db, org_id=org_id)
    return OrgSummary(
        id=int(org.id),
        name=org.name,
        slug=org.slug,
        owner_user_id=org.owner_user_id,
        is_active=bool(org.is_active),
        member_count=len(members),
        site_count=len(sites),
    )


@router.patch("/organizations/{org_id}", response_model=OrgSummary)
async def update_organization(
    org_id: int,
    payload: UpdateOrgRequest,
    db: AsyncSession = Depends(get_async_db),
):
    org_repo = OrganizationRepository()
    org = await org_repo.get_by_id(db, org_id)
    if org is None:
        raise HTTPException(status_code=404, detail=ORGANIZATION_NOT_FOUND)

    try:
        await org_repo.update_organization(
            db,
            org_id,
            OrganizationUpdateDTO(**payload.model_dump(exclude_unset=True)),
        )
        await db.commit()
        await db.refresh(org)
    except Exception:
        await db.rollback()
        raise

    return await get_organization(org_id, db)


@router.delete(
    "/organizations/{org_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_organization(
    org_id: int, db: AsyncSession = Depends(get_async_db)
):
    """Hard delete the tenant. CASCADE removes memberships and sites."""
    org_repo = OrganizationRepository()
    deleted = await org_repo.delete_organization(db, org_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=ORGANIZATION_NOT_FOUND)
    await db.commit()
    return None


@router.get("/sites", response_model=List[SiteSummary])
async def list_all_sites(
    org_id: Optional[int] = None,
    db: AsyncSession = Depends(get_async_db)):
    """Every site (across every org) that is not soft-deleted.

    Pass `?org_id=` to scope to a single tenant.
    """
    stmt = select(Site).where(Site.is_deleted.is_(False))
    if org_id is not None:
        stmt = stmt.where(Site.org_id == int(org_id))
    stmt = stmt.order_by(Site.created_at.desc())
    res = await db.execute(stmt)
    return [_to_site_summary(s) for s in res.scalars().all()]


@router.get("/devices", response_model=List[DeviceSummary])
async def list_all_devices(
    org_id: Optional[int] = None,
    db: AsyncSession = Depends(get_async_db)):
    """Every device across every organization. Optional `?org_id=` filter."""
    stmt = select(Device)
    if org_id is not None:
        stmt = stmt.where(Device.org_id == int(org_id))
    stmt = stmt.order_by(Device.created_at.desc())
    res = await db.execute(stmt)
    return [_to_device_summary(d) for d in res.scalars().all()]


@router.get("/cameras", response_model=List[CameraSummary])
async def list_all_cameras(
    org_id: Optional[int] = None,
    db: AsyncSession = Depends(get_async_db)):
    """Every camera across every organization. Optional `?org_id=` filter."""
    stmt = select(Camera)
    if org_id is not None:
        stmt = stmt.where(Camera.org_id == int(org_id))
    stmt = stmt.order_by(Camera.created_at.desc())
    res = await db.execute(stmt)
    return [_to_camera_summary(c) for c in res.scalars().all()]
