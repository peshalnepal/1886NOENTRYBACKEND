# routes/notification_email_routes.py
"""
API routes for managing user notification emails.
"""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import NotificationRepository
from application.repositories.site_repository import SiteRepository
from application.dtos import NotificationEmailCreateDTO
from core.schemas import (
    NotificationEmailCreate,
    NotificationEmailCreateResult,
    NotificationEmailOut,
)
from application.services.authz_service import AuthzService
from dependencies import (
    get_async_db,
    RequirePermission,
    OrgContext,
)
from core.security.roles import Permission


def _email_owner_id(site, ctx: OrgContext) -> int:
    """Recipient emails are keyed by the site owner so the notification
    flusher (which resolves recipients by the camera/site owner) finds them."""
    owner = getattr(site, "user_id", None)
    return int(owner) if owner is not None else int(ctx.user.id)

router = APIRouter(prefix="/notification-emails", tags=["notification-emails"])

notif_repo = NotificationRepository()
site_repo = SiteRepository()


def _get_notification_service(request: Request):
    """The notification service if the app has one, else None (never raises)."""
    svc = getattr(request.app.state, "notification_service", None)
    if svc is not None:
        return svc
    manager = getattr(request.app.state, "manager", None)
    return getattr(manager, "notification_service", None) if manager else None


def _invalidate_notification_email_cache(
    request: Request,
    *,
    user_id: Optional[int] = None,
    site_uuid: Optional[uuid.UUID] = None,
) -> None:
    if request is None:
        return

    svc = _get_notification_service(request)
    if svc is None or not hasattr(svc, "invalidate_recipient_cache"):
        return

    # `user_id` is only an audit value here and may be NULL; when a site is
    # given the flusher clears that site across every user, which is what a
    # site-owned recipient list needs.
    svc.invalidate_recipient_cache(user_id=int(user_id or 0), site_uuid=site_uuid)

# -------------------------
# Endpoints
# -------------------------
@router.get("", response_model=List[NotificationEmailOut])
async def list_notification_emails(
    site_uuid: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    """
    List notification emails for the caller's organization.
    Optional site_uuid filter narrows results to one site.
    Members are limited to sites they can access.
    """
    accessible = await AuthzService.accessible_site_uuids(
        db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
    )
    if site_uuid is not None:
        if accessible is not None and site_uuid not in accessible:
            raise HTTPException(status_code=404, detail="Site not found")
        target = [site_uuid]
    else:
        org_sites = await site_repo.list_site_uuids(db, org_id=ctx.org_id)
        target = org_sites if accessible is None else [s for s in org_sites if s in accessible]

    rows = await notif_repo.list_notification_email_rows(db, site_uuids=target)
    return [
        NotificationEmailOut(
            id=row.id,
            user_id=row.user_id,
            site_uuid=row.site_uuid,
            email=row.email,
            is_enabled=bool(row.is_enabled),
        )
        for row in rows
    ]


@router.post("", response_model=NotificationEmailCreateResult, status_code=status.HTTP_201_CREATED)
async def add_notification_email(
    payload: NotificationEmailCreate,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SETTINGS)),
):
    """
    Add a notification email (org admin only).
    If site_uuid is missing, apply it to all sites in the organization.
    """
    normalized_email = payload.email.lower().strip()

    if payload.site_uuid is not None:
        site = await site_repo.get_site(
            db, site_uuid=payload.site_uuid, org_id=ctx.org_id, raise_if_missing=False
        )
        target_sites = [site] if site is not None else []
    else:
        target_sites = await site_repo.get_sites(db, org_id=ctx.org_id)

    if not target_sites:
        if payload.site_uuid is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found for this organization",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Create at least one site before adding notification emails",
        )

    created_rows = []
    for target_site in target_sites:
        # Recipients are resolved by site, so store the actual adder as an audit
        # trail rather than impersonating the site owner.
        if await notif_repo.notification_email_exists(
            db, site_uuid=target_site.site_uuid, email=normalized_email
        ):
            continue

        row = await notif_repo.create_notification_email(
            db,
            dto=NotificationEmailCreateDTO(
                user_id=int(ctx.user.id),
                site_uuid=target_site.site_uuid,
                email=normalized_email,
                is_enabled=True,
            ),
        )
        created_rows.append(row)

    if not created_rows:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists for the selected site scope",
        )

    await db.commit()
    for row in created_rows:
        _invalidate_notification_email_cache(request, user_id=row.user_id, site_uuid=row.site_uuid)
    for row in created_rows:
        await db.refresh(row)

    return NotificationEmailCreateResult(
        created=[
            NotificationEmailOut(
                id=row.id,
                user_id=row.user_id,
                site_uuid=row.site_uuid,
                email=row.email,
                is_enabled=bool(row.is_enabled),
            )
            for row in created_rows
        ],
        created_count=len(created_rows),
        skipped_count=len(target_sites) - len(created_rows),
        target_site_count=len(target_sites),
    )


@router.delete("/{email_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification_email(
    email_id: int,
    all_sites: bool = False,
    request: Request = None,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SETTINGS)),
):
    """
    Delete one notification email entry by ID (org admin only).
    If all_sites=true, remove that address from every site in the caller's org.

    Any org admin may delete any recipient: recipients belong to the site, not
    to the admin who happened to add them.
    """
    email = await notif_repo.get_notification_email(db, email_id=email_id)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification email not found",
        )

    # Confirm the recipient's site belongs to the caller's organization.
    owning_site = await site_repo.get_site(
        db, site_uuid=email.site_uuid, org_id=ctx.org_id, raise_if_missing=False
    )
    if owning_site is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification email not found",
        )

    email_user_id = email.user_id
    email_email = email.email

    if all_sites:
        # Scope by the caller's org sites, not by `user_id`: the stored user is
        # only an audit trail (and may be NULL), and matching on it would both
        # miss rows other admins added and reach into other tenants.
        affected_site_uuids = await site_repo.list_site_uuids(db, org_id=ctx.org_id)
        for target_site_uuid in affected_site_uuids:
            await notif_repo.delete_notification_email(
                db, site_uuid=target_site_uuid, email=email_email
            )
    else:
        affected_site_uuids = [email.site_uuid]
        await notif_repo.delete_notification_email(db, email_id=email_id)

    await db.commit()
    if request is not None:
        # Clear per affected site: the cache is site-keyed, so a single
        # user-keyed sweep would miss other members' entries.
        for target_site_uuid in affected_site_uuids:
            _invalidate_notification_email_cache(
                request, user_id=email_user_id, site_uuid=target_site_uuid
            )
    return None
