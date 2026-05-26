# routes/notification_email_routes.py
"""
API routes for managing user notification emails.
"""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import NotificationRepository
from application.repositories.site_repository import SiteRepository
from application.dtos import NotificationEmailCreateDTO
from dependencies import get_async_db, get_current_user
from core.database_orm import User

router = APIRouter(prefix="/notification-emails", tags=["notification-emails"])

notif_repo = NotificationRepository()
site_repo = SiteRepository()


def _get_notification_service(request: Request):
    svc = getattr(request.app.state, "notification_service", None)
    if svc is not None:
        return svc

    manager = getattr(request.app.state, "manager", None)
    if manager is not None:
        svc = getattr(manager, "_notification_service", None)
        if svc is not None:
            return svc

    return None


def _invalidate_notification_email_cache(
    request: Request,
    *,
    user_id: int,
    site_uuid: Optional[uuid.UUID] = None,
) -> None:
    if request is None:
        return

    svc = _get_notification_service(request)
    if svc is None or not hasattr(svc, "invalidate_recipient_cache"):
        return

    svc.invalidate_recipient_cache(user_id=int(user_id), site_uuid=site_uuid)

# -------------------------
# Schemas
# -------------------------
class NotificationEmailCreate(BaseModel):
    email: EmailStr
    # If omitted, email is applied to all sites owned by the user.
    site_uuid: Optional[uuid.UUID] = None


class NotificationEmailOut(BaseModel):
    id: int
    user_id: int
    site_uuid: uuid.UUID
    email: str
    is_enabled: bool


class NotificationEmailCreateResult(BaseModel):
    created: List[NotificationEmailOut]
    created_count: int
    skipped_count: int
    target_site_count: int


# -------------------------
# Endpoints
# -------------------------
@router.get("", response_model=List[NotificationEmailOut])
async def list_notification_emails(
    site_uuid: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    """
    List notification emails for a user.
    Optional site_uuid filter narrows results to one site.
    """
    rows = await notif_repo.list_notification_email_rows(
        db, user_id=int(user.id), site_uuid=site_uuid
    )
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
    user: User = Depends(get_current_user),
):
    """
    Add a notification email.
    If site_uuid is missing, apply it to all sites for the user.
    """
    normalized_email = payload.email.lower().strip()

    if payload.site_uuid is not None:
        site = await site_repo.get_site(
            db, site_uuid=payload.site_uuid, user_id=int(user.id), raise_if_missing=False
        )
        target_site_uuids = [site.site_uuid] if site is not None else []
    else:
        target_site_uuids = await site_repo.list_site_uuids(db, user_id=int(user.id))

    if not target_site_uuids:
        if payload.site_uuid is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Site not found for this user",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Create at least one site before adding notification emails",
        )

    created_rows = []
    for target_site_uuid in target_site_uuids:
        if await notif_repo.notification_email_exists(
            db, user_id=int(user.id), site_uuid=target_site_uuid, email=normalized_email
        ):
            continue

        row = await notif_repo.create_notification_email(
            db,
            dto=NotificationEmailCreateDTO(
                user_id=int(user.id),
                site_uuid=target_site_uuid,
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
        skipped_count=len(target_site_uuids) - len(created_rows),
        target_site_count=len(target_site_uuids),
    )


@router.delete("/{email_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification_email(
    email_id: int,
    all_sites: bool = False,
    request: Request = None,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    """
    Delete one notification email entry by ID.
    If all_sites=true, remove the same user/email pair from all sites.
    """
    email = await notif_repo.get_notification_email(
        db, email_id=email_id, user_id=int(user.id)
    )
    if not email:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification email not found",
        )

    email_user_id = email.user_id
    email_email = email.email
    email_site_uuid = email.site_uuid

    if all_sites:
        await notif_repo.delete_notification_email(
            db, user_id=email_user_id, email=email_email
        )
    else:
        await notif_repo.delete_notification_email(db, email_id=email_id)

    await db.commit()
    if request is not None:
        _invalidate_notification_email_cache(
            request,
            user_id=email_user_id,
            site_uuid=None if all_sites else email_site_uuid,
        )
    return None
