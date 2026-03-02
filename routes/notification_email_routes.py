# routes/notification_email_routes.py
"""
API routes for managing user notification emails.
"""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_async_db, get_current_user
from core.database_orm import NotificationEmail, Site, User

router = APIRouter(prefix="/notification-emails", tags=["notification-emails"])


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
    stmt = select(NotificationEmail).where(NotificationEmail.user_id == int(user.id))
    if site_uuid is not None:
        stmt = stmt.where(NotificationEmail.site_uuid == site_uuid)
    stmt = stmt.order_by(NotificationEmail.email.asc(), NotificationEmail.site_uuid.asc())

    result = await db.execute(stmt)
    rows = result.scalars().all()
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
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    """
    Add a notification email.
    If site_uuid is missing, apply it to all sites for the user.
    """
    normalized_email = payload.email.lower().strip()

    site_stmt = select(Site.site_uuid).where(Site.user_id == int(user.id))
    if payload.site_uuid is not None:
        site_stmt = site_stmt.where(Site.site_uuid == payload.site_uuid)

    site_rows = (await db.execute(site_stmt)).all()
    target_site_uuids = [row[0] for row in site_rows]

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

    existing_stmt = select(NotificationEmail.site_uuid).where(
        NotificationEmail.user_id == int(user.id),
        NotificationEmail.email == normalized_email,
        NotificationEmail.site_uuid.in_(target_site_uuids),
    )
    existing_rows = (await db.execute(existing_stmt)).all()
    existing_site_uuids = {row[0] for row in existing_rows}

    created_rows: List[NotificationEmail] = []
    for target_site_uuid in target_site_uuids:
        if target_site_uuid in existing_site_uuids:
            continue

        row = NotificationEmail(
            user_id=int(user.id),
            site_uuid=target_site_uuid,
            email=normalized_email,
            is_enabled=True,
        )
        db.add(row)
        created_rows.append(row)

    if not created_rows:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists for the selected site scope",
        )

    await db.commit()
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
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    """
    Delete one notification email entry by ID.
    If all_sites=true, remove the same user/email pair from all sites.
    """
    result = await db.execute(
        select(NotificationEmail).where(
            NotificationEmail.id == email_id,
            NotificationEmail.user_id == int(user.id),
        )
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification email not found",
        )

    if all_sites:
        await db.execute(
            delete(NotificationEmail).where(
                NotificationEmail.user_id == email.user_id,
                NotificationEmail.email == email.email,
            )
        )
    else:
        await db.delete(email)

    await db.commit()
    return None
