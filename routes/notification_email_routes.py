# routes/notification_email_routes.py
"""
API routes for managing user notification emails.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from dependencies import get_async_db
from core.database_orm import NotificationEmail

router = APIRouter(prefix="/notification-emails", tags=["notification-emails"])


# -------------------------
# Schemas
# -------------------------
class NotificationEmailCreate(BaseModel):
    user_id: int
    email: EmailStr


class NotificationEmailOut(BaseModel):
    id: int
    user_id: int
    email: str


# -------------------------
# Endpoints
# -------------------------
@router.get("", response_model=List[NotificationEmailOut])
async def list_notification_emails(
    user_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """
    List all notification emails for a user.
    """
    result = await db.execute(
        select(NotificationEmail).where(NotificationEmail.user_id == user_id)
    )
    emails = result.scalars().all()
    return [
        NotificationEmailOut(id=e.id, user_id=e.user_id, email=e.email)
        for e in emails
    ]


@router.post("", response_model=NotificationEmailOut, status_code=status.HTTP_201_CREATED)
async def add_notification_email(
    payload: NotificationEmailCreate,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Add a new notification email for a user.
    """
    # Check for duplicates
    result = await db.execute(
        select(NotificationEmail).where(
            NotificationEmail.user_id == payload.user_id,
            NotificationEmail.email == payload.email,
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists for this user",
        )

    new_email = NotificationEmail(
        user_id=payload.user_id,
        email=payload.email,
    )
    db.add(new_email)
    await db.commit()
    await db.refresh(new_email)

    return NotificationEmailOut(
        id=new_email.id,
        user_id=new_email.user_id,
        email=new_email.email,
    )


@router.delete("/{email_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_notification_email(
    email_id: int,
    db: AsyncSession = Depends(get_async_db),
):
    """
    Delete a notification email by ID.
    """
    result = await db.execute(
        select(NotificationEmail).where(NotificationEmail.id == email_id)
    )
    email = result.scalar_one_or_none()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification email not found",
        )

    await db.delete(email)
    await db.commit()
    return None
