from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from application.services.user_snapshot_cache import UserSnapshotCache

from core.database_orm import User
from core.security.hashing import get_password_hash, verify_password
import asyncio
import logging
from typing import Optional

from dependencies import get_async_db, get_current_user,get_manager
from application.services.manager import Manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


def _invalidate_user_snapshot_cache(request: Request, user_id: int) -> None:
    cache = getattr(request.app.state, "user_snapshot_cache", None)
    if cache is None:
        return
    cache.invalidate(int(user_id))
    
class UserOut(BaseModel):
    id: int
    user_name: str
    user_email: EmailStr


class UserProfileUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str | None = Field(default=None, min_length=1, max_length=255)
    user_email: EmailStr | None = None

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr | None) -> str | None:
        if value is None:
            return None
        return str(value).lower().strip()

    @model_validator(mode="after")
    def validate_has_fields(self):
        if self.user_name is None and self.user_email is None:
            raise ValueError("Provide at least one field to update")
        return self


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: SecretStr
    new_password: SecretStr

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 8:
            raise ValueError("New password must be at least 8 characters long")
        return value


class DeleteAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: SecretStr


def _to_user_out(user: User) -> UserOut:
    return UserOut(
        id=int(user.id),
        user_name=user.user_name,
        user_email=user.email,
    )


@router.get("/me", response_model=UserOut)
async def get_me(
    current_user: User = Depends(get_current_user),
):
    return _to_user_out(current_user)


@router.patch("/me", response_model=UserOut)
async def update_me(
    payload: UserProfileUpdateRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    changed = False

    if payload.user_name is not None and payload.user_name != current_user.user_name:
        current_user.user_name = payload.user_name
        changed = True

    if payload.user_email is not None:
        next_email = str(payload.user_email).lower().strip()
        current_email = str(current_user.email).lower().strip()

        if next_email != current_email:
            exists = (
                await db.execute(
                    select(User.id).where(
                        User.email == next_email,
                        User.id != int(current_user.id),
                    ).limit(1)
                )
            ).scalar_one_or_none()
            if exists is not None:
                raise HTTPException(status_code=409, detail="Email is already in use")

            current_user.email = next_email
            changed = True

    if not changed:
        return _to_user_out(current_user)

    try:
        await db.commit()
        _invalidate_user_snapshot_cache(request, int(current_user.id))
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Email is already in use")
    except Exception:
        await db.rollback()
        raise

    await db.refresh(current_user)
    return _to_user_out(current_user)


@router.patch("/me/password")
async def change_my_password(
    payload: ChangePasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    current_password = payload.current_password.get_secret_value()
    new_password = payload.new_password.get_secret_value()

    if not verify_password(current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if verify_password(new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="New password must be different from current password",
        )

    current_user.hashed_password = get_password_hash(new_password)

    try:
        await db.commit()
        _invalidate_user_snapshot_cache(request, int(current_user.id))
    except Exception:
        await db.rollback()
        raise

    return {"message": "Password updated successfully"}


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_account(
    payload: DeleteAccountRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
    manager: Manager = Depends(get_manager),
):
    if not verify_password(payload.password.get_secret_value(), current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Password is incorrect")

    try:
        if manager is not None:
            cleanup = await asyncio.wait_for(
                manager.cleanup_user_resources(db, user_id=int(current_user.id)),
                timeout=15.0,
            )
            if cleanup.get("errors"):
                logger.warning(
                    "Partial cleanup errors for user %s: %s",
                    current_user.id, cleanup["errors"],
                )

        await db.delete(current_user)
        await db.commit()
        _invalidate_user_snapshot_cache(request, int(current_user.id))
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        await db.rollback()
        raise

    return Response(status_code=status.HTTP_204_NO_CONTENT)
