from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import User, utc_now
from core.security.hashing import get_password_hash, verify_password
from core.security.tokens import create_access_token
from dependencies import get_async_db, get_current_user


router = APIRouter(prefix="/auth", tags=["auth"])


class AuthUserOut(BaseModel):
    id: int
    user_name: str
    user_email: EmailStr


class AuthTokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserOut


class SignupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_name: str = Field(..., min_length=1, max_length=255)
    user_email: EmailStr
    password: SecretStr

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_email: EmailStr
    password: SecretStr

    @field_validator("user_email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).lower().strip()


def _to_user_out(user: User) -> AuthUserOut:
    return AuthUserOut(
        id=int(user.id),
        user_name=user.user_name,
        user_email=user.email,
    )


def _build_access_token(user: User) -> str:
    return create_access_token(
        {
            "sub": str(user.id),
            "user_id": int(user.id),
            "email": user.email,
        }
    )


async def _get_users_table_columns(db: AsyncSession) -> set[str]:
    res = await db.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'users'
            """
        )
    )
    return {str(row[0]) for row in res.fetchall()}


async def _insert_user_compat(
    db: AsyncSession,
    *,
    user_name: str,
    email: str,
    password_hash: str,
) -> User:
    cols = await _get_users_table_columns(db)
    now = utc_now()

    values: dict[str, object] = {}
    if "user_name" in cols:
        values["user_name"] = user_name
    if "email" in cols:
        values["email"] = email
    if "hashed_password" in cols:
        values["hashed_password"] = password_hash
    if "contact_phone" in cols:
        values["contact_phone"] = None
    if "created_at" in cols:
        values["created_at"] = now
    if "email_verified" in cols:
        values["email_verified"] = 1
    if "verified_at" in cols:
        values["verified_at"] = now
    if "last_login_at" in cols:
        values["last_login_at"] = None

    # Legacy columns present in some deployed databases.
    if "user_email" in cols:
        values["user_email"] = email
    if "password_hash" in cols:
        values["password_hash"] = password_hash
    if "is_verified" in cols:
        values["is_verified"] = 1
    if "business_url" in cols:
        values["business_url"] = os.getenv("DEFAULT_BUSINESS_URL", "https://1886noentry.com")
    if "contact_name" in cols:
        values["contact_name"] = user_name
    if "business_address" in cols:
        values["business_address"] = os.getenv("DEFAULT_BUSINESS_ADDRESS", "")

    if "email" not in values and "user_email" not in values:
        raise HTTPException(status_code=500, detail="Users table is missing email columns")
    if "hashed_password" not in values and "password_hash" not in values:
        raise HTTPException(status_code=500, detail="Users table is missing password columns")

    columns_sql = ", ".join(f"`{k}`" for k in values.keys())
    values_sql = ", ".join(f":{k}" for k in values.keys())
    await db.execute(text(f"INSERT INTO users ({columns_sql}) VALUES ({values_sql})"), values)
    await db.commit()

    user = (
        await db.execute(select(User).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=500, detail="Failed to create user")
    return user


@router.post("/signup", response_model=AuthTokenOut, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    db: AsyncSession = Depends(get_async_db),
):
    email = str(payload.user_email).lower().strip()

    existing = (
        await db.execute(select(User.id).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Account already exists")

    try:
        user = await _insert_user_compat(
            db,
            user_name=payload.user_name.strip(),
            email=email,
            password_hash=get_password_hash(payload.password.get_secret_value()),
        )
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Account already exists")

    token = _build_access_token(user)
    return AuthTokenOut(access_token=token, user=_to_user_out(user))


@router.post("/login", response_model=AuthTokenOut)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_async_db),
):
    email = str(payload.user_email).lower().strip()
    user = (
        await db.execute(select(User).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(payload.password.get_secret_value(), user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user.last_login_at = utc_now()
    await db.commit()
    await db.refresh(user)

    token = _build_access_token(user)
    return AuthTokenOut(access_token=token, user=_to_user_out(user))


@router.get("/me", response_model=AuthUserOut)
async def me(current_user: User = Depends(get_current_user)):
    return _to_user_out(current_user)
