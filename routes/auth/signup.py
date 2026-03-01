from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator
from sqlalchemy import select
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
    def normalize_email(cls, value: EmailStr) -> EmailStr:
        return EmailStr(str(value).lower().strip())

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
    def normalize_email(cls, value: EmailStr) -> EmailStr:
        return EmailStr(str(value).lower().strip())


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

    user = User(
        user_name=payload.user_name.strip(),
        email=email,
        hashed_password=get_password_hash(payload.password.get_secret_value()),
        email_verified=True,
        verified_at=utc_now(),
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

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
