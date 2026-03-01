from __future__ import annotations

import asyncio
import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.verify_repository import EmailVerificationRepository
from core.database_orm import SignupTempData, User, utc_now
from core.security.hashing import get_password_hash, verify_password
from core.security.tokens import create_access_token
from dependencies import get_async_db, get_current_user
from routes.auth.oauth import generate_otp, hash_otp, verify_otp_hash


router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


OTP_TTL_SECONDS = max(60, _env_int("OTP_TTL_SECONDS", 600))
OTP_MAX_ATTEMPTS = max(1, _env_int("OTP_MAX_ATTEMPTS", 5))
OTP_RESEND_COOLDOWN_SECONDS = max(0, _env_int("OTP_RESEND_COOLDOWN_SECONDS", 30))
SIGNUP_TEMP_TTL_SECONDS = max(
    OTP_TTL_SECONDS,
    _env_int("SIGNUP_TEMP_TTL_SECONDS", OTP_TTL_SECONDS),
)
OTP_SECRET_PEPPER = os.getenv("OTP_SECRET_PEPPER") or os.getenv("SECRET_KEY", "your_secret_key")
AUTH_DEBUG_RETURN_OTP = _env_bool("AUTH_DEBUG_RETURN_OTP", False)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = _env_int("SMTP_PORT", 587)
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = (os.getenv("FROM_EMAIL") or os.getenv("SMTP_FROM") or SMTP_USERNAME).strip()
SMTP_USE_TLS = _env_bool("SMTP_USE_TLS", True)


class AuthUserOut(BaseModel):
    id: int
    user_name: str
    user_email: EmailStr


class AuthTokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserOut


class SignupRequestCode(BaseModel):
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


class SignupVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    signup_token: str = Field(..., min_length=16, max_length=255)
    code: str = Field(..., min_length=6, max_length=10)

    @field_validator("signup_token")
    @classmethod
    def normalize_token(cls, value: str) -> str:
        token = value.strip()
        if not token:
            raise ValueError("signup_token is required")
        return token

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        code = value.strip()
        if not code.isdigit():
            raise ValueError("Verification code must contain digits only")
        return code


class SignupCodeOut(BaseModel):
    message: str
    signup_token: str
    expires_at: datetime
    debug_code: str | None = None


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
    email_verified: bool,
    verified_at: datetime | None,
    commit: bool = True,
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
        values["email_verified"] = 1 if email_verified else 0
    if "verified_at" in cols:
        values["verified_at"] = verified_at
    if "last_login_at" in cols:
        values["last_login_at"] = None

    # Legacy columns present in some deployed databases.
    if "user_email" in cols:
        values["user_email"] = email
    if "password_hash" in cols:
        values["password_hash"] = password_hash
    if "is_verified" in cols:
        values["is_verified"] = 1 if email_verified else 0
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
    if commit:
        await db.commit()

    user = (
        await db.execute(select(User).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=500, detail="Failed to create user")
    return user


def _build_signup_email_body(user_name: str, code: str) -> str:
    return (
        f"Hello {user_name},\n\n"
        "Use the verification code below to complete your 1886NoEntry signup:\n\n"
        f"{code}\n\n"
        f"The code expires in {OTP_TTL_SECONDS // 60} minutes.\n"
        "If you did not request this, you can ignore this email.\n"
    )


def _send_signup_code_email_sync(email: str, user_name: str, code: str) -> None:
    if not SMTP_HOST or not SMTP_FROM:
        raise RuntimeError("SMTP is not configured for signup verification")

    message = MIMEText(_build_signup_email_body(user_name=user_name, code=code), "plain", "utf-8")
    message["Subject"] = "1886NoEntry signup verification code"
    message["From"] = SMTP_FROM
    message["To"] = email

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
        if SMTP_USE_TLS:
            server.starttls()
        if SMTP_USERNAME and SMTP_PASSWORD:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_FROM, [email], message.as_string())


async def _send_signup_code_email(email: str, user_name: str, code: str) -> None:
    await asyncio.to_thread(_send_signup_code_email_sync, email, user_name, code)


def _request_ip(req: Request) -> str | None:
    fwd = req.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip() or None
    return req.client.host if req.client else None


@router.post("/signup/request-code", response_model=SignupCodeOut, status_code=status.HTTP_202_ACCEPTED)
@router.post("/signup", response_model=SignupCodeOut, status_code=status.HTTP_202_ACCEPTED)
async def signup_request_code(
    payload: SignupRequestCode,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    email = str(payload.user_email).lower().strip()
    user_name = payload.user_name.strip()

    existing = (
        await db.execute(select(User.id).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Account already exists")

    verify_repo = EmailVerificationRepository(db)
    active_verification = await verify_repo.get_latest_active(email)
    now = utc_now()
    if active_verification is not None and active_verification.sent_at is not None:
        age = (now - _as_utc(active_verification.sent_at)).total_seconds()
        if age < OTP_RESEND_COOLDOWN_SECONDS:
            retry_after = OTP_RESEND_COOLDOWN_SECONDS - int(age)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {max(1, retry_after)} seconds before requesting a new code.",
            )

    code = generate_otp()
    code_hash = hash_otp(email=email, otp=code, secret_pepper=OTP_SECRET_PEPPER)
    signup_token = secrets.token_urlsafe(32)
    signup_expires_at = now + timedelta(seconds=SIGNUP_TEMP_TTL_SECONDS)

    try:
        await verify_repo.invalidate_active(email)
        await verify_repo.create(
            email=email,
            code_hash=code_hash,
            ip=_request_ip(request),
            user_agent=request.headers.get("user-agent"),
        )

        # Mark older signup tokens for this email as used.
        await db.execute(
            text(
                """
                UPDATE signup_temp_data
                SET used = 1
                WHERE used = 0
                  AND JSON_UNQUOTE(JSON_EXTRACT(data, '$.user_email')) = :email
                """
            ),
            {"email": email},
        )

        db.add(
            SignupTempData(
                token=signup_token,
                data={
                    "user_name": user_name,
                    "user_email": email,
                    "password_hash": get_password_hash(payload.password.get_secret_value()),
                },
                created_at=now,
                expires_at=signup_expires_at,
                used=False,
            )
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Account already exists")
    except Exception:
        await db.rollback()
        raise

    debug_code: str | None = None
    try:
        await _send_signup_code_email(email=email, user_name=user_name, code=code)
    except Exception as exc:
        logger.exception("Failed to send signup verification email to %s", email)
        if AUTH_DEBUG_RETURN_OTP:
            debug_code = code
        else:
            raise HTTPException(
                status_code=503,
                detail="Failed to send verification code. Please try again.",
            ) from exc

    message = f"Verification code sent to {email}. Enter the code to finish signup."
    return SignupCodeOut(
        message=message,
        signup_token=signup_token,
        expires_at=signup_expires_at,
        debug_code=debug_code,
    )


@router.post("/signup/verify", response_model=AuthTokenOut, status_code=status.HTTP_201_CREATED)
async def signup_verify(
    payload: SignupVerifyRequest,
    db: AsyncSession = Depends(get_async_db),
):
    temp = (
        await db.execute(
            select(SignupTempData).where(
                SignupTempData.token == payload.signup_token,
                SignupTempData.used == False,  # noqa: E712
                SignupTempData.expires_at > utc_now(),
            ).limit(1)
        )
    ).scalar_one_or_none()
    if temp is None:
        raise HTTPException(status_code=400, detail="Signup token is invalid or expired. Request a new code.")

    raw = temp.data if isinstance(temp.data, dict) else {}
    user_name = str(raw.get("user_name", "")).strip()
    email = str(raw.get("user_email", "")).strip().lower()
    password_hash = str(raw.get("password_hash", "")).strip()

    if not user_name or not email or not password_hash:
        temp.used = True
        await db.commit()
        raise HTTPException(status_code=400, detail="Signup token payload is invalid. Request a new code.")

    existing = (
        await db.execute(select(User.id).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if existing is not None:
        temp.used = True
        await db.commit()
        raise HTTPException(status_code=409, detail="Account already exists")

    verify_repo = EmailVerificationRepository(db)
    verification = await verify_repo.get_latest_active(email)
    if verification is None:
        raise HTTPException(status_code=400, detail="Verification code expired or missing. Request a new code.")

    if verification.attempts >= OTP_MAX_ATTEMPTS:
        await verify_repo.consume(int(verification.id))
        await db.commit()
        raise HTTPException(
            status_code=429,
            detail="Too many invalid attempts. Request a new verification code.",
        )

    is_valid_code = verify_otp_hash(
        email=email,
        otp=payload.code,
        secret_pepper=OTP_SECRET_PEPPER,
        stored_hash=verification.code_hash,
    )
    if not is_valid_code:
        await verify_repo.increment_attempts(int(verification.id))
        if verification.attempts + 1 >= OTP_MAX_ATTEMPTS:
            await verify_repo.consume(int(verification.id))
        await db.commit()
        raise HTTPException(status_code=400, detail="Invalid verification code")

    temp.used = True
    await verify_repo.consume(int(verification.id))

    try:
        user = await _insert_user_compat(
            db,
            user_name=user_name,
            email=email,
            password_hash=password_hash,
            email_verified=True,
            verified_at=utc_now(),
            commit=False,
        )
        user.last_login_at = utc_now()
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Account already exists")
    except Exception:
        await db.rollback()
        raise

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

    if not bool(getattr(user, "email_verified", True)):
        raise HTTPException(
            status_code=403,
            detail="Email is not verified. Complete signup verification first.",
        )

    user.last_login_at = utc_now()
    await db.commit()
    await db.refresh(user)

    token = _build_access_token(user)
    return AuthTokenOut(access_token=token, user=_to_user_out(user))


@router.get("/me", response_model=AuthUserOut)
async def me(current_user: User = Depends(get_current_user)):
    return _to_user_out(current_user)
