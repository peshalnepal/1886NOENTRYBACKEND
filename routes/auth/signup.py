from __future__ import annotations

import asyncio
import logging
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import OrganizationCreateDTO, OrgMembershipUpsertDTO
from application.repositories.organization_repository import OrganizationRepository
from application.repositories.verify_repository import (
    PURPOSE_PASSWORD_RESET,
    EmailVerificationRepository,
)
from core.database_orm import SignupTempData, User,EmailVerification, utc_now
from core.security.roles import OrgRole
from core.env import env_bool, env_int
from core.schemas import (
    AuthOrgMembershipOut,
    AuthTokenOut,
    AuthUserOut,
    LoginRequest,
    PasswordResetCodeOut,
    PasswordResetConfirm,
    PasswordResetRequest,
    SignupCodeOut,
    SignupRequestCode,
    SignupVerifyRequest,
)
from core.security.hashing import get_password_hash, verify_password
from core.security.tokens import create_access_token
from dependencies import get_async_db, get_current_user
from routes.auth.oauth import generate_otp, hash_otp, verify_otp_hash


router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


OTP_TTL_SECONDS = env_int("OTP_TTL_SECONDS", 600, minimum=60)
OTP_MAX_ATTEMPTS = env_int("OTP_MAX_ATTEMPTS", 5, minimum=1)
OTP_RESEND_COOLDOWN_SECONDS = env_int("OTP_RESEND_COOLDOWN_SECONDS", 30, minimum=0)
SIGNUP_TEMP_TTL_SECONDS = env_int(
    "SIGNUP_TEMP_TTL_SECONDS", OTP_TTL_SECONDS, minimum=OTP_TTL_SECONDS,
)
OTP_SECRET_PEPPER = os.getenv("SECRET_KEY", "")
AUTH_DEBUG_RETURN_OTP = env_bool("AUTH_DEBUG_RETURN_OTP", False)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = env_int("SMTP_PORT", 587)
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = (os.getenv("FROM_EMAIL") or os.getenv("SMTP_FROM") or SMTP_USERNAME).strip()
SMTP_USE_TLS = env_bool("SMTP_USE_TLS", True)


def _slug_from_email(email: str) -> str:
    """Build a URL-safe organization slug from the signup email.

    Example: "Jane.Doe@Example.com" -> "jane-doe-example-com". A short
    random suffix is appended at the caller to guarantee uniqueness.
    """
    local = email.split("@", 1)[0]
    cleaned = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in email.replace("@", "-")
    )
    cleaned = "-".join(part for part in cleaned.split("-") if part) or local.lower()
    return cleaned[:48] or "org"


async def _bootstrap_personal_organization(
    db: AsyncSession, *, user: User
) -> None:
    """Create an Organization owned by `user` and grant them Org Admin.

    Called immediately after a signup verification succeeds. The new
    organization is named after the user (`<name>'s Organization`) and
    given a slug derived from their email. If the derived slug is
    already taken (e.g. they signed up before, the org was deleted,
    and someone else grabbed the slug) a 6-char hex suffix is appended.

    Does NOT commit; the caller owns the transaction.
    """
    org_repo = OrganizationRepository()

    base_slug = _slug_from_email(user.email)
    slug = base_slug
    # Try a handful of suffixes to avoid an infinite loop in the rare
    # case of repeated collisions.
    for attempt in range(5):
        existing = await org_repo.get_by_slug(db, slug)
        if existing is None:
            break
        slug = f"{base_slug}-{secrets.token_hex(3)}"

    org = await org_repo.create_organization(
        db,
        OrganizationCreateDTO(
            name=f"{user.user_name}'s Organization",
            slug=slug,
            owner_user_id=int(user.id),
        ),
    )
    await org_repo.upsert_org_membership(
        db,
        OrgMembershipUpsertDTO(
            user_id=int(user.id),
            org_id=int(org.id),
            role=OrgRole.ADMIN.value,
        ),
    )
    await db.commit()
    await db.refresh(org)


def _to_user_out(user: User) -> AuthUserOut:
    """Project a User row to the auth payload (no org info)."""
    return AuthUserOut(
        id=int(user.id),
        user_name=user.user_name,
        user_email=user.email,
        is_platform_admin=bool(getattr(user, "is_platform_admin", False)),
        organizations=[],
    )


async def _to_user_out_with_orgs(db: AsyncSession, user: User) -> AuthUserOut:
    """Project a User row, additionally hydrating their organizations.

    Used by login/signup/me so the client receives the full role
    picture in one round-trip and can pick which dashboard to land
    the user on.
    """
    rows = await OrganizationRepository().list_user_orgs(db, user_id=int(user.id))
    return AuthUserOut(
        id=int(user.id),
        user_name=user.user_name,
        user_email=user.email,
        is_platform_admin=bool(getattr(user, "is_platform_admin", False)),
        organizations=[
            AuthOrgMembershipOut(
                org_id=int(o.id),
                org_name=o.name,
                org_slug=o.slug,
                role=role_name,
            )
            for (role_name, o) in rows
        ],
    )


def _build_access_token(user: User) -> str:
    return create_access_token(
        {
            "sub": str(user.id),
            "user_id": int(user.id),
            "email": user.email,
        }
    )

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

    verify_repo = EmailVerificationRepository()
    active_verification = await verify_repo.get_latest_active(db, email)
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
        await verify_repo.invalidate_active(db, email)
        await verify_repo.create(
            db,
            email=email,
            code_hash=code_hash,
            ip=_request_ip(request),
            user_agent=request.headers.get("user-agent"),
        )

        stmt = (
            update(SignupTempData)
            .where(
                SignupTempData.used.is_(False),
                SignupTempData.data["user_email"].as_string() == email, 
            )
            .values(used=True)
        )

        await db.execute(stmt)
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
                SignupTempData.used.is_(False),
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

    verify_repo = EmailVerificationRepository()
    verification = await verify_repo.get_latest_active(db, email)
    if verification is None:
        raise HTTPException(status_code=400, detail="Verification code expired or missing. Request a new code.")

    if verification.attempts >= OTP_MAX_ATTEMPTS:
        await verify_repo.consume(db, int(verification.id))
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
        await verify_repo.increment_attempts(db, int(verification.id))
        if verification.attempts + 1 >= OTP_MAX_ATTEMPTS:
            await verify_repo.consume(db, int(verification.id))
        await db.commit()
        raise HTTPException(status_code=400, detail="Invalid verification code")

    temp.used = True
    await verify_repo.consume(db, int(verification.id))

    try:
        user = User(
            user_name=user_name,
            email=email,
            hashed_password=password_hash,
            email_verified=True,
            verified_at=utc_now(),
            created_at=utc_now(),
        )
        db.add(user)
        user.last_login_at = utc_now()
        await db.flush()
        await _bootstrap_personal_organization(db, user=user)
        await db.commit()
        await db.refresh(user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Account already exists")
    except Exception:
        await db.rollback()
        raise
    token = _build_access_token(user)
    return AuthTokenOut(
        access_token=token, user=await _to_user_out_with_orgs(db, user)
    )


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
    return AuthTokenOut(
        access_token=token, user=await _to_user_out_with_orgs(db, user)
    )


@router.get("/me", response_model=AuthUserOut)
async def me(
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    return await _to_user_out_with_orgs(db, current_user)


# =====================================================================
# Forgot password
# =====================================================================
def _build_reset_email_body(user_name: str, code: str) -> str:
    return (
        f"Hello {user_name},\n\n"
        "Use the code below to reset your 1886NoEntry password:\n\n"
        f"{code}\n\n"
        f"The code expires in {OTP_TTL_SECONDS // 60} minutes.\n"
        "If you did not request a password reset, you can ignore this email — "
        "your password has not been changed.\n"
    )


def _send_reset_code_email_sync(email: str, user_name: str, code: str) -> None:
    if not SMTP_HOST or not SMTP_FROM:
        raise RuntimeError("SMTP is not configured for password reset")

    message = MIMEText(
        _build_reset_email_body(user_name=user_name, code=code), "plain", "utf-8"
    )
    message["Subject"] = "1886NoEntry password reset code"
    message["From"] = SMTP_FROM
    message["To"] = email

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=12) as server:
        if SMTP_USE_TLS:
            server.starttls()
        if SMTP_USERNAME and SMTP_PASSWORD:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_FROM, [email], message.as_string())


@router.post(
    "/password/forgot",
    response_model=PasswordResetCodeOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def password_forgot(
    payload: PasswordResetRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
):
    """Email a password-reset code.

    Always reports success. Telling an anonymous caller whether an address has
    an account turns this endpoint into a user-enumeration oracle, so the
    unknown-email path returns the same body (and takes the same visible path)
    as the real one.
    """
    email = str(payload.user_email).lower().strip()
    now = utc_now()
    expires_at = now + timedelta(seconds=OTP_TTL_SECONDS)
    generic = PasswordResetCodeOut(
        message=(
            f"If an account exists for {email}, a reset code has been sent. "
            "Enter it below to choose a new password."
        ),
        expires_at=expires_at,
    )

    user = (
        await db.execute(select(User).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if user is None:
        logger.info("Password reset requested for unknown email %s", email)
        return generic

    verify_repo = EmailVerificationRepository()
    active = await verify_repo.get_latest_active(db, email, purpose=PURPOSE_PASSWORD_RESET)
    if active is not None and active.sent_at is not None:
        age = (now - _as_utc(active.sent_at)).total_seconds()
        if age < OTP_RESEND_COOLDOWN_SECONDS:
            retry_after = OTP_RESEND_COOLDOWN_SECONDS - int(age)
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {max(1, retry_after)} seconds before requesting a new code.",
            )

    code = generate_otp()
    code_hash = hash_otp(email=email, otp=code, secret_pepper=OTP_SECRET_PEPPER)

    try:
        await verify_repo.invalidate_active(db, email, purpose=PURPOSE_PASSWORD_RESET)
        await verify_repo.create(
            db,
            email=email,
            code_hash=code_hash,
            ip=_request_ip(request),
            user_agent=request.headers.get("user-agent"),
            purpose=PURPOSE_PASSWORD_RESET,
        )
        await db.commit()
    except Exception:
        await db.rollback()
        raise

    try:
        await asyncio.to_thread(
            _send_reset_code_email_sync, email, user.user_name, code
        )
    except Exception as exc:
        logger.exception("Failed to send password reset email to %s", email)
        if AUTH_DEBUG_RETURN_OTP:
            return generic.model_copy(update={"debug_code": code})
        raise HTTPException(
            status_code=503,
            detail="Failed to send reset code. Please try again.",
        ) from exc

    return generic


@router.post("/password/reset", response_model=AuthTokenOut)
async def password_reset(
    payload: PasswordResetConfirm,
    db: AsyncSession = Depends(get_async_db),
):
    """Verify the reset code and set the new password.

    On success the user is signed in directly — they just proved control of
    the mailbox and chose the password, so a second login step adds nothing.
    """
    email = str(payload.user_email).lower().strip()
    new_password = payload.new_password.get_secret_value()

    verify_repo = EmailVerificationRepository()
    verification = await verify_repo.get_latest_active(
        db, email, purpose=PURPOSE_PASSWORD_RESET
    )
    if verification is None:
        raise HTTPException(
            status_code=400,
            detail="Reset code expired or missing. Request a new code.",
        )

    if verification.attempts >= OTP_MAX_ATTEMPTS:
        await verify_repo.consume(db, int(verification.id))
        await db.commit()
        raise HTTPException(
            status_code=429,
            detail="Too many invalid attempts. Request a new reset code.",
        )

    if not verify_otp_hash(
        email=email,
        otp=payload.code,
        secret_pepper=OTP_SECRET_PEPPER,
        stored_hash=verification.code_hash,
    ):
        await verify_repo.increment_attempts(db, int(verification.id))
        if verification.attempts + 1 >= OTP_MAX_ATTEMPTS:
            await verify_repo.consume(db, int(verification.id))
        await db.commit()
        raise HTTPException(status_code=400, detail="Invalid reset code")

    user = (
        await db.execute(select(User).where(User.email == email).limit(1))
    ).scalar_one_or_none()
    if user is None:
        # The account was deleted between request and confirm.
        await verify_repo.consume(db, int(verification.id))
        await db.commit()
        raise HTTPException(status_code=400, detail="Invalid reset code")

    try:
        await verify_repo.consume(db, int(verification.id))
        user.hashed_password = get_password_hash(new_password)
        if not bool(getattr(user, "email_verified", False)):
            user.email_verified = True
            user.verified_at = utc_now()
        user.last_login_at = utc_now()
        await db.commit()
        await db.refresh(user)
    except Exception:
        await db.rollback()
        raise

    logger.info("Password reset completed for user=%s", int(user.id))
    token = _build_access_token(user)
    return AuthTokenOut(
        access_token=token, user=await _to_user_out_with_orgs(db, user)
    )
