import os
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import EmailVerification, utc_now


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


OTP_TTL_SECONDS = max(60, _env_int("OTP_TTL_SECONDS", 600))


class EmailVerificationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def invalidate_active(self, email: str) -> None:
        stmt = (
            update(EmailVerification)
            .where(EmailVerification.email == email, EmailVerification.used == False)  # noqa: E712
            .values(used=True, consumed_at=utc_now())
            .execution_options(synchronize_session=False)
        )
        await self.db.execute(stmt)

    async def create(
        self,
        email: str,
        code_hash: str,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> EmailVerification:
        now = utc_now()
        v = EmailVerification(
            email=email,
            code_hash=code_hash,
            sent_at=now,
            expires_at=now + timedelta(seconds=OTP_TTL_SECONDS),
            attempts=0,
            used=False,
            ip=ip,
            user_agent=user_agent,
        )
        self.db.add(v)
        await self.db.flush()
        return v

    async def get_latest_active(self, email: str) -> EmailVerification | None:
        now = utc_now()
        stmt = (
            select(EmailVerification)
            .where(
                EmailVerification.email == email,
                EmailVerification.used == False,          # noqa: E712
                EmailVerification.expires_at > now,
            )
            .order_by(EmailVerification.sent_at.desc())
            .limit(1)
        )
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

    async def increment_attempts(self, verification_id: int) -> None:
        stmt = (
            update(EmailVerification)
            .where(EmailVerification.id == verification_id)
            .values(attempts=EmailVerification.attempts + 1)
            .execution_options(synchronize_session=False)
        )
        await self.db.execute(stmt)

    async def consume(self, verification_id: int) -> None:
        stmt = (
            update(EmailVerification)
            .where(EmailVerification.id == verification_id)
            .values(used=True, consumed_at=utc_now())
            .execution_options(synchronize_session=False)
        )
        await self.db.execute(stmt)
