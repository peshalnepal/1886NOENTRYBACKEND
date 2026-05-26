from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import EmailVerification, utc_now
from core.env import env_int


OTP_TTL_SECONDS = env_int("OTP_TTL_SECONDS", 600, minimum=60)


class EmailVerificationRepository:
    async def invalidate_active(self, db: AsyncSession, email: str) -> None:
        stmt = (
            update(EmailVerification)
            .where(EmailVerification.email == email, EmailVerification.used == False)
            .values(used=True, consumed_at=utc_now())
        )
        await db.execute(stmt)

    async def create(
        self,
        db: AsyncSession,
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
        db.add(v)
        await db.flush()
        return v

    async def get_latest_active(self, db: AsyncSession, email: str) -> EmailVerification | None:
        now = utc_now()
        stmt = (
            select(EmailVerification)
            .where(
                EmailVerification.email == email,
                EmailVerification.used == False,
                EmailVerification.expires_at > now,
            )
            .order_by(EmailVerification.id.desc())
            .limit(1)
        )
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    async def increment_attempts(self, db: AsyncSession, verification_id: int) -> None:
        stmt = (
            update(EmailVerification)
            .where(EmailVerification.id == verification_id)
            .values(attempts=EmailVerification.attempts + 1)
        )
        await db.execute(stmt)

    async def consume(self, db: AsyncSession, verification_id: int) -> None:
        stmt = (
            update(EmailVerification)
            .where(EmailVerification.id == verification_id)
            .values(used=True, consumed_at=utc_now())
        )
        await db.execute(stmt)
