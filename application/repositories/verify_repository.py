from datetime import timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import EmailVerification, SignupTempData, utc_now
from core.env import env_int


OTP_TTL_SECONDS = env_int("OTP_TTL_SECONDS", 600, minimum=60)

# Stored in `EmailVerification.additional_data` so the two OTP flows that share
# this table can never consume each other's codes.
PURPOSE_SIGNUP = "signup"
PURPOSE_PASSWORD_RESET = "password_reset"


class EmailVerificationRepository:
    async def invalidate_active(
        self, db: AsyncSession, email: str, purpose: str = PURPOSE_SIGNUP
    ) -> None:
        stmt = (
            update(EmailVerification)
            .where(
                EmailVerification.email == email,
                EmailVerification.used == False,
                self._purpose_clause(purpose),
            )
            .values(used=True, consumed_at=utc_now())
            .execution_options(synchronize_session=False)
        )
        await db.execute(stmt)

    async def create(
        self,
        db: AsyncSession,
        email: str,
        code_hash: str,
        ip: str | None = None,
        user_agent: str | None = None,
        purpose: str = PURPOSE_SIGNUP,
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
            additional_data=purpose,
        )
        db.add(v)
        await db.flush()
        return v

    @staticmethod
    def _purpose_clause(purpose: str):
        """Match rows of one purpose only.

        Signup and password-reset codes share this table, so every lookup must
        be scoped or a reset code would satisfy a signup (and vice versa).
        Rows written before `additional_data` carried a purpose are NULL and
        are treated as signup, which is all that existed then.
        """
        if purpose == PURPOSE_SIGNUP:
            return (EmailVerification.additional_data.is_(None)) | (
                EmailVerification.additional_data == PURPOSE_SIGNUP
            )
        return EmailVerification.additional_data == purpose

    async def get_latest_active(
        self, db: AsyncSession, email: str, purpose: str = PURPOSE_SIGNUP
    ) -> EmailVerification | None:
        now = utc_now()
        stmt = (
            select(EmailVerification)
            .where(
                EmailVerification.email == email,
                EmailVerification.used == False,
                EmailVerification.expires_at > now,
                self._purpose_clause(purpose),
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

    async def purge_for_email(self, db: AsyncSession, email: str) -> int:
        """Delete every OTP / pending-signup row belonging to `email`.

        Neither table has a FK to `users` — they are keyed by the raw email
        string because both are written *before* the user row exists. So no
        CASCADE reaches them on account deletion and the rows would linger
        forever, which is also why a re-signup with the same address could
        collide with a stale pending token.

        Does NOT commit; the caller owns the transaction.
        """
        email = email.strip().lower()

        removed = await db.execute(
            delete(EmailVerification)
            .where(EmailVerification.email == email)
            .execution_options(synchronize_session=False)
        )
        await db.execute(
            delete(SignupTempData)
            .where(SignupTempData.data["user_email"].as_string() == email)
            .execution_options(synchronize_session=False)
        )
        return removed.rowcount or 0
