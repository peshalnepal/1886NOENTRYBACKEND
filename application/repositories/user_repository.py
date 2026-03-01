from __future__ import annotations

from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from core.database_orm import User  # adjust import to your project structure


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()


    async def get_by_id(self, user_id: int) -> Optional[User]:
        stmt = select(User).where(User.id == user_id).limit(1)
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        email = self.normalize_email(email)
        stmt = select(User).where(User.email == email).limit(1)
        res = await self.db.execute(stmt)
        return res.scalar_one_or_none()

    async def exists_email(self, email: str) -> bool:
        return (await self.get_by_email(email)) is not None

    async def create_user(
        self,
        user_name: str,
        email: str,
        password_hash: str,
        contact_phone: str | None = None,
        email_verified: bool = False,
        verified_at: datetime | None = None,
    ) -> User:
        """
        Creates a user row. Does NOT commit automatically.
        Call await db.commit() in your service/route when ready.
        """
        email = self.normalize_email(email)

        user = User(
            user_name=user_name,
            email=email,
            hashed_password=password_hash,
            contact_phone=contact_phone,
            email_verified=email_verified,
            verified_at=verified_at,
            created_at=utc_now(),
        )

        self.db.add(user)
        try:
            await self.db.flush()  
        except IntegrityError:
            raise ValueError("Email already exists")

        return user

    async def mark_email_verified(self, user_id: int) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(email_verified=True, verified_at=utc_now())
        )
        await self.db.execute(stmt)

    async def update_last_login(self, user_id: int) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(last_login_at=utc_now())
        )
        await self.db.execute(stmt)

    async def update_password_hash(self, user_id: int, new_password_hash: str) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(hashed_password=new_password_hash)
        )
        await self.db.execute(stmt)


    async def update_profile(
        self,
        user_id: int,
        user_name: str | None = None,
        contact_phone: str | None = None,
    ) -> None:
        values = {}
        if user_name is not None:
            values["user_name"] = user_name
        if contact_phone is not None:
            values["contact_phone"] = contact_phone

        if not values:
            return

        stmt = update(User).where(User.id == user_id).values(**values)
        await self.db.execute(stmt)
