"""User persistence. Never commits: the caller owns the transaction."""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import UserCreateDTO, UserProfileUpdateDTO
from application.repositories._helpers import model_patch
from core.database_orm import User, utc_now


class UserRepository:
    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    async def get_by_id(self, db: AsyncSession, user_id: int) -> Optional[User]:
        stmt = select(User).where(User.id == user_id).limit(1)
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        email = self.normalize_email(email)
        stmt = select(User).where(User.email == email).limit(1)
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    async def exists_email(self, db: AsyncSession, email: str) -> bool:
        return (await self.get_by_email(db, email)) is not None

    async def list_user_ids(self, db: AsyncSession) -> List[int]:
        res = await db.execute(select(User.id))
        return list(res.scalars().all())

    async def create_user(self, db: AsyncSession, dto: UserCreateDTO) -> User:
        """Insert a user row. Flush only; the caller commits."""
        user = User(
            user_name=dto.user_name,
            email=self.normalize_email(dto.email),
            hashed_password=dto.password_hash,
            contact_phone=dto.contact_phone,
            email_verified=dto.email_verified,
            verified_at=dto.verified_at,
            created_at=utc_now(),
        )

        db.add(user)
        try:
            await db.flush()
        except IntegrityError:
            raise ValueError("Email already exists")

        return user

    async def _update(self, db: AsyncSession, user_id: int, **values) -> None:
        await db.execute(update(User).where(User.id == int(user_id)).values(**values))

    async def mark_email_verified(self, db: AsyncSession, user_id: int) -> None:
        await self._update(db, user_id, email_verified=True, verified_at=utc_now())

    async def update_last_login(self, db: AsyncSession, user_id: int) -> None:
        await self._update(db, user_id, last_login_at=utc_now())

    async def update_password_hash(
        self, db: AsyncSession, user_id: int, new_password_hash: str
    ) -> None:
        await self._update(db, user_id, hashed_password=new_password_hash)

    async def update_profile(
        self, db: AsyncSession, user_id: int, dto: UserProfileUpdateDTO
    ) -> None:
        """Update the User columns set on `dto`."""
        values = model_patch(dto, drop_none=True)
        if "email" in values:
            values["email"] = self.normalize_email(values["email"])
        if values:
            await self._update(db, user_id, **values)

    async def delete_by_id(self, db: AsyncSession, user_id: int) -> int:
        """Hard-delete a user row. Does NOT commit; caller owns the transaction."""
        result = await db.execute(
            delete(User)
            .where(User.id == user_id)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount or 0
