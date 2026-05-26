from __future__ import annotations

from typing import Optional, List

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from application.dtos import UserCreateDTO, UserProfileUpdateDTO
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

    async def get_exisiting_users_id(self, db: AsyncSession) -> List[int]:
        stmt = select(User.id).where(User.id.is_not(None))
        res = await db.execute(stmt)
        return res.scalars().all()

    async def create_user(self, db: AsyncSession, dto: UserCreateDTO) -> User:
        """
        Creates a user row from a `UserCreateDTO`. Does NOT commit automatically.
        Call await db.commit() in your service/route when ready.
        """
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

    async def mark_email_verified(self, db: AsyncSession, user_id: int) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(email_verified=True, verified_at=utc_now())
        )
        await db.execute(stmt)

    async def update_last_login(self, db: AsyncSession, user_id: int) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(last_login_at=utc_now())
        )
        await db.execute(stmt)

    async def update_password_hash(self, db: AsyncSession, user_id: int, new_password_hash: str) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(hashed_password=new_password_hash)
        )
        await db.execute(stmt)

    async def update_profile(self, db: AsyncSession, user_id: int, dto: UserProfileUpdateDTO) -> None:
        """Update the User columns set on `dto`."""
        values = dto.model_dump(exclude_unset=True)
        if "email" in values and values["email"] is not None:
            values["email"] = self.normalize_email(values["email"])
        values = {k: v for k, v in values.items() if v is not None}
        if not values:
            return

        stmt = update(User).where(User.id == user_id).values(**values)
        await db.execute(stmt)

    async def delete_by_id(self, db: AsyncSession, user_id: int) -> int:
        """Hard-delete a user row. Does NOT commit; caller owns the transaction."""
        result = await db.execute(
            delete(User)
            .where(User.id == user_id)
            .execution_options(synchronize_session=False)
        )
        return result.rowcount or 0
