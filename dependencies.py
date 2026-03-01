import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import User
from core.database import db_manager
from core.security.tokens import decode_access_token
from application.services.manager import Manager

security = HTTPBearer(auto_error=False)

logger = logging.getLogger(__name__)


def get_db():
    if not db_manager.SessionLocal:
        raise Exception("Sync database has not been initialized.")
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncSession:
    if not db_manager.AsyncSessionLocal:
        raise Exception("Async database has not been initialized.")
    async with db_manager.AsyncSessionLocal() as session:
        yield session


def get_manager(request: Request) -> Manager:
    return request.app.state.manager


def _auth_error(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise _auth_error("Missing bearer token")

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise _auth_error(str(exc)) from exc

    raw_user_id = payload.get("user_id") or payload.get("sub")
    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        raise _auth_error("Invalid token payload")

    try:
        user = (
            await db.execute(select(User).where(User.id == user_id).limit(1))
        ).scalar_one_or_none()
        if user is None:
            raise _auth_error("User not found")
        return user
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auth dependency failed: %s", exc)
        raise _auth_error()


async def get_current_user_id(
    current_user: User = Depends(get_current_user),
) -> int:
    return int(current_user.id)
