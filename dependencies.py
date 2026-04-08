import asyncio
import logging
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from application.services.manager import Manager
from application.services.notification import NotificationService, WebNotificationHub
from application.services.retention import RetentionService
from application.services.user_snapshot_cache import UserSnapshotCache
from core.database import db_manager
from core.database_orm import User
from core.security.tokens import decode_access_token

security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# database
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# generic app.state helper
# -------------------------------------------------------------------
def _require_app_state(request: Request, attr_name: str, detail: str) -> Any:
    value = getattr(request.app.state, attr_name, None)
    if value is None:
        raise HTTPException(status_code=503, detail=detail)
    return value


# -------------------------------------------------------------------
# app singletons / shared services
# -------------------------------------------------------------------
def get_manager(request: Request) -> Manager:
    return _require_app_state(request, "manager", "Manager not available")


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is not None:
        return session_factory

    # fallback for safety
    if getattr(db_manager, "AsyncSessionLocal", None) is not None:
        return db_manager.AsyncSessionLocal

    raise HTTPException(status_code=503, detail="Session factory not available")


def get_notification_service(request: Request) -> NotificationService:
    manager = getattr(request.app.state, "manager", None)
    if manager is not None:
        svc = getattr(manager, "_notification_service", None)
        if svc is not None:
            return svc
    raise HTTPException(status_code=503, detail="Notification service not available")


def get_notification_hub(request: Request) -> WebNotificationHub:
    hub = getattr(request.app.state, "notification_hub", None)
    if hub is not None:
        return hub
    manager = getattr(request.app.state, "manager", None)
    if manager is not None:
        svc = getattr(manager, "_notification_service", None)
        if svc is not None:
            hub = getattr(svc, "hub", None)
            if hub is not None:
                return hub
    raise HTTPException(status_code=503, detail="Notification hub not available")


def get_user_snapshot_cache(request: Request) -> UserSnapshotCache:
    cache = getattr(request.app.state, "user_snapshot_cache", None)
    if cache is None:
        cache = UserSnapshotCache()
        request.app.state.user_snapshot_cache = cache
    return cache


def get_retention_service(request: Request) -> RetentionService:
    return _require_app_state(request, "retention_service", "Retention service not available")


def get_alert_blob_cleanup_tasks(request: Request) -> set:
    tasks = getattr(request.app.state, "alert_blob_cleanup_tasks", None)
    if tasks is None:
        tasks = set()
        request.app.state.alert_blob_cleanup_tasks = tasks
    return tasks


# -------------------------------------------------------------------
# auth
# -------------------------------------------------------------------
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