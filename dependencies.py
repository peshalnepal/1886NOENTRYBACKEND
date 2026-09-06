"""FastAPI dependencies: database sessions, app singletons and authorization."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from application.repositories.user_repository import UserRepository
from application.services.authz_service import AuthzService
from application.services.manager import Manager
from application.services.notification import NotificationService, WebNotificationHub
from application.services.retention import RetentionService
from application.services.user_snapshot_cache import UserSnapshotCache
from core.database import db_manager
from core.database_orm import User
from core.security.roles import OrgRole, Permission
from core.security.tokens import decode_access_token

security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# database
# -------------------------------------------------------------------
def get_db():
    if not db_manager.SessionLocal:
        raise RuntimeError("Sync database has not been initialized.")
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncSession:
    if not db_manager.AsyncSessionLocal:
        raise RuntimeError("Async database has not been initialized.")
    async with db_manager.AsyncSessionLocal() as session:
        yield session


# -------------------------------------------------------------------
# app singletons / shared services
# -------------------------------------------------------------------
def _require_app_state(request: Request, attr_name: str, detail: str) -> Any:
    value = getattr(request.app.state, attr_name, None)
    if value is None:
        raise HTTPException(status_code=503, detail=detail)
    return value


def _app_state_default(request: Request, attr_name: str, factory) -> Any:
    """Read an app-state singleton, creating and storing it on first use."""
    value = getattr(request.app.state, attr_name, None)
    if value is None:
        value = factory()
        setattr(request.app.state, attr_name, value)
    return value


def get_manager(request: Request) -> Manager:
    return _require_app_state(request, "manager", "Manager not available")


def get_manager_optional(request: Request) -> Optional[Manager]:
    """The Manager if it is up, else None — never 503.

    Used by routes whose DB work must succeed even before the pipeline / edge
    layer has finished starting.
    """
    return getattr(request.app.state, "manager", None)


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is not None:
        return session_factory
    if getattr(db_manager, "AsyncSessionLocal", None) is not None:
        return db_manager.AsyncSessionLocal
    raise HTTPException(status_code=503, detail="Session factory not available")


def get_notification_service(request: Request) -> NotificationService:
    svc = getattr(request.app.state, "notification_service", None)
    if svc is None:
        manager = getattr(request.app.state, "manager", None)
        svc = getattr(manager, "notification_service", None) if manager else None
    if svc is None:
        raise HTTPException(status_code=503, detail="Notification service not available")
    return svc


def get_notification_hub(request: Request) -> WebNotificationHub:
    hub = getattr(request.app.state, "notification_hub", None)
    if hub is None:
        manager = getattr(request.app.state, "manager", None)
        svc = getattr(manager, "notification_service", None) if manager else None
        hub = getattr(svc, "hub", None)
    if hub is None:
        raise HTTPException(status_code=503, detail="Notification hub not available")
    return hub


def get_user_snapshot_cache(request: Request) -> UserSnapshotCache:
    return _app_state_default(request, "user_snapshot_cache", UserSnapshotCache)


def get_retention_service(request: Request) -> RetentionService:
    return _require_app_state(
        request, "retention_service", "Retention service not available"
    )


def get_alert_blob_cleanup_tasks(request: Request) -> set:
    return _app_state_default(request, "alert_blob_cleanup_tasks", set)


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

    try:
        user_id = int(payload.get("user_id") or payload.get("sub"))
    except (TypeError, ValueError):
        raise _auth_error("Invalid token payload")

    try:
        user = await UserRepository().get_by_id(db, user_id)
    except Exception as exc:
        logger.error("Auth dependency failed: %s", exc)
        raise _auth_error()

    if user is None:
        raise _auth_error("User not found")
    return user


async def get_current_user_id(current_user: User = Depends(get_current_user)) -> int:
    return int(current_user.id)


async def require_platform_admin(user: User = Depends(get_current_user)) -> User:
    """Allow only users flagged as platform admins."""
    AuthzService.ensure_platform_admin(user)
    return user


@dataclass
class OrgContext:
    """The organization + role a request operates in.

    Resolved from the caller's single access grant, so the flat resource routes
    (`/sites`, `/cameras`, `/devices`) need no `org_id` in their path.
    `is_admin` already folds in the operator-as-admin elevation (see
    `AuthzService.is_effective_org_admin`).

    `org_id is None` is the platform-admin "super context": the caller has no
    single-tenant scope, and repositories/helpers skip org filtering when they
    see it.
    """

    user: User
    org_id: Optional[int]
    role: str
    is_admin: bool

    @property
    def is_platform_admin(self) -> bool:
        return self.org_id is None and bool(
            getattr(self.user, "is_platform_admin", False)
        )


async def get_org_context(
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
) -> OrgContext:
    """Resolve the caller's organization and role for flat resource routes.

    Platform admins get the super context (`org_id=None`), which the flat routes
    read as "every org". A non-platform user with no membership gets a 403.
    """
    if bool(getattr(user, "is_platform_admin", False)):
        return OrgContext(user=user, org_id=None, role=OrgRole.ADMIN.value, is_admin=True)

    resolved = await AuthzService.resolve_user_org(db, user=user)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not belong to any organization.",
        )

    org_id, role = resolved
    is_admin = await AuthzService.is_effective_org_admin(db, user=user, org_id=org_id)
    return OrgContext(user=user, org_id=org_id, role=role, is_admin=is_admin)


class RequirePermission:
    """The single authorization dependency for resource routes.

    A route declares the permission it needs and nothing else::

        ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_CAMERAS))

    It both gates the request and returns the `OrgContext` the body consumes.
    You pass only a permission — never a role; the caller's role comes from
    their access grant and the `ROLE_PERMISSIONS` catalog decides whether that
    role carries the permission (via `AuthzService.has_permission`, which also
    short-circuits platform admins and effective org admins).

    Flat routes (no `org_id`/`site_uuid` in the path) resolve the caller's own
    organization; path routes (`/orgs/{org_id}/…`, `/sites/{site_uuid}/…`)
    check against the path context and return a minimal context, since such
    routes gate via `dependencies=[...]` and read their path params directly.
    """

    def __init__(self, permission: Permission | str):
        self.permission = permission

    async def __call__(
        self,
        org_id: Optional[int] = None,
        site_uuid: Optional[uuid.UUID] = None,
        db: AsyncSession = Depends(get_async_db),
        user: User = Depends(get_current_user),
    ) -> OrgContext:
        if org_id is None and site_uuid is None:
            ctx = await get_org_context(db=db, user=user)
            await AuthzService.require_permission(
                db, user=user, permission=self.permission, org_id=ctx.org_id
            )
            return ctx

        # Platform admins hold no org grant, so the org-resolution and org-match
        # gates below would 403 them. `require_permission` bypasses every check
        # for them anyway, so short-circuit here.
        if bool(getattr(user, "is_platform_admin", False)):
            await AuthzService.require_permission(
                db,
                user=user,
                permission=self.permission,
                org_id=org_id,
                site_uuid=site_uuid,
            )
            return OrgContext(
                user=user, org_id=org_id, role=OrgRole.ADMIN.value, is_admin=True
            )

        resolved = await AuthzService.resolve_user_org(db, user=user)
        user_org_id, user_role = resolved if resolved else (None, "")
        if user_org_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not belong to any organization.",
            )

        # An org-scoped path route must be acted on within the caller's own org.
        # Site-scoped routes carry no `org_id` — their org is resolved from the
        # site inside `require_permission` — so they skip this match.
        if org_id is not None and user_org_id != org_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this organization.",
            )

        await AuthzService.require_permission(
            db, user=user, permission=self.permission, org_id=org_id, site_uuid=site_uuid
        )
        return OrgContext(
            user=user,
            org_id=org_id if org_id is not None else user_org_id,
            role=user_role,
            is_admin=False,
        )
