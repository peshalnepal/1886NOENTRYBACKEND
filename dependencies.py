import asyncio
import logging
from typing import Any, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from application.repositories.user_repository import UserRepository
from application.services.manager import Manager
from application.services.notification import NotificationService, WebNotificationHub
from application.services.retention import RetentionService
from application.services.user_snapshot_cache import UserSnapshotCache
from core.database import db_manager
from core.database_orm import User
from core.security.tokens import decode_access_token
import uuid as _uuid
from dataclasses import dataclass
from application.services.authz_service import AuthzService
from core.security.roles import OrgRole, Permission

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


def get_manager_optional(request: Request) -> Optional[Manager]:
    """Returns the Manager if available, None otherwise.  Never raises 503.
    Use this on routes where the DB operation must succeed even when the
    manager (pipeline / edge-device layer) has not fully started up."""
    return getattr(request.app.state, "manager", None)


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
        user = await UserRepository().get_by_id(db, user_id)

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

async def require_platform_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Allow only users flagged as platform admins."""
    AuthzService.ensure_platform_admin(user)
    return user

@dataclass
class OrgContext:
    """The organization + role a flat-route request operates in.

    Resolved from the authenticated user's single `OrgMembership` so the
    existing flat resource routes (`/sites`, `/cameras`, `/devices`) need
    no `org_id` in their path. `is_admin` already folds in the
    operator-as-admin elevation (see `AuthzService.is_effective_org_admin`).

    `org_id is None` is the platform-admin "super context": the caller has
    no single-tenant scope and the route should treat every org as in
    scope. Repos and helpers check for this sentinel and skip org
    filtering when they see it.
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

    Platform admins get a "super context" with `org_id=None`, which the
    flat routes interpret as "every org" so the same endpoints work as
    both per-tenant routes and god-mode views. A non-platform user with
    no organization membership still gets a 403.
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

    Declare the permission a route needs and nothing else::

        ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_CAMERAS))

    It both **gates** the request and **returns the `OrgContext`** the route
    body consumes (org_id / role / is_admin / user), collapsing the old
    `get_org_context` + `require_org_admin_ctx` pair into one. You pass only a
    permission — never a role; the caller's role is resolved from their
    `access_grant` and the `ROLE_PERMISSIONS` catalog answers whether that
    role carries the permission (via `AuthzService.has_permission`, which also
    short-circuits platform admins and effective org admins).

    Context resolution:
      * **Flat routes** (no `org_id`/`site_uuid` in the path): the caller's
        single organization is resolved via `get_org_context` (preserving the
        platform-admin super-context and the no-org -> 403 behaviour), and the
        permission is checked in that org.
      * **Path routes** (`/orgs/{org_id}/...`, `/sites/{site_uuid}/...`): the
        permission is checked against the path context. The returned context
        is minimal because such routes gate via ``dependencies=[...]`` and
        read their path params directly.
    """

    def __init__(self, permission: Permission | str):
        self.permission = permission

    async def __call__(
        self,
        org_id: Optional[int] = None,
        site_uuid: Optional[_uuid.UUID] = None,
        db: AsyncSession = Depends(get_async_db),
        user: User = Depends(get_current_user),
    ) -> OrgContext:
        if org_id is None and site_uuid is None:
            # Flat route: resolve the caller's single-org context, then check.
            ctx = await get_org_context(db=db, user=user)
            await AuthzService.require_permission(
                db, user=user, permission=self.permission, org_id=ctx.org_id
            )
            return ctx

        # Platform admins have no single-org grant; they operate in the
        # super context and `require_permission` bypasses every check, so
        # short-circuit before the org-resolution / org-match gates below
        # (which would otherwise 403 them for lacking an org grant).
        if bool(getattr(user, "is_platform_admin", False)):
            await AuthzService.require_permission(
                db, user=user, permission=self.permission, org_id=org_id, site_uuid=site_uuid
            )
            return OrgContext(
                user=user,
                org_id=org_id,
                role=OrgRole.ADMIN.value,
                is_admin=True,
            )

        resolved = await AuthzService.resolve_user_org(db, user=user)
        user_org_id, user_role = resolved if resolved else (None, "")
        if user_org_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not belong to any organization.",
            )
        # Org-scoped path routes (`/orgs/{org_id}/...`) must be acted on
        # within the caller's own organization. Site-scoped routes
        # (`/sites/{site_uuid}/...`) carry no `org_id`; their org is
        # resolved from the site inside `require_permission`, so skip the
        # explicit org match for them.
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