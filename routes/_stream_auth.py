"""Bearer-token resolution for SSE/stream endpoints.

EventSource cannot set an Authorization header, so these endpoints also accept
the token as an `access_token` query parameter. Shared by camera and
notification stream routes.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from application.services.user_snapshot_cache import (
    CachedUserSnapshot,
    UserSnapshotCache,
    UserSnapshotLookupError,
)
from core.security.tokens import decode_access_token

DB_UNAVAILABLE = "Database not available"


def extract_bearer_token(auth_header: str, access_token: Optional[str]) -> str:
    token = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        token = str(access_token or "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return token


async def resolve_stream_user(
    *,
    auth_header: str,
    access_token: Optional[str],
    cache: UserSnapshotCache,
    session_factory: async_sessionmaker[AsyncSession],
) -> CachedUserSnapshot:
    token = extract_bearer_token(auth_header, access_token)

    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    raw_user_id = payload.get("user_id") or payload.get("sub")
    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    try:
        user = await cache.get(session_factory=session_factory, user_id=user_id)
    except UserSnapshotLookupError:
        raise HTTPException(status_code=503, detail=DB_UNAVAILABLE)

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user
