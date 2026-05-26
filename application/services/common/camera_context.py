"""Shared camera-context resolver.

Both ``ModelPipeline`` and ``NotificationService`` needed the same thing:
resolve user/site/device/camera names for a camera_uuid, with a TTL cache,
single-flight de-duplication of concurrent lookups, and a concurrency cap on
DB sessions. That logic lived twice; it now lives here once.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional, Tuple

from application.repositories.notification_repository import (
    CameraContext,
    NotificationRepository,
)

logger = logging.getLogger(__name__)


class CameraContextResolver:
    """TTL-cached, single-flight resolver for :class:`CameraContext`."""

    def __init__(
        self,
        *,
        repo: Optional[NotificationRepository] = None,
        hit_ttl_s: float = 60.0,
        miss_ttl_s: float = 5.0,
        max_concurrent_lookups: int = 4,
    ) -> None:
        self._repo = repo or NotificationRepository()
        self._hit_ttl_s = float(hit_ttl_s)
        self._miss_ttl_s = max(0.0, float(miss_ttl_s))
        self._session_factory = None

        # key -> (expires_at_monotonic, ctx)
        self._cache: Dict[str, Tuple[float, Optional[CameraContext]]] = {}
        self._cache_lock = asyncio.Lock()
        self._inflight: Dict[str, asyncio.Future] = {}
        self._lookup_limit = asyncio.Semaphore(max(1, int(max_concurrent_lookups)))

    def set_session_factory(self, session_factory) -> None:
        self._session_factory = session_factory

    def invalidate(self, camera_uuid: str) -> None:
        """Drop any cached / in-flight entry for a camera (e.g. after edit/delete)."""
        key = str(camera_uuid)
        self._cache.pop(key, None)
        future = self._inflight.pop(key, None)
        if future is not None and not future.done():
            future.cancel()

    async def resolve(self, camera_uuid: str) -> Optional[CameraContext]:
        sf = self._session_factory
        if sf is None:
            return None

        key = str(camera_uuid)
        now = time.monotonic()
        leader = False
        pending: Optional[asyncio.Future] = None

        async with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
            pending = self._inflight.get(key)
            if pending is None:
                pending = asyncio.get_running_loop().create_future()
                self._inflight[key] = pending
                leader = True

        try:
            cam_uuid = uuid.UUID(key)
        except Exception:
            if leader:
                async with self._cache_lock:
                    future = self._inflight.pop(key, None)
                    if future is not None and not future.done():
                        future.set_result(None)
            return None

        if not leader and pending is not None:
            return await pending

        ctx: Optional[CameraContext] = None
        cancelled = False
        try:
            async with self._lookup_limit:
                async with sf() as db:
                    ctx = await self._repo.get_camera_context(db, camera_uuid=cam_uuid)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("Failed to load CameraContext camera=%s", key)
            ctx = None
        finally:
            async with self._cache_lock:
                if not cancelled:
                    ttl_s = self._hit_ttl_s if ctx else self._miss_ttl_s
                    if ttl_s > 0.0:
                        self._cache[key] = (now + ttl_s, ctx)
                    else:
                        self._cache.pop(key, None)
                future = self._inflight.pop(key, None)
                if future is not None and not future.done():
                    if cancelled:
                        future.cancel()
                    else:
                        future.set_result(ctx)

        return ctx
