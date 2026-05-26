import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.user_repository import UserRepository


@dataclass(frozen=True)
class CachedUserSnapshot:
    id: int
    user_name: str
    email: str
    email_verified: bool


SessionFactory = Callable[[], AsyncSession]


class UserSnapshotLookupError(RuntimeError):
    """Raised when the cache cannot load a user snapshot due to lookup failure."""


def _configured_worker_count() -> int:
    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"):
        raw = str(os.getenv(name) or "").strip()
        if not raw:
            continue
        try:
            return max(1, int(raw))
        except ValueError:
            continue

    gunicorn_args = str(os.getenv("GUNICORN_CMD_ARGS") or "")
    match = re.search(r"(?:^|\s)(?:-w|--workers)(?:\s+|=)(\d+)", gunicorn_args)
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return 1

    return 1


def _default_cache_ttl_s() -> float:
    return 0.0 if _configured_worker_count() > 1 else 60.0


def _default_miss_ttl_s() -> float:
    return 0.0 if _configured_worker_count() > 1 else 5.0


class UserSnapshotCache:
    def __init__(
        self,
        *,
        ttl_s: float = _default_cache_ttl_s(),
        miss_ttl_s: float = _default_miss_ttl_s(),
        max_concurrent_db_lookups: int = 8,
    ) -> None:
        self._ttl_s = float(ttl_s)
        self._miss_ttl_s = float(miss_ttl_s)
        self._cache: Dict[int, Tuple[float, Optional[CachedUserSnapshot]]] = {}
        self._lock = asyncio.Lock()
        self._generation: Dict[int, int] = {}
        self._inflight: Dict[int, Tuple[int, asyncio.Future]] = {}
        self._lookup_limit = asyncio.Semaphore(max(1, int(max_concurrent_db_lookups)))

    def invalidate(self, user_id: int) -> None:
        uid = int(user_id)
        self._generation[uid] = self._generation.get(uid, 0) + 1
        self._cache.pop(uid, None)

    async def get(
        self,
        *,
        session_factory: SessionFactory,
        user_id: int,
    ) -> Optional[CachedUserSnapshot]:
        uid = int(user_id)
        now = time.monotonic()
        leader = False
        pending: Optional[asyncio.Future] = None
        generation = 0

        async with self._lock:
            generation = self._generation.get(uid, 0)
            hit = self._cache.get(uid)
            if hit and hit[0] > now:
                return hit[1]

            inflight = self._inflight.get(uid)
            if inflight is not None and inflight[0] == generation:
                pending = inflight[1]
            else:
                pending = asyncio.get_running_loop().create_future()
                self._inflight[uid] = (generation, pending)
                leader = True

        if not leader and pending is not None:
            result = await pending
            if isinstance(result, BaseException):
                raise result
            return result

        snapshot: Optional[CachedUserSnapshot] = None
        cancelled = False
        lookup_error: Optional[BaseException] = None

        try:
            async with self._lookup_limit:
                async with session_factory() as db:
                    row = await UserRepository().get_by_id(db, uid)

                if row is not None:
                    snapshot = CachedUserSnapshot(
                        id=int(row.id),
                        user_name=str(row.user_name or ""),
                        email=str(row.email or ""),
                        email_verified=bool(row.email_verified),
                    )
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            lookup_error = UserSnapshotLookupError(str(exc) or "Failed to load user snapshot")
        finally:
            async with self._lock:
                current_generation = self._generation.get(uid, 0)
                if not cancelled and lookup_error is None and current_generation == generation:
                    ttl_s = self._ttl_s if snapshot is not None else self._miss_ttl_s
                    if ttl_s > 0:
                        self._cache[uid] = (time.monotonic() + ttl_s, snapshot)
                    else:
                        self._cache.pop(uid, None)

                current_inflight = self._inflight.get(uid)
                if (
                    current_inflight is not None
                    and current_inflight[1] is pending
                ):
                    self._inflight.pop(uid, None)

                future = pending
                if future is not None and not future.done():
                    if cancelled:
                        future.cancel()
                    elif lookup_error is not None:
                        future.set_result(lookup_error)
                    else:
                        future.set_result(snapshot)

        if lookup_error is not None:
            raise lookup_error

        return snapshot
