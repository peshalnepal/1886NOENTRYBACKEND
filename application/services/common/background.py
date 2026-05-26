"""Background-task helpers.

Previously each service spawned ``asyncio.create_task`` directly (or had its
own private ``_fire_and_forget``). A bare ``create_task`` whose result is not
referenced anywhere can be garbage-collected mid-flight, so tasks are tracked
in a set until they finish.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Optional, Set

logger = logging.getLogger(__name__)

# Module-level registry so tasks spawned via ``fire_and_forget`` keep a strong
# reference until completion.
_BACKGROUND_TASKS: Set[asyncio.Task] = set()


def fire_and_forget(coro: Awaitable, *, name: Optional[str] = None) -> asyncio.Task:
    """Run ``coro`` detached, logging (never raising) any exception."""

    async def _runner() -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background task failed name=%s", name)

    task = asyncio.create_task(_runner(), name=name)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


class BackgroundTasks:
    """A per-owner task registry with cooperative shutdown.

    Use this when a service needs to cancel/await its own background work on
    shutdown instead of leaking it into the module-level registry.
    """

    def __init__(self) -> None:
        self._tasks: Set[asyncio.Task] = set()

    def spawn(self, coro: Awaitable, *, name: Optional[str] = None) -> asyncio.Task:
        async def _runner() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Background task failed name=%s", name)

        task = asyncio.create_task(_runner(), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Background task failed during shutdown")
