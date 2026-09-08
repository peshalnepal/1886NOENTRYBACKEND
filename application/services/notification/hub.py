"""In-process notification hub: per-user queues fed by the flusher, drained by SSE."""

from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List, Set

from application.services.notification.types import NotificationMessage


def _put_latest(
    queue: asyncio.Queue[NotificationMessage], msg: NotificationMessage
) -> None:
    try:
        queue.put_nowait(msg)
        return
    except asyncio.QueueFull:
        pass

    try:
        queue.get_nowait()
        queue.put_nowait(msg)
    except Exception:
        pass


class WebNotificationHub:
    def __init__(self, max_q: int = 200):
        self._subs_by_user: Dict[int, Set[asyncio.Queue[NotificationMessage]]] = {}
        self._lock = asyncio.Lock()
        self._max_q = max_q

    async def subscribe(self, *, user_id: int) -> asyncio.Queue[NotificationMessage]:
        q: asyncio.Queue[NotificationMessage] = asyncio.Queue(maxsize=self._max_q)
        async with self._lock:
            uid = int(user_id)
            bucket = self._subs_by_user.get(uid)
            if bucket is None:
                bucket = set()
                self._subs_by_user[uid] = bucket
            bucket.add(q)
        return q

    async def unsubscribe(self, *, user_id: int, q: asyncio.Queue[NotificationMessage]) -> None:
        async with self._lock:
            uid = int(user_id)
            bucket = self._subs_by_user.get(uid)
            if not bucket:
                return
            bucket.discard(q)
            if not bucket:
                self._subs_by_user.pop(uid, None)

    async def publish(self, msg: NotificationMessage) -> None:
        """Deliver to the message's owner (`msg.user_id`)."""
        await self.publish_to_users([int(msg.user_id)], msg)

    async def publish_to_users(self, user_ids: Iterable[int], msg: NotificationMessage) -> None:
        """Deliver to an explicit set of users.

        Used to route operator-gated alerts to the org's operators (whose ids
        differ from `msg.user_id`) instead of the end user.
        """
        ids = {int(uid) for uid in (user_ids or [])}
        if not ids:
            return

        async with self._lock:
            subs: List[asyncio.Queue[NotificationMessage]] = []
            for uid in ids:
                subs.extend(self._subs_by_user.get(uid, set()))

        for q in subs:
            _put_latest(q, msg)
