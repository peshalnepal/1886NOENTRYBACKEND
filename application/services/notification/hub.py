"""WebSocket-like in-process notification hub.

Extracted verbatim from the former monolithic application/services/notification.py.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Set

from application.services.notification.types import NotificationMessage


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
        uid = int(msg.user_id)
        async with self._lock:
            subs = list(self._subs_by_user.get(uid, set()))

        for q in subs:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(msg)
                except Exception:
                    pass
