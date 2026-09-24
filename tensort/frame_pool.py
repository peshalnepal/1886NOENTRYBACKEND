"""Bounded, fair buffering between camera capture and GPU dispatch."""

import asyncio
import collections
import time


class FramePool(object):
    """Keep recent frames in a FIFO queue per camera.

    Drop expired frames first. At capacity, evict the oldest frame from the
    camera with the largest backlog, breaking ties by oldest timestamp.
    Batch selection gives each ready camera a turn before taking a second frame.
    All access happens on the asyncio thread, so no locks are needed.
    """

    def __init__(self, capacity=30, max_age_s=0.7):
        self._frames = {}                 # camera_key -> deque[RTSPEvent]
        self._total = 0
        self._capacity = max(1, int(capacity))
        self._max_age_s = max(0.0, float(max_age_s))
        self._data_evt = asyncio.Event()
        self.evicted_total = 0
        self.expired_total = 0

    # -- internals ----------------------------------------------------------

    def _drop_expired(self):
        """Drop frames older than max_age_s. Deques are FIFO so the head is the
        oldest — stop scanning a camera as soon as its head is fresh."""
        if self._max_age_s <= 0.0 or self._total == 0:
            return
        cutoff_ms = (time.time() - self._max_age_s) * 1000.0
        for dq in self._frames.values():
            while dq and float(dq[0].ts_ms) < cutoff_ms:
                dq.popleft()
                self._total -= 1
                self.expired_total += 1

    def _evict_one(self):
        """Free exactly one slot using the fairness rule above."""
        victim_key = None
        victim_len = 0
        victim_ts = 0.0
        for key, dq in self._frames.items():
            if not dq:
                continue
            head_ts = float(dq[0].ts_ms)
            # Most frames wins; tie-break on the oldest head so eviction is
            # deterministic rather than dict-iteration-order dependent.
            if victim_key is None or len(dq) > victim_len or (len(dq) == victim_len and head_ts < victim_ts):
                victim_key = key
                victim_len = len(dq)
                victim_ts = head_ts

        if victim_key is None:
            return

        # When every camera holds exactly one frame the scan above degenerates
        # to "oldest head wins", which is exactly the desired fallback: nobody
        # is over-represented, so the globally oldest frame is the one to drop.
        self._frames[victim_key].popleft()
        self._total -= 1
        self.evicted_total += 1

    # -- producer side ------------------------------------------------------

    def put(self, ev):
        """Add one frame. Called from the event loop (capture thread -> loop)."""
        self._drop_expired()
        if self._total >= self._capacity:
            self._evict_one()

        key = str(ev.camera_uuid)
        dq = self._frames.get(key)
        if dq is None:
            # A newly added camera starts contributing immediately — no
            # scheduler registration, no rebalancing step.
            dq = collections.deque()
            self._frames[key] = dq
        dq.append(ev)
        self._total += 1
        self._data_evt.set()

    def discard_camera(self, camera_key):
        dq = self._frames.pop(str(camera_key), None)
        if dq:
            self._total -= len(dq)

    # -- consumer side ------------------------------------------------------

    async def get_batch(self, max_n, linger_s=0.0):
        """
        Draw up to `max_n` frames, round-robin across cameras, oldest first.

        Never waits for all cameras: it returns whatever is pooled. The linger
        is a small top-up window used ONLY when the pool is underfilled (idle or
        cold start); in steady state the pool has already accumulated frames
        while the GPU was busy with the previous batch, so batches self-fill.
        """
        max_n = max(1, int(max_n))

        while True:
            self._drop_expired()
            if self._total > 0:
                break
            self._data_evt.clear()
            await self._data_evt.wait()

        if self._total < max_n and linger_s > 0.0:
            await asyncio.sleep(linger_s)
            self._drop_expired()

        # Round-robin: every camera with pending frames contributes one frame
        # before any camera contributes a second. This is what makes per-camera
        # detection FPS equal under load instead of first-come-first-served.
        evs = []
        while len(evs) < max_n:
            ready = [k for k, dq in self._frames.items() if dq]
            if not ready:
                break
            ready.sort(key=lambda k: self._frames[k][0].ts_ms)
            for key in ready:
                if len(evs) >= max_n:
                    break
                evs.append(self._frames[key].popleft())
                self._total -= 1

        # Drop entries for cameras that are now empty, so a long-running service
        # that cycles through many cameras does not keep scanning dead keys.
        if self._total == 0:
            self._frames.clear()

        return evs

    def depth(self):
        return self._total
