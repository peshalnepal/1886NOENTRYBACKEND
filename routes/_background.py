"""Shared background-task helpers for the route layer.

Both helpers are used by the camera, site and user delete paths, which spawn
work that must outlive the HTTP response.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)

# Blob deletes run in parallel batches of this size against Azure Storage.
_BLOB_DELETE_BATCH = 10


def _spawn_bg_task(coro, *, name: str) -> None:
    """Run `coro` detached, logging how it finished.

    The done-callback also keeps a reference alive until completion — asyncio
    only holds a weak one, so a bare `create_task` can be garbage-collected
    mid-flight.
    """
    task = asyncio.create_task(coro, name=name)

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
            logger.info("[Background Task] %s: SUCCESS", name)
        except asyncio.CancelledError:
            logger.info("[Background Task] %s: CANCELLED", name)
        except Exception:
            logger.exception("[Background Task] %s: FAILED", name)

    task.add_done_callback(_on_done)


async def _delete_blobs_background(
    keys: List[str], *, service_cls: type, label: str
) -> None:
    """Delete blobs in parallel batches, best effort.

    A single failure is logged and skipped: these blobs are already orphaned by
    the delete that scheduled this, so leaving one behind costs storage, never
    correctness.
    """
    unique = list(dict.fromkeys(k for k in keys if k))
    if not unique:
        return

    logger.info("[Blob Cleanup] Deleting %s %s blobs", len(unique), label)
    svc = service_cls()
    deleted = failed = 0

    try:
        for i in range(0, len(unique), _BLOB_DELETE_BATCH):
            batch = unique[i : i + _BLOB_DELETE_BATCH]
            results = await asyncio.gather(
                *(svc.delete_blob(blob_name=k) for k in batch),
                return_exceptions=True,
            )
            for key, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.warning(
                        "[Blob Cleanup] Failed to delete %s blob %s: %s",
                        label,
                        key,
                        result,
                    )
                else:
                    deleted += 1
    finally:
        try:
            await svc.close()
        except Exception:
            logger.debug("[Blob Cleanup] Service close failed", exc_info=True)

    logger.info(
        "[Blob Cleanup] COMPLETE: deleted %s %s blobs, %s failed", deleted, label, failed
    )
