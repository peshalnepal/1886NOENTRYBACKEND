"""Shared route background-task helpers.

Consolidates `_spawn_bg_task` and `_delete_blobs_background` previously
duplicated across routes/user_routes.py, routes/site_routes.py and
routes/camera_routes.py. The merged versions use the most informative
logging (SUCCESS/CANCELLED on bg tasks; batch progress on blob cleanup).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List

logger = logging.getLogger(__name__)


def _spawn_bg_task(coro, *, name: str) -> None:
    """Spawn a background task with completion logging."""
    task = asyncio.create_task(coro, name=name)

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            done_task.result()
            logger.info(f"[Background Task] {name}: SUCCESS")
        except asyncio.CancelledError:
            logger.info(f"[Background Task] {name}: CANCELLED")
        except Exception as e:
            logger.exception(f"[Background Task] {name}: FAILED with error: {e}")

    task.add_done_callback(_on_done)


async def _delete_blobs_background(keys: List[str], *, service_cls: type, label: str) -> None:
    """Delete blobs in parallel batches of 10."""
    unique = list(dict.fromkeys(k for k in keys if k))
    if not unique:
        return

    logger.info(f"[Blob Cleanup] Starting deletion of {len(unique)} {label} blobs (parallel, batch size=10)")
    svc = service_cls()
    deleted = 0
    failed = 0
    batch_size = 10

    try:
        for i in range(0, len(unique), batch_size):
            batch = unique[i: i + batch_size]
            results = await asyncio.gather(
                *[svc.delete_blob(blob_name=k) for k in batch],
                return_exceptions=True,
            )
            for key, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.warning(f"[Blob Cleanup] Failed to delete {label} blob {key}: {result}")
                else:
                    deleted += 1
            logger.info(
                f"[Blob Cleanup] Batch {i // batch_size + 1}: "
                f"deleted {sum(1 for r in results if not isinstance(r, Exception))}/{len(batch)} {label} blobs"
            )
    finally:
        try:
            await svc.close()
        except Exception:
            pass
    logger.info(f"[Blob Cleanup] COMPLETE: deleted {deleted} {label} blobs, {failed} failed")
