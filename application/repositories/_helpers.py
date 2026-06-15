"""Shared persistence helpers for the repository layer.

These were previously copy-pasted (byte-for-byte) into each repository module.
Keeping a single definition avoids the copies drifting apart over time.
"""
from __future__ import annotations

import uuid
from typing import Any, List, Optional


def as_uuid(value: Any) -> Optional[uuid.UUID]:
    """Coerce a value to ``uuid.UUID`` (None stays None)."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def normalize_uuid_list(values: Any) -> List[uuid.UUID]:
    """Parse an iterable of values into a deduped list of ``uuid.UUID``.

    Non-iterable input yields an empty list; entries that cannot be parsed are
    skipped. Order of first appearance is preserved.
    """
    if not isinstance(values, (list, tuple, set)):
        return []

    seen: set[str] = set()
    out: List[uuid.UUID] = []
    for item in values:
        try:
            parsed = as_uuid(item)
        except Exception:
            parsed = None
        if parsed is None:
            continue
        key = str(parsed)
        if key in seen:
            continue
        seen.add(key)
        out.append(parsed)
    return out
