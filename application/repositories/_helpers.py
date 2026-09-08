"""Shared persistence helpers for the repository layer.

These were previously copy-pasted (byte-for-byte) into each repository module.
Keeping a single definition avoids the copies drifting apart over time.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


def as_uuid(value: Any) -> Optional[uuid.UUID]:
    """Coerce a value to ``uuid.UUID`` (None stays None)."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def require_uuid(value: Any, name: str) -> uuid.UUID:
    """Coerce to ``uuid.UUID``, raising a caller-friendly ValueError."""
    try:
        parsed = as_uuid(value)
    except Exception as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc
    if parsed is None:
        raise ValueError(f"Invalid {name}: {value}")
    return parsed


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


def model_patch(model: Any, *, drop_none: bool = False) -> Dict[str, Any]:
    """Return fields explicitly supplied to a Pydantic update model."""
    values = model.model_dump(exclude_unset=True)
    if drop_none:
        values = {key: value for key, value in values.items() if value is not None}
    return values
