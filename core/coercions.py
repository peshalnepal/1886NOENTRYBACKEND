"""Shared value-coercion helpers.

Consolidates small normalizers that were duplicated across routes,
repositories, and services. Follows the same pattern as `core/env.py`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

TRIGGER_MODES = ("inherit", "roi_enter", "any_detection")
PLAYBACK_MODES = ("inherit", "always", "never")


def is_blank(value: Optional[str]) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def gen_code(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:6]}"


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def coerce_trigger_mode(raw: Any) -> str:
    """Normalize a per-camera notification trigger to a valid tri-state."""
    value = str(raw or "").strip().lower()
    return value if value in TRIGGER_MODES else "inherit"


def coerce_playback_mode(raw: Any) -> str:
    """Normalize a per-camera playback override to a valid tri-state.

    Legacy rows stored a bool here, so True/False still map to always/never.
    """
    if raw is True:
        return "always"
    if raw is False:
        return "never"
    value = str(raw or "").strip().lower()
    return value if value in PLAYBACK_MODES else "inherit"
