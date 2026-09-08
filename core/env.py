"""Shared environment-variable parsing helpers.

Consolidates `_env_bool` / `_env_int` / `_env_float` previously duplicated
across main.py, routes/auth/signup.py, application/services/edgeinference.py,
application/services/notification/service.py, application/channels/channel.py,
application/repositories/verify_repository.py, and core/database.py.
"""

from __future__ import annotations

import os
from typing import Optional


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return value if minimum is None else max(minimum, value)


def env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return value if minimum is None else max(minimum, value)
