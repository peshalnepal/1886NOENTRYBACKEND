# env_utils.py  (Python 3.6)
"""Environment parsing shared by discovery.py and service.py.

Every getter is total: a malformed value falls back to the default rather than
raising, because a typo in .env must not stop the Jetson from booting.
"""

import os


def env_str(name, default=""):
    v = os.getenv(name)
    return default if v is None else str(v).strip()


def env_int(name, default, minimum=None, maximum=None):
    try:
        v = int(env_str(name, str(default)))
    except Exception:
        v = int(default)
    if minimum is not None:
        v = max(minimum, v)
    if maximum is not None:
        v = min(maximum, v)
    return v


def env_float(name, default, minimum=None):
    try:
        v = float(env_str(name, str(default)))
    except Exception:
        v = float(default)
    if minimum is not None:
        v = max(minimum, v)
    return v


def env_bool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")
