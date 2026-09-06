"""JWT access tokens (HS256)."""

import logging
import os
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"

logger = logging.getLogger(__name__)


def create_access_token(
    data: dict, expires_delta: timedelta = timedelta(hours=24)
) -> str:
    payload = {**data, "exp": datetime.now(timezone.utc) + expires_delta}
    try:
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    except Exception:
        logger.exception("Error creating access token")
        raise


def decode_access_token(token: str) -> dict:
    """Decode and validate a token, raising ValueError when it is unusable."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
