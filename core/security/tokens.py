import logging
import os
import traceback
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"

logger = logging.getLogger(__name__)


def create_access_token(
    data: dict, expires_delta: timedelta = timedelta(hours=24)
) -> str:
    """Creates a JWT access token."""
    try:
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + expires_delta
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def decode_access_token(token: str) -> dict:
    """Decodes and validates a JWT access token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")
