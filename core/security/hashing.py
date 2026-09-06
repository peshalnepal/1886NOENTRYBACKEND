"""Password hashing (bcrypt) and API-key helpers."""

import hashlib
import secrets

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def generate_random_key(length: int = 32) -> str:
    """A URL-safe random string, used as an API key."""
    return secrets.token_urlsafe(length)


def hash_key(key: str) -> str:
    """Deterministic SHA-256 of an API key, for lookup by hash."""
    return hashlib.sha256(key.encode()).hexdigest()
