import hashlib
import secrets

from passlib.context import CryptContext

# Configure hashing algorithm
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generates a secure hash for a password."""
    return pwd_context.hash(password)


def generate_random_key(length: int = 32) -> str:
    """Generates a secure, URL-safe string for use as an API key."""
    return secrets.token_urlsafe(length)


def hash_key(key: str) -> str:
    """
    Creates a deterministic SHA-256 hash of an API key.
    Used for looking up keys in the database.
    """
    return hashlib.sha256(key.encode()).hexdigest()
