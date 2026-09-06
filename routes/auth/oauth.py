"""One-time-password generation and verification for the auth flows.

An OTP is never stored in plaintext: it is HMAC'd with the server pepper and
the recipient's email, so a code minted for one address cannot be replayed
against another. TTL and attempt limits live with the callers that enforce
them (`routes/auth/signup.py`, `application/repositories/verify_repository.py`).
"""

import hashlib
import hmac
import secrets


def generate_otp(length: int = 6) -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(length))


def hash_otp(email: str, otp: str, secret_pepper: str) -> str:
    """Bind an OTP to (email, server secret) and return the hash to store."""
    return hmac.new(
        secret_pepper.encode("utf-8"),
        f"{email}|{otp}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_otp_hash(email: str, otp: str, secret_pepper: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_otp(email, otp, secret_pepper), stored_hash)
