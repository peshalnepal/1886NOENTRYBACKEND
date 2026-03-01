import secrets
import hashlib
import hmac
import os 
OTP_TTL_SECONDS = os.getenv("OTP_TTL_SECONDS",600)

OTP_MAX_ATTEMPTS = os.getenv("OTP_MAX_ATTEMPTS",5)

def generate_otp(length: int = 6) -> str:
    # numeric OTP
    return "".join(str(secrets.randbelow(10)) for _ in range(length))

def hash_otp(email: str, otp: str, secret_pepper: str) -> str:
    """
    Bind OTP to email + server secret so it can't be reused across emails.
    Store this hash in DB.
    """
    msg = f"{email}|{otp}".encode("utf-8")
    key = secret_pepper.encode("utf-8")
    digest = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return digest

def verify_otp_hash(email: str, otp: str, secret_pepper: str, stored_hash: str) -> bool:
    candidate = hash_otp(email, otp, secret_pepper)
    return hmac.compare_digest(candidate, stored_hash)