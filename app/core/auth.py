"""
Real backend authentication - replaces the old frontend-only localStorage fake login.

- Passwords hashed with bcrypt (never stored or compared in plain text).
- Sessions are stateless JWTs signed with settings.secret_key, sent as
  `Authorization: Bearer <token>` and verified on every protected request.
- get_current_user_id is a FastAPI dependency: any endpoint that takes it as a
  parameter automatically requires a valid token, and gets back the caller's user_id.
"""
import bcrypt
import jwt
import time
from typing import Optional
from fastapi import Header, HTTPException

from app.config import settings

TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days
JWT_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_token(user_id: str) -> str:
    payload = {"sub": user_id, "iat": int(time.time()), "exp": int(time.time()) + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, settings.secret_key, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    """Returns the user_id encoded in a valid, non-expired token, or None."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None


def get_current_user_id(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency - require a valid Bearer token, return the user_id.
    Use as: def endpoint(user_id: str = Depends(get_current_user_id))"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated - missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id


def get_current_user_id_optional(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Same as get_current_user_id but returns None instead of raising - for endpoints
    that work for anonymous visitors too (e.g. public marketplace listings) but can
    personalize behavior when a valid token IS present."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return decode_token(authorization.removeprefix("Bearer ").strip())
