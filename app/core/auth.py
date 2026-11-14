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
    now = time.time()
    # iat_ms: second-resolution `iat` cannot tell "issued before the reset" from "issued right after it" when both
    # happen in the same second (see GenesisDB.invalidate_sessions).
    payload = {"sub": user_id, "iat": int(now), "iat_ms": int(now * 1000), "exp": int(now) + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, settings.secret_key, algorithm=JWT_ALGORITHM)


def _issued_ms(payload: dict) -> int:
    """Issue time in ms; tokens from before iat_ms existed fall back to their `iat` seconds."""
    return int(payload.get("iat_ms") or int(payload.get("iat") or 0) * 1000)


def _session_revoked(user_id: str, issued_ms: int) -> bool:
    """Отсутствующий или отключённый владелец не может продолжать сессию."""
    try:
        from app.db.database import GenesisDB
        state = GenesisDB().get_session_state(user_id)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Authentication storage is unavailable") from error
    if state is None:
        return True
    valid_after, disabled = state
    return bool(disabled) or bool(valid_after and issued_ms < valid_after)


def decode_token(token: str) -> Optional[str]:
    """Returns the user_id encoded in a valid, non-expired, non-revoked token, or None."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    user_id = payload.get("sub")
    if user_id and _session_revoked(user_id, _issued_ms(payload)):
        return None
    return user_id


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


def require_admin(authorization: Optional[str] = Header(None)) -> str:
    """Dependency for /api/admin/*: a valid session AND a verified owner e-mail (GenesisDB.is_admin). Strict: a failing
    session lookup is an error here, not a pass."""
    user_id = get_current_user_id(authorization)
    from app.db.database import GenesisDB
    db = GenesisDB()
    payload = jwt.decode(authorization.removeprefix("Bearer ").strip(), settings.secret_key, algorithms=[JWT_ALGORITHM])
    state = db.get_session_state(user_id)
    if state is None or state[1] or (state[0] and _issued_ms(payload) < state[0]):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not db.is_admin(user_id):
        raise HTTPException(status_code=403, detail="Admin only")
    return user_id


def get_current_user_id_optional(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Same as get_current_user_id but returns None instead of raising - for endpoints
    that work for anonymous visitors too (e.g. public marketplace listings) but can
    personalize behavior when a valid token IS present.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return decode_token(authorization.removeprefix("Bearer ").strip())


def get_api_key_from_request(authorization: Optional[str] = Header(None),
                              x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """FastAPI dependency for the public developer API (POST /api/v1/chat/completions):
    require a Genesis API key, return the raw key string (the caller still needs to look it
    up via GenesisDB.validate_api_key to get the owning user_id and confirm it's active).

    Accepts the key two ways, checked in this order:
      1. `Authorization: Bearer gen-xxxx` - matches the convention the official OpenAI and
         Groq Python SDKs use for their own API keys, so a developer can point either SDK's
         `base_url` at this platform and their existing `Authorization: Bearer <key>` client
         code just works unchanged, with a Genesis key in place of an OpenAI/Groq one.
      2. `X-API-Key: gen-xxxx` - same header the older /api/models/{id}/predict endpoint
         uses, for consistency with the rest of this platform's API-key surface.

    NOTE this is intentionally a DIFFERENT dependency from get_current_user_id: that one
    reads `Authorization: Bearer <JWT>` for browser session logins, and a Genesis API key
    (`gen-...`) is never a valid JWT, so there's no ambiguity between the two even though
    they read the same header - a route takes one dependency or the other, never both.
    """
    if authorization and authorization.startswith("Bearer "):
        key = authorization.removeprefix("Bearer ").strip()
        if key:
            return key
    if x_api_key:
        return x_api_key
    raise HTTPException(status_code=401, detail="Missing API key - send it as 'Authorization: Bearer "
                                                  "<your key>' or the X-API-Key header")
