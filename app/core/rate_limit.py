"""
Request-volume rate limiting - distinct from app.core.quota, which caps *Groq token spend*
per account per month. This module caps raw *request count* per client per short window,
which quota does nothing to prevent: someone hammering /api/auth/login to brute-force a
password, or scripting /api/agents/search in a tight loop, never touches Groq at all, so the
token quota never sees it and never blocks it.

Implementation is a plain in-memory sliding window, deliberately simple:
  - No new dependency - works with what's already installed.
  - Correct for a single-process deployment (the common case for this project today).
  - NOT correct across multiple processes/workers or multiple machines - each process has
    its own counters, so real capacity under N workers is N times these numbers. When this
    app is actually scaled horizontally, swap the in-memory store below for Redis (the
    settings.redis_url field already exists in app/config.py for exactly this - it is
    currently unused) so all workers share one counter. Don't mistake this module for that
    upgrade; it buys real protection today and documents its own ceiling.
"""
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class _SlidingWindowLimiter:
    def __init__(self):
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
        """Returns (allowed, seconds_until_next_slot_frees_up)."""
        now = time.monotonic()
        q = self._hits[key]
        cutoff = now - window_seconds
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= limit:
            retry_after = int(window_seconds - (now - q[0])) + 1
            return False, max(retry_after, 1)
        q.append(now)
        return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Two tiers:
      - AUTH_LIMIT: tight, IP-keyed, applied to /api/auth/* - these are the brute-force /
        credential-stuffing / signup-spam targets, and IP is the only identity available
        before a request is authenticated.
      - DEFAULT_LIMIT: looser, applied to everything under /api/ - keyed by the caller's
        Bearer token when present (so one user's usage doesn't throttle another user
        behind the same NAT/proxy IP), falling back to IP for anonymous requests.
    Static files, docs, and non-API routes are never limited.
    """

    AUTH_PATHS = ("/api/auth/login", "/api/auth/register")
    AUTH_LIMIT, AUTH_WINDOW = 10, 60          # 10 attempts / minute / IP
    DEFAULT_LIMIT, DEFAULT_WINDOW = 120, 60   # 120 requests / minute / caller

    def __init__(self, app):
        super().__init__(app)
        self._limiter = _SlidingWindowLimiter()

    @staticmethod
    def _client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _caller_key(self, request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and len(auth) > 12:
            # Keyed on the token itself, not the decoded user_id - avoids importing/decoding
            # JWTs in the middleware layer (that stays the auth dependency's job) while still
            # giving each distinct caller their own bucket.
            return "tok:" + auth[7:39]
        return "ip:" + self._client_ip(request)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        if path in self.AUTH_PATHS:
            key = "auth:" + self._client_ip(request)
            allowed, retry_after = self._limiter.allow(key, self.AUTH_LIMIT, self.AUTH_WINDOW)
        else:
            key = self._caller_key(request)
            allowed, retry_after = self._limiter.allow(key, self.DEFAULT_LIMIT, self.DEFAULT_WINDOW)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests - slow down and try again shortly."},
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
