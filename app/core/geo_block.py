"""
Geo-blocking - restricts access from a configured list of countries (default: Russia,
per product decision - many upstream providers this app depends on, including Stripe and
Groq, already restrict or complicate service for Russia-based accounts under sanctions, so
blocking at the app level avoids a user hitting a confusing failure three steps into signup).

Country detection, in priority order:
  1. Cloudflare's `CF-IPCountry` header - zero cost, zero latency, set at the edge by
     Cloudflare before the request ever reaches this server. This is the ONLY reliable path
     in production and is what you get for free just by putting the app behind Cloudflare's
     proxy (orange-cloud DNS). This is the recommended real-world setup.
  2. A fallback call to a free IP-geolocation API (ip-api.com, no key required) for when the
     app is NOT behind Cloudflare (e.g. local testing of this feature, or a deployment behind
     a different/no CDN). Results are cached in memory per IP for GEOBLOCK_CACHE_TTL seconds
     to avoid hammering the external API on every request from the same visitor. This
     fallback has a real request-count limit (45/min on ip-api.com's free tier) and a real
     network dependency - it is NOT a substitute for #1 at real traffic volume.

Fail-open policy: if country cannot be determined (private/loopback IP, external API
unreachable, rate-limited, malformed response), the request is ALLOWED, never blocked. A
false negative (a Russian visitor slips through because lookup failed) is an acceptable
product trade-off; a false positive (blocking a legitimate visitor because our geolocation
dependency had a bad moment) is not - that would deny service to someone we have no actual
reason to deny.

Private/loopback/link-local IPs (127.0.0.1, 192.168.x.x, 10.x.x.x, etc.) always skip the
check entirely - this is what makes local development and testing work without any config.
"""
import ipaddress
import json
import time
import urllib.request
import urllib.error
from typing import Dict, Optional, Set, Tuple
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse

from app.config import settings

_CACHE_TTL_SECONDS = 3600  # how long a fallback-API country lookup is trusted per IP
_geo_cache: Dict[str, Tuple[str, float]] = {}  # ip -> (country_code, expires_at_monotonic)

# Routes that must always be reachable regardless of geography - health checks for
# monitoring/uptime services, which themselves often run from arbitrary cloud regions and
# would otherwise flap the app's health status based on where the monitor happens to be.
_ALWAYS_ALLOWED_PATHS = {"/health"}

_BLOCKED_PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Genesis AI - Not available in your region</title>
<style>
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
       background:#141416;color:#f2f2f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       padding:24px;text-align:center}
  .card{max-width:440px}
  h1{font-size:22px;margin:0 0 12px}
  p{font-size:15px;line-height:1.6;color:#a8a8ac;margin:0 0 8px}
  .icon{font-size:40px;margin-bottom:16px}
</style></head>
<body><div class="card">
  <div class="icon">🌐</div>
  <h1>Genesis AI isn't available in your region yet</h1>
  <p>We're currently unable to offer service to visitors connecting from this location.</p>
  <p>If you're using a VPN or proxy, try disabling it and reloading. If you're not, connecting
     through a VPN set to a supported region may allow access.</p>
</div></body></html>"""


def _is_private_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return True  # unparseable "IP" (e.g. a unix socket path in some deployments) - don't block on it


def _client_ip(request: Request) -> Optional[str]:
    # Prefer a CDN/proxy-set header (first hop = original client) over the raw socket peer,
    # which behind any reverse proxy or CDN would just be the proxy's own IP, not the
    # visitor's - see also app/core/rate_limit.py, which has this same concern for a
    # different reason (rate-limiting the proxy instead of the real client).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _lookup_country_via_fallback_api(ip: str) -> Optional[str]:
    cached = _geo_cache.get(ip)
    if cached and cached[1] > time.monotonic():
        return cached[0]
    try:
        req = urllib.request.Request(
            f"http://ip-api.com/json/{ip}?fields=countryCode",
            headers={"User-Agent": "GenesisAI/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            country = json.loads(resp.read().decode("utf-8")).get("countryCode")
        if country:
            _geo_cache[ip] = (country, time.monotonic() + _CACHE_TTL_SECONDS)
        return country
    except Exception:
        # Network error, timeout, rate limit, malformed response - all fail open (see module
        # docstring). Deliberately not logged at error level: this is expected to happen
        # occasionally under normal operation and should not page anyone.
        return None


def _blocked_country_codes() -> Set[str]:
    return {c.strip().upper() for c in settings.geoblock_countries.split(",") if c.strip()}


class GeoBlockMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.geoblock_enabled or request.url.path in _ALWAYS_ALLOWED_PATHS:
            return await call_next(request)

        ip = _client_ip(request)
        if not ip or _is_private_ip(ip):
            return await call_next(request)

        country = request.headers.get("cf-ipcountry")  # Cloudflare, if present - see module docstring
        if not country:
            country = _lookup_country_via_fallback_api(ip)

        if country and country.upper() in _blocked_country_codes():
            if "text/html" in request.headers.get("accept", ""):
                return HTMLResponse(_BLOCKED_PAGE_HTML, status_code=451)
            return JSONResponse(
                {"detail": "This service is not available in your region."}, status_code=451)

        return await call_next(request)
