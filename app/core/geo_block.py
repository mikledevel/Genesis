"""
Geo-blocking - restricts access from a configured list of countries and, for Ukraine
specifically, a configured list of oblasts/regions (Crimea and the other territories under
comprehensive OFAC sanctions - see below). Many upstream providers this app depends on,
including Stripe and Groq, already restrict or complicate service for sanctioned
countries/regions, so blocking at the app level avoids a user hitting a confusing failure
three steps into signup, and keeps the platform itself compliant rather than just relying on
those upstream providers to reject things downstream.

Country detection, in priority order:
  1. Cloudflare's `CF-IPCountry` header - zero cost, zero latency, set at the edge by
     Cloudflare before the request ever reaches this server. This is the ONLY reliable path
     in production and is what you get for free just by putting the app behind Cloudflare's
     proxy (orange-cloud DNS). This is the recommended real-world setup.
  2. A fallback call to a free IP-geolocation API (ip-api.com, no key required) for when the
     app is NOT behind Cloudflare (e.g. local testing of this feature, or a deployment behind
     a different/no CDN). Results are cached in memory per IP for _CACHE_TTL_SECONDS to avoid
     hammering the external API on every request from the same visitor. This fallback has a
     real request-count limit (45/min on ip-api.com's free tier) and a real network
     dependency - it is NOT a substitute for #1 at real traffic volume.

Region detection (Ukraine oblasts only): Cloudflare's standard (non-Enterprise) product only
ever exposes country-level geography via CF-IPCountry - there is no free "CF-IPRegion"
equivalent, so oblast-level detail always requires the ip-api.com fallback lookup, REGARDLESS
of whether Cloudflare's country header is present. When CF-IPCountry says "UA", an extra
lookup against ip-api.com is made specifically to read regionName (this is the one exception
to "skip the external call when Cloudflare already answered" - Cloudflare's answer just isn't
granular enough for this one case). Region names from geolocation providers are matched
case-insensitively against several spellings, including the Russian-transliterated forms
(Lugansk/Zaporozhye) some databases still use for these regions.

LIMITATION - read before relying on this for compliance sign-off: oblast-level IP geolocation
is meaningfully less reliable than country-level. It depends entirely on the region field
whatever provider is configured (self-reports and can be stale, especially for territories
where internet infrastructure has been rerouted since 2022). Note also that a real, practical
side effect of that rerouting: a substantial share of traffic actually originating from
Crimea and the occupied parts of Donetsk, Luhansk, Zaporizhzhia and Kherson oblasts is by now
routed through Russian telecom infrastructure and geolocates as country=RU outright - which
the plain country-code block below already catches, independent of the region-matching logic
here. Treat the region list as a meaningful improvement over country-only blocking, not as a
guarantee - if a compliance sign-off requires a specific accuracy bar, pair this with a
commercial geolocation provider (e.g. MaxMind GeoIP2 Precision) that warrants region-level
accuracy, rather than relying solely on a free lookup service.

Fail-open policy: if country/region cannot be determined (private/loopback IP, external API
unreachable, rate-limited, malformed response), the request is ALLOWED, never blocked. A
false negative (a visitor from a blocked location slips through because lookup failed) is an
acceptable product trade-off; a false positive (blocking a legitimate visitor because our
geolocation dependency had a bad moment) is not - that would deny service to someone we have
no actual reason to deny.

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

_CACHE_TTL_SECONDS = 3600  # how long a fallback-API geo lookup is trusted per IP
_geo_cache: Dict[str, Tuple[Optional[str], Optional[str], float]] = {}  # ip -> (country, region, expires_at_monotonic)

# Routes that must always be reachable regardless of geography - health checks for
# monitoring/uptime services, which themselves often run from arbitrary cloud regions and
# would otherwise flap the app's health status based on where the monitor happens to be.
_ALWAYS_ALLOWED_PATHS = {"/health"}

# Alternate spellings/transliterations geolocation providers use for these regions - matched
# as a substring of the lowercased regionName, so "Donetsk Oblast", "Donetska oblast", and
# "Donetsk" all match the same "donetsk" entry.
_UA_REGION_ALIASES = {
    "crimea": "Crimea",
    "avtonomna respublika krym": "Crimea",
    "sevastopol": "Crimea",  # federal-city-status but part of the same sanctioned territory
    "donetsk": "Donetsk (DPR)",
    "luhansk": "Luhansk (LPR)",
    "lugansk": "Luhansk (LPR)",  # Russian transliteration some providers still use
    "zaporizhzhia": "Zaporizhzhia",
    "zaporozhye": "Zaporizhzhia",  # Russian transliteration
    "kherson": "Kherson",
}

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
  <h1>Genesis AI isn't available in your region</h1>
  <p>Due to export control and sanctions requirements, we're unable to offer service to
     visitors connecting from this location.</p>
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


def _lookup_geo_via_fallback_api(ip: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (country_code, region_name). Both come from the same request, so this is the
    ONE roundtrip needed whether the caller wants just the country or country+region."""
    cached = _geo_cache.get(ip)
    if cached and cached[2] > time.monotonic():
        return cached[0], cached[1]
    try:
        req = urllib.request.Request(
            f"http://ip-api.com/json/{ip}?fields=countryCode,regionName",
            headers={"User-Agent": "GenesisAI/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        country, region = data.get("countryCode"), data.get("regionName")
        _geo_cache[ip] = (country, region, time.monotonic() + _CACHE_TTL_SECONDS)
        return country, region
    except Exception:
        # Network error, timeout, rate limit, malformed response - all fail open (see module
        # docstring). Deliberately not logged at error level: this is expected to happen
        # occasionally under normal operation and should not page anyone.
        return None, None


def _blocked_country_codes() -> Set[str]:
    return {c.strip().upper() for c in settings.geoblock_countries.split(",") if c.strip()}


def _blocked_ua_region_match(region_name: Optional[str]) -> Optional[str]:
    """Returns the canonical name of the blocked region region_name matches (for logging/the
    denial message), or None if it doesn't match any configured blocked region."""
    if not region_name:
        return None
    configured = {r.strip().lower() for r in settings.geoblock_ua_regions.split(",") if r.strip()}
    if not configured:
        return None
    haystack = region_name.strip().lower()
    for alias, canonical in _UA_REGION_ALIASES.items():
        if alias in haystack and (alias in configured or canonical.split(" ")[0].lower() in configured):
            return canonical
    return None


class GeoBlockMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.geoblock_enabled or request.url.path in _ALWAYS_ALLOWED_PATHS:
            return await call_next(request)

        ip = _client_ip(request)
        if not ip or _is_private_ip(ip):
            if settings.geoblock_debug_log:
                print(f"[geo_block] {request.url.path} - resolved client ip={ip!r} -> "
                      f"private/unknown, SKIPPING check (this is why local/direct-to-uvicorn "
                      f"testing never blocks - see docs/PUBLIC_API.md's Availability section "
                      f"if this is unexpected in production: it usually means a reverse proxy "
                      f"in front of this app isn't forwarding X-Forwarded-For)")
            return await call_next(request)

        cf_header = request.headers.get("cf-ipcountry")
        country = cf_header
        region = None
        source = "cf-ipcountry header"
        if not country:
            country, region = _lookup_geo_via_fallback_api(ip)
            source = "ip-api.com fallback"
        elif country.upper() == "UA" and settings.geoblock_ua_regions.strip():
            # Cloudflare's header alone can't distinguish Crimea/DPR/LPR/Zaporizhzhia/Kherson
            # from the rest of Ukraine - the extra lookup is only made for this one case.
            _, region = _lookup_geo_via_fallback_api(ip)
            source = "cf-ipcountry header + ip-api.com fallback for region"

        blocked_reason = None
        if country and country.upper() in _blocked_country_codes():
            blocked_reason = country.upper()
        elif country and country.upper() == "UA":
            region_match = _blocked_ua_region_match(region)
            if region_match:
                blocked_reason = region_match

        if settings.geoblock_debug_log:
            print(f"[geo_block] {request.url.path} - client ip={ip!r} country={country!r} "
                  f"region={region!r} (source: {source}) -> "
                  f"{'BLOCKED (' + blocked_reason + ')' if blocked_reason else 'allowed'}")

        if blocked_reason:
            if "text/html" in request.headers.get("accept", ""):
                return HTMLResponse(_BLOCKED_PAGE_HTML, status_code=451)
            return JSONResponse(
                {"detail": "This service is not available in your region."}, status_code=451)

        return await call_next(request)
