#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenesisAI - Geo-block verification script
===========================================

Checks that GeoBlockMiddleware is actually blocking the countries/regions it's supposed to,
against a REAL running server - the same way I verified it manually with curl earlier in this
conversation, just automated and covering the full country list in one run.

HOW TO RUN
----------
1. Make sure the server is running (locally or on your real domain) and GEOBLOCK_ENABLED=true
   in whatever .env it actually loaded, and that you RESTARTED it after any .env change.
2. python3 verify_geoblock.py
   python3 verify_geoblock.py --base-url https://your-real-domain.example.com

WHAT IT TESTS
-------------
- Every configured blocked country (RU,BY,CU,IR,KP,SY by default) returns 451.
- A few clearly-allowed countries (US, DE, JP) do NOT return 451.
- A request with no geo headers at all and no X-Forwarded-For (i.e. how a DIRECT localhost
  request looks) is allowed - this is the private-IP exemption, and its result here is exactly
  what tells you whether your reverse proxy (if any) is forwarding real client IPs. See the
  note this script prints if that test's outcome looks suspicious.
- /health is reachable regardless of the CF-IPCountry header (always-exempt path).

All requests target GET /api/marketplace/listings, a real public endpoint that needs no auth
and no body, so a 451 unambiguously means "blocked by geography" and nothing else could have
produced that status code.

Ukraine-region (Crimea/Donetsk/Luhansk/Zaporizhzhia/Kherson) blocking is NOT re-tested live
here, because that logic depends on a real ip-api.com lookup by IP, which can't be forced via
a header the way country-level CF-IPCountry can - it's already covered by
tests/test_geo_block.py's mocked unit tests, which test the matching logic directly. This
script only proves your deployment's CONFIGURATION (env vars loaded, middleware wired in,
restarted) is actually taking effect - which was exactly the gap in your case.
"""
import argparse
import sys

try:
    import requests
except ImportError:
    print("This script needs the 'requests' package: pip install requests")
    sys.exit(1)

ENDPOINT = "/api/marketplace/listings"
FAKE_PUBLIC_IP = "1.2.3.4"  # any non-private IP - only used to defeat the private-IP exemption
DEFAULT_BLOCKED = ["RU", "BY", "CU", "IR", "KP", "SY"]
ALLOWED_SAMPLE = ["US", "DE", "JP"]


def check(base_url: str, country: str = None, forwarded_for: str = FAKE_PUBLIC_IP) -> requests.Response:
    headers = {}
    if country:
        headers["CF-IPCountry"] = country
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    return requests.get(base_url + ENDPOINT, headers=headers, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--blocked", default=",".join(DEFAULT_BLOCKED),
                         help="Comma-separated country codes expected to be blocked (default: your configured list)")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    blocked_countries = [c.strip().upper() for c in args.blocked.split(",") if c.strip()]

    print(f"Testing geo-block against {base_url}\n")
    failures = 0

    # 0. Server reachable at all?
    try:
        requests.get(base_url + "/health", timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"CANNOT REACH {base_url} - is the server running? ({e})")
        sys.exit(2)

    # 1. /health always reachable, even with a blocked country header.
    r = requests.get(base_url + "/health", headers={"CF-IPCountry": "RU"}, timeout=10)
    ok = r.status_code != 451
    print(f"{'PASS' if ok else 'FAIL'}  /health stays reachable regardless of geography "
          f"(got {r.status_code})")
    failures += 0 if ok else 1

    # 2. Every blocked country -> 451.
    print()
    for country in blocked_countries:
        r = check(base_url, country=country)
        ok = r.status_code == 451
        print(f"{'PASS' if ok else 'FAIL'}  CF-IPCountry={country} -> {r.status_code} "
              f"{'(blocked, correct)' if ok else '(expected 451!)'}")
        failures += 0 if ok else 1

    # 3. Allowed countries -> NOT 451.
    print()
    for country in ALLOWED_SAMPLE:
        r = check(base_url, country=country)
        ok = r.status_code != 451
        print(f"{'PASS' if ok else 'FAIL'}  CF-IPCountry={country} -> {r.status_code} "
              f"{'(allowed, correct)' if ok else '(WRONGLY blocked!)'}")
        failures += 0 if ok else 1

    # 4. Direct request, no forwarded-for, no CF header - simulates hitting the app directly
    #    with no proxy in front translating the real client IP. Result here is diagnostic,
    #    not strictly pass/fail, since "allowed" is CORRECT when testing straight to
    #    localhost, but is the exact symptom of a misconfigured reverse proxy in production.
    print()
    r = check(base_url, country=None, forwarded_for=None)
    print(f"INFO  Direct request, no CF-IPCountry, no X-Forwarded-For -> {r.status_code}")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        print("      (expected: allowed - this is the private-IP exemption working as intended")
        print("       for local testing)")
    else:
        print("      This request looked exactly like a request from a private/local IP to the")
        print("      server. If you're hitting a real public domain here and NOT running curl")
        print("      from the server's own machine, and this returned anything other than 451")
        print("      for what should be a blocked visitor in production, it's a strong signal")
        print("      your reverse proxy isn't forwarding the real client IP - see")
        print("      GEOBLOCK_DEBUG_LOG in your server's logs for the definitive answer.")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
