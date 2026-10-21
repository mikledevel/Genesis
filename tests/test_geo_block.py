"""Tests for app/core/geo_block.py.

Covers the pure logic directly (country-code parsing, Ukraine region-name matching, private-IP
detection, client-IP extraction) plus full middleware dispatch scenarios against a minimal
stand-in for starlette (this sandbox doesn't have the real package installed, but the dispatch
logic itself has no real dependency on starlette beyond the Request/Response shapes it reads
and returns, both duck-typed here).
"""
import os
import sys
import types
import time
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Minimal starlette stand-in, only if the real package isn't installed ---
try:
    import starlette  # noqa
except ImportError:
    fake_starlette = types.ModuleType("starlette")
    fake_mw = types.ModuleType("starlette.middleware")
    fake_mw_base = types.ModuleType("starlette.middleware.base")
    fake_requests = types.ModuleType("starlette.requests")
    fake_responses = types.ModuleType("starlette.responses")

    class _BaseHTTPMiddleware:
        def __init__(self, app=None):
            self.app = app

    class _Request:
        pass

    class _Response:
        def __init__(self, content, status_code=200):
            self.body, self.status_code = content, status_code

    class _HTMLResponse(_Response):
        pass

    class _JSONResponse(_Response):
        def __init__(self, content, status_code=200):
            self.content, self.status_code = content, status_code

    fake_mw_base.BaseHTTPMiddleware = _BaseHTTPMiddleware
    fake_requests.Request = _Request
    fake_responses.HTMLResponse = _HTMLResponse
    fake_responses.JSONResponse = _JSONResponse
    sys.modules["starlette"] = fake_starlette
    sys.modules["starlette.middleware"] = fake_mw
    sys.modules["starlette.middleware.base"] = fake_mw_base
    sys.modules["starlette.requests"] = fake_requests
    sys.modules["starlette.responses"] = fake_responses

try:
    import pydantic_settings  # noqa
except ImportError:
    fake_ps = types.ModuleType("pydantic_settings")
    class _BaseSettings:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    fake_ps.BaseSettings = _BaseSettings
    fake_ps.SettingsConfigDict = lambda **kw: kw
    sys.modules["pydantic_settings"] = fake_ps

from app.config import settings
from app.core import geo_block


class _FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, headers=None, client_ip="8.8.8.8", path="/api/agent/message"):
        self.headers = _FakeHeaders({k.lower(): v for k, v in (headers or {}).items()})
        self.client = _FakeClient(client_ip) if client_ip else None
        self.url = types.SimpleNamespace(path=path)


def setup_module(module):
    settings.geoblock_enabled = True
    settings.geoblock_countries = "RU,BY,CU,IR,KP,SY"
    settings.geoblock_ua_regions = "Crimea,Donetsk,Luhansk,Zaporizhzhia,Kherson"
    geo_block._geo_cache.clear()


# ---------------------------------------------------------------- pure functions ----

def test_blocked_country_codes_parses_full_list():
    assert geo_block._blocked_country_codes() == {"RU", "BY", "CU", "IR", "KP", "SY"}


def test_is_private_ip():
    assert geo_block._is_private_ip("127.0.0.1") is True
    assert geo_block._is_private_ip("192.168.1.5") is True
    assert geo_block._is_private_ip("10.0.0.1") is True
    assert geo_block._is_private_ip("8.8.8.8") is False
    assert geo_block._is_private_ip("not-an-ip") is True  # unparseable -> don't block on it


def test_client_ip_prefers_x_forwarded_for():
    req = _FakeRequest(headers={"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}, client_ip="10.0.0.1")
    assert geo_block._client_ip(req) == "203.0.113.5"


def test_client_ip_falls_back_to_socket_peer():
    req = _FakeRequest(headers={}, client_ip="203.0.113.9")
    assert geo_block._client_ip(req) == "203.0.113.9"


class TestUaRegionMatching:
    def test_crimea_variants_match(self):
        for name in ["Crimea", "Republic of Crimea", "Avtonomna Respublika Krym", "Sevastopol"]:
            assert geo_block._blocked_ua_region_match(name) == "Crimea", name

    def test_donetsk_and_luhansk_variants_match(self):
        assert geo_block._blocked_ua_region_match("Donetsk Oblast") == "Donetsk (DPR)"
        assert geo_block._blocked_ua_region_match("Luhansk Oblast") == "Luhansk (LPR)"
        assert geo_block._blocked_ua_region_match("Lugansk Oblast") == "Luhansk (LPR)"  # RU transliteration

    def test_zaporizhzhia_and_kherson_variants_match(self):
        assert geo_block._blocked_ua_region_match("Zaporizhzhia Oblast") == "Zaporizhzhia"
        assert geo_block._blocked_ua_region_match("Zaporozhye Oblast") == "Zaporizhzhia"  # RU transliteration
        assert geo_block._blocked_ua_region_match("Kherson Oblast") == "Kherson"

    def test_unblocked_ukrainian_regions_do_not_match(self):
        for name in ["Kyiv City", "Lviv Oblast", "Odessa Oblast", "Kharkiv Oblast", None, ""]:
            assert geo_block._blocked_ua_region_match(name) is None, name

    def test_empty_configured_list_disables_region_blocking(self):
        settings.geoblock_ua_regions = ""
        try:
            assert geo_block._blocked_ua_region_match("Crimea") is None
        finally:
            settings.geoblock_ua_regions = "Crimea,Donetsk,Luhansk,Zaporizhzhia,Kherson"


# ---------------------------------------------------------------- middleware dispatch ----

class _DummyMiddleware(geo_block.GeoBlockMiddleware):
    """dispatch() is what we're testing; skip BaseHTTPMiddleware.__init__'s ASGI app wiring."""
    def __init__(self):
        pass


async def _call_next_ok(request):
    return "PASSED_THROUGH"


def _run(coro):
    import asyncio
    return asyncio.run(coro)


class TestMiddlewareDispatch:
    def test_disabled_allows_everything(self):
        settings.geoblock_enabled = False
        try:
            req = _FakeRequest(headers={"CF-IPCountry": "RU"})
            result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
            assert result == "PASSED_THROUGH"
        finally:
            settings.geoblock_enabled = True

    def test_health_check_always_allowed(self):
        req = _FakeRequest(headers={"CF-IPCountry": "RU"}, path="/health")
        result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result == "PASSED_THROUGH"

    def test_private_ip_always_allowed_even_with_blocked_header(self):
        req = _FakeRequest(headers={"CF-IPCountry": "RU"}, client_ip="127.0.0.1")
        result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result == "PASSED_THROUGH"

    def test_cf_header_blocked_country_is_blocked(self):
        req = _FakeRequest(headers={"CF-IPCountry": "RU", "Accept": "application/json"})
        result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result != "PASSED_THROUGH"
        assert result.status_code == 451

    def test_cf_header_allowed_country_passes(self):
        req = _FakeRequest(headers={"CF-IPCountry": "US"})
        result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result == "PASSED_THROUGH"

    def test_html_accept_header_gets_html_response(self):
        req = _FakeRequest(headers={"CF-IPCountry": "IR", "Accept": "text/html,application/xhtml+xml"})
        result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result.status_code == 451
        assert isinstance(result, geo_block.HTMLResponse)

    def test_cf_says_ukraine_and_fallback_says_crimea_is_blocked(self):
        req = _FakeRequest(headers={"CF-IPCountry": "UA"})
        with patch("app.core.geo_block._lookup_geo_via_fallback_api", return_value=("UA", "Crimea")):
            result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result.status_code == 451

    def test_cf_says_ukraine_and_fallback_says_kyiv_passes(self):
        req = _FakeRequest(headers={"CF-IPCountry": "UA"})
        with patch("app.core.geo_block._lookup_geo_via_fallback_api", return_value=("UA", "Kyiv City")):
            result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result == "PASSED_THROUGH"

    def test_no_cf_header_uses_fallback_country(self):
        req = _FakeRequest(headers={})
        with patch("app.core.geo_block._lookup_geo_via_fallback_api", return_value=("BY", None)):
            result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result.status_code == 451

    def test_fallback_api_failure_fails_open(self):
        req = _FakeRequest(headers={})
        with patch("app.core.geo_block._lookup_geo_via_fallback_api", return_value=(None, None)):
            result = _run(_DummyMiddleware().dispatch(req, _call_next_ok))
        assert result == "PASSED_THROUGH"

    def test_geo_cache_avoids_a_second_lookup_for_the_same_ip(self):
        geo_block._geo_cache.clear()
        call_count = {"n": 0}
        def _mock_urlopen(req, timeout=2.0):
            call_count["n"] += 1
            class _Resp:
                def read(self_inner):
                    return b'{"countryCode": "RU", "regionName": null}'
                def __enter__(self_inner):
                    return self_inner
                def __exit__(self_inner, *a):
                    return False
            return _Resp()
        with patch("app.core.geo_block.urllib.request.urlopen", side_effect=_mock_urlopen):
            geo_block._lookup_geo_via_fallback_api("203.0.113.50")
            geo_block._lookup_geo_via_fallback_api("203.0.113.50")
        assert call_count["n"] == 1, "second lookup for the same IP should have hit the cache"


if __name__ == "__main__":
    import inspect
    passed, failed = 0, 0
    setup_module(sys.modules[__name__])
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                failed += 1
        elif isinstance(obj, type) and name.startswith("Test"):
            inst = obj()
            for meth in dir(inst):
                if meth.startswith("test_"):
                    try:
                        getattr(inst, meth)()
                        print(f"PASS  {name}.{meth}")
                        passed += 1
                    except Exception as e:
                        print(f"FAIL  {name}.{meth}: {e}")
                        failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
