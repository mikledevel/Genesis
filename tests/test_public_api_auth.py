"""Tests for app.core.auth.get_api_key_from_request - the auth dependency for the public
developer API. Stubs just enough of fastapi (Header/HTTPException) and its transitive deps
to exercise the real function, since this sandbox doesn't have fastapi installed.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import fastapi  # noqa
except ImportError:
    fake_fastapi = types.ModuleType("fastapi")

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            self.status_code, self.detail = status_code, detail
            super().__init__(detail)

    def Header(default=None, alias=None):
        return default  # in real fastapi this is resolved via DI; here callers pass args directly

    fake_fastapi.HTTPException = HTTPException
    fake_fastapi.Header = Header
    sys.modules["fastapi"] = fake_fastapi

try:
    import jwt  # noqa
except ImportError:
    fake_jwt = types.ModuleType("jwt")
    class PyJWTError(Exception): pass
    fake_jwt.PyJWTError = PyJWTError
    fake_jwt.encode = lambda *a, **kw: "fake.jwt.token"
    fake_jwt.decode = lambda *a, **kw: (_ for _ in ()).throw(PyJWTError())
    sys.modules["jwt"] = fake_jwt

try:
    import bcrypt  # noqa
except ImportError:
    fake_bcrypt = types.ModuleType("bcrypt")
    fake_bcrypt.hashpw = lambda pw, salt: pw
    fake_bcrypt.gensalt = lambda: b""
    fake_bcrypt.checkpw = lambda pw, h: pw == h
    sys.modules["bcrypt"] = fake_bcrypt

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

import fastapi
from app.core.auth import get_api_key_from_request


def test_bearer_header_is_used_when_present():
    key = get_api_key_from_request(authorization="Bearer gen-abc123", x_api_key=None)
    assert key == "gen-abc123"


def test_x_api_key_header_used_when_no_bearer():
    key = get_api_key_from_request(authorization=None, x_api_key="gen-xyz789")
    assert key == "gen-xyz789"


def test_bearer_takes_priority_over_x_api_key_when_both_present():
    key = get_api_key_from_request(authorization="Bearer gen-fromheader", x_api_key="gen-fromxapikey")
    assert key == "gen-fromheader"


def test_missing_both_raises_401():
    try:
        get_api_key_from_request(authorization=None, x_api_key=None)
        assert False, "should have raised"
    except fastapi.HTTPException as e:
        assert e.status_code == 401


def test_non_bearer_authorization_header_falls_back_to_x_api_key():
    # A JWT session token (or anything not starting with "Bearer ") in this header must NOT
    # be treated as the API key - falls through to X-API-Key instead.
    key = get_api_key_from_request(authorization="Basic somejwt", x_api_key="gen-fallback")
    assert key == "gen-fallback"


def test_empty_bearer_value_falls_back_to_x_api_key():
    key = get_api_key_from_request(authorization="Bearer ", x_api_key="gen-fallback2")
    assert key == "gen-fallback2"


if __name__ == "__main__":
    passed, failed = 0, 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
