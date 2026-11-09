"""E-mail through Resend and the bot protection. Network calls are faked: the dev sandbox cannot reach api.resend.com or
challenges.cloudflare.com, so the REAL provider behaviour (accepted keys, verified domains, Turnstile keys) is NOT covered here."""
import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.core import antibot, email_sender  # noqa: E402

OWNER = "owner@genesis.test"
_IP = 0
KEY = "re_TESTKEY_1234567890"


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body if body is not None else {"id": "msg_1"}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.main import app
    from app import deps
    deps._agents.clear()
    monkeypatch.setattr(settings, "admin_emails", OWNER)
    monkeypatch.setattr(email_sender, "_SYNC", True)
    monkeypatch.setattr(email_sender.time, "sleep", lambda s: None)
    c = TestClient(app)
    global _IP
    _IP += 1
    c.headers["x-forwarded-for"] = f"10.8.{_IP // 250}.{_IP % 250}"     # own address per test: the real rate limiter stays on
    return c


def _resend_on(monkeypatch, handler):
    """Resend configured; every httpx.post goes to handler(url, **kw) -> FakeResp (or raises)."""
    monkeypatch.setattr(settings, "resend_api_key", KEY)
    monkeypatch.setattr(settings, "email_from", "Genesis <hello@mail.genesis.test>")
    calls = []

    def post(url, **kw):
        calls.append({"url": url, **kw})
        return handler(url, **kw)
    monkeypatch.setattr(email_sender.httpx, "post", post)
    return calls


def _reg(client, email="u@shop.test", **extra):
    body = {"email": email, "password": "secret12", "name": "T", **extra}
    return client.post("/api/auth/register", json=body)


def _db():
    from app.db.database import GenesisDB
    return GenesisDB()


# ======================================================================= e-mail: provider and request
def test_provider_priority_and_sender_fallback(monkeypatch):
    assert email_sender.email_provider() == "dev"
    monkeypatch.setattr(settings, "smtp_host", "smtp.x")
    assert email_sender.email_provider() == "smtp"
    monkeypatch.setattr(settings, "resend_api_key", KEY)
    assert email_sender.email_provider() == "resend"                      # Resend wins over SMTP
    assert email_sender.from_address() == settings.smtp_from
    monkeypatch.setattr(settings, "email_from", "A <a@b.test>")
    assert email_sender.from_address() == "A <a@b.test>"
    assert KEY not in json.dumps(email_sender.status())                   # the key never leaves through status()


def test_resend_request_shape(monkeypatch):
    calls = _resend_on(monkeypatch, lambda url, **kw: FakeResp())
    ok, detail = email_sender.deliver("to@x.io", "Hello", "plain body", "<b>html</b>")
    assert ok and detail == "msg_1" and len(calls) == 1
    c = calls[0]
    assert c["url"] == "https://api.resend.com/emails" and c["headers"]["Authorization"] == f"Bearer {KEY}"
    assert c["json"] == {"from": "Genesis <hello@mail.genesis.test>", "to": ["to@x.io"], "subject": "Hello", "text": "plain body", "html": "<b>html</b>"}


def test_a_client_error_is_not_retried_and_the_key_is_scrubbed(monkeypatch):
    calls = _resend_on(monkeypatch, lambda url, **kw: FakeResp(403, {"message": f"domain not verified (key {KEY})"}))
    ok, detail = email_sender.deliver("to@x.io", "S", "b")
    assert not ok and len(calls) == 1 and "domain not verified" in detail and KEY not in detail


def test_rate_limit_and_server_errors_are_retried_then_succeed(monkeypatch):
    seq = [FakeResp(429, {"message": "slow down"}, {"retry-after": "1"}), FakeResp(503, {"message": "down"}), FakeResp(200)]
    slept = []
    monkeypatch.setattr(email_sender.time, "sleep", lambda s: slept.append(s))
    calls = _resend_on(monkeypatch, lambda url, **kw: seq[len(calls) - 1])
    assert email_sender.deliver("to@x.io", "S", "b")[0] is True
    assert len(calls) == 3 and slept[0] == 1.0                            # Retry-After honoured


def test_network_failures_give_up_after_three_attempts_without_raising(monkeypatch):
    import httpx

    def boom(url, **kw):
        raise httpx.ConnectTimeout("t")
    calls = _resend_on(monkeypatch, boom)
    ok, detail = email_sender.deliver("to@x.io", "S", "b")
    assert not ok and len(calls) == 3 and "network error" in detail
    _resend_on(monkeypatch, lambda url, **kw: (_ for _ in ()).throw(RuntimeError("weird")))
    assert email_sender.deliver("to@x.io", "S", "b")[0] is False          # never raises


def test_header_injection_is_flattened(monkeypatch):
    calls = _resend_on(monkeypatch, lambda url, **kw: FakeResp())
    email_sender.deliver("to@x.io\r\nBcc: evil@x.io", "Hi\r\nBcc: evil@x.io", "b")
    j = calls[0]["json"]
    assert "\n" not in j["subject"] and "\r" not in j["subject"] and "\n" not in j["to"][0]


def test_html_mail_escapes_everything_it_is_given():
    h = email_sender._html_mail('<script>x</script>', ['<b>p</b>'], 'https://a.b/?t="><script>alert(1)</script>', '<i>go</i>', '<u>f</u>')
    assert "<script>" not in h and "<b>p</b>" not in h and "<i>go</i>" not in h
    assert '&quot;&gt;&lt;script&gt;' in h


def test_with_a_real_provider_the_send_runs_in_a_background_thread(monkeypatch):
    monkeypatch.setattr(email_sender, "_SYNC", False)
    seen = {}
    done = threading.Event()

    def handler(url, **kw):
        seen["thread"] = threading.current_thread()
        done.set()
        return FakeResp()
    _resend_on(monkeypatch, handler)
    email_sender.send_verification_email("to@x.io", "tok")
    assert done.wait(3) and seen["thread"] is not threading.main_thread()


def test_dev_mode_prints_the_link_and_reports_nothing_was_sent(capsys):
    ok, detail = email_sender.deliver("to@x.io", "S", "link: http://x/?verify_email=abc")
    assert ok and "dev mode" in detail and "verify_email=abc" in capsys.readouterr().out


# ======================================================================= e-mail: the flows
def test_registration_sends_the_verification_mail_through_resend(client, monkeypatch):
    calls = _resend_on(monkeypatch, lambda url, **kw: FakeResp())
    assert _reg(client, "alice@shop.test").status_code == 200
    j = calls[0]["json"]
    assert j["to"] == ["alice@shop.test"] and "verify_email=" in j["text"] and "verify_email=" in j["html"]


def test_a_failing_provider_never_breaks_registration(client, monkeypatch):
    _resend_on(monkeypatch, lambda url, **kw: FakeResp(401, {"message": "invalid api key"}))
    assert _reg(client, "alice@shop.test").status_code == 200


def test_password_reset_is_capped_per_address_and_the_answer_never_changes(client, monkeypatch):
    calls = _resend_on(monkeypatch, lambda url, **kw: FakeResp())
    assert _reg(client, "alice@shop.test").status_code == 200
    n0 = len(calls)
    answers = []
    for _ in range(5):
        r = client.post("/api/auth/password-reset/request", json={"email": "alice@shop.test"})
        answers.append((r.status_code, r.json()))
    assert len({json.dumps(a) for a in answers}) == 1                              # identical response every time
    assert len(calls) - n0 == settings.email_per_address_per_hour                  # ... but only 3 mails went out
    r = client.post("/api/auth/password-reset/request", json={"email": "nobody@shop.test"})
    assert (r.status_code, r.json()) == answers[0] and len(calls) - n0 == settings.email_per_address_per_hour


def test_resend_verification_is_capped(client, monkeypatch):
    _resend_on(monkeypatch, lambda url, **kw: FakeResp())
    h = {"Authorization": "Bearer " + _reg(client, "alice@shop.test").json()["token"]}
    codes = [client.post("/api/auth/verify-email/resend", headers=h).status_code for _ in range(4)]
    assert codes[-1] == 429 and codes[0] == 200


# ======================================================================= bots
def test_honeypot_and_speed_checks(client):
    assert _reg(client, "a1@shop.test", website="http://spam.example").status_code == 400
    assert _reg(client, "a2@shop.test", form_elapsed_ms=300).status_code == 400            # 0.3 s: a script
    assert _reg(client, "a3@shop.test", form_elapsed_ms="abc").status_code in (400, 422)       # refused by validation or by us
    assert _reg(client, "a4@shop.test", form_elapsed_ms=9000).status_code == 200           # a person
    assert _reg(client, "a5@shop.test").status_code == 200                                  # old clients that send neither
    refused = _reg(client, "a6@shop.test", website="x").json()["detail"]
    assert refused == antibot.GENERIC_REFUSAL                                               # no hint about WHICH check failed


def test_disposable_addresses_are_refused_unless_switched_off(client, monkeypatch):
    for e in ("x@mailinator.com", "x@sub.guerrillamail.com", "X@YOPMAIL.COM"):
        r = _reg(client, e)
        assert r.status_code == 400 and "permanent" in r.json()["detail"]
    monkeypatch.setattr(settings, "disposable_email_domains_extra", "my-temp.test, other.test")
    assert _reg(client, "x@my-temp.test").status_code == 400
    assert _reg(client, "real@gmail.com").status_code == 200
    monkeypatch.setattr(settings, "antibot_block_disposable", False)
    assert _reg(client, "x@mailinator.com").status_code == 200


def test_one_mailbox_is_one_account(client):
    assert _reg(client, "ab@gmail.com").status_code == 200
    for alias in ("a.b@gmail.com", "ab+promo@gmail.com", "a.b+1@googlemail.com", "AB@Gmail.com"):
        r = _reg(client, alias)
        assert r.status_code == 409, alias
    assert _reg(client, "ann.lee@shop.test").status_code == 200
    assert _reg(client, "annlee@shop.test").status_code == 200               # dots only matter for Gmail
    assert _reg(client, "ann.lee+x@shop.test").status_code == 409            # +tag does


def test_canonical_backfill_for_accounts_created_before_the_column_existed(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.chdir(tmp_path)
    from app.db.database import GenesisDB
    db = GenesisDB(db_path=str(tmp_path / "t.db"))
    db.create_user("old.user@gmail.com", "x", "Old")
    with sqlite3.connect(str(tmp_path / "t.db")) as conn:
        conn.execute("UPDATE users SET email_canonical = NULL")
    db2 = GenesisDB(db_path=str(tmp_path / "t.db"))                           # opening the database runs the migration
    assert db2.canonical_email_exists("olduser@gmail.com") and not db2.canonical_email_exists("other@gmail.com")


@pytest.mark.parametrize("bad", ["a@b", "a b@c.io", "a@b..io", "@x.io", "a@@x.io", "a,b@x.io", "x" * 250 + "@a.io", "plain"])
def test_bad_email_formats(client, bad):
    assert _reg(client, bad).status_code == 400


def test_overlong_name_is_refused(client):
    assert client.post("/api/auth/register", json={"email": "n@shop.test", "password": "secret12", "name": "n" * 101}).status_code == 400


# ======================================================================= Turnstile
def _turnstile(monkeypatch, handler):
    monkeypatch.setattr(settings, "turnstile_site_key", "SITEKEY")
    monkeypatch.setattr(settings, "turnstile_secret_key", "SECRETKEY")
    calls = []

    def post(url, **kw):
        calls.append({"url": url, **kw})
        return handler(url, **kw)
    monkeypatch.setattr(antibot.httpx, "post", post)
    return calls


def test_captcha_is_off_unless_both_keys_are_set(client, monkeypatch):
    assert antibot.captcha_enabled() is False and client.get("/api/config/public").json() == {
        "turnstile_site_key": "", "stripe_subscriptions_enabled": False, "crypto_payments_enabled": False}
    monkeypatch.setattr(settings, "turnstile_site_key", "SITEKEY")
    assert antibot.captcha_enabled() is False                                     # half-configured -> off, not a lockout
    assert client.get("/api/config/public").json() == {
        "turnstile_site_key": "", "stripe_subscriptions_enabled": False, "crypto_payments_enabled": False}
    assert _reg(client, "a@shop.test").status_code == 200
    monkeypatch.setattr(settings, "turnstile_secret_key", "SECRETKEY")
    assert client.get("/api/config/public").json() == {
        "turnstile_site_key": "SITEKEY", "stripe_subscriptions_enabled": False, "crypto_payments_enabled": False}  # the SECRET never leaves


def test_captcha_is_required_and_verified_server_side(client, monkeypatch):
    calls = _turnstile(monkeypatch, lambda url, **kw: FakeResp(200, {"success": kw["data"]["response"] == "good"}))
    assert _reg(client, "a1@shop.test").status_code == 400                        # no token: Cloudflare is not even asked
    assert calls == []
    r = _reg(client, "a2@shop.test", captcha_token="bad")
    assert r.status_code == 400 and "human check" in r.json()["detail"]
    assert _reg(client, "a3@shop.test", captcha_token="good").status_code == 200
    c = calls[-1]
    assert c["url"] == "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    assert c["data"]["secret"] == "SECRETKEY" and c["data"]["response"] == "good" and c["data"]["remoteip"].startswith("10.8.")


def test_captcha_fails_closed_when_cloudflare_is_unreachable(client, monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("no route")
    _turnstile(monkeypatch, boom)
    r = _reg(client, "a@shop.test", captcha_token="good")
    assert r.status_code == 503 and "try again" in r.json()["detail"]
    assert _db().get_user_by_email("a@shop.test") is None                          # nobody was waved through


def test_the_cheap_checks_run_before_cloudflare_is_called(client, monkeypatch):
    calls = _turnstile(monkeypatch, lambda url, **kw: FakeResp(200, {"success": True}))
    assert _reg(client, "a@shop.test", website="x", captcha_token="good").status_code == 400
    assert _reg(client, "b@mailinator.com", captcha_token="good").status_code == 400
    assert calls == []


def test_the_password_reset_request_is_protected_too(client, monkeypatch):
    _turnstile(monkeypatch, lambda url, **kw: FakeResp(200, {"success": kw["data"]["response"] == "good"}))
    assert client.post("/api/auth/password-reset/request", json={"email": "a@shop.test"}).status_code == 400
    assert client.post("/api/auth/password-reset/request", json={"email": "a@shop.test", "website": "x", "captcha_token": "good"}).status_code == 400
    assert client.post("/api/auth/password-reset/request", json={"email": "a@shop.test", "captcha_token": "good"}).status_code == 200


# ======================================================================= admin: test e-mail and status
def _owner(client):
    r = _reg(client, OWNER)
    uid, h = r.json()["user"]["id"], {"Authorization": "Bearer " + r.json()["token"]}
    tok = _db().create_auth_token(uid, "verify_email", 3600)
    assert client.post("/api/auth/verify-email/confirm", json={"token": tok}, headers=h).status_code == 200
    login = client.post("/api/auth/login", json={"email": OWNER, "password": "secret12"}).json()["token"]
    return uid, {"Authorization": "Bearer " + login}


def test_admin_test_email_reports_the_real_outcome(client, monkeypatch):
    uid, ah = _owner(client)
    r = client.post("/api/admin/email/test", headers=ah).json()
    assert r["ok"] is True and r["provider"] == "dev" and r["to"] == OWNER and "dev mode" in r["detail"]
    _resend_on(monkeypatch, lambda url, **kw: FakeResp(403, {"message": "domain not verified"}))
    r = client.post("/api/admin/email/test", headers=ah).json()
    assert r["ok"] is False and r["provider"] == "resend" and "domain not verified" in r["detail"] and KEY not in json.dumps(r)
    ov = client.get("/api/admin/overview", headers=ah).json()
    assert ov["email"]["provider"] == "resend" and ov["email"]["key_set"] is True and ov["captcha_enabled"] is False and KEY not in json.dumps(ov)
    assert client.get("/api/admin/audit", headers=ah).json()["actions"][0]["action"] == "email_test"


def test_only_an_admin_can_send_the_test_email(client):
    _u, h = _reg(client, "plain@shop.test").json()["user"]["id"], None
    tok = _reg(client, "plain2@shop.test").json()["token"]
    assert client.post("/api/admin/email/test", headers={"Authorization": "Bearer " + tok}).status_code == 403
    assert client.post("/api/admin/email/test").status_code == 401
