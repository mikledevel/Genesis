"""Plans, limits and the owner/admin panel. The security-critical part: admin = owner e-mail in settings.admin_emails
AND that e-mail VERIFIED (and sessions issued before the verification are void) - never the e-mail string alone."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402

OWNER = "owner@genesis.test"
_IP = 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.main import app
    from app import deps
    from app.config import settings
    deps._agents.clear()
    monkeypatch.setattr(settings, "admin_emails", OWNER)
    c = TestClient(app)
    # The rate limiter is process-wide and keyed by IP: give every test its own address so dozens of registrations and
    # logins across this file never trip the real limit (which stays on, and untouched).
    global _IP
    _IP += 1
    c.headers["x-forwarded-for"] = f"10.9.{_IP // 250}.{_IP % 250}"
    return c


def _db():
    from app.db.database import GenesisDB
    return GenesisDB()


def _register(client, email, password="secret12"):
    r = client.post("/api/auth/register", json={"email": email, "password": password, "name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["user"]["id"], {"Authorization": "Bearer " + r.json()["token"]}


def _verify(client, user_id, headers=None):
    tok = _db().create_auth_token(user_id, "verify_email", 3600)
    r = client.post("/api/auth/verify-email/confirm", json={"token": tok}, headers=headers or {})
    assert r.status_code == 200, r.text


def _login(client, email, password="secret12"):
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def _owner(client):
    uid, h = _register(client, OWNER)
    _verify(client, uid, h)                # an owner e-mail is verified from its own logged-in session
    return uid, _login(client, OWNER)


# ------------------------------------------------------------------ who is an admin
def test_an_unverified_owner_email_is_not_an_admin(client):
    uid, h = _register(client, OWNER)
    assert client.get("/api/admin/overview", headers=h).status_code == 403
    assert client.get("/api/auth/me", headers=h).json()["user"]["is_admin"] is False
    assert _db().is_admin(uid) is False


def test_a_verified_owner_is_an_admin(client):
    uid, h = _owner(client)
    assert client.get("/api/admin/overview", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).json()["user"]["is_admin"] is True


def test_owner_email_cannot_be_verified_without_its_own_session(client):
    """Whoever registered the owner's address first has the password; the real owner has the mailbox. Neither alone may
    complete the verification, so the owner's click on a mail he did not request does not make anyone an admin."""
    uid, mine = _register(client, OWNER)
    tok = _db().create_auth_token(uid, "verify_email", 3600)
    anon = client.post("/api/auth/verify-email/confirm", json={"token": tok})
    assert anon.status_code == 401 and "log in first" in anon.json()["detail"]
    _other_uid, other = _register(client, "other@x.io")
    assert client.post("/api/auth/verify-email/confirm", json={"token": tok}, headers=other).status_code == 401
    assert _db().is_email_verified(uid) is False and _db().is_admin(uid) is False
    assert client.post("/api/auth/verify-email/confirm", json={"token": tok}, headers=mine).status_code == 200   # not burnt by the refusals
    assert _db().is_admin(uid) is True


def test_a_squatter_who_registered_the_owner_address_is_locked_out_by_password_reset(client):
    uid, squatter = _register(client, OWNER, password="attacker1")
    reset = _db().create_auth_token(uid, "password_reset", 3600)          # the owner uses the mailbox he really controls
    assert client.post("/api/auth/password-reset/confirm", json={"token": reset, "new_password": "ownerpass1"}).status_code == 200
    assert client.get("/api/auth/me", headers=squatter).status_code == 401           # the squatter's session is void ...
    assert client.post("/api/auth/login", json={"email": OWNER, "password": "attacker1"}).status_code == 401   # ... and so is his password
    mine = _login(client, OWNER, "ownerpass1")
    _verify(client, uid, mine)
    assert client.get("/api/admin/overview", headers=_login(client, OWNER, "ownerpass1")).status_code == 200
    assert client.get("/api/admin/overview", headers=squatter).status_code == 401


def test_non_owner_emails_still_verify_without_a_session(client):
    uid, _h = _register(client, "plain@x.io")
    _verify(client, uid)
    assert _db().is_email_verified(uid) is True


def test_other_users_are_never_admins_even_verified(client):
    uid, h = _register(client, "someone@x.io")
    _verify(client, uid)
    h = _login(client, "someone@x.io")
    for path in ("/api/admin/overview", "/api/admin/users", "/api/admin/audit"):
        assert client.get(path, headers=h).status_code == 403


def test_owner_match_is_case_insensitive_but_not_alias_based(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "admin_emails", "Owner@Genesis.Test")
    uid, h = _register(client, "OWNER@genesis.test")
    _verify(client, uid, h)
    assert _db().is_admin(uid) is True
    uid2, _h = _register(client, "own.er@genesis.test")      # a different string: never an admin
    _verify(client, uid2)
    assert _db().is_admin(uid2) is False


def test_empty_admin_list_means_nobody(client, monkeypatch):
    from app.config import settings
    uid, h = _owner(client)
    monkeypatch.setattr(settings, "admin_emails", "")
    assert client.get("/api/admin/overview", headers=h).status_code == 403


def test_admin_routes_need_a_session(client):
    for method, path in (("get", "/api/admin/overview"), ("get", "/api/admin/users"), ("post", "/api/admin/users/x/plan")):
        assert getattr(client, method)(path).status_code == 401


def test_password_reset_voids_existing_sessions(client):
    uid, old = _register(client, "r@x.io")
    tok = _db().create_auth_token(uid, "password_reset", 3600)
    assert client.post("/api/auth/password-reset/confirm", json={"token": tok, "new_password": "brandnew1"}).status_code == 200
    assert client.get("/api/auth/me", headers=old).status_code == 401
    assert client.get("/api/auth/me", headers=_login(client, "r@x.io", "brandnew1")).status_code == 200


# ------------------------------------------------------------------ the admin actions
def test_set_plan_validates_and_is_audited(client):
    admin, ah = _owner(client)
    uid, uh = _register(client, "u@x.io")
    assert client.post(f"/api/admin/users/{uid}/plan", json={"plan": "gold"}, headers=ah).status_code == 400
    assert client.post("/api/admin/users/nope/plan", json={"plan": "pro"}, headers=ah).status_code == 404
    assert client.post(f"/api/admin/users/{uid}/plan", json={"plan": "pro"}, headers=ah).status_code == 200
    assert client.get("/api/me/plan", headers=uh).json()["plan"]["id"] == "pro"
    log = client.get("/api/admin/audit", headers=ah).json()["actions"]
    assert log[0]["action"] == "set_plan" and log[0]["target_id"] == uid and log[0]["detail"]["to"] == "pro" and log[0]["admin_id"] == admin


@pytest.mark.parametrize("bad", [-1, True, "5", 1.5, 10 ** 13])
def test_token_limit_rejects_garbage(client, bad):
    _a, ah = _owner(client)
    uid, _h = _register(client, "u@x.io")
    assert client.post(f"/api/admin/users/{uid}/token-limit", json={"limit": bad}, headers=ah).status_code == 400


def test_token_limit_override_and_clear(client):
    from app.core.quota import check_quota
    _a, ah = _owner(client)
    uid, _h = _register(client, "u@x.io")
    assert client.post(f"/api/admin/users/{uid}/token-limit", json={"limit": 1234}, headers=ah).status_code == 200
    assert check_quota(_db(), uid)[2] == 1234
    assert client.post(f"/api/admin/users/{uid}/token-limit", json={"limit": 0}, headers=ah).status_code == 200
    assert check_quota(_db(), uid) == (True, 0, 0)                      # 0 = unlimited
    assert client.post(f"/api/admin/users/{uid}/token-limit", json={"limit": None}, headers=ah).status_code == 200
    assert check_quota(_db(), uid)[2] == 300_000                        # back to the free plan's limit


def test_disable_blocks_the_user_and_enable_restores_and_admins_cannot_be_disabled(client):
    admin, ah = _owner(client)
    uid, uh = _register(client, "u@x.io")
    assert client.post(f"/api/admin/users/{uid}/disable", json={"disabled": "yes"}, headers=ah).status_code == 400
    assert client.post(f"/api/admin/users/{uid}/disable", json={"disabled": True}, headers=ah).status_code == 200
    assert client.get("/api/auth/me", headers=uh).status_code == 401
    assert client.post(f"/api/admin/users/{uid}/disable", json={"disabled": False}, headers=ah).status_code == 200
    assert client.get("/api/auth/me", headers=uh).status_code == 200
    assert client.post(f"/api/admin/users/{admin}/disable", json={"disabled": True}, headers=ah).status_code == 400


def test_users_list_search_and_overview(client):
    _a, ah = _owner(client)
    _register(client, "alice@x.io")
    _register(client, "bob@x.io")
    allu = client.get("/api/admin/users", headers=ah).json()
    assert allu["total"] == 3 and {u["email"] for u in allu["users"]} == {OWNER, "alice@x.io", "bob@x.io"}
    assert [u["email"] for u in client.get("/api/admin/users?q=ali", headers=ah).json()["users"]] == ["alice@x.io"]
    assert client.get("/api/admin/users?q=%25", headers=ah).json()["total"] == 0       # a literal %, not "match everything"
    assert "password_hash" not in str(allu)
    ov = client.get("/api/admin/overview", headers=ah).json()
    assert ov["users"] == 3 and ov["verified"] == 1 and ov["by_plan"]["free"] == 3 and ov["list_price_mrr_usd"] == 0


# ------------------------------------------------------------------ plans and limits
def test_public_plan_list_hides_business_and_matches_the_enforced_limits(client):
    plans = client.get("/api/plans").json()["plans"]
    assert [p["id"] for p in plans] == ["free", "pro"]
    from app.core import plans as pl
    assert plans[1]["limits"] == pl.get_plan("pro")["limits"] and plans[0]["price_usd_month"] == 0


def test_my_plan_reports_limits_and_usage(client):
    uid, h = _register(client, "u@x.io")
    me = client.get("/api/me/plan", headers=h).json()
    assert me["plan"]["id"] == "free" and me["is_admin"] is False
    assert me["effective_limits"]["max_bots"] == 2 and me["usage"] == {"tokens_this_month": 0, "bots": 0, "models": 0, "api_keys": 0}


def test_plan_token_limits_and_admin_unlimited(client):
    from app.core.quota import check_quota
    admin, _ah = _owner(client)
    uid, _h = _register(client, "u@x.io")
    db = _db()
    assert check_quota(db, uid)[2] == 300_000
    db.set_user_plan(uid, "pro")
    assert check_quota(db, uid)[2] == 5_000_000
    assert check_quota(db, admin) == (True, 0, 0)


def test_free_plan_bot_limit_and_upgrade(client):
    admin, ah = _owner(client)
    uid, h = _register(client, "u@x.io")
    for _ in range(2):
        assert client.post("/api/agents/create", json={"name": "b"}, headers=h).status_code == 200
    r = client.post("/api/agents/create", json={"name": "b"}, headers=h)
    assert r.status_code == 403 and "Free plan allows 2 bots" in r.json()["detail"] and "Upgrade to Pro" in r.json()["detail"]
    client.post(f"/api/admin/users/{uid}/plan", json={"plan": "pro"}, headers=ah)
    assert client.post("/api/agents/create", json={"name": "b"}, headers=h).status_code == 200
    for _ in range(5):                                                   # the owner is never limited
        assert client.post("/api/agents/create", json={"name": "o"}, headers=ah).status_code == 200


def test_api_key_count_and_calls_are_capped_by_plan(client):
    admin, ah = _owner(client)
    uid, h = _register(client, "u@x.io")
    k = client.post("/api/keys/create", json={"calls_limit": 10 ** 9}, headers=h)
    assert k.status_code == 200 and k.json()["calls_limit"] == 100       # capped to the free plan
    r = client.post("/api/keys/create", json={}, headers=h)
    assert r.status_code == 403 and "1 API keys" in r.json()["detail"]
    client.post(f"/api/admin/users/{uid}/plan", json={"plan": "pro"}, headers=ah)
    k2 = client.post("/api/keys/create", json={"calls_limit": 10 ** 9}, headers=h)
    assert k2.status_code == 200 and k2.json()["calls_limit"] == 5_000
    assert client.post("/api/keys/create", json={"calls_limit": "oops"}, headers=h).json()["calls_limit"] == 5_000


def test_upload_size_is_capped_by_plan(client):
    admin, ah = _owner(client)
    uid, h = _register(client, "u@x.io")
    big = ("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(600_000))).encode()       # ~7 MB
    assert 5 * 1024 * 1024 < len(big) < 25 * 1024 * 1024
    r = client.post("/api/agent/upload", headers=h, files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413 and "your plan" in r.json()["detail"]
    client.post(f"/api/admin/users/{uid}/plan", json={"plan": "pro"}, headers=ah)
    assert client.post("/api/agent/upload", headers=h, files={"file": ("big.csv", big, "text/csv")}).status_code == 200


def test_custom_code_is_a_paid_feature_and_model_slots_are_counted(client, monkeypatch):
    from app.core import plans
    uid, _h = _register(client, "u@x.io")
    db = _db()
    assert "not part of the Free plan" in plans.gate_message(db, uid, "custom_code")
    assert plans.gate_message(db, uid, "model") is None
    monkeypatch.setattr(plans, "_count_models", lambda user_id: 3)
    assert "Free plan allows 3 saved models and you have 3" in plans.gate_message(db, uid, "model")
    db.set_user_plan(uid, "pro")
    assert plans.gate_message(db, uid, "custom_code") is None and plans.gate_message(db, uid, "model") is None
    assert plans.gate_message(db, "anonymous", "model") is None


def test_the_agent_refuses_a_third_bot_on_the_free_plan(client):
    from app.core.agent import GenesisAgent
    from app.core.agent_store import AgentStore, AgentConfig
    uid, _h = _register(client, "u@x.io")
    for i in range(2):
        cfg = AgentConfig(name=f"b{i}", system_prompt="x")
        cfg.author = uid
        AgentStore().create_agent(cfg)
    agent = GenesisAgent()
    agent.current_user_id = uid
    res = agent._do_create_bot("make me a bot for my shop")
    assert res.success is False and "Free plan allows 2 bots" in res.message


def test_the_admin_page_and_its_script_are_served_and_locked_down(client, monkeypatch):
    monkeypatch.chdir(os.path.join(os.path.dirname(__file__), ".."))      # pages are served relative to the project root
    page = client.get("/admin")
    assert page.status_code == 200 and "GenesisAI admin" in page.text
    assert "script-src 'self'" in page.text and "<script>" not in page.text      # CSP, and no inline script
    js = client.get("/static/admin.js")
    assert js.status_code == 200 and "innerHTML" not in js.text                 # server text is only ever set as text
