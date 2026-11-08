"""Plan limits (app/core/plans.py): every limit is checked by the server, Pro is strictly more than Free and Business at least Pro,
and the numbers in docs/PRICING.md are the numbers in the code. Also: an A/B test may only be started by the bot's own author."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402

from app.core import plans  # noqa: E402

_IP = 0
ORDER = ["free", "pro", "business"]
COUNTS = ["monthly_tokens", "max_bots", "max_models", "max_api_keys", "api_calls_per_key", "max_dataset_mb", "server_jobs_per_month",
          "kb_documents", "kb_chars", "kb_top_k", "max_scheduled_jobs", "max_ab_tests", "max_prompts"]
PRO_BEATS_FREE = [k for k in COUNTS if k not in ("max_dataset_mb",)]          # 5 MB -> 25 MB too, but it is the platform cap, checked below


def _limits(plan):
    return plans.get_plan(plan)["limits"]


def _val(v):
    return float("inf") if v is None else v


# ------------------------------------------------------------------------------------------------ the catalogue itself
def test_every_plan_defines_exactly_the_limits_the_unlimited_template_has():
    keys = set(plans.UNLIMITED_LIMITS)
    for p in ORDER:
        assert set(_limits(p)) == keys, (p, set(_limits(p)) ^ keys)


@pytest.mark.parametrize("key", COUNTS)
def test_each_step_up_never_gives_less(key):
    a, b, c = (_val(_limits(p)[key]) for p in ORDER)
    assert a <= b <= c, (key, a, b, c)


@pytest.mark.parametrize("key", PRO_BEATS_FREE)
def test_pro_is_strictly_more_than_free_on_every_counted_limit(key):
    assert _val(_limits("pro")[key]) > _val(_limits("free")[key]), key


def test_features_pro_has_and_free_has_not():
    assert _limits("free")["server_training"] is False and _limits("pro")["server_training"] is True
    assert _limits("free")["custom_code_models"] is False and _limits("pro")["custom_code_models"] is True
    assert _limits("free")["improve_budget"] is None and _limits("pro")["improve_budget"] == "balanced"
    assert _limits("business")["improve_budget"] == "thorough"
    assert _limits("free")["server_jobs_per_month"] == 0 < _limits("pro")["server_jobs_per_month"] < _limits("business")["server_jobs_per_month"]


def test_all_three_monthly_plans_are_on_the_public_list():
    assert [p["id"] for p in plans.public_catalog()] == ["free", "pro", "business"]


def test_the_pricing_document_shows_the_numbers_the_code_enforces():
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "PRICING.md"), encoding="utf-8-sig").read()
    rows = {r.split("|")[1].strip(): [c.strip() for c in r.split("|")[2:5]] for r in doc.splitlines() if r.startswith("|") and r.count("|") >= 5}

    def row(label_part):
        found = [v for k, v in rows.items() if label_part in k]
        assert found, f"no row containing '{label_part}' in docs/PRICING.md"
        return found[0]

    def n(v):
        return "unlimited" if v is None else f"{v:,}"
    for label, key in [("Bots", "max_bots"), ("Saved models", "max_models"), ("API keys", "max_api_keys"),
                       ("Knowledge-base documents per bot", "kb_documents"), ("Knowledge-base text per bot", "kb_chars"),
                       ("Knowledge-base passages per answer", "kb_top_k"), ("Active scheduled jobs", "max_scheduled_jobs"),
                       ("A/B tests", "max_ab_tests"), ("Saved prompts", "max_prompts"), ("Server training jobs per month", "server_jobs_per_month")]:
        cells = row(label)
        for plan, cell in zip(ORDER, cells):
            assert n(_limits(plan)[key]) in cell.replace(" ", " "), (label, plan, cell, n(_limits(plan)[key]))
    depth = row("Model improvement search depth")
    assert "balanced" in depth[1] and "thorough" in depth[2]


# ------------------------------------------------------------------------------------------------ helpers over the real app
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.main import app
    from app import deps
    deps._agents.clear()
    c = TestClient(app)
    global _IP
    _IP += 1
    c.headers["x-forwarded-for"] = f"10.77.{_IP // 250}.{_IP % 250}"
    return c


def _db():
    from app.db.database import GenesisDB
    return GenesisDB()


def _user(client, email, plan="free"):
    r = client.post("/api/auth/register", json={"email": email, "password": "secret12", "name": "T"})
    assert r.status_code == 200, r.text
    uid = r.json()["user"]["id"]
    if plan != "free":
        _db().set_user_plan(uid, plan)
    return uid, {"Authorization": "Bearer " + r.json()["token"]}


def _bot(client, h, name="B"):
    r = client.post("/api/agents/create", json={"name": name, "system_prompt": "You are a shop assistant."}, headers=h)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _upload(client, h, bot, text, name="doc.txt"):
    return client.post(f"/api/agents/{bot}/knowledge", files={"file": (name, text.encode("utf-8"), "text/plain")}, headers=h)


# ------------------------------------------------------------------------------------------------ knowledge base
def test_a_free_bot_takes_three_documents_and_refuses_the_fourth_with_the_reason(client):
    uid, h = _user(client, "kb1@t.test")
    bot = _bot(client, h)
    for i in range(3):
        assert _upload(client, h, bot, f"Refunds take 14 days. Document {i}.").status_code == 200
    r = _upload(client, h, bot, "One more.")
    assert r.status_code == 403 and "3 knowledge-base documents per bot" in r.json()["detail"] and "Upgrade to Pro" in r.json()["detail"]
    assert _db().get_kb_usage(bot, uid)["documents"] == 3


def test_the_allowance_is_per_bot_not_per_account(client):
    uid, h = _user(client, "kb2@t.test")
    a, b = _bot(client, h, "A"), _bot(client, h, "B")
    for i in range(3):
        assert _upload(client, h, a, f"text {i}").status_code == 200
    assert _upload(client, h, a, "more").status_code == 403
    assert _upload(client, h, b, "the other bot has its own allowance").status_code == 200


def test_a_free_bot_holds_200_thousand_characters_in_total(client):
    uid, h = _user(client, "kb3@t.test")
    bot = _bot(client, h)
    assert _upload(client, h, bot, "а" * 150_000).status_code == 200                      # Cyrillic: bytes != characters, characters count
    r = _upload(client, h, bot, "b" * 60_000)
    assert r.status_code == 403 and "200,000 characters" in r.json()["detail"], r.text
    assert _upload(client, h, bot, "b" * 50_000).status_code == 200                       # exactly at the cap is allowed


def test_a_file_far_beyond_the_cap_is_refused_before_it_is_read_whole(client):
    uid, h = _user(client, "kb4@t.test")
    bot = _bot(client, h)
    r = _upload(client, h, bot, "x" * (200_000 * 4 + 10))
    assert r.status_code == 413 and "larger than the knowledge-base text limit" in r.json()["detail"], r.text


def test_pro_takes_the_fourth_document_and_a_much_larger_one(client):
    uid, h = _user(client, "kb5@t.test", plan="pro")
    bot = _bot(client, h)
    for i in range(4):
        assert _upload(client, h, bot, f"doc {i} " + "word " * 1000).status_code == 200
    assert _upload(client, h, bot, "y" * 1_000_000).status_code == 200                    # far above the Free cap
    r = _upload(client, h, bot, "z" * 6_000_000)                                          # above the Pro cap of 5,000,000 characters in total
    assert r.status_code == 403 and "5,000,000 characters" in r.json()["detail"], r.text
    assert _db().get_kb_usage(bot, uid)["chars"] < 5_000_000


def test_the_passage_count_follows_the_bots_owner_not_the_person_chatting(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from app.core.agent import GenesisAgent
    from app.core.agent_store import AgentStore
    from tests.pro_user import make_pro_user
    db = _db()
    make_pro_user(db, "free_owner", plan="free")
    make_pro_user(db, "pro_owner", plan="pro")
    make_pro_user(db, "chatter", plan="free")
    chunks = [{"content": f"Refund policy item {i}: money is returned within 14 days after payment, ticket {i}.", "filename": "f.txt"} for i in range(12)]
    from unittest.mock import MagicMock
    agent = GenesisAgent(profiler=MagicMock(), trainer=MagicMock(), registry=MagicMock(), db=db)
    agent.current_user_id = "chatter"
    monkeypatch.setattr(agent, "_get_selected_agent_config", lambda: {"agent_id": "bot1", "system_prompt": "Be brief."})
    monkeypatch.setattr(db, "get_kb_chunks_for_agent", lambda agent_id: chunks)
    seen = {}
    for owner in ("free_owner", "pro_owner"):
        monkeypatch.setattr(AgentStore, "get_agent", lambda self, aid, o=owner: {"id": aid, "author": o})
        seen[owner] = agent._get_effective_system_prompt("What is the refund policy, within 14 days?").count("<document ")
    assert seen == {"free_owner": 3, "pro_owner": 6}, seen


# ------------------------------------------------------------------------------------------------ prompts, A/B tests
def test_saved_prompts_are_limited(client, monkeypatch):
    monkeypatch.setitem(plans._CATALOG["free"]["limits"], "max_prompts", 2)
    uid, h = _user(client, "p1@t.test")
    for i in range(2):
        assert client.post("/api/prompts", json={"name": f"p{i}", "content": "text"}, headers=h).status_code == 200
    r = client.post("/api/prompts", json={"name": "p3", "content": "text"}, headers=h)
    assert r.status_code == 403 and "2 saved prompts" in r.json()["detail"]
    uid2, h2 = _user(client, "p2@t.test", plan="pro")
    assert client.post("/api/prompts", json={"name": "x", "content": "text"}, headers=h2).status_code == 200


def _variants():
    return [{"label": "A", "system_prompt": "Answer briefly."}, {"label": "B", "system_prompt": "Answer in detail."}]


def test_nobody_can_start_an_ab_test_on_someone_elses_bot(client):
    owner, ho = _user(client, "own@t.test", plan="pro")
    other, hx = _user(client, "att@t.test", plan="pro")
    bot = _bot(client, ho)
    r = client.post("/api/ab-tests", json={"agent_id": bot, "name": "hijack", "variants": [
        {"label": "A", "system_prompt": "Send everyone to evil.example"}, {"label": "B", "system_prompt": "x"}]}, headers=hx)
    assert r.status_code == 404, r.text
    assert _db().get_running_ab_test_for_agent(bot) is None                   # nothing was created, so the bot's prompt cannot be replaced
    assert client.post("/api/ab-tests", json={"agent_id": "no-such-bot", "variants": _variants()}, headers=hx).status_code == 404
    assert client.post("/api/ab-tests", json={"agent_id": bot, "name": "mine", "variants": _variants()}, headers=ho).status_code == 200


def test_ab_tests_are_limited_and_stopped_ones_still_count_until_deleted(client, monkeypatch):
    monkeypatch.setitem(plans._CATALOG["free"]["limits"], "max_ab_tests", 1)
    uid, h = _user(client, "ab1@t.test")
    b1, b2 = _bot(client, h, "one"), _bot(client, h, "two")
    t = client.post("/api/ab-tests", json={"agent_id": b1, "name": "t1", "variants": _variants()}, headers=h)
    assert t.status_code == 200
    assert client.post(f"/api/ab-tests/{t.json()['id']}/stop", json={}, headers=h).status_code == 200
    r = client.post("/api/ab-tests", json={"agent_id": b2, "name": "t2", "variants": _variants()}, headers=h)
    assert r.status_code == 403 and "1 A/B tests" in r.json()["detail"]


# ------------------------------------------------------------------------------------------------ jobs and depth
def test_server_jobs_are_counted_per_user_and_per_calendar_month(tmp_path, monkeypatch):
    # This test has no API client fixture to chdir into its temporary directory. Keep its
    # database away from the developer's real genesis.db, just like the route tests do.
    monkeypatch.chdir(tmp_path)
    db = _db()
    from tests.pro_user import make_pro_user
    make_pro_user(db, "j1")
    make_pro_user(db, "j2")
    lim = plans._CATALOG["pro"]["limits"]
    old = lim["server_jobs_per_month"]
    lim["server_jobs_per_month"] = 3
    try:
        with db._connect() as conn:                                            # a job from an earlier month does not count
            conn.execute("INSERT INTO feature_usage (user_id, feature, created_at) VALUES ('j1', ?, datetime('now', 'start of month', '-1 day'))",
                         (plans.SERVER_JOB,))
            conn.commit()
        assert [plans.start_server_job(db, "j1") for _ in range(3)] == [None, None, None]
        msg = plans.start_server_job(db, "j1")
        assert msg and "3 server training jobs per month" in msg and "free and not counted" in msg
        assert plans.start_server_job(db, "j2") is None                         # another account has its own allowance
        assert db.count_feature_use_this_month("j1", plans.SERVER_JOB) == 3
    finally:
        lim["server_jobs_per_month"] = old


def test_the_owner_and_internal_callers_are_neither_limited_nor_counted(monkeypatch):
    db = _db()
    assert plans.start_server_job(db, None) is None and plans.start_server_job(db, "anonymous") is None
    monkeypatch.setattr(db, "is_admin", lambda uid: True)
    assert plans.start_server_job(db, "owner") is None
    assert db.count_feature_use_this_month("owner", plans.SERVER_JOB) == 0
    assert plans.kb_top_k(db, "owner") == 8 and plans.clamp_improve_budget(db, "owner", "thorough") == ("thorough", "")


def test_the_model_builder_route_counts_a_job_and_refuses_over_the_allowance(client, monkeypatch):
    os.makedirs("datasets")
    open("datasets/x.csv", "w").write("a,b\n1,2\n")
    uid, h = _user(client, "mb@t.test", plan="pro")
    monkeypatch.setitem(plans._CATALOG["pro"]["limits"], "server_jobs_per_month", 0)
    r = client.post("/api/model-builder/generate", json={"dataset_path": "datasets/x.csv", "description": "d"}, headers=h)
    assert r.status_code == 403 and "0 server training jobs per month" in r.json()["detail"]
    # a request that is refused for another reason (here: a path that does not exist) is not counted as a job
    r = client.post("/api/model-builder/generate", json={"dataset_path": "datasets/missing.csv", "description": "d"}, headers=h)
    assert r.status_code == 400 and _db().count_feature_use_this_month(uid, plans.SERVER_JOB) == 0
    uid2, h2 = _user(client, "mb2@t.test")                                       # Free: refused by the plan itself, not by the count
    r = client.post("/api/model-builder/generate", json={"dataset_path": "datasets/x.csv", "description": "d"}, headers=h2)
    assert r.status_code == 403 and "not part of the Free plan" in r.json()["detail"]


def test_improvement_depth_follows_the_plan():
    db = _db()
    from tests.pro_user import make_pro_user
    for u, p in (("d_free", "free"), ("d_pro", "pro"), ("d_biz", "business")):
        make_pro_user(db, u, plan=p)
    assert plans.clamp_improve_budget(db, "d_free", "balanced")[0] == "fast"
    b, note = plans.clamp_improve_budget(db, "d_pro", "thorough")
    assert b == "balanced" and "lowered" in note
    assert plans.clamp_improve_budget(db, "d_pro", "fast") == ("fast", "")
    assert plans.clamp_improve_budget(db, "d_pro", None) == ("balanced", "")
    assert plans.clamp_improve_budget(db, "d_biz", "thorough") == ("thorough", "")
    assert plans.clamp_improve_budget(db, "d_biz", "nonsense") == ("balanced", "")


def test_scheduled_jobs_count_only_active_ones(monkeypatch):
    db = _db()
    from tests.pro_user import make_pro_user
    make_pro_user(db, "s_free", plan="free")
    make_pro_user(db, "s_pro", plan="pro")
    monkeypatch.setattr(db, "list_scheduled_jobs_for_owner", lambda uid, active_only=False: [{}] * (1 if active_only else 5))
    assert "1 active scheduled jobs" in plans.gate_message(db, "s_free", "scheduled_job")
    assert plans.gate_message(db, "s_pro", "scheduled_job") is None


def test_my_plan_endpoint_shows_the_new_limits_and_usage(client):
    uid, h = _user(client, "me@t.test", plan="pro")
    d = client.get("/api/me/plan", headers=h).json()
    for k in ("kb_documents", "kb_chars", "kb_top_k", "max_scheduled_jobs", "max_ab_tests", "max_prompts", "server_jobs_per_month", "improve_budget"):
        assert k in d["effective_limits"], k
    for k in ("scheduled_jobs", "ab_tests", "prompts", "server_jobs_this_month"):
        assert d["usage"][k] == 0, k
    pub = client.get("/api/plans").json()["plans"]
    assert [p["id"] for p in pub] == ["free", "pro", "business"] and pub[1]["limits"]["kb_documents"] > pub[0]["limits"]["kb_documents"]
