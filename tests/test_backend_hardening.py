import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

import pytest
import app.main as main_module
from fastapi.testclient import TestClient
from app.core.auth import create_token, decode_token
from app.core.paths import is_owned_path
from app.core.handoff import is_plausible_webhook_url
from app.core.rate_limit import RateLimitMiddleware
from app.db.database import GenesisDB
from app.db.reliability import UsageRejected


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = GenesisDB()
    a = db.create_user("a@example.test", "unused")["id"]
    b = db.create_user("b@example.test", "unused")["id"]
    return db, a, b


def test_owner_paths_reject_foreign_files_and_symlinks(accounts, tmp_path):
    _, a, b = accounts
    folder = tmp_path / "datasets" / "uploads" / a
    folder.mkdir(parents=True)
    file = folder / "data.csv"
    file.write_text("x,y\n1,2")
    assert is_owned_path(str(file), a)
    assert not is_owned_path(str(file), b)
    other = tmp_path / "datasets" / "uploads" / b
    other.mkdir()
    link = other / "alias.csv"
    link.symlink_to(file)
    assert not is_owned_path(str(link), b)


def test_anonymous_llm_endpoints_are_rejected(accounts):
    client = TestClient(main_module.app)
    for endpoint in ("/api/bot-builder/chat", "/api/bot-builder/generate", "/api/bot-builder/revise", "/api/pipeline/suggest"):
        with patch("app.core.groq_client.call_llm") as llm:
            assert client.post(endpoint, json={"description": "test", "message": "test"}).status_code == 401
            llm.assert_not_called()


def test_selected_bot_uses_server_permissions_and_price(accounts):
    from fastapi import HTTPException
    from app.core.agent_store import AgentStore, AgentConfig
    from app.core.selection import resolve_selection
    db, owner, buyer = accounts
    bot = AgentConfig("Private", "Server prompt")
    bot.author = owner
    store = AgentStore(db=db)
    store.create_agent(bot)
    with pytest.raises(HTTPException) as denied:
        resolve_selection(db, buyer, {"agent_id": bot.id})
    assert denied.value.status_code == 404
    store.publish_agent(bot.id, 0.10, owner_user_id=owner)
    selected = resolve_selection(db, buyer, {"agent_id": bot.id, "price_per_call": 0,
                                           "author": buyer, "system_prompt": "Forged"})
    assert selected["price_per_call"] == 0.10
    assert selected["author"] == owner
    assert selected["system_prompt"] == "Server prompt"
    db.set_user_disabled(owner, True)
    with pytest.raises(HTTPException):
        resolve_selection(db, buyer, {"agent_id": bot.id})


def test_external_bot_cannot_change_owner_capabilities(accounts):
    from app.core.agent import GenesisAgent
    db, owner, _ = accounts
    agent = GenesisAgent(db=db, agent_override={"external_channel": "telegram", "allow_management": False})
    agent.current_user_id = owner
    with patch("app.core.agent_store.AgentStore.get_agent") as lookup:
        assert not agent._do_add_skill("owner-bot", {"description": "change capabilities"}).success
        lookup.assert_not_called()


def test_foreign_model_paths_and_orphan_chats_are_rejected(accounts, tmp_path):
    db, owner, other = accounts
    folder = tmp_path / "models" / owner
    folder.mkdir(parents=True)
    model = folder / "model.pkl"
    model.write_bytes(b"test")
    assert is_owned_path(str(model), owner)
    assert not is_owned_path(str(model), other)
    db.create_chat("orphan")
    with pytest.raises(PermissionError):
        db.create_chat("orphan", user_id=owner)


def test_disabled_and_missing_accounts_cannot_authenticate(accounts):
    db, a, _ = accounts
    token = create_token(a)
    key = db.create_api_key(a)["api_key"]
    from app.core.secrets_store import key_digest
    assert db.validate_api_key(key_digest(key)) is None
    assert decode_token(token) == a
    db.set_user_disabled(a, True)
    assert decode_token(token) is None
    assert db.validate_api_key(key) is None
    assert not db.use_api_key(key, "model")
    assert decode_token(create_token("deleted")) is None


def test_auth_storage_failure_is_not_an_allowed_session(accounts):
    from fastapi import HTTPException
    _, a, _ = accounts
    with patch.object(GenesisDB, "get_session_state", side_effect=OSError("unavailable")):
        with pytest.raises(HTTPException) as caught:
            decode_token(create_token(a))
    assert caught.value.status_code == 503


def test_exhausted_key_does_not_charge_or_credit(accounts):
    db, a, b = accounts
    db.add_transaction(a, "topup", 100)
    key = db.create_api_key(a, calls_limit=0)["api_key"]
    with pytest.raises(UsageRejected) as caught:
        db.reserve_usage(a, 10, author_id=b, api_key=key)
    assert caught.value.status == 429
    assert db.get_balance(a) == 100
    assert db.get_balance(b) == 0


def test_usage_is_idempotent_and_refund_is_atomic(accounts):
    db, a, b = accounts
    db.add_transaction(a, "topup", 100)
    key = db.create_api_key(a)["api_key"]
    op = db.reserve_usage(a, 10, author_id=b, api_key=key, operation_id="one")
    assert db.get_balance(a) == 90 and db.get_balance(b) == 0
    db.finish_usage(op["id"], refund=True)
    assert db.get_balance(a) == 100 and db.get_balance(b) == 0
    assert db.validate_api_key(key)["calls_used"] == 0
    assert not db.finish_usage(op["id"], refund=True)
    db.reserve_usage(a, 10, author_id=b, operation_id="two")
    db.finish_usage("two", result={"prediction": [1]})
    replay = db.reserve_usage(a, 10, author_id=b, operation_id="two")
    assert replay["replayed"] and replay["result"] == {"prediction": [1]}
    assert db.get_balance(a) == 90 and db.get_balance(b) == 8


def test_monthly_limit_is_atomic(accounts):
    db, a, _ = accounts
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: db.claim_monthly_use(a, "train", 1), range(8)))
    assert sum(results) == 1
    assert db.count_feature_use_this_month(a, "train") == 1


def test_chat_pins_selection_and_replays_after_spending_last_balance(accounts):
    from app.routers.chat_agent import _agent_message_sync
    from app.api.contracts import MessageRequest
    from app.core.agent import AgentResponse
    from unittest.mock import Mock
    db, buyer, author = accounts
    db.add_transaction(buyer, "topup", 10)
    selected = {"agent_id": "paid", "model": "openai/gpt-oss-120b", "author": author,
                "price_per_call": 0.10, "system_prompt": "Pinned"}
    agent = Mock()
    def process(*args, **kwargs):
        assert agent.agent_override == selected
        return AgentResponse("Done")
    agent.process.side_effect = process
    req = MessageRequest(message="Hello", conversation_id="paid-chat", request_id="retry-one")
    with patch.object(GenesisDB, "get_selected_model", return_value=selected), \
         patch("app.routers.chat_agent.get_agent", return_value=agent):
        first = _agent_message_sync(req, buyer)
        assert db.get_balance(buyer) == 0
        assert _agent_message_sync(req, buyer) == first
    agent.process.assert_called_once()


def test_expired_final_outgoing_attempt_becomes_terminal(accounts):
    db, owner, _ = accounts
    event_id = db.enqueue_outgoing(owner, "telegram", {"text": "test"})
    with db.write_transaction() as conn:
        conn.execute("UPDATE outgoing_events SET attempts=5,lease_until=1 WHERE id=?", (event_id,))
    assert db.claim_outgoing() is None
    with db._connect() as conn:
        assert conn.execute("SELECT status FROM outgoing_events WHERE id=?", (event_id,)).fetchone()[0] == "failed"


def test_stale_marketplace_instances_do_not_drop_publications(accounts):
    from app.core.marketplace import Marketplace
    _, a, _ = accounts
    first, second = Marketplace(), Marketplace()
    one = first.publish(a, "one", "One", "", "classification", "test", {}, 0.01)
    two = second.publish(a, "two", "Two", "", "classification", "test", {}, 0.01)
    assert {one, two} <= Marketplace().listings.keys()
    active,stale = Marketplace(),Marketplace()
    third = active.publish(a,"three","Three","","classification","test",{},0.01)
    stale.deactivate_listing(one)
    assert third in Marketplace().listings
    assert Marketplace().listings[one]["active"] is False


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://[::1]/", "http://10.0.0.1/", "http://x:password@example.com/", "http://example.com:8080/"])
def test_webhook_rejects_internal_and_ambiguous_urls(url):
    assert not is_plausible_webhook_url(url)


def test_invalid_tokens_do_not_rotate_rate_limit_bucket():
    from starlette.requests import Request
    middleware = object.__new__(RateLimitMiddleware)
    def request(token):
        return Request({"type": "http", "headers": [(b"authorization", ("Bearer " + token).encode()), (b"x-forwarded-for", b"1.1.1.1")], "client": ("192.0.2.2", 1234)})
    assert middleware._caller_key(request("invalid-one")) == middleware._caller_key(request("invalid-two"))
    assert middleware._client_ip(request("test")) == "192.0.2.2"
