"""Tests for app/core/agent_browser.py and the agent_browser_sessions DB table.

What's covered here (no Docker daemon needed - this sandbox doesn't have one, and these
tests don't try to fake having one): the pure cost math, the action-count/expiry enforcement
logic, the file-based "recover next action index from disk" logic (real tempdir, real files),
and the billing DB methods (real sqlite3, same approach as test_charge_for_usage_race.py).

What's NOT covered here, and can't be from this sandbox: actually starting a container,
actually running docker/agent_browser_harness.py against a real Playwright browser, or the
file-bridge protocol's behavior end-to-end. See agent_browser.py's module docstring - this
needs a real Docker daemon to verify before trusting it with real users.
"""
import json
import os
import sys
import tempfile
import types
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from app.core.agent_browser import (
    AgentBrowserSession, AgentBrowserError, compute_browser_cost_cents, ALLOWED_ACTION_TYPES,
)
from app.db.database import GenesisDB


def setup_module(module):
    settings.agent_browser_price_cents_per_minute = 5
    settings.agent_browser_max_session_minutes = 10
    settings.agent_browser_max_actions_per_session = 60


# ---------------------------------------------------------------- compute_browser_cost_cents ----

def test_cost_basic_math():
    # 1 minute @ 5c/min = 5 cents
    assert compute_browser_cost_cents(60) == 5
    # 2.5 minutes @ 5c/min = 12.5 -> ceil -> 13
    assert compute_browser_cost_cents(150) == 13


def test_cost_rounds_up_never_down():
    # A few seconds should never cost 0 - minimum 1 cent
    assert compute_browser_cost_cents(1) == 1
    assert compute_browser_cost_cents(0) == 1


def test_cost_never_below_minimum():
    assert compute_browser_cost_cents(0.001) == 1


# ---------------------------------------------------------------- session-limit logic ----
# These construct an AgentBrowserSession directly (bypassing .create()/.resume(), which need
# a real Docker daemon) with a fake container object, to test perform_action's OWN guard
# logic (action type allowlist, expiry, action cap) in isolation from Docker/Playwright.

def _make_session(workdir, created_at_offset_seconds=0):
    import time
    fake_container = MagicMock()
    return AgentBrowserSession(
        session_id="agb_test", workdir=workdir, container=fake_container,
        user_id="u1", created_at=time.time() - created_at_offset_seconds)


def test_unknown_action_type_is_rejected_before_touching_the_filesystem():
    with tempfile.TemporaryDirectory() as workdir:
        session = _make_session(workdir)
        try:
            session.perform_action("delete_system32", {})
            assert False, "should have raised"
        except AgentBrowserError as e:
            assert "Unknown action type" in str(e)
        # Nothing should have been written - rejected before any file I/O.
        assert os.listdir(workdir) == []


def test_all_documented_action_types_are_in_the_allowlist():
    expected = {"navigate", "click", "type", "press_key", "scroll", "extract_text", "screenshot"}
    assert ALLOWED_ACTION_TYPES == expected


def test_expired_session_rejects_new_actions():
    with tempfile.TemporaryDirectory() as workdir:
        # Created 11 minutes ago, cap is 10 minutes -> expired.
        session = _make_session(workdir, created_at_offset_seconds=11 * 60)
        try:
            session.perform_action("navigate", {"url": "https://example.com"})
            assert False, "should have raised"
        except AgentBrowserError as e:
            assert "exceeded" in str(e).lower() or "limit" in str(e).lower()


def test_session_at_action_cap_rejects_further_actions():
    with tempfile.TemporaryDirectory() as workdir:
        session = _make_session(workdir)
        session._next_n = settings.agent_browser_max_actions_per_session  # simulate having used them all
        try:
            session.perform_action("navigate", {"url": "https://example.com"})
            assert False, "should have raised"
        except AgentBrowserError as e:
            assert "action limit" in str(e).lower()


def test_is_expired_property():
    with tempfile.TemporaryDirectory() as workdir:
        fresh = _make_session(workdir, created_at_offset_seconds=5)
        assert fresh.is_expired is False
        stale = _make_session(workdir, created_at_offset_seconds=999999)
        assert stale.is_expired is True


# ---------------------------------------------------------------- file-based recovery ----

def test_count_existing_actions_recovers_correctly_from_disk():
    """This is what makes a session resumable across separate HTTP requests / worker
    processes - _next_n must be reconstructed purely from what's on disk, never from
    in-memory state that wouldn't exist in a fresh process."""
    with tempfile.TemporaryDirectory() as workdir:
        for n in range(3):
            with open(os.path.join(workdir, f"action_{n}.json"), "w") as f:
                json.dump({"type": "navigate", "params": {}}, f)
        session = _make_session(workdir)
        assert session._next_n == 3


def test_count_existing_actions_is_zero_for_a_brand_new_session():
    with tempfile.TemporaryDirectory() as workdir:
        session = _make_session(workdir)
        assert session._next_n == 0


def test_perform_action_writes_action_file_and_reads_result_file():
    """Exercises the actual file-bridge protocol (write action_N.json, poll for
    result_N.json) without a real harness process - a background thread plays the harness's
    role by writing the result file shortly after the action file appears, proving the
    host-side polling/matching logic itself is correct."""
    import threading
    with tempfile.TemporaryDirectory() as workdir:
        session = _make_session(workdir)

        def fake_harness():
            import time as _t
            action_path = os.path.join(workdir, "action_0.json")
            while not os.path.exists(action_path):
                _t.sleep(0.02)
            with open(action_path) as f:
                action = json.load(f)
            assert action == {"type": "navigate", "params": {"url": "https://example.com"}}
            with open(os.path.join(workdir, "result_0.json"), "w") as f:
                json.dump({"success": True, "data": {"url": "https://example.com", "title": "Example"},
                          "error": None}, f)

        t = threading.Thread(target=fake_harness)
        t.start()
        result = session.perform_action("navigate", {"url": "https://example.com"})
        t.join(timeout=5)

        assert result["success"] is True
        assert result["data"]["title"] == "Example"
        assert session._next_n == 1


def test_perform_action_screenshot_reads_and_base64_encodes_the_png():
    import threading
    with tempfile.TemporaryDirectory() as workdir:
        session = _make_session(workdir)
        fake_png_bytes = b"\x89PNG\r\n\x1a\nFAKE_PNG_DATA"

        def fake_harness():
            import time as _t
            action_path = os.path.join(workdir, "action_0.json")
            while not os.path.exists(action_path):
                _t.sleep(0.02)
            with open(os.path.join(workdir, "screenshot_0.png"), "wb") as f:
                f.write(fake_png_bytes)
            with open(os.path.join(workdir, "result_0.json"), "w") as f:
                json.dump({"success": True, "data": {"screenshot_file": "screenshot_0.png",
                          "url": "https://example.com"}, "error": None}, f)

        t = threading.Thread(target=fake_harness)
        t.start()
        result = session.perform_action("screenshot", {})
        t.join(timeout=5)

        import base64
        assert result["success"] is True
        assert base64.b64decode(result["data"]["screenshot_base64"]) == fake_png_bytes


# ---------------------------------------------------------------- DB layer ----

def _fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return GenesisDB(db_path=path), path


def test_db_session_lifecycle_round_trip():
    db, path = _fresh_db()
    try:
        user = db.create_user("browseruser@example.com", "hash")["id"]
        db.create_agent_browser_session_record("agb_abc123", user)

        record = db.get_agent_browser_session("agb_abc123")
        assert record["status"] == "active"
        assert record["action_count"] == 0
        assert record["charged_cents"] is None

        db.record_agent_browser_action("agb_abc123")
        db.record_agent_browser_action("agb_abc123")
        record = db.get_agent_browser_session("agb_abc123")
        assert record["action_count"] == 2

        db.close_agent_browser_session_record("agb_abc123", duration_seconds=125.4, charged_cents=11)
        record = db.get_agent_browser_session("agb_abc123")
        assert record["status"] == "closed"
        assert record["duration_seconds"] == 125.4
        assert record["charged_cents"] == 11
        assert record["ended_at"] is not None
    finally:
        os.unlink(path)


def test_db_list_sessions_only_returns_that_users_own():
    db, path = _fresh_db()
    try:
        u1 = db.create_user("owner1@example.com", "hash")["id"]
        u2 = db.create_user("owner2@example.com", "hash")["id"]
        db.create_agent_browser_session_record("agb_mine", u1)
        db.create_agent_browser_session_record("agb_not_mine", u2)

        sessions = db.list_agent_browser_sessions(u1)
        assert len(sessions) == 1
        assert sessions[0]["id"] == "agb_mine"
    finally:
        os.unlink(path)


def test_db_get_nonexistent_session_returns_none():
    db, path = _fresh_db()
    try:
        assert db.get_agent_browser_session("agb_does_not_exist") is None
    finally:
        os.unlink(path)


def test_full_billing_flow_charges_the_right_amount():
    """Mirrors exactly what POST /api/agent-browser/sessions/{id}/close does."""
    db, path = _fresh_db()
    try:
        user = db.create_user("payer@example.com", "hash")["id"]
        db.add_transaction(user, "test_topup", 1000, "seed")
        db.create_agent_browser_session_record("agb_billed", user)

        duration_seconds = 185.0  # just over 3 minutes
        cost_cents = compute_browser_cost_cents(duration_seconds)  # ceil(185/60*5) = ceil(15.4) = 16
        assert cost_cents == 16
        charge = db.charge_for_usage(user, cost_cents, "Agent browser session (185s)", related_id="agb_billed")
        assert charge["ok"] is True
        db.close_agent_browser_session_record("agb_billed", duration_seconds, cost_cents)

        assert db.get_balance(user) == 1000 - 16
        record = db.get_agent_browser_session("agb_billed")
        assert record["charged_cents"] == 16
    finally:
        os.unlink(path)


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
                import traceback; traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
