"""
Tests for the GenesisDB Telegram link methods (upsert/get/delete, event logging). Uses a
throwaway SQLite file per test, never touches the real genesis.db.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.db.database import GenesisDB


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    d = GenesisDB(db_path=path)
    yield d
    os.remove(path)
    for ext in ("-wal", "-shm"):
        try:
            os.remove(path + ext)
        except FileNotFoundError:
            pass


class TestTelegramLinkStorage:
    def test_upsert_creates_new_link(self, db):
        link_id = db.upsert_telegram_link("agent_1", "user_1", "123:ABC", 999, "mybot", "secret1", "connected")
        link = db.get_telegram_link("agent_1", "user_1")
        assert link is not None
        assert link["bot_username"] == "mybot"
        assert link["status"] == "connected"

    def test_token_is_masked_never_returned_in_full(self, db):
        db.upsert_telegram_link("agent_1", "user_1", "123456789:AAAA-verysecret-token", 999, "mybot", "secret1", "connected")
        link = db.get_telegram_link("agent_1", "user_1")
        assert "verysecret" not in link["token_masked"]
        assert link["token_masked"].startswith("...")

    def test_reconnecting_same_agent_updates_not_duplicates(self, db):
        db.upsert_telegram_link("agent_1", "user_1", "old_token", 111, "oldname", "old_secret", "connected")
        db.upsert_telegram_link("agent_1", "user_1", "new_token", 222, "newname", "new_secret", "connected")
        link = db.get_telegram_link("agent_1", "user_1")
        assert link["bot_id"] == 222
        assert link["bot_username"] == "newname"
        # confirm no duplicate row exists
        with db._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM telegram_links WHERE agent_id = ?", ("agent_1",)).fetchone()[0]
        assert count == 1

    def test_get_link_ownership_checked(self, db):
        db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "sec1", "connected")
        assert db.get_telegram_link("agent_1", "someone_else") is None
        assert db.get_telegram_link("agent_1", "owner_user") is not None

    def test_get_link_by_secret_not_ownership_checked(self, db):
        """Deliberately different - this is how the webhook receiver (no user session at
        all, just Telegram's own request) identifies which bot/owner an update is for."""
        db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "the_secret", "connected")
        link = db.get_telegram_link_by_secret("the_secret")
        assert link is not None
        assert link["agent_id"] == "agent_1"
        assert link["user_id"] == "owner_user"
        assert link["bot_token"] == "tok"  # unmasked here - this is the internal lookup used to actually call Telegram

    def test_get_link_by_wrong_secret_returns_none(self, db):
        db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "the_real_secret", "connected")
        assert db.get_telegram_link_by_secret("wrong_secret") is None

    def test_delete_ownership_checked(self, db):
        db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "sec1", "connected")
        assert db.delete_telegram_link("agent_1", "someone_else") is None
        result = db.delete_telegram_link("agent_1", "owner_user")
        assert result is not None
        assert result["bot_token"] == "tok"
        assert db.get_telegram_link("agent_1", "owner_user") is None

    def test_delete_removes_events_too(self, db):
        link_id = db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "sec1", "connected")
        db.log_telegram_event(link_id, "message_in", "hello")
        db.delete_telegram_link("agent_1", "owner_user")
        assert db.get_telegram_events(link_id) == []

    def test_update_status(self, db):
        link_id = db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "sec1", "connected")
        db.update_telegram_link_status(link_id, "error", "webhook delivery failed")
        link = db.get_telegram_link("agent_1", "owner_user")
        assert link["status"] == "error"
        assert link["last_error"] == "webhook delivery failed"

    def test_events_logged_and_retrieved_newest_first(self, db):
        link_id = db.upsert_telegram_link("agent_1", "owner_user", "tok", 1, "bot1", "sec1", "connected")
        db.log_telegram_event(link_id, "connected", "@bot1")
        db.log_telegram_event(link_id, "message_in", "hi")
        db.log_telegram_event(link_id, "message_out", "hello there")
        events = db.get_telegram_events(link_id)
        assert len(events) == 3
        assert events[0]["event_type"] == "message_out"  # newest first
        assert events[-1]["event_type"] == "connected"

    def test_different_bots_dont_share_links(self, db):
        db.upsert_telegram_link("agent_1", "user_1", "tok1", 1, "bot1", "sec1", "connected")
        db.upsert_telegram_link("agent_2", "user_1", "tok2", 2, "bot2", "sec2", "connected")
        link1 = db.get_telegram_link("agent_1", "user_1")
        link2 = db.get_telegram_link("agent_2", "user_1")
        assert link1["bot_id"] == 1
        assert link2["bot_id"] == 2
