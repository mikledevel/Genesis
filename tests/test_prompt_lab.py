"""
Tests for the Prompt Library (versioned) and A/B Testing additions to GenesisDB.
Uses a throwaway SQLite file per test (tempfile) - never touches the real genesis.db
or agents/store.json.
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


class TestPromptLibrary:
    def test_create_and_list(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        p = db.create_prompt(user["id"], "Greeting", "You are helpful.", "desc", ["support"])
        assert p["current_version"] == 1
        prompts = db.list_prompts(user["id"])
        assert len(prompts) == 1
        assert prompts[0]["content"] == "You are helpful."

    def test_edit_creates_new_version_not_overwrite(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        p = db.create_prompt(user["id"], "Greeting", "v1 content")
        db.add_prompt_version(p["id"], user["id"], "v2 content", "tweaked tone")
        full = db.get_prompt(p["id"], user["id"])
        assert full["current_version"] == 2
        assert len(full["versions"]) == 2
        assert full["versions"][0]["content"] == "v2 content"  # newest first
        assert full["versions"][1]["content"] == "v1 content"  # original preserved, not lost

    def test_revert_copies_forward_as_new_version(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        p = db.create_prompt(user["id"], "Greeting", "v1 content")
        db.add_prompt_version(p["id"], user["id"], "v2 content")
        db.revert_prompt(p["id"], user["id"], 1)
        full = db.get_prompt(p["id"], user["id"])
        assert full["current_version"] == 3  # revert is a NEW version, not a rewind
        assert full["versions"][0]["content"] == "v1 content"
        assert len(full["versions"]) == 3  # nothing was deleted

    def test_isolation_other_user_cannot_read_or_edit(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        db.create_user("b@x.com", "hash", "Bob")
        alice = db.get_user_by_email("a@x.com")
        bob = db.get_user_by_email("b@x.com")
        p = db.create_prompt(alice["id"], "Alice's prompt", "secret content")
        assert db.get_prompt(p["id"], bob["id"]) is None
        assert db.add_prompt_version(p["id"], bob["id"], "hacked") is None
        assert db.delete_prompt(p["id"], bob["id"]) is False
        # Alice's prompt is untouched
        assert db.get_prompt(p["id"], alice["id"])["versions"][0]["content"] == "secret content"

    def test_delete(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        p = db.create_prompt(user["id"], "Greeting", "content")
        assert db.delete_prompt(p["id"], user["id"]) is True
        assert db.get_prompt(p["id"], user["id"]) is None


class TestABTesting:
    def _make_user_and_test(self, db, n_variants=2):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        variants = [{"label": chr(65 + i), "system_prompt": f"prompt {chr(65+i)}"} for i in range(n_variants)]
        test = db.create_ab_test(user["id"], "agent_123", "My test", variants)
        return user, test

    def test_create_and_get_running(self, db):
        user, test = self._make_user_and_test(db)
        running = db.get_running_ab_test_for_agent("agent_123")
        assert running is not None
        assert len(running["variants"]) == 2
        assert {v["label"] for v in running["variants"]} == {"A", "B"}

    def test_no_running_test_for_unrelated_agent(self, db):
        self._make_user_and_test(db)
        assert db.get_running_ab_test_for_agent("some_other_agent") is None

    def test_sticky_assignment(self, db):
        user, test = self._make_user_and_test(db)
        running = db.get_running_ab_test_for_agent("agent_123")
        v1 = db.get_or_assign_variant(test["id"], running["variants"], "conv_1")
        v2 = db.get_or_assign_variant(test["id"], running["variants"], "conv_1")
        v3 = db.get_or_assign_variant(test["id"], running["variants"], "conv_1")
        assert v1["id"] == v2["id"] == v3["id"]

    def test_different_conversations_can_get_different_variants(self, db):
        user, test = self._make_user_and_test(db)
        running = db.get_running_ab_test_for_agent("agent_123")
        # With enough samples, both variants should show up at least once (not a strict
        # guarantee, but essentially certain with 40 draws at 50/50 weight).
        assigned = {db.get_or_assign_variant(test["id"], running["variants"], f"conv_{i}")["label"]
                   for i in range(40)}
        assert assigned == {"A", "B"}

    def test_stats_aggregate_messages_and_ratings_per_variant(self, db):
        user, test = self._make_user_and_test(db)
        running = db.get_running_ab_test_for_agent("agent_123")
        variant = db.get_or_assign_variant(test["id"], running["variants"], "conv_1")
        db.create_chat("conv_1", "test chat", user["id"])
        db.add_message("conv_1", "user", "hi")
        db.add_message("conv_1", "assistant", "hello there")
        # find the assistant message id and rate it
        msgs = db.get_messages("conv_1")
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        db.rate_message(assistant_msg["id"], "conv_1", 1)

        stats = db.get_ab_test_stats(test["id"])
        assert stats[variant["id"]]["conversations"] == 1
        assert stats[variant["id"]]["assistant_messages"] == 1
        assert stats[variant["id"]]["thumbs_up"] == 1
        assert stats[variant["id"]]["thumbs_down"] == 0

    def test_only_one_running_test_per_agent_is_the_caller_contract(self, db):
        """create_ab_test itself doesn't block a second running test - the API layer
        (main.py) checks get_running_ab_test_for_agent first and rejects with 409. Verify
        the query that guard depends on behaves correctly with two DIFFERENT agents."""
        user, test1 = self._make_user_and_test(db)
        variants2 = [{"label": "A", "system_prompt": "x"}, {"label": "B", "system_prompt": "y"}]
        test2 = db.create_ab_test(user["id"], "agent_456", "Another bot's test", variants2)
        assert db.get_running_ab_test_for_agent("agent_123")["id"] == test1["id"]
        assert db.get_running_ab_test_for_agent("agent_456")["id"] == test2["id"]

    def test_stop_test_records_winner_and_clears_running_status(self, db):
        user, test = self._make_user_and_test(db)
        running = db.get_running_ab_test_for_agent("agent_123")
        winner_id = running["variants"][0]["id"]
        assert db.stop_ab_test(test["id"], user["id"], winner_id) is True
        assert db.get_running_ab_test_for_agent("agent_123") is None  # no longer "running"
        tests = db.list_ab_tests(user["id"])
        assert tests[0]["status"] == "completed"
        assert tests[0]["winner_variant_id"] == winner_id

    def test_stop_test_wrong_owner_rejected(self, db):
        user, test = self._make_user_and_test(db)
        db.create_user("b@x.com", "hash", "Bob")
        bob = db.get_user_by_email("b@x.com")
        assert db.stop_ab_test(test["id"], bob["id"]) is False
        assert db.get_running_ab_test_for_agent("agent_123") is not None  # untouched

    def test_delete_test_removes_variants_and_assignments(self, db):
        user, test = self._make_user_and_test(db)
        running = db.get_running_ab_test_for_agent("agent_123")
        db.get_or_assign_variant(test["id"], running["variants"], "conv_1")
        assert db.delete_ab_test(test["id"], user["id"]) is True
        assert db.get_running_ab_test_for_agent("agent_123") is None
        assert db.list_ab_tests(user["id"]) == []
