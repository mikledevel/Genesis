"""
Tests for app.core.quota - checked BEFORE each Groq call is made across the codebase.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from unittest.mock import patch
from app.db.database import GenesisDB
from app.core.quota import check_quota, quota_exceeded_message


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


class TestCheckQuota:
    def test_none_user_id_always_allowed(self, db):
        allowed, used, limit = check_quota(db, None)
        assert allowed is True

    def test_under_platform_default_limit_is_allowed(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=1000, success=True)
        with patch("app.config.settings.default_monthly_groq_token_limit", 2_000_000):
            allowed, used, limit = check_quota(db, user["id"])
        assert allowed is True
        assert used == 1000
        assert limit == 2_000_000

    def test_over_platform_default_limit_is_blocked(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=3_000_000, success=True)
        with patch("app.config.settings.default_monthly_groq_token_limit", 2_000_000):
            allowed, used, limit = check_quota(db, user["id"])
        assert allowed is False
        assert used == 3_000_000

    def test_per_user_override_takes_precedence_over_platform_default(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        db.set_user_groq_limit(user["id"], 500)
        db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=600, success=True)
        with patch("app.config.settings.default_monthly_groq_token_limit", 2_000_000):
            allowed, used, limit = check_quota(db, user["id"])
        assert allowed is False  # blocked by the LOWER per-user override, not the high default
        assert limit == 500

    def test_zero_limit_means_unlimited(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        db.set_user_groq_limit(user["id"], 0)
        db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=99_999_999, success=True)
        allowed, used, limit = check_quota(db, user["id"])
        assert allowed is True

    def test_negative_limit_means_unlimited(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        db.set_user_groq_limit(user["id"], -1)
        db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=99_999_999, success=True)
        allowed, used, limit = check_quota(db, user["id"])
        assert allowed is True

    def test_failed_calls_dont_count_toward_quota(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=5000, success=False, error="timeout")
        allowed, used, limit = check_quota(db, user["id"])
        assert used == 0  # the failed call's tokens (if any were even reported) don't count

    def test_usage_from_previous_month_doesnt_count(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        user = db.get_user_by_email("a@x.com")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO groq_usage_log (user_id, call_site, model, total_tokens, success, created_at) "
                "VALUES (?, ?, ?, ?, 1, datetime('now', '-2 months'))",
                (user["id"], "chat_routing", "m", 5_000_000))
            conn.commit()
        allowed, used, limit = check_quota(db, user["id"])
        assert used == 0  # old usage from 2 months ago shouldn't count against this month

    def test_different_users_dont_share_quota(self, db):
        db.create_user("a@x.com", "hash", "Alice")
        db.create_user("b@x.com", "hash", "Bob")
        alice = db.get_user_by_email("a@x.com")
        bob = db.get_user_by_email("b@x.com")
        db.log_groq_usage(alice["id"], "chat_routing", "m", total_tokens=5_000_000, success=True)
        allowed_alice, used_alice, _ = check_quota(db, alice["id"])
        allowed_bob, used_bob, _ = check_quota(db, bob["id"])
        assert used_bob == 0
        assert allowed_bob is True  # Bob is unaffected by Alice's heavy usage


class TestQuotaExceededMessage:
    def test_message_includes_numbers(self):
        msg = quota_exceeded_message(2_500_000, 2_000_000)
        assert "2,500,000" in msg
        assert "2,000,000" in msg
