"""
Tests for internal Groq usage tracking (GenesisDB.log_groq_usage / get_groq_usage_summary).
This is separate from usage_log/api_keys, which meters the external developer
/api/models/predict API - this table exists because there was previously NO visibility at
all into the platform's own Groq API calls (chat routing, bot builder, codegen, pipeline
suggestions), all of which draw from the platform's own GROQ_API_KEY.
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


class TestGroqUsageLogging:
    def test_log_and_summarize_a_single_successful_call(self, db):
        db.log_groq_usage("user_1", "chat_routing", "openai/gpt-oss-120b",
                          prompt_tokens=100, completion_tokens=50, total_tokens=150, success=True)
        summary = db.get_groq_usage_summary("user_1")
        assert summary["total_calls"] == 1
        assert summary["successful_calls"] == 1
        assert summary["failed_calls"] == 0
        assert summary["total_prompt_tokens"] == 100
        assert summary["total_completion_tokens"] == 50
        assert summary["total_tokens"] == 150

    def test_log_a_failed_call_with_no_token_counts(self, db):
        db.log_groq_usage("user_1", "chat_routing", "openai/gpt-oss-120b", success=False, error="RateLimitError")
        summary = db.get_groq_usage_summary("user_1")
        assert summary["total_calls"] == 1
        assert summary["failed_calls"] == 1
        assert summary["successful_calls"] == 0
        assert summary["total_tokens"] == 0  # SUM of NULL rows -> 0, not None
        assert summary["recent"][0]["error"] == "RateLimitError"

    def test_summary_scoped_per_user(self, db):
        db.log_groq_usage("user_1", "chat_routing", "m", total_tokens=100, success=True)
        db.log_groq_usage("user_2", "chat_routing", "m", total_tokens=999, success=True)
        summary1 = db.get_groq_usage_summary("user_1")
        summary2 = db.get_groq_usage_summary("user_2")
        assert summary1["total_tokens"] == 100
        assert summary2["total_tokens"] == 999

    def test_summary_with_no_user_id_aggregates_everyone(self, db):
        """Groq calls made outside a logged-in session (e.g. an unauthenticated bot-builder
        preview) log with user_id=None - the global (no-filter) summary must still include them."""
        db.log_groq_usage(None, "bot_builder_generate_spec", "m", total_tokens=50, success=True)
        db.log_groq_usage("user_1", "chat_routing", "m", total_tokens=100, success=True)
        summary = db.get_groq_usage_summary(user_id=None)
        assert summary["total_calls"] == 2
        assert summary["total_tokens"] == 150

    def test_by_call_site_breakdown(self, db):
        db.log_groq_usage("user_1", "chat_routing", "m", total_tokens=100, success=True)
        db.log_groq_usage("user_1", "chat_routing", "m", total_tokens=100, success=True)
        db.log_groq_usage("user_1", "bot_builder_fix_spec", "m", total_tokens=300, success=True)
        summary = db.get_groq_usage_summary("user_1")
        by_site = {row["call_site"]: row for row in summary["by_call_site"]}
        assert by_site["chat_routing"]["calls"] == 2
        assert by_site["chat_routing"]["total_tokens"] == 200
        assert by_site["bot_builder_fix_spec"]["calls"] == 1
        assert by_site["bot_builder_fix_spec"]["total_tokens"] == 300

    def test_recent_rows_capped_by_limit(self, db):
        for i in range(10):
            db.log_groq_usage("user_1", "chat_routing", "m", total_tokens=1, success=True)
        summary = db.get_groq_usage_summary("user_1", limit=3)
        assert len(summary["recent"]) == 3

    def test_logging_never_raises_on_bad_input(self, db):
        """A logging call must never be able to break the actual Groq call it's tracking -
        this exercises the DB method directly; the call-site try/except wrappers are the
        second line of defense, tested via the mocked BotBuilder/AICodeGenerator tests."""
        db.log_groq_usage("user_1", "chat_routing", "m")  # no token counts at all
        summary = db.get_groq_usage_summary("user_1")
        assert summary["total_calls"] == 1


class TestUsageLoggingWiring:
    """Confirms the actual call sites (not just the DB layer) log through correctly,
    using mocked Groq responses shaped like the real SDK's usage object."""

    def test_bot_builder_logs_successful_call_with_real_token_counts(self, db):
        from app.core.bot_builder import BotBuilder
        from unittest.mock import patch, MagicMock

        bb = BotBuilder(user_id="user_1")
        bb.groq_key = "fake-key"
        bb._db = db

        fake_completion = MagicMock()
        fake_completion.choices[0].message.content = '{"hallucinated": false, "reason": "ok"}'
        fake_completion.usage.prompt_tokens = 42
        fake_completion.usage.completion_tokens = 8
        fake_completion.usage.total_tokens = 50

        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = fake_completion
            bb._groq_json("system", "user", call_site="test_call_site")

        summary = db.get_groq_usage_summary("user_1")
        assert summary["total_calls"] == 1
        assert summary["total_tokens"] == 50
        assert summary["by_call_site"][0]["call_site"] == "test_call_site"

    def test_bot_builder_logs_failed_call(self, db):
        from app.core.bot_builder import BotBuilder
        from unittest.mock import patch

        bb = BotBuilder(user_id="user_1")
        bb.groq_key = "fake-key"
        bb._db = db

        with patch("groq.Groq", side_effect=RuntimeError("connection refused")):
            result = bb._groq_json("system", "user", call_site="test_call_site")

        assert result is None
        summary = db.get_groq_usage_summary("user_1")
        assert summary["total_calls"] == 1
        assert summary["failed_calls"] == 1
        assert "connection refused" in summary["recent"][0]["error"]

    def test_no_groq_key_does_not_log_at_all(self, db):
        """If there's no API key, the call short-circuits before ever reaching Groq - that's
        a configuration issue, not a call attempt, so it correctly shouldn't appear in usage."""
        from app.core.bot_builder import BotBuilder
        bb = BotBuilder(user_id="user_1")
        bb.groq_key = ""
        bb._db = db
        result = bb._groq_json("system", "user")
        assert result is None
        summary = db.get_groq_usage_summary("user_1")
        assert summary["total_calls"] == 0
