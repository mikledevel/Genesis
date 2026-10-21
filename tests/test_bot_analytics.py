"""
Tests for GenesisDB.get_bot_analytics and tag_chat_with_agent.

Includes a regression test for a real ordering bug caught during development:
tag_chat_with_agent does an UPDATE, which silently affects zero rows if the chats row
doesn't exist yet - and for a brand-new conversation, the chats row is only created inside
GenesisAgent.process() (via create_chat), which happens AFTER the point in main.py where
agent_id was resolved. Tagging before process() runs would silently lose the very first
message of every new conversation with a bot.
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


class TestChatTagging:
    def test_tagging_before_chat_exists_is_a_silent_no_op(self, db):
        """Documents the exact failure mode found during development - tagging a chat_id
        that doesn't exist yet in the chats table does NOT create it or raise an error, it
        just silently updates zero rows. This is why main.py calls tag_chat_with_agent AFTER
        agent.process() (which creates the chat), not before."""
        db.tag_chat_with_agent("conv_never_created", "agent_1")
        analytics = db.get_bot_analytics("agent_1")
        assert analytics["total_conversations"] == 0  # confirms the tag was lost

    def test_tagging_after_chat_exists_works_correctly(self, db):
        db.create_chat("conv_1", user_id="user_1")
        db.tag_chat_with_agent("conv_1", "agent_1")
        analytics = db.get_bot_analytics("agent_1")
        assert analytics["total_conversations"] == 1

    def test_tag_is_idempotent_and_sticky(self, db):
        """A conversation's bot attribution shouldn't change after the fact, even if
        tag_chat_with_agent gets called again on a later message in the same conversation."""
        db.create_chat("conv_1", user_id="user_1")
        db.tag_chat_with_agent("conv_1", "agent_1")
        db.tag_chat_with_agent("conv_1", "agent_2")  # should NOT overwrite
        analytics_1 = db.get_bot_analytics("agent_1")
        analytics_2 = db.get_bot_analytics("agent_2")
        assert analytics_1["total_conversations"] == 1
        assert analytics_2["total_conversations"] == 0


class TestBotAnalytics:
    def _setup_conversation(self, db, chat_id, agent_id, user_id, messages):
        """messages: list of (role, content, rating)"""
        db.create_chat(chat_id, user_id=user_id)
        db.tag_chat_with_agent(chat_id, agent_id)
        for role, content, rating in messages:
            mid = db.add_message(chat_id, role, content)
            if rating is not None:
                db.rate_message(mid, chat_id, rating)

    def test_empty_bot_has_zeroed_analytics(self, db):
        analytics = db.get_bot_analytics("agent_never_used")
        assert analytics["total_conversations"] == 0
        assert analytics["total_messages"] == 0
        assert analytics["thumbs_up"] == 0
        assert analytics["thumbs_down"] == 0
        assert analytics["recent_opening_questions"] == []

    def test_counts_conversations_and_messages_correctly(self, db):
        self._setup_conversation(db, "conv_1", "agent_1", "user_a", [
            ("user", "What are your hours?", None),
            ("assistant", "9am to 6pm.", 1),
        ])
        self._setup_conversation(db, "conv_2", "agent_1", "user_b", [
            ("user", "Do you ship internationally?", None),
            ("assistant", "Yes, worldwide.", -1),
        ])
        analytics = db.get_bot_analytics("agent_1")
        assert analytics["total_conversations"] == 2
        assert analytics["total_messages"] == 4
        assert analytics["assistant_messages"] == 2
        assert analytics["thumbs_up"] == 1
        assert analytics["thumbs_down"] == 1

    def test_different_bots_dont_mix_analytics(self, db):
        self._setup_conversation(db, "conv_1", "agent_1", "user_a", [("user", "hi agent 1", None)])
        self._setup_conversation(db, "conv_2", "agent_2", "user_a", [("user", "hi agent 2", None)])
        a1 = db.get_bot_analytics("agent_1")
        a2 = db.get_bot_analytics("agent_2")
        assert a1["total_conversations"] == 1
        assert a2["total_conversations"] == 1

    def test_recent_opening_questions_captures_first_user_message_per_conversation(self, db):
        self._setup_conversation(db, "conv_1", "agent_1", "user_a", [
            ("user", "What is your return policy?", None),
            ("assistant", "30 days.", None),
            ("user", "And for sale items?", None),  # NOT the opening question - must be excluded
        ])
        analytics = db.get_bot_analytics("agent_1")
        questions = [q["content"] for q in analytics["recent_opening_questions"]]
        assert "What is your return policy?" in questions
        assert "And for sale items?" not in questions

    def test_unrated_messages_dont_count_as_thumbs_up_or_down(self, db):
        self._setup_conversation(db, "conv_1", "agent_1", "user_a", [
            ("user", "hi", None), ("assistant", "hello", None),
        ])
        analytics = db.get_bot_analytics("agent_1")
        assert analytics["thumbs_up"] == 0
        assert analytics["thumbs_down"] == 0
