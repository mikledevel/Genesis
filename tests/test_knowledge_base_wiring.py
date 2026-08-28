"""
Integration test: GenesisAgent._get_effective_system_prompt actually retrieves and injects
knowledge base context when chatting AS a specific bot, and correctly does NOT do so for
the platform's own default assistant (no agent_id selected) or for queries the knowledge
base doesn't actually cover - this is the exact wiring manually verified live during
development; formalized here so it can't silently regress.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.agent import GenesisAgent
from app.db.database import GenesisDB
from app.core.knowledge_base import chunk_text


def _make_agent_with_kb(workdir, agent_id="agent_test", docs=None):
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    for filename, content in (docs or {}).items():
        db.add_kb_document(agent_id, "owner_user", filename, content, chunk_text(content))
    agent = GenesisAgent(db=db)
    agent.current_user_id = "chatting_user"
    return agent


class TestKnowledgeBaseWiring:
    def test_relevant_query_gets_kb_context_injected(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent_with_kb(workdir, docs={
                "policy.txt": "Our return policy allows returns within 30 days of purchase with a receipt.",
            })
            agent.agent_override = {"model": "openai/gpt-oss-120b", "system_prompt": "You are ShopBot.", "agent_id": "agent_test"}
            prompt = agent._get_effective_system_prompt("What is your return policy?")
            assert "KNOWLEDGE BASE CONTEXT" in prompt
            assert "30 days" in prompt
            assert "You are ShopBot." in prompt  # persona overlay still present too
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_unrelated_query_does_not_inject_noise(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent_with_kb(workdir, docs={
                "policy.txt": "Our return policy allows returns within 30 days of purchase with a receipt.",
            })
            agent.agent_override = {"model": "openai/gpt-oss-120b", "system_prompt": "You are ShopBot.", "agent_id": "agent_test"}
            prompt = agent._get_effective_system_prompt("What is the airspeed velocity of a swallow?")
            assert "KNOWLEDGE BASE CONTEXT" not in prompt
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_default_assistant_never_gets_kb_injection(self):
        """Critical isolation: KB retrieval must only apply when chatting AS a published bot
        (agent_id present), never for the platform's own general-purpose assistant."""
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent_with_kb(workdir, docs={
                "policy.txt": "Our return policy allows returns within 30 days.",
            })
            agent.agent_override = None  # no bot selected - the plain Genesis AI assistant
            prompt = agent._get_effective_system_prompt("What is your return policy?")
            assert "KNOWLEDGE BASE CONTEXT" not in prompt
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_bot_with_no_uploaded_documents_is_unaffected(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent_with_kb(workdir, docs={})  # no documents at all
            agent.agent_override = {"model": "openai/gpt-oss-120b", "system_prompt": "You are ShopBot.", "agent_id": "agent_test"}
            prompt = agent._get_effective_system_prompt("What is your return policy?")
            assert "KNOWLEDGE BASE CONTEXT" not in prompt
            assert "You are ShopBot." in prompt  # everything else still works normally
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_kb_lookup_failure_never_breaks_the_chat(self):
        """If DB access for KB chunks throws for any reason, the chat must still get a
        usable system prompt back - a knowledge base problem should never take down the
        whole bot."""
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent_with_kb(workdir, docs={"doc.txt": "some content"})
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "agent_test"}
            # simulate a broken DB call
            agent.db.get_kb_chunks_for_agent = lambda agent_id: (_ for _ in ()).throw(RuntimeError("db exploded"))
            prompt = agent._get_effective_system_prompt("What is your return policy?")
            assert "You are ShopBot." in prompt  # still got a usable prompt back
            assert "KNOWLEDGE BASE CONTEXT" not in prompt
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    def test_different_bots_only_see_their_own_knowledge(self):
        workdir = tempfile.mkdtemp()
        try:
            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            db.add_kb_document("bot_a", "owner", "a.txt", "Bot A's secret return policy is 30 days.",
                               chunk_text("Bot A's secret return policy is 30 days."))
            db.add_kb_document("bot_b", "owner", "b.txt", "Bot B's shipping takes 5 days via courier.",
                               chunk_text("Bot B's shipping takes 5 days via courier."))
            agent = GenesisAgent(db=db)
            agent.current_user_id = "chatting_user"

            agent.agent_override = {"model": "m", "system_prompt": "You are Bot A.", "agent_id": "bot_a"}
            prompt_a = agent._get_effective_system_prompt("What is the return policy?")
            assert "30 days" in prompt_a
            assert "courier" not in prompt_a  # bot A must not see bot B's knowledge
        finally:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
