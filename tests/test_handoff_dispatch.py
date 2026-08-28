"""
Tests for the request_handoff tool dispatch/handler - the piece that actually notifies a
human when a bot can't help, closing the loop on the existing anti-hallucination deflection
behavior. Network calls (the actual webhook POST) are mocked here; app.core.handoff itself is
tested separately in test_handoff.py against real (mocked) HTTP semantics.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch
from app.core.agent import GenesisAgent, AgentResponse
from app.core.engine.registry import ModelRegistry
from app.core.engine.trainer import DatasetTrainer
from app.core.agent_store import AgentStore, AgentConfig
from app.db.database import GenesisDB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_agent(workdir):
    registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
    trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    agent = GenesisAgent(registry=registry, trainer=trainer, db=db)
    agent.current_user_id = "test_user"
    return agent


class TestHandoffToolScoping:
    def test_request_handoff_is_in_bot_scoped_allowed_tools(self):
        assert "request_handoff" in GenesisAgent.BOT_SCOPED_ALLOWED_TOOLS

    def test_bot_scoped_conversation_can_dispatch_handoff(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={"tool": "request_handoff", "args": {"reason": "x"}, "message": "ok"}), \
                 patch.object(agent, "_do_request_handoff", return_value=AgentResponse(message="handed off")) as mock_handoff:
                agent.process("where is my order?")
            mock_handoff.assert_called_once()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestRequestHandoffHandler:
    def test_no_webhook_configured_gives_honest_answer_not_false_promise(self):
        """Critical: if there's no webhook, the bot must NOT tell the user someone will
        follow up, since no one actually will."""
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent = _make_agent(workdir)
            cfg = AgentConfig(name="PlainBot", system_prompt="Just chat.")
            cfg.author = "test_user"  # no handoff_webhook_url set
            AgentStore().create_agent(cfg)

            resp = agent._do_request_handoff(cfg.id, "conv_1", "where's my order?", {"reason": "order status"})
            assert resp.data["delivered"] is False
            assert "follow up" not in resp.message.lower()  # doesn't falsely promise
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_webhook_configured_and_delivered_confirms_to_user(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent = _make_agent(workdir)
            cfg = AgentConfig(name="ShopBot", system_prompt="You help with orders.")
            cfg.author = "test_user"
            cfg.handoff_webhook_url = "https://hooks.example.com/abc"
            AgentStore().create_agent(cfg)

            with patch("app.core.handoff.send_handoff_webhook", return_value={"delivered": True, "error": None}) as mock_send:
                resp = agent._do_request_handoff(cfg.id, "conv_1", "where's my order?", {"reason": "order status"})
            assert resp.data["delivered"] is True
            mock_send.assert_called_once()
            # payload sent to the webhook should carry the real conversation context
            call_args = mock_send.call_args[0]
            assert call_args[0] == "https://hooks.example.com/abc"
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_webhook_delivery_failure_is_reported_honestly(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent = _make_agent(workdir)
            cfg = AgentConfig(name="ShopBot", system_prompt="You help with orders.")
            cfg.author = "test_user"
            cfg.handoff_webhook_url = "https://hooks.example.com/dead"
            AgentStore().create_agent(cfg)

            with patch("app.core.handoff.send_handoff_webhook", return_value={"delivered": False, "error": "HTTP 404"}):
                resp = agent._do_request_handoff(cfg.id, "conv_1", "where's my order?", {"reason": "order status"})
            assert resp.data["delivered"] is False
            assert not resp.success or "delivery failed" in resp.message.lower()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_handoff_event_is_logged_regardless_of_delivery_outcome(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent = _make_agent(workdir)
            cfg = AgentConfig(name="ShopBot", system_prompt="You help with orders.")
            cfg.author = "test_user"
            cfg.handoff_webhook_url = "https://hooks.example.com/abc"
            AgentStore().create_agent(cfg)

            with patch("app.core.handoff.send_handoff_webhook", return_value={"delivered": True, "error": None}):
                agent._do_request_handoff(cfg.id, "conv_1", "where's my order?", {"reason": "order status"})

            events = agent.db.get_handoff_events(cfg.id)
            assert len(events) == 1
            assert events[0]["delivered"] is True
            assert events[0]["user_message"] == "where's my order?"
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_nonexistent_bot_fails_clearly(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent = _make_agent(workdir)
            resp = agent._do_request_handoff("does_not_exist", "conv_1", "hi", {})
            assert not resp.success
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_no_agent_id_fails_clearly(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent(workdir)
            resp = agent._do_request_handoff(None, "conv_1", "hi", {})
            assert not resp.success
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestSystemPromptHandoffInjection:
    def test_handoff_capability_always_described_for_any_bot(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent = _make_agent(workdir)
            cfg = AgentConfig(name="PlainBot", system_prompt="Just chat.")
            cfg.author = "test_user"  # no webhook configured at all
            AgentStore().create_agent(cfg)

            agent.agent_override = {"model": "m", "system_prompt": "Just chat.", "agent_id": cfg.id}
            prompt = agent._get_effective_system_prompt("hi")
            assert "HANDOFF CAPABILITY" in prompt
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_no_handoff_text_for_unscoped_platform_assistant(self):
        workdir = tempfile.mkdtemp()
        try:
            agent = _make_agent(workdir)
            agent.agent_override = None
            prompt = agent._get_effective_system_prompt("hi")
            assert "HANDOFF CAPABILITY" not in prompt
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
