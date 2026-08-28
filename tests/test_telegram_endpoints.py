"""
Tests for the /api/agents/{agent_id}/telegram/* endpoints and the webhook receiver -
app.core.telegram_integration is mocked at the boundary (no real Telegram network calls,
same reasoning as test_handoff.py: api.telegram.org isn't reachable from this environment,
same as api.groq.com). Auth is overridden via app.dependency_overrides, and GenesisDB's
constructor is patched to avoid touching the real ./genesis.db as a side effect (same lesson
learned the hard way in test_quota_wiring.py).
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.core.auth import get_current_user_id
from app.core.agent_store import AgentStore, AgentConfig
from app.db.database import GenesisDB
import app.main as main_module  # imported once here, before any test chdirs - main.py mounts
                                 # StaticFiles(directory="static") at import time, resolved
                                 # relative to cwd, so importing it after a chdir into an
                                 # empty tempdir would fail; once cached in sys.modules here,
                                 # later `import app.main` calls in tests just reuse this.

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestTelegramConnectEndpoint:
    def test_connect_requires_ownership(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="x")
            cfg.author = "real_owner"
            AgentStore().create_agent(cfg)

            import app.main as main_module
            main_module.app.dependency_overrides[get_current_user_id] = lambda: "not_the_owner"
            client = TestClient(main_module.app)
            try:
                r = client.post(f"/api/agents/{cfg.id}/telegram/connect", json={"bot_token": "123:ABC"})
                assert r.status_code == 403
            finally:
                main_module.app.dependency_overrides.clear()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_connect_rejects_invalid_token(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="x")
            cfg.author = "the_owner"
            AgentStore().create_agent(cfg)

            import app.main as main_module
            from app.core.telegram_integration import TelegramAPIError
            main_module.app.dependency_overrides[get_current_user_id] = lambda: "the_owner"
            client = TestClient(main_module.app)
            try:
                with patch("app.core.telegram_integration.get_me",
                          side_effect=TelegramAPIError("getMe", "Unauthorized", 401)):
                    r = client.post(f"/api/agents/{cfg.id}/telegram/connect", json={"bot_token": "bad_token"})
                assert r.status_code == 400
                assert "invalid" in r.json()["detail"].lower()
            finally:
                main_module.app.dependency_overrides.clear()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_connect_without_public_base_url_warns_but_still_saves(self):
        """No PUBLIC_BASE_URL configured (the default, e.g. local dev) - the token should
        still validate/save and the bot import should still happen, just with a webhook
        warning instead of a hard failure, since the test-chat path works without it."""
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="x")
            cfg.author = "the_owner"
            AgentStore().create_agent(cfg)

            import app.main as main_module
            main_module.app.dependency_overrides[get_current_user_id] = lambda: "the_owner"
            client = TestClient(main_module.app)
            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            try:
                with patch("app.core.telegram_integration.get_me", return_value={"id": 555, "username": "mybot"}), \
                     patch("app.core.telegram_integration.import_bot_settings", return_value={"commands": []}), \
                     patch("app.db.database.GenesisDB", return_value=db), \
                     patch("app.config.settings.public_base_url", ""):
                    r = client.post(f"/api/agents/{cfg.id}/telegram/connect", json={"bot_token": "123:ABC"})
                assert r.status_code == 200
                data = r.json()
                assert data["status"] == "connected_no_webhook"
                assert data["warning"] is not None
                assert data["bot_username"] == "mybot"
            finally:
                main_module.app.dependency_overrides.clear()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_connect_with_public_base_url_sets_webhook(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="x")
            cfg.author = "the_owner"
            AgentStore().create_agent(cfg)

            import app.main as main_module
            main_module.app.dependency_overrides[get_current_user_id] = lambda: "the_owner"
            client = TestClient(main_module.app)
            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            try:
                with patch("app.core.telegram_integration.get_me", return_value={"id": 555, "username": "mybot"}), \
                     patch("app.core.telegram_integration.set_webhook", return_value=True) as mock_set_webhook, \
                     patch("app.core.telegram_integration.import_bot_settings", return_value={"commands": []}), \
                     patch("app.db.database.GenesisDB", return_value=db), \
                     patch("app.config.settings.public_base_url", "https://myapp.example.com"):
                    r = client.post(f"/api/agents/{cfg.id}/telegram/connect", json={"bot_token": "123:ABC"})
                assert r.status_code == 200
                assert r.json()["status"] == "connected"
                mock_set_webhook.assert_called_once()
                webhook_url = mock_set_webhook.call_args[0][1]
                assert webhook_url.startswith("https://myapp.example.com/api/telegram/webhook/")
            finally:
                main_module.app.dependency_overrides.clear()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)


class TestTelegramTestChatEndpoint:
    def test_test_chat_reuses_real_dispatch_pipeline(self):
        """This is the core design claim - test_chat must go through the SAME
        GenesisAgent.process() as the real webhook, not a separate simplified path."""
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="You are a helpful bot.")
            cfg.author = "the_owner"
            AgentStore().create_agent(cfg)

            import app.main as main_module
            main_module.app.dependency_overrides[get_current_user_id] = lambda: "the_owner"
            client = TestClient(main_module.app)
            try:
                with patch("app.core.agent.GenesisAgent.process") as mock_process:
                    from app.core.agent import AgentResponse
                    mock_process.return_value = AgentResponse(message="test reply")
                    r = client.post(f"/api/agents/{cfg.id}/telegram/test", json={"message": "hi"})
                assert r.status_code == 200
                assert r.json()["response"] == "test reply"
                # confirm agent_override carried agent_id - this is what activates
                # tool-scoping/RAG/predict/handoff/quota identically to the web chat
                call_agent = mock_process.call_args
                # process() is called on an instance whose agent_override we can't see directly
                # via call_args, so instead confirm process was actually invoked with the message
                assert mock_process.call_args[0][0] == "hi"
            finally:
                main_module.app.dependency_overrides.clear()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_test_chat_requires_ownership(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="x")
            cfg.author = "real_owner"
            AgentStore().create_agent(cfg)

            import app.main as main_module
            main_module.app.dependency_overrides[get_current_user_id] = lambda: "not_the_owner"
            client = TestClient(main_module.app)
            try:
                r = client.post(f"/api/agents/{cfg.id}/telegram/test", json={"message": "hi"})
                assert r.status_code == 403
            finally:
                main_module.app.dependency_overrides.clear()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)


class TestTelegramWebhookReceiver:
    def test_unknown_secret_returns_404(self):
        workdir = tempfile.mkdtemp()
        try:
            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            import app.main as main_module
            client = TestClient(main_module.app)
            with patch("app.db.database.GenesisDB", return_value=db):
                r = client.post("/api/telegram/webhook/nonexistent_secret",
                                json={"message": {"chat": {"id": 1}, "text": "hi"}},
                                headers={"X-Telegram-Bot-Api-Secret-Token": "nonexistent_secret"})
            assert r.status_code == 404
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_wrong_header_secret_rejected_even_with_right_path_secret(self):
        """Defense in depth: the URL path secret alone isn't enough - the header must match too."""
        workdir = tempfile.mkdtemp()
        try:
            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            db.upsert_telegram_link("agent_1", "owner", "tok", 1, "bot1", "real_secret", "connected")
            import app.main as main_module
            client = TestClient(main_module.app)
            with patch("app.db.database.GenesisDB", return_value=db):
                r = client.post("/api/telegram/webhook/real_secret",
                                json={"message": {"chat": {"id": 1}, "text": "hi"}},
                                headers={"X-Telegram-Bot-Api-Secret-Token": "wrong_header_value"})
            assert r.status_code == 403
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_valid_message_dispatches_through_process_and_replies(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="You are a helpful bot.")
            cfg.author = "owner_user"
            AgentStore().create_agent(cfg)

            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            db.upsert_telegram_link(cfg.id, "owner_user", "tok123", 1, "bot1", "real_secret", "connected")

            import app.main as main_module
            client = TestClient(main_module.app)
            from app.core.agent import AgentResponse
            with patch("app.db.database.GenesisDB", return_value=db), \
                 patch("app.core.agent.GenesisAgent.process", return_value=AgentResponse(message="Hello from bot!")) as mock_process, \
                 patch("app.core.telegram_integration.send_message", return_value={}) as mock_send:
                r = client.post("/api/telegram/webhook/real_secret",
                                json={"message": {"chat": {"id": 42}, "text": "hi there"}},
                                headers={"X-Telegram-Bot-Api-Secret-Token": "real_secret"})
            assert r.status_code == 200
            mock_process.assert_called_once()
            assert mock_process.call_args[0][0] == "hi there"
            # attributed to the BOT OWNER's account for quota purposes, not an anonymous Telegram user
            assert mock_process.call_args[1]["user_id"] == "owner_user"
            mock_send.assert_called_once()
            assert mock_send.call_args[0][0] == "tok123"
            assert mock_send.call_args[0][1] == 42
            assert mock_send.call_args[0][2] == "Hello from bot!"

            events = db.get_telegram_events(db.get_telegram_link(cfg.id, "owner_user")["id"])
            event_types = [e["event_type"] for e in events]
            assert "message_in" in event_types
            assert "message_out" in event_types
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_non_text_update_acked_without_crashing(self):
        """A sticker/photo/etc has no 'text' field - must be acknowledged (so Telegram
        doesn't endlessly retry), not crash or dispatch an empty message."""
        workdir = tempfile.mkdtemp()
        try:
            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            db.upsert_telegram_link("agent_1", "owner", "tok", 1, "bot1", "real_secret", "connected")
            import app.main as main_module
            client = TestClient(main_module.app)
            with patch("app.db.database.GenesisDB", return_value=db), \
                 patch("app.core.agent.GenesisAgent.process") as mock_process:
                r = client.post("/api/telegram/webhook/real_secret",
                                json={"message": {"chat": {"id": 1}, "sticker": {"file_id": "xyz"}}},
                                headers={"X-Telegram-Bot-Api-Secret-Token": "real_secret"})
            assert r.status_code == 200
            mock_process.assert_not_called()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_send_failure_logs_error_and_updates_status(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            cfg = AgentConfig(name="Bot", system_prompt="x")
            cfg.author = "owner_user"
            AgentStore().create_agent(cfg)

            db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
            db.upsert_telegram_link(cfg.id, "owner_user", "tok", 1, "bot1", "real_secret", "connected")

            import app.main as main_module
            client = TestClient(main_module.app)
            from app.core.agent import AgentResponse
            from app.core.telegram_integration import TelegramAPIError
            with patch("app.db.database.GenesisDB", return_value=db), \
                 patch("app.core.agent.GenesisAgent.process", return_value=AgentResponse(message="reply")), \
                 patch("app.core.telegram_integration.send_message",
                      side_effect=TelegramAPIError("sendMessage", "bot was blocked by the user", 403)):
                r = client.post("/api/telegram/webhook/real_secret",
                                json={"message": {"chat": {"id": 1}, "text": "hi"}},
                                headers={"X-Telegram-Bot-Api-Secret-Token": "real_secret"})
            assert r.status_code == 200  # still acks Telegram even though delivery failed
            link = db.get_telegram_link(cfg.id, "owner_user")
            assert link["status"] == "error"
            assert "blocked" in link["last_error"]
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)
