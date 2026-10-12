"""
Tests confirming the quota gate (app.core.quota) is actually wired into its real entry
points - GenesisAgent.process() for the chat-driven path, and the REST layer for
BotBuilder/ModelBuilder endpoints that never go through process() at all. Complements
test_quota.py (module logic) and the live E2E already run manually against a real server.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.core.agent import GenesisAgent
from app.core.engine.registry import ModelRegistry
from app.core.engine.trainer import DatasetTrainer
from app.db.database import GenesisDB


def _make_agent(workdir):
    registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
    trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    agent = GenesisAgent(registry=registry, trainer=trainer, db=db)
    return agent, db


class TestProcessQuotaGate:
    def test_over_quota_blocks_before_any_llm_call(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, db = _make_agent(workdir)
            db.create_user("a@x.com", "hash", "Alice")
            user = db.get_user_by_email("a@x.com")
            db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=3_000_000, success=True)

            with patch.object(agent, "_call_llm") as mock_call_llm:
                resp = agent.process("hello", conversation_id="c1", user_id=user["id"])
            mock_call_llm.assert_not_called()  # blocked BEFORE any LLM call was attempted
            assert not resp.success
            assert "лимит" in resp.message.lower()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_under_quota_proceeds_normally(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, db = _make_agent(workdir)
            db.create_user("a@x.com", "hash", "Alice")
            user = db.get_user_by_email("a@x.com")
            db.log_groq_usage(user["id"], "chat_routing", "m", total_tokens=100, success=True)

            with patch.object(agent, "_call_llm", return_value={"tool": None, "message": "hi there"}) as mock_call_llm:
                resp = agent.process("hello", conversation_id="c1", user_id=user["id"])
            mock_call_llm.assert_called_once()
            assert resp.message == "hi there"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_anonymous_user_never_blocked(self):
        """No account to attribute/throttle usage to - always allowed, matching
        check_quota's documented behavior for user_id=None."""
        workdir = tempfile.mkdtemp()
        try:
            agent, db = _make_agent(workdir)
            with patch.object(agent, "_call_llm", return_value={"tool": None, "message": "hi"}) as mock_call_llm:
                resp = agent.process("hello", conversation_id="c1", user_id=None)
            mock_call_llm.assert_called_once()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestRestLayerQuotaGate:
    """Uses FastAPI's TestClient directly against the app - these hit the real endpoint
    routing/dependency-injection layer (not process()), which is the part that previously
    had zero quota protection for BotBuilder/ModelBuilder. Auth is overridden via FastAPI's
    own app.dependency_overrides mechanism - patching the imported function reference by
    module-attribute name does NOT work here, since Depends() captures the callable object
    at route-registration time, not by a name that gets re-looked-up per request.
    GenesisDB.__init__ is also patched to a no-op - _enforce_groq_quota constructs its own
    GenesisDB() internally, and even with its methods mocked, the real constructor would
    still touch the real ./genesis.db on disk as a side effect of just being instantiated."""

    def test_bot_builder_generate_blocked_over_quota(self):
        import app.main as main_module
        from app.core.auth import get_current_user_id_optional
        main_module.app.dependency_overrides[get_current_user_id_optional] = lambda: "user_over_quota"
        client = TestClient(main_module.app)
        try:
            with patch("app.db.database.GenesisDB.__init__", return_value=None), \
                 patch("app.db.database.GenesisDB.get_groq_usage_this_month", return_value=5_000_000), \
                 patch("app.db.database.GenesisDB.get_user_groq_limit", return_value=None), \
                 patch("app.config.settings.default_monthly_groq_token_limit", 2_000_000):
                r = client.post("/api/bot-builder/generate", json={"description": "a test bot"})
            assert r.status_code == 429
            assert "лимит" in r.json()["detail"].lower()
        finally:
            main_module.app.dependency_overrides.clear()

    def test_model_builder_generate_blocked_over_quota(self):
        import app.main as main_module
        from app.core.auth import get_current_user_id
        main_module.app.dependency_overrides[get_current_user_id] = lambda: "user_over_quota"
        client = TestClient(main_module.app)
        try:
            with patch("app.db.database.GenesisDB.__init__", return_value=None), \
                 patch("app.db.database.GenesisDB.get_groq_usage_this_month", return_value=5_000_000), \
                 patch("app.db.database.GenesisDB.get_user_groq_limit", return_value=None), \
                 patch("app.config.settings.default_monthly_groq_token_limit", 2_000_000):
                r = client.post("/api/model-builder/generate", json={"description": "x", "dataset_path": "datasets/x.csv"})
            assert r.status_code == 429
        finally:
            main_module.app.dependency_overrides.clear()

    def test_bot_builder_generate_not_blocked_under_quota(self):
        """Regression guard: the gate must not block legitimate under-quota requests."""
        import app.main as main_module
        from app.core.auth import get_current_user_id_optional
        main_module.app.dependency_overrides[get_current_user_id_optional] = lambda: "user_under_quota"
        client = TestClient(main_module.app)
        try:
            with patch("app.db.database.GenesisDB.__init__", return_value=None), \
                 patch("app.db.database.GenesisDB.get_groq_usage_this_month", return_value=100), \
                 patch("app.db.database.GenesisDB.get_user_groq_limit", return_value=None), \
                 patch("app.config.settings.default_monthly_groq_token_limit", 2_000_000), \
                 patch("app.core.bot_builder.BotBuilder.build", return_value={"spec": None, "success": False}):
                r = client.post("/api/bot-builder/generate", json={"description": "a test bot"})
            assert r.status_code == 200  # reached the (mocked) BotBuilder, not blocked at 429
        finally:
            main_module.app.dependency_overrides.clear()
