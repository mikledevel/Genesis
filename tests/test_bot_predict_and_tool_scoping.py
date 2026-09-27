"""
Tests for two related additions:
  1. Tool scoping: a published-bot conversation must NOT be able to reach platform-builder
     tools (train_model, create_bot, publish_marketplace, etc.) - only plain conversation and
     predict_with_model. Without this, chatting with someone else's published bot could get
     the underlying LLM to train models or publish marketplace listings under the CHATTING
     user's own account while they think they're just talking to a support bot.
  2. predict_with_model: a bot with a linked_model_id can run a real prediction from its
     registered ML model mid-conversation.

AgentStore hardcodes "agents/store.json" relative to cwd (no constructor injection point),
so tests that need it use the same chdir-into-isolated-tempdir pattern already established in
test_retrain_integration.py - never touches the real ./agents/store.json.

Real training (DatasetTrainer) and a real registry are used with isolated tempdirs
throughout. These test the dispatch/handler logic directly, not the LLM's tool-choosing
behavior itself (which needs a live key - see live-groq-testing-checklist.md for that layer).
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from unittest.mock import patch
from app.core.agent import GenesisAgent
from app.core.engine.registry import ModelRegistry
from app.core.engine.trainer import DatasetTrainer
from app.core.agent_store import AgentStore, AgentConfig
from app.db.database import GenesisDB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _make_dataset(path, n=100, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"age": rng.randint(20, 65, n), "salary": rng.randint(30000, 120000, n),
                       "churn": rng.randint(0, 2, n)})
    df.to_csv(path, index=False)


def _make_agent(workdir):
    registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
    trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    agent = GenesisAgent(registry=registry, trainer=trainer, db=db)
    agent.current_user_id = "test_user"
    return agent, registry


class TestToolScoping:
    def test_bot_scoped_conversation_blocks_train_model_tool(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={
                "tool": "train_model", "args": {"path": "datasets/x.csv"}, "message": "Sure, training now!"}), \
                 patch.object(agent, "_do_train") as mock_do_train:
                agent.process("please train a model on my data")
            mock_do_train.assert_not_called()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_bot_scoped_conversation_blocks_create_bot_tool(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={"tool": "create_bot", "args": {}, "message": "ok"}), \
                 patch.object(agent, "_do_create_bot") as mock_create_bot:
                agent.process("build me a new bot")
            mock_create_bot.assert_not_called()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_bot_scoped_conversation_blocks_publish_marketplace_tool(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={"tool": "publish_marketplace", "args": {}, "message": "ok"}), \
                 patch.object(agent, "_do_publish_marketplace") as mock_publish:
                agent.process("publish this to the marketplace")
            mock_publish.assert_not_called()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_blocked_tool_attempt_never_leaks_the_models_own_reasoning_text(self):
        """A user chatting with someone else's published bot asking it to do a
        platform-builder thing (create a bot, train a model, etc.) must get a clean,
        in-character decline - never the raw "message" text the model wrote while it was
        reasoning about calling that now-blocked tool, which can (and did, in production)
        leak internal/meta details like "I'm a bot someone else made, not the real
        assistant"."""
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are TenderScout.", "agent_id": "some_bot"}
            leaky_message = ("I can't create a bot because I'm actually just a bot that a "
                              "user built on this platform, not the original Genesis AI assistant.")
            with patch.object(agent, "_call_llm", return_value={"tool": "create_bot", "args": {}, "message": leaky_message}), \
                 patch.object(agent, "_do_create_bot") as mock_create_bot:
                resp = agent.process("create a bot for me")
            mock_create_bot.assert_not_called()
            assert leaky_message not in resp.message
            assert "user" not in resp.message.lower() or "genesis" not in resp.message.lower()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_genuine_no_tool_response_still_passes_the_models_message_through(self):
        """Regression guard: the fix must only intercept BLOCKED tool attempts, not the
        ordinary case where the model just decides plain conversation is the right answer
        (tool=None was its own genuine choice, not a demotion) - that message is real
        conversational output and must still reach the user unchanged."""
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are TenderScout.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={"tool": None, "message": "Sure, tenders in IT are currently..."}):
                resp = agent.process("what tenders are open?")
            assert resp.message == "Sure, tenders in IT are currently..."
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_bot_scoped_conversation_still_allows_predict(self):
        workdir = tempfile.mkdtemp()
        try:
            from app.core.agent import AgentResponse
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={"tool": "predict_with_model", "args": {}, "message": "ok"}), \
                 patch.object(agent, "_do_bot_predict", return_value=AgentResponse(message="predicted!")) as mock_predict:
                agent.process("will this customer churn?")
            mock_predict.assert_called_once()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_bot_scoped_conversation_allows_plain_chat(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            agent.agent_override = {"model": "m", "system_prompt": "You are ShopBot.", "agent_id": "some_bot"}
            with patch.object(agent, "_call_llm", return_value={"tool": None, "message": "Hello there!"}):
                resp = agent.process("hi")
            assert resp.message == "Hello there!"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def test_unscoped_platform_assistant_still_has_full_tool_access(self):
        """Regression guard: this scoping must NOT affect the platform's own default
        assistant (no agent_id selected) - it should still dispatch train_model normally."""
        workdir = tempfile.mkdtemp()
        try:
            from app.core.agent import AgentResponse
            agent, registry = _make_agent(workdir)
            agent.agent_override = None
            with patch.object(agent, "_call_llm", return_value={
                "tool": "train_model", "args": {"path": "datasets/x.csv"}, "message": "ok"}), \
                 patch.object(agent, "_do_train", return_value=AgentResponse(message="trained!")) as mock_do_train:
                agent.process("train a model on my data")
            mock_do_train.assert_called_once()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestBotPredict:
    def test_predict_with_all_features_returns_real_prediction(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            os.makedirs("datasets")
            _make_dataset("datasets/churn.csv")
            agent, registry = _make_agent(workdir)
            train_resp = agent._do_train("train datasets/churn.csv", {"target_column": "churn"})
            assert train_resp.success, train_resp.message
            model_id = train_resp.data["model_id"]

            cfg = AgentConfig(name="ChurnBot", system_prompt="You predict churn.")
            cfg.author = "test_user"
            cfg.linked_model_id = model_id
            AgentStore().create_agent(cfg)

            resp = agent._do_bot_predict(cfg.id, {"features": {"age": 35, "salary": 60000}})
            assert resp.success, resp.message
            assert "prediction" in resp.data
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_predict_with_missing_features_asks_for_them(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            os.makedirs("datasets")
            _make_dataset("datasets/churn.csv")
            agent, registry = _make_agent(workdir)
            train_resp = agent._do_train("train datasets/churn.csv", {"target_column": "churn"})
            model_id = train_resp.data["model_id"]

            cfg = AgentConfig(name="ChurnBot", system_prompt="You predict churn.")
            cfg.author = "test_user"
            cfg.linked_model_id = model_id
            AgentStore().create_agent(cfg)

            resp = agent._do_bot_predict(cfg.id, {"features": {"age": 35}})
            assert not resp.success
            assert "salary" in resp.message
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_predict_with_no_linked_model_fails_clearly(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent, registry = _make_agent(workdir)
            cfg = AgentConfig(name="PlainBot", system_prompt="Just chat.")
            cfg.author = "test_user"
            AgentStore().create_agent(cfg)

            resp = agent._do_bot_predict(cfg.id, {"features": {}})
            assert not resp.success
            assert "isn't linked" in resp.message.lower()
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_predict_with_nonexistent_bot_fails_clearly(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent, registry = _make_agent(workdir)
            resp = agent._do_bot_predict("does_not_exist", {"features": {}})
            assert not resp.success
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_predict_with_no_agent_id_at_all_fails_clearly(self):
        workdir = tempfile.mkdtemp()
        try:
            agent, registry = _make_agent(workdir)
            resp = agent._do_bot_predict(None, {"features": {}})
            assert not resp.success
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class TestSystemPromptPredictInjection:
    def test_linked_model_features_appear_in_system_prompt(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            os.makedirs("datasets")
            _make_dataset("datasets/churn.csv")
            agent, registry = _make_agent(workdir)
            train_resp = agent._do_train("train datasets/churn.csv", {"target_column": "churn"})
            model_id = train_resp.data["model_id"]

            cfg = AgentConfig(name="ChurnBot", system_prompt="You predict churn.")
            cfg.author = "test_user"
            cfg.linked_model_id = model_id
            AgentStore().create_agent(cfg)

            agent.agent_override = {"model": "m", "system_prompt": "You predict churn.", "agent_id": cfg.id}
            prompt = agent._get_effective_system_prompt("will this customer churn?")
            assert "PREDICTION CAPABILITY" in prompt
            assert "age" in prompt and "salary" in prompt
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)

    def test_no_linked_model_no_prediction_capability_text(self):
        workdir = tempfile.mkdtemp()
        try:
            os.chdir(workdir)
            agent, registry = _make_agent(workdir)
            cfg = AgentConfig(name="PlainBot", system_prompt="Just chat.")
            cfg.author = "test_user"
            AgentStore().create_agent(cfg)

            agent.agent_override = {"model": "m", "system_prompt": "Just chat.", "agent_id": cfg.id}
            prompt = agent._get_effective_system_prompt("hi")
            assert "PREDICTION CAPABILITY" not in prompt
        finally:
            os.chdir(PROJECT_ROOT)
            shutil.rmtree(workdir, ignore_errors=True)
