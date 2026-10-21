"""
Core GenesisAgent.process() dispatch tests.

This file previously imported `Intent` from app.core.agent, a class that no longer exists in
the module - every test here failed to even COLLECT (pytest reported a collection error, not
a test failure), which meant the largest, most central module in the codebase (agent.py, the
entire conversational/tool-calling brain) had zero enforced regression coverage.

These tests call agent.process() with real DatasetProfiler/DatasetTrainer/ModelRegistry
instances (so _do_analyze/_do_train genuinely run), but mock GenesisAgent._call_llm - the one
external network boundary (Groq) - exactly like tests/test_bot_predict_and_tool_scoping.py
already does. That is a deliberate, standard test-isolation boundary (we don't own Groq's
uptime and don't want these tests to depend on network access or a real API key), not a fake
implementation of the agent's own logic: process()'s dispatch, _do_analyze, _do_train, and
_do_find_best all run for real.

LLM failure-handling regression tests (TestLLMFailureHandling) exercise the fix in
app.core.llm_errors / GenesisAgent._call_groq / _call_local / _call_llm: a genuine provider
failure must raise a typed LLMError, not return a fake successful reply.
"""
import pytest, pandas as pd, numpy as np, tempfile, shutil, os, sys
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.agent import GenesisAgent, AgentResponse
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry
from app.core.llm_errors import LLMError, LLMConfigError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from app.db.database import GenesisDB


def _make_agent(workdir):
    registry = ModelRegistry(storage_dir=os.path.join(workdir, "registry"))
    trainer = DatasetTrainer(model_dir=os.path.join(workdir, "models"))
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    agent = GenesisAgent(profiler=DatasetProfiler(), trainer=trainer, registry=registry, db=db)
    agent.current_user_id = "test_user"
    return agent


class TestGenesisAgent:
    @pytest.fixture
    def workdir(self):
        d = tempfile.mkdtemp()
        prev_cwd = os.getcwd()
        os.chdir(d)
        os.makedirs("datasets", exist_ok=True)
        yield d
        os.chdir(prev_cwd)
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def agent(self, workdir):
        return _make_agent(workdir)

    @pytest.fixture
    def sample_csv(self, workdir):
        # Must live under datasets/ (relative to the chdir'd workdir) - GenesisAgent._validate_path
        # only allows datasets/, models/, static/, generated/, and correctly rejects everything
        # else (e.g. plain /tmp). That's a real security control, not a test inconvenience.
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 28, 33, 38, 43, 48],
                            "salary": [50, 60, 75, 90, 110, 55, 70, 85, 100, 120],
                            "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]})
        p = "datasets/agent_test_sample.csv"
        df.to_csv(p, index=False)
        yield p

    def test_agent_creation(self, agent):
        assert agent is not None

    def test_process_analyze(self, agent, sample_csv):
        with patch.object(agent, "_call_llm", return_value={
                "tool": "analyze_dataset", "args": {"path": sample_csv}, "message": "Analyzing..."}):
            resp = agent.process(f"analyze {sample_csv}")
        assert resp.success, resp.message

    def test_process_train(self, agent, sample_csv):
        with patch.object(agent, "_call_llm", return_value={
                "tool": "analyze_dataset", "args": {"path": sample_csv}, "message": "Analyzing..."}):
            agent.process(f"analyze {sample_csv}")
        with patch.object(agent, "_call_llm", return_value={
                "tool": "train_model", "args": {"path": sample_csv, "target": "churn"}, "message": "Training..."}):
            resp = agent.process("train xgboost on churn")
        assert resp.success, resp.message
        assert "model_id" in resp.data

    def test_process_find_best(self, agent, sample_csv):
        with patch.object(agent, "_call_llm", return_value={
                "tool": "train_model", "args": {"path": sample_csv, "target": "churn"}, "message": "Training..."}):
            agent.process("train xgboost on churn")
        with patch.object(agent, "_call_llm", return_value={
                "tool": "find_best_model", "args": {}, "message": "Looking..."}):
            resp = agent.process("show the best model")
        assert isinstance(resp, AgentResponse)

    def test_process_search(self, agent):
        with patch.object(agent, "_call_llm", return_value={
                "tool": "search_datasets", "args": {"query": "churn prediction"}, "message": "Searching..."}):
            resp = agent.process("find a dataset for churn prediction")
        assert isinstance(resp, AgentResponse)

    def test_process_unknown_falls_back_to_plain_reply(self, agent):
        with patch.object(agent, "_call_llm", return_value={"tool": None, "message": "It's sunny today!"}):
            resp = agent.process("what's the weather like today")
        assert isinstance(resp, AgentResponse)
        assert resp.message == "It's sunny today!"


class TestLLMFailureHandling:
    """Regression tests for the silent-failure fix: a real provider failure must be reported
    as a real error, not swallowed into a fake HTTP-200-equivalent success (see CHANGES.md /
    app.core.llm_errors)."""

    @pytest.fixture
    def workdir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def agent(self, workdir):
        return _make_agent(workdir)

    def test_no_groq_key_and_no_local_model_raises_config_error(self, agent):
        agent.groq_key = None
        agent.llm = None
        with pytest.raises(LLMConfigError):
            agent._call_llm("hello")

    def test_groq_provider_error_propagates_when_no_local_fallback(self, agent):
        agent.groq_key = "fake-key-for-test"
        agent.llm = None
        with patch.object(agent, "_call_groq", side_effect=LLMProviderError("connection reset")):
            with pytest.raises(LLMProviderError):
                agent._call_llm("hello")

    def test_groq_rate_limit_error_propagates(self, agent):
        agent.groq_key = "fake-key-for-test"
        agent.llm = None
        with patch.object(agent, "_call_groq", side_effect=LLMRateLimitError("rate limited")):
            with pytest.raises(LLMRateLimitError):
                agent._call_llm("hello")

    def test_groq_timeout_error_propagates(self, agent):
        agent.groq_key = "fake-key-for-test"
        agent.llm = None
        with patch.object(agent, "_call_groq", side_effect=LLMTimeoutError("timed out")):
            with pytest.raises(LLMTimeoutError):
                agent._call_llm("hello")

    def test_falls_back_to_real_local_model_when_one_is_actually_loaded(self, agent):
        """If Groq fails but a real local model IS loaded, that's a genuine degraded-mode
        fallback and should succeed - this is different from the old bug, where there was no
        real local model at all and a canned reply was returned anyway."""
        agent.groq_key = "fake-key-for-test"
        agent.llm = object()  # stand-in for "a local model really is loaded"
        with patch.object(agent, "_call_groq", side_effect=LLMProviderError("connection reset")), \
             patch.object(agent, "_call_local", return_value={"tool": None, "message": "local reply"}) as mock_local:
            result = agent._call_llm("hello")
        mock_local.assert_called_once()
        assert result == {"tool": None, "message": "local reply"}

    def test_process_lets_llm_error_propagate_instead_of_faking_success(self, agent):
        """process() must NOT catch LLMError and turn it into a normal AgentResponse - the
        route layer (main.py's /api/agent/message) is what maps it to an HTTP error, and it
        can only do that if the exception actually reaches it."""
        with patch.object(agent, "_call_llm", side_effect=LLMProviderError("provider down")):
            with pytest.raises(LLMProviderError):
                agent.process("hello")

    def test_local_call_without_a_loaded_model_raises_instead_of_greeting(self, agent):
        agent.llm = None
        with pytest.raises(LLMConfigError):
            agent._call_local("hello")
