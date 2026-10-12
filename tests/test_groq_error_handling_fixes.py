"""
Regression tests for two real bugs found in live production logs (a real Groq key, not a
mock): a 403 PermissionDeniedError from Groq (the API key lacks access to a specific model)
being mis-mapped to a generic "try again" error, and BotBuilder crashing when certain models
(openai/gpt-oss-120b observed in practice) spontaneously try to call an undeclared tool.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import httpx
import groq
from unittest.mock import patch, MagicMock
from app.core.agent import GenesisAgent
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry
from app.db.database import GenesisDB
from app.core.llm_errors import LLMConfigError
from app.core.bot_builder import BotBuilder


def _make_groq_error(exc_class, status_code: int, message: str, code: str = None):
    """Builds a real instance of a groq SDK exception, the same shape the SDK itself raises -
    not a generic Exception standing in for it - so these tests actually exercise the `except
    groq_sdk.<SpecificClass>` branch being tested, not just `except Exception`. `code`, when
    given, is placed in the structured error body the way Groq's real API does (e.g.
    "tool_use_failed") - see BotBuilder._is_tool_use_failed_error, which reads exactly this."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": message, "type": "invalid_request_error"}}
    if code:
        body["error"]["code"] = code
    response = httpx.Response(status_code, request=request, json=body)
    return exc_class(message, response=response, body=body)


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp()
    prev = os.getcwd()
    os.chdir(d)
    os.makedirs("datasets", exist_ok=True)
    yield d
    os.chdir(prev)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def agent(workdir):
    db = GenesisDB(db_path=os.path.join(workdir, "test.db"))
    a = GenesisAgent(profiler=DatasetProfiler(), trainer=DatasetTrainer(),
                      registry=ModelRegistry(storage_dir=os.path.join(workdir, "registry")), db=db)
    a.current_user_id = "test_user"
    a.groq_key = "fake-key-for-test"
    a.llm = None
    return a


class TestGroqPermissionDeniedHandling:
    def test_403_permission_denied_maps_to_config_error_not_generic_provider_error(self, agent):
        """Confirmed from production logs: Error code: 403 - {'error': {'message':
        'Forbidden'}} was being caught by the generic APIStatusError branch and mapped to
        LLMProviderError (502, "try again") - misleading, since retrying does nothing for a
        model the key genuinely isn't allowed to use."""
        error = _make_groq_error(groq.PermissionDeniedError, 403, "Forbidden")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = error
            with pytest.raises(LLMConfigError) as exc_info:
                agent._call_groq("hello")
        assert "doesn't have access" in str(exc_info.value)
        assert "console.groq.com" in str(exc_info.value)

    def test_permission_denied_message_names_the_actual_model(self, agent):
        error = _make_groq_error(groq.PermissionDeniedError, 403, "Forbidden")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = error
            with pytest.raises(LLMConfigError) as exc_info:
                agent._call_groq("hello")
        # the model id agent.py resolves to (see MODEL_MAP / _resolve_groq_model) should
        # appear in the error so whoever reads the log/response knows which model to check
        assert agent._resolve_groq_model(agent._get_selected_model()) in str(exc_info.value)

    def test_401_authentication_error_still_maps_correctly(self, agent):
        """Regression guard: adding the new 403 branch right next to the existing 401 one
        must not break the existing, already-correct handling."""
        error = _make_groq_error(groq.AuthenticationError, 401, "Invalid API Key")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = error
            with pytest.raises(LLMConfigError) as exc_info:
                agent._call_groq("hello")
        assert "credentials" in str(exc_info.value)


class TestBotBuilderToolUseFailedHandling:
    def test_groq_chat_returns_honest_deflection_instead_of_none(self):
        """Confirmed from production logs: model='openai/gpt-oss-120b' repeatedly tried to
        invoke fictional tools (repo_browser.open_file, repo_browser.github_lookup,
        repo_browser.search) despite no tools ever being declared in the request, and Groq
        rejected each one with 400 tool_use_failed - previously this made _groq_chat return
        None, which test_draft() would then mark as a hard failure ("empty response or API
        call failed"), even though the honest, correct answer to what the model was trying to
        do (look something up in real time) is exactly the deflection we want anyway."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key-for-test"
        error = _make_groq_error(
            groq.BadRequestError, 400,
            'Tool choice is none, but model called a tool: {"name": "repo_browser.github_lookup"}', code="tool_use_failed")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = error
            result = bb._groq_chat("system prompt", "does this repo exist?")
        assert result is not None
        assert "real time" in result.lower() or "check" in result.lower()

    def test_groq_json_returns_none_cleanly_on_tool_use_failed(self):
        """_groq_json (used for spec generation, not conversation) can't fabricate a fake
        JSON spec the way _groq_chat can fabricate a plausible deflection - it must still
        return None so the caller's existing "generation failed" handling takes over, just
        without crashing or raising an unhandled exception."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key-for-test"
        error = _make_groq_error(groq.BadRequestError, 400, 'Tool choice is none, but model called a tool', code='tool_use_failed')
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = error
            result = bb._groq_json("system prompt", "design a bot")
        assert result is None

    def test_test_draft_trap_question_passes_via_deflection_instead_of_hard_failing(self):
        """End-to-end: the exact failure mode from the logs (trap question about a GitHub
        repo triggering tool_use_failed) should now make the trap question PASS its test
        (deflection = correct, non-hallucinated behavior), not fail with "empty response"."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key-for-test"
        spec = {
            "greeting": "Hi!",
            "system_prompt": "You help with GitHub questions.",
            "test_questions": ["What is Git?", "Does octocat/Hello-World exist?", "How do I clone a repo?"],
            "trap_question_index": 1,
        }
        tool_use_error = _make_groq_error(
            groq.BadRequestError, 400,
            'Tool choice is none, but model called a tool: {"name": "repo_browser.github_lookup"}', code="tool_use_failed")

        def fake_chat_completion(*args, **kwargs):
            messages = kwargs.get("messages", [])
            question = messages[-1]["content"] if messages else ""
            if "octocat" in question:
                raise tool_use_error
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = f"Answer to: {question}"
            mock_resp.usage = None
            return mock_resp

        with patch("groq.Groq") as MockGroq, \
             patch.object(bb, "_check_hallucination", return_value={"hallucinated": False, "reason": "", "verified": True}):
            MockGroq.return_value.chat.completions.create.side_effect = fake_chat_completion
            results = bb.test_draft(spec)

        trap_result = results[2]  # index 0 is greeting, so trap_question_index=1 is results[2]
        assert trap_result["ok"] is True
        assert "real time" in trap_result["response"].lower() or "check" in trap_result["response"].lower()
