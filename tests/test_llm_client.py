"""Tests for app/core/groq_client.py: the shared Groq client factory + call_llm() fallback.

Three things are pinned down here:

1. A REGRESSION: an earlier version of groq_client.py did `from groq import Groq` at module
   level, which silently breaks `unittest.mock.patch("groq.Groq")` everywhere else in this
   suite (test_groq_error_handling_fixes.py, test_bot_builder_tool_calling.py,
   test_groq_usage_tracking.py - 14 patch sites total) - the mock is applied to the `groq`
   module's attribute, but `from X import Y` had already copied a separate reference into
   groq_client's own namespace before the patch runs, so the real (network-calling,
   unmockable in CI) class gets used instead. See groq_client.py's module docstring for the
   full explanation. test_get_groq_client_is_mockable_via_patch_groq_Groq below is the direct
   regression guard for this.

2. call_llm()'s fallback behavior: try Groq, and only if an OpenRouter key is configured, fall
   back to it on ANY Groq-side failure - transparently on success, transparently re-raising
   the ORIGINAL Groq exception (not OpenRouter's) if the fallback also fails, so every
   existing call site's exception handling keeps working unchanged.

3. The heavy/light read-timeout tiers actually select different timeout values.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx
import groq
import pytest
from unittest.mock import patch, MagicMock

from app.core import groq_client
from app.config import settings


def _make_groq_error(exc_class, status_code: int, message: str):
    """Builds a real instance of a groq SDK exception - see
    tests/test_groq_error_handling_fixes.py for why this matters (a generic Exception
    wouldn't exercise the same `except groq.<SpecificClass>` branches production code uses)."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": message, "type": "invalid_request_error"}}
    response = httpx.Response(status_code, request=request, json=body)
    return exc_class(message, response=response, body=body)


@pytest.fixture(autouse=True)
def _reset_openrouter_setting():
    """Every test starts from "OpenRouter not configured" (this repo's real default), and
    restores it afterwards regardless of what a test sets it to - settings is a process-wide
    singleton, so a test that forgets to clean up would otherwise leak into unrelated tests."""
    original = settings.openrouter_api_key
    settings.openrouter_api_key = ""
    yield
    settings.openrouter_api_key = original


class TestGetGroqClientIsMockable:
    def test_get_groq_client_is_mockable_via_patch_groq_Groq(self):
        """THE regression guard: if groq_client.py ever goes back to `from groq import Groq`
        at module level, this fails, because the real (unmockable) class gets constructed
        instead of the patched one."""
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value = "the-mock-instance"
            client = groq_client.get_groq_client("fake-key")
        assert client == "the-mock-instance"

    def test_light_vs_heavy_use_different_read_timeouts(self):
        with patch("groq.Groq") as MockGroq:
            groq_client.get_groq_client("fake-key", heavy=False)
            light_timeout = MockGroq.call_args.kwargs["timeout"]
            groq_client.get_groq_client("fake-key", heavy=True)
            heavy_timeout = MockGroq.call_args.kwargs["timeout"]
        assert light_timeout.read == settings.groq_read_timeout_seconds
        assert heavy_timeout.read == settings.groq_read_timeout_heavy_seconds
        assert heavy_timeout.read > light_timeout.read


class TestCallLlmFallback:
    def test_no_openrouter_key_reraises_original_groq_error_unchanged(self):
        """This repo's default (openrouter_api_key=""): behavior must be identical to before
        call_llm existed - a Groq failure is just a Groq failure."""
        settings.openrouter_api_key = ""
        error = _make_groq_error(groq.APITimeoutError, 504, "timed out")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = error
            with pytest.raises(groq.APITimeoutError):
                groq_client.call_llm("fake-key", model="m", messages=[])

    def test_openrouter_fallback_succeeds_transparently(self):
        settings.openrouter_api_key = "or-fake-key"
        error = _make_groq_error(groq.APITimeoutError, 504, "timed out")
        fake_completion = MagicMock()
        with patch("groq.Groq") as MockGroq, patch("openai.OpenAI") as MockOpenAI:
            MockGroq.return_value.chat.completions.create.side_effect = error
            MockOpenAI.return_value.chat.completions.create.return_value = fake_completion
            result = groq_client.call_llm("fake-key", model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "hi"}])
        assert result is fake_completion
        # OpenRouter must have been asked for the SAME model - that's the whole point (Groq
        # and OpenRouter serve the same open-weight model id, e.g. "openai/gpt-oss-120b").
        assert MockOpenAI.return_value.chat.completions.create.call_args.kwargs["model"] == "openai/gpt-oss-120b"

    def test_fallback_also_failing_reraises_the_ORIGINAL_groq_error_not_openrouters(self):
        """Critical for every existing call site's exception handling: agent.py's `_call_groq`
        has an `except groq.PermissionDeniedError` branch (see
        test_groq_error_handling_fixes.py) that must still see a real groq.* exception even
        when a fallback was attempted and also failed - not some unrelated openai.* exception
        type it has no branch for."""
        settings.openrouter_api_key = "or-fake-key"
        groq_error = _make_groq_error(groq.PermissionDeniedError, 403, "forbidden")
        with patch("groq.Groq") as MockGroq, patch("openai.OpenAI") as MockOpenAI:
            MockGroq.return_value.chat.completions.create.side_effect = groq_error
            MockOpenAI.return_value.chat.completions.create.side_effect = RuntimeError("openrouter also down")
            with pytest.raises(groq.PermissionDeniedError):
                groq_client.call_llm("fake-key", model="m", messages=[])

    def test_groq_only_kwargs_not_forwarded_to_openrouter(self):
        """reasoning_effort is a Groq-specific extension - sending it to a different provider
        risks THAT request also being rejected, turning a would-be-successful fallback into a
        guaranteed failure."""
        settings.openrouter_api_key = "or-fake-key"
        error = _make_groq_error(groq.APITimeoutError, 504, "timed out")
        with patch("groq.Groq") as MockGroq, patch("openai.OpenAI") as MockOpenAI:
            MockGroq.return_value.chat.completions.create.side_effect = error
            MockOpenAI.return_value.chat.completions.create.return_value = MagicMock()
            groq_client.call_llm("fake-key", model="m", messages=[], groq_only_kwargs={"reasoning_effort": "low"})
        assert "reasoning_effort" not in MockOpenAI.return_value.chat.completions.create.call_args.kwargs
        # ...but Groq itself must still have received it.
        assert MockGroq.return_value.chat.completions.create.call_args.kwargs["reasoning_effort"] == "low"

    def test_openrouter_configured_but_openai_package_missing_degrades_gracefully(self):
        """If someone sets OPENROUTER_API_KEY without running `pip install -r
        requirements.txt` again, the app must not crash with ImportError - it should just
        behave as if the fallback weren't configured."""
        settings.openrouter_api_key = "or-fake-key"
        error = _make_groq_error(groq.APITimeoutError, 504, "timed out")
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *a, **kw):
            if name == "openai":
                raise ImportError("no module named openai")
            return real_import(name, *a, **kw)

        with patch("groq.Groq") as MockGroq, patch("builtins.__import__", side_effect=fake_import):
            MockGroq.return_value.chat.completions.create.side_effect = error
            with pytest.raises(groq.APITimeoutError):
                groq_client.call_llm("fake-key", model="m", messages=[])
