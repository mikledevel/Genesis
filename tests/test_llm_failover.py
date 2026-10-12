"""Groq -> OpenRouter -> Gemini failover chain with a per-provider cooldown (circuit breaker)."""
import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import groq, httpx, openai, pytest
from unittest.mock import patch, MagicMock
from app.config import settings
from app.core import groq_client, progress


def _groq_err(cls, status, retry_after=None):
    req = httpx.Request("POST", "https://api.groq.com/x")
    if cls is groq.APITimeoutError:
        return cls(request=req)
    headers = {"retry-after": str(retry_after)} if retry_after else {}
    resp = httpx.Response(status, request=req, json={"error": {"message": "boom"}}, headers=headers)
    return cls("boom", response=resp, body={"error": {"message": "boom"}})


def _openai_err(cls, status):
    req = httpx.Request("POST", "https://openrouter.ai/x")
    return cls("nope", response=httpx.Response(status, request=req, json={"error": {"message": "nope"}}), body=None)


class Env:
    """Three fake providers behind patch('groq.Groq') and a patched openai.OpenAI that routes by base_url."""
    def __init__(self):
        self.groq = MagicMock(); self.openrouter = MagicMock(); self.gemini = MagicMock()
        for m, text in ((self.groq, "from-groq"), (self.openrouter, "from-openrouter"), (self.gemini, "from-gemini")):
            m.chat.completions.create.return_value = MagicMock(provider_text=text)
        self.ctor_calls = []

    def openai_factory(self, **kw):
        self.ctor_calls.append(kw)
        return self.gemini if "generativelanguage" in kw["base_url"] else self.openrouter

    def __enter__(self):
        settings.openrouter_api_key, settings.gemini_api_key = "or-key", "gm-key"
        self._p = [patch("groq.Groq", return_value=self.groq), patch("openai.OpenAI", side_effect=self.openai_factory)]
        [p.start() for p in self._p]; return self

    def __exit__(self, *a): [p.stop() for p in self._p]

    def call(self, **kw):
        return groq_client.call_llm("gk", model="openai/gpt-oss-120b", messages=[{"role": "user", "content": "hi"}], **kw)

    def who(self, result):
        return result.provider_text


def test_groq_is_used_first_and_alone_when_healthy():
    with Env() as e:
        assert e.who(e.call()) == "from-groq"
        e.openrouter.chat.completions.create.assert_not_called(); e.gemini.chat.completions.create.assert_not_called()


def test_groq_fails_openrouter_answers():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        assert e.who(e.call()) == "from-openrouter"
        e.gemini.chat.completions.create.assert_not_called()


def test_groq_and_openrouter_fail_gemini_answers():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.APITimeoutError, 504)
        e.openrouter.chat.completions.create.side_effect = _openai_err(openai.InternalServerError, 502)
        assert e.who(e.call()) == "from-gemini"


def test_everything_fails_raises_the_original_groq_error():
    with Env() as e:
        err = _groq_err(groq.PermissionDeniedError, 403)
        e.groq.chat.completions.create.side_effect = err
        e.openrouter.chat.completions.create.side_effect = RuntimeError("or down")
        e.gemini.chat.completions.create.side_effect = RuntimeError("gem down")
        with pytest.raises(groq.PermissionDeniedError) as exc:
            e.call()
        assert exc.value is err


def test_rate_limited_groq_is_skipped_on_the_next_call():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        e.call(); e.groq.chat.completions.create.reset_mock()
        assert e.who(e.call()) == "from-openrouter"
        e.groq.chat.completions.create.assert_not_called()           # cooling down: no wasted attempt


def test_cooldown_expires_and_groq_is_tried_again():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        e.call()
        with patch("app.core.groq_client.time.monotonic", return_value=1e12):    # far in the future
            e.groq.chat.completions.create.side_effect = None
            assert e.who(e.call()) == "from-groq"
        assert groq_client.status("gk")["providers"][0]["cooling_seconds"] == 0


def test_retry_after_header_sets_the_cooldown():
    assert groq_client._cooldown_for(_groq_err(groq.RateLimitError, 429, retry_after=120)) == 120
    assert groq_client._cooldown_for(_groq_err(groq.RateLimitError, 429)) == settings.llm_cooldown_seconds


def test_cooldown_classes():
    assert groq_client._cooldown_for(_groq_err(groq.AuthenticationError, 401)) == settings.llm_auth_cooldown_seconds
    assert groq_client._cooldown_for(_groq_err(groq.BadRequestError, 400)) == 0          # request-specific
    assert groq_client._cooldown_for(_groq_err(groq.NotFoundError, 404)) == 0
    assert 0 < groq_client._cooldown_for(_groq_err(groq.APITimeoutError, 504)) <= settings.llm_cooldown_seconds
    settings.llm_cooldown_seconds = 0
    assert groq_client._cooldown_for(_groq_err(groq.RateLimitError, 429)) == 0           # breaker disabled


def test_bad_request_does_not_poison_the_provider():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = [_groq_err(groq.BadRequestError, 400), MagicMock(provider_text="from-groq")]
        assert e.who(e.call()) == "from-openrouter"           # this request fell through...
        assert e.who(e.call()) == "from-groq"                 # ...but Groq is not on cooldown for the next one


def test_a_single_configured_provider_is_never_locked_out():
    settings.openrouter_api_key = settings.gemini_api_key = ""
    with patch("groq.Groq") as G:
        G.return_value.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        for _ in range(2):
            with pytest.raises(groq.RateLimitError):
                groq_client.call_llm("gk", model="m", messages=[])
        assert G.return_value.chat.completions.create.call_count == 2    # tried both times despite the cooldown


def test_all_cooling_still_tries_everything_and_raises_groqs_error():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        e.openrouter.chat.completions.create.side_effect = _openai_err(openai.RateLimitError, 429)
        e.gemini.chat.completions.create.side_effect = _openai_err(openai.RateLimitError, 429)
        with pytest.raises(groq.RateLimitError):
            e.call()
        with pytest.raises(groq.RateLimitError):             # second call: all cooling -> tried again, same outcome
            e.call()
        assert e.gemini.chat.completions.create.call_count == 2


def test_groq_skipped_by_cooldown_and_others_fail_still_raises_the_remembered_groq_error():
    with Env() as e:
        g = _groq_err(groq.RateLimitError, 429)
        e.groq.chat.completions.create.side_effect = g
        e.call()                                                         # trips Groq (OpenRouter answered)
        e.openrouter.chat.completions.create.side_effect = RuntimeError("down")
        e.gemini.chat.completions.create.side_effect = RuntimeError("down")
        with pytest.raises(groq.RateLimitError) as exc:
            e.call()
        assert exc.value is g


def test_models_per_provider_and_groq_only_kwargs():
    settings.openrouter_model = "openai/gpt-oss-120b:free"
    settings.gemini_model = "gemini-flash-latest"
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.APITimeoutError, 504)
        e.openrouter.chat.completions.create.side_effect = _openai_err(openai.InternalServerError, 500)
        e.call(groq_only_kwargs={"reasoning_effort": "low"}, temperature=0.1)
        assert e.groq.chat.completions.create.call_args.kwargs["reasoning_effort"] == "low"
        assert e.openrouter.chat.completions.create.call_args.kwargs["model"] == "openai/gpt-oss-120b:free"
        assert e.gemini.chat.completions.create.call_args.kwargs["model"] == "gemini-flash-latest"
        for m in (e.openrouter, e.gemini):
            assert "reasoning_effort" not in m.chat.completions.create.call_args.kwargs
            assert m.chat.completions.create.call_args.kwargs["temperature"] == 0.1


def test_model_lists_are_tried_in_order_on_one_provider():
    settings.gemini_model = "gem-a,gem-b"; settings.openrouter_api_key = ""
    with Env() as e:
        settings.openrouter_api_key = ""
        e.groq.chat.completions.create.side_effect = _groq_err(groq.APITimeoutError, 504)
        e.gemini.chat.completions.create.side_effect = [_openai_err(openai.NotFoundError, 404), MagicMock(provider_text="from-gemini-b")]
        assert e.who(e.call()) == "from-gemini-b"
        assert [c.kwargs["model"] for c in e.gemini.chat.completions.create.call_args_list] == ["gem-a", "gem-b"]


def test_gemini_uses_its_openai_compatible_endpoint_and_key():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.APITimeoutError, 504)
        e.openrouter.chat.completions.create.side_effect = _openai_err(openai.InternalServerError, 500)
        e.call()
        gem = [c for c in e.ctor_calls if "generativelanguage" in c["base_url"]][0]
        assert gem["api_key"] == "gm-key" and gem["base_url"].endswith("/v1beta/openai/")


def test_response_format_rejected_by_a_fallback_is_retried_without_it():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.APITimeoutError, 504)
        e.openrouter.chat.completions.create.side_effect = [_openai_err(openai.BadRequestError, 400), MagicMock(provider_text="from-openrouter")]
        assert e.who(e.call(response_format={"type": "json_object"})) == "from-openrouter"
        calls = e.openrouter.chat.completions.create.call_args_list
        assert "response_format" in calls[0].kwargs and "response_format" not in calls[1].kwargs
        e.gemini.chat.completions.create.assert_not_called()


def test_custom_order_can_demote_groq():
    settings.llm_provider_order = "gemini,groq"
    with Env() as e:
        assert e.who(e.call()) == "from-gemini"
    assert groq_client.provider_order() == ["gemini", "groq"]
    settings.llm_provider_order = "nonsense"
    assert groq_client.provider_order() == list(groq_client.PROVIDERS)


def test_user_is_told_in_the_progress_stream():
    events = []; progress.set_callback(events.append)
    try:
        with Env() as e:
            e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
            e.call()
    finally:
        progress.set_callback(None)
    assert any(ev["type"] == "note" and "Groq" in ev["text"] and "OpenRouter" in ev["text"] for ev in events)


def test_missing_openai_package_degrades_to_groq_only():
    settings.openrouter_api_key, settings.gemini_api_key = "or", "gm"
    real = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    def fake(name, *a, **k):
        if name == "openai": raise ImportError("no openai")
        return real(name, *a, **k)
    with patch("groq.Groq") as G, patch("builtins.__import__", side_effect=fake):
        G.return_value.chat.completions.create.side_effect = _groq_err(groq.APITimeoutError, 504)
        with pytest.raises(groq.APITimeoutError):
            groq_client.call_llm("gk", model="m", messages=[])


def test_status_reports_configuration_and_cooldown_without_secrets():
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        e.call()
        st = groq_client.status("gk")
    by = {p["provider"]: p for p in st["providers"]}
    assert st["order"] == ["groq", "openrouter", "gemini"]
    assert by["groq"]["cooling_seconds"] > 0 and by["groq"]["last_error_type"] == "RateLimitError"
    assert by["openrouter"]["configured"] and by["gemini"]["configured"] and by["gemini"]["cooling_seconds"] == 0
    assert "key" not in str(st).lower().replace("last_error_type", "")


def test_agent_survives_a_groq_outage_end_to_end(tmp_path):
    """The whole path the chat uses: agent routing call -> Groq 429 -> Gemini answers -> JSON parsed."""
    from app.core.agent import GenesisAgent
    from app.core.engine.registry import ModelRegistry
    from app.db.database import GenesisDB
    a = GenesisAgent(registry=ModelRegistry(storage_dir=str(tmp_path / "r")), db=GenesisDB(db_path=str(tmp_path / "t.db")))
    a.groq_key = "gk"
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.RateLimitError, 429)
        e.openrouter.chat.completions.create.side_effect = _openai_err(openai.InternalServerError, 500)
        good = MagicMock(); good.choices[0].message.content = '{"tool": null, "message": "answered by gemini"}'
        good.usage.prompt_tokens = good.usage.completion_tokens = good.usage.total_tokens = 1
        e.gemini.chat.completions.create.return_value = good
        out = a._call_groq("hello")
    assert out["message"] == "answered by gemini"


def test_llm_check_reports_each_provider_and_exit_code(capsys):
    from app.core import llm_check
    with Env() as e:
        e.groq.chat.completions.create.side_effect = _groq_err(groq.AuthenticationError, 401)
        ok = MagicMock(); ok.choices[0].message.content = "ok"
        e.openrouter.chat.completions.create.return_value = ok
        settings.gemini_api_key = ""                           # not configured
        code = llm_check.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Groq" in out and "FAILED" in out and "AuthenticationError" in out
    assert "OpenRouter" in out and "OK" in out
    assert "Gemini" in out and "not configured (set GEMINI_API_KEY in .env)" in out
    assert "or-key" not in out and "gk" not in out.replace("Groq", "")     # never prints keys


def test_llm_check_exit_code_1_when_nothing_answers(capsys):
    from app.core import llm_check
    with Env() as e:
        for m in (e.groq, e.openrouter, e.gemini):
            m.chat.completions.create.side_effect = RuntimeError("down")
        settings.groq_api_key, saved = "gk", settings.groq_api_key
        try:
            assert llm_check.main() == 1
        finally:
            settings.groq_api_key = saved


def test_status_endpoint_requires_login_and_returns_the_chain(tmp_path, monkeypatch):
    import app.main as main_module
    from fastapi.testclient import TestClient
    from app.deps import get_current_user_id
    monkeypatch.chdir(tmp_path)
    client = TestClient(main_module.app)
    main_module.app.dependency_overrides.pop(get_current_user_id, None)
    assert client.get("/api/llm/status").status_code in (401, 403)
    main_module.app.dependency_overrides[get_current_user_id] = lambda: "u1"
    try:
        settings.openrouter_api_key = "or"
        j = client.get("/api/llm/status").json()
    finally:
        main_module.app.dependency_overrides.pop(get_current_user_id, None)
    assert j["order"] == ["groq", "openrouter", "gemini"]
    assert {p["provider"]: p["configured"] for p in j["providers"]}["openrouter"] is True
