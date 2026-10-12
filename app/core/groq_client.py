"""Single shared entry point for every outbound LLM call in the codebase.

Before this existed, each call site (agent.py x2, bot_builder.py x2, ai_generator.py,
skill_generator.py) constructed its own `Groq(...)` client with the timeout/retry settings
copy-pasted inline. That duplication is exactly how the timeout tuning drifted out of sync with
reality once before - fix it in one place next time by editing this file, not by grepping for
every `Groq(api_key=` in the repo.

IMPORTANT (regression fixed here): this module imports the SDKs as `import groq` / lazily
`import openai`, and always calls through the module attribute (`groq.Groq(...)`), never via
`from groq import Groq`. The existing test suite mocks the provider with
`unittest.mock.patch("groq.Groq")`, which replaces the `Groq` attribute on the `groq` module
object. `from groq import Groq` at import time copies a *separate* reference into this module's
own namespace - patching `groq.Groq` afterwards does not touch that copy, so the mock is silently
never used and a real (failing, unmockable) network call is attempted instead. An earlier version
of this file did exactly that and broke all 14 `patch("groq.Groq")` sites across
tests/test_groq_error_handling_fixes.py, tests/test_bot_builder_tool_calling.py, and
tests/test_groq_usage_tracking.py. `import groq; groq.Groq(...)` looks up the attribute fresh on
every call, so a patch applied before the call is always seen. See tests/test_llm_client.py for a
regression test that pins this down directly (constructs the client through this module while
`groq.Groq` is mocked, and asserts the mock - not the real class - was used).

See app/config.py for *why* the timeout is split into connect/read/heavy-read, and for the
OpenRouter fallback settings used by `call_llm` below.
"""
import threading
import time
from typing import Dict, List, Optional, Tuple

import httpx

from app.config import settings


def get_groq_client(api_key: str, heavy: bool = False):
    """Build a Groq client with this platform's shared timeout/retry policy applied.

    heavy=True uses the longer read timeout (see settings.groq_read_timeout_heavy_seconds) for
    calls known to generate a lot of structured output in one shot (a full bot spec, generated
    code) rather than a short conversational reply.
    """
    import groq
    timeout = httpx.Timeout(
        connect=settings.groq_connect_timeout_seconds,
        read=settings.groq_read_timeout_heavy_seconds if heavy else settings.groq_read_timeout_seconds,
        write=10.0,
        pool=5.0,
    )
    return groq.Groq(api_key=api_key, timeout=timeout, max_retries=settings.groq_max_retries)


# ───────────────────────── multi-provider chain: Groq -> OpenRouter -> Gemini ─────────────────────────
# call_llm() below tries the configured providers in order (settings.llm_provider_order, default
# "groq,openrouter,gemini"). Groq stays the primary; a provider is only used if its API key is set.
# OpenRouter and Gemini are reached through their OpenAI-compatible endpoints with the `openai`
# package. A provider that just failed is put on a short COOLDOWN (a circuit breaker), so the next
# requests skip it instead of paying for another doomed attempt - e.g. after a Groq 429 or timeout
# the following messages go straight to OpenRouter for a minute rather than each waiting on Groq.
# If every provider fails, the ORIGINAL Groq error is raised, so each caller's `except groq.X`
# branches keep working unchanged.
PROVIDERS = ("groq", "openrouter", "gemini")
PRETTY = {"groq": "Groq", "openrouter": "OpenRouter", "gemini": "Gemini"}
_lock = threading.Lock()
_cooldown_until: Dict[Tuple[str, str], float] = {}     # (provider, model) -> time.monotonic() deadline
_last_error: Dict[Tuple[str, str], Exception] = {}


def _timeout(heavy: bool) -> "httpx.Timeout":
    return httpx.Timeout(
        connect=settings.groq_connect_timeout_seconds,
        read=settings.groq_read_timeout_heavy_seconds if heavy else settings.groq_read_timeout_seconds,
        write=10.0, pool=5.0)


def _openai_compatible_client(api_key: str, base_url: str, heavy: bool):
    """None when the key is missing or the `openai` package isn't installed (degrade, don't crash)."""
    if not api_key:
        return None
    try:
        import openai
    except ImportError:
        print("[LLM] a fallback key is set but the 'openai' package is not installed - "
              "run `pip install -r requirements.txt`. Fallback disabled.")
        return None
    return openai.OpenAI(api_key=api_key, base_url=base_url, timeout=_timeout(heavy),
                         max_retries=settings.groq_max_retries)


def _get_openrouter_client(heavy: bool = False):
    return _openai_compatible_client(settings.openrouter_api_key, settings.openrouter_base_url, heavy)


def _get_gemini_client(heavy: bool = False):
    return _openai_compatible_client(settings.gemini_api_key, settings.gemini_base_url, heavy)


def _split(value) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def provider_order() -> List[str]:
    order = []
    for name in _split(settings.llm_provider_order.lower()):
        if name in PROVIDERS and name not in order:
            order.append(name)
    return order or list(PROVIDERS)


def models_for(name: str, groq_model: str) -> List[str]:
    """Groq gets the caller's model. OpenRouter defaults to the SAME model id (Groq and OpenRouter
    both serve e.g. openai/gpt-oss-120b) unless OPENROUTER_MODEL says otherwise; Gemini always uses
    GEMINI_MODEL. Both settings accept a comma-separated list: the models are tried in order."""
    if name == "groq":
        return [groq_model]
    if name == "openrouter":
        return _split(settings.openrouter_model) or [groq_model]
    return _split(settings.gemini_model) or ["gemini-flash-latest"]


def provider_plan(api_key: str, model: str, heavy: bool = False) -> List[Tuple[str, str, object]]:
    """[(provider, model, client)] for every configured provider, in order."""
    plan = []
    for name in provider_order():
        client = (get_groq_client(api_key, heavy) if api_key else None) if name == "groq" else \
                 _get_openrouter_client(heavy) if name == "openrouter" else _get_gemini_client(heavy)
        if client is None:
            continue
        for m in models_for(name, model):
            plan.append((name, m, client))
    return plan


def reset_breaker() -> None:
    with _lock:
        _cooldown_until.clear()
        _last_error.clear()


def _cooling_seconds(key) -> float:
    with _lock:
        return max(0.0, _cooldown_until.get(key, 0.0) - time.monotonic())


def _cooldown_for(error: Exception) -> float:
    """How long to leave a provider alone after this error. 0 = request-specific failure
    (bad request, unknown model...), which says nothing about the provider's health."""
    if settings.llm_cooldown_seconds <= 0:
        return 0.0
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    if status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        return float(settings.llm_auth_cooldown_seconds)
    if status == 429 or "RateLimit" in name:
        retry_after = None
        try:
            retry_after = float(error.response.headers.get("retry-after"))
        except Exception:
            pass
        return min(max(retry_after or float(settings.llm_cooldown_seconds), 5.0), 600.0)
    if status in (400, 404, 413, 422) or any(x in name for x in ("BadRequest", "NotFound", "Unprocessable")):
        return 0.0
    return max(5.0, settings.llm_cooldown_seconds / 2)      # timeouts, connection errors, 5xx


def _trip(key, error: Exception) -> None:
    seconds = _cooldown_for(error)
    with _lock:
        _last_error[key] = error
        if seconds > 0:
            _cooldown_until[key] = time.monotonic() + seconds


def _clear(key) -> None:
    with _lock:
        _cooldown_until.pop(key, None)


def _is_bad_request(error: Exception) -> bool:
    return getattr(error, "status_code", None) in (400, 422) or "BadRequest" in type(error).__name__


def _create(client, name: str, model: str, messages, kwargs: dict, groq_only_kwargs: Optional[dict]):
    kw = dict(kwargs)
    if name == "groq" and groq_only_kwargs:
        kw.update(groq_only_kwargs)       # reasoning_effort etc. are Groq extensions: never sent elsewhere
    try:
        return client.chat.completions.create(model=model, messages=messages, **kw)
    except Exception as e:
        # Some OpenRouter free models / Gemini variants reject response_format; the prompts already
        # demand JSON, so retry once without it before declaring the provider failed.
        if name != "groq" and "response_format" in kw and _is_bad_request(e):
            kw.pop("response_format")
            return client.chat.completions.create(model=model, messages=messages, **kw)
        raise


def call_llm(api_key: str, *, model: str, messages, heavy: bool = False,
             groq_only_kwargs: Optional[dict] = None, **kwargs):
    """Drop-in replacement for `client.chat.completions.create(...)` with automatic failover.

    `api_key` is the GROQ key (every existing call site already has it). Returns the first
    successful completion; raises the original Groq error if nothing worked."""
    plan = provider_plan(api_key, model, heavy)
    if not plan:        # nothing configured at all: fail exactly as a bare Groq call would
        return get_groq_client(api_key, heavy).chat.completions.create(
            model=model, messages=messages, **dict(kwargs, **(groq_only_kwargs or {})))

    active = [p for p in plan if _cooling_seconds((p[0], p[1])) <= 0] or plan   # all cooling -> try anyway
    skipped = [p for p in plan if p not in active]
    errors: List[Tuple[str, str, Exception]] = []
    for name, m, client in active:
        key = (name, m)
        try:
            completion = _create(client, name, m, messages, kwargs, groq_only_kwargs)
        except Exception as e:
            errors.append((name, m, e))
            _trip(key, e)
            print(f"[LLM] {PRETTY[name]} failed for model='{m}' ({type(e).__name__}: {str(e)[:160]})"
                  f"{' - trying the next provider' if (name, m, client) != active[-1] else ''}")
            continue
        _clear(key)
        if errors or skipped:
            first = (errors[0][0] if errors else skipped[0][0])
            print(f"[LLM] {PRETTY[first]} unavailable - answered by {PRETTY[name]} (model='{m}').")
            try:
                from app.core import progress
                progress.note(f"{PRETTY[first]} is unavailable right now \u2014 answered by {PRETTY[name]} instead.")
            except Exception:
                pass
        return completion

    # Everything failed. Prefer the Groq error (callers map groq.* types), even if Groq was skipped this
    # time because of its cooldown - the remembered error is the true reason it was skipped.
    for name, m, _ in plan:
        if name == "groq":
            err = next((e for n, mm, e in errors if n == "groq"), None) or _last_error.get((name, m))
            if err is not None:
                raise err
    raise errors[0][2]


def status(api_key: str) -> Dict:
    """Which providers are configured and which are cooling down (no secrets, no error texts)."""
    out = []
    for name in provider_order():
        configured = bool(api_key) if name == "groq" else bool(
            settings.openrouter_api_key if name == "openrouter" else settings.gemini_api_key)
        models = models_for(name, "(the model the caller asks Groq for)") if configured else []
        with _lock:                       # cooldowns are keyed (provider, model): report the provider's worst
            now = time.monotonic()
            cooling = max([d - now for (n, _), d in _cooldown_until.items() if n == name] or [0.0])
            last = next((type(e).__name__ for (n, _), e in _last_error.items() if n == name), None)
        cooling = max(0.0, cooling)
        out.append({"provider": name, "configured": configured, "models": models,
                    "cooling_seconds": round(cooling), "last_error_type": last if cooling > 0 else None})
    return {"order": provider_order(), "providers": out}
