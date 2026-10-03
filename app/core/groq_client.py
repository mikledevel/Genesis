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


def _get_openrouter_client(heavy: bool = False):
    """Build an OpenRouter client, or return None if it isn't configured/installed.

    OpenRouter exposes an OpenAI-compatible `/chat/completions` endpoint, so the official
    `openai` package (not `groq`) is what talks to it - same request/response shape as Groq's
    SDK (`.chat.completions.create(...)` -> an object with `.choices[0].message.content` and
    `.usage`), which is exactly why `call_llm` below can fail over between the two without the
    caller having to know or care which provider actually answered.
    """
    if not settings.openrouter_api_key:
        return None
    try:
        import openai
    except ImportError:
        print("[LLM] openrouter_api_key is set but the 'openai' package isn't installed "
              "(pip install -r requirements.txt) - skipping fallback.")
        return None
    timeout = httpx.Timeout(
        connect=settings.groq_connect_timeout_seconds,
        read=settings.groq_read_timeout_heavy_seconds if heavy else settings.groq_read_timeout_seconds,
        write=10.0,
        pool=5.0,
    )
    return openai.OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url,
                          timeout=timeout, max_retries=0)


def call_llm(api_key: str, *, model: str, messages: list, heavy: bool = False,
             groq_only_kwargs: dict = None, **kwargs):
    """Chat-completion call with automatic cross-provider fallback.

    Drop-in replacement for `get_groq_client(api_key).chat.completions.create(model=model,
    messages=messages, **kwargs)`: tries Groq first, and - only if settings.openrouter_api_key
    is configured - automatically retries the exact same request via OpenRouter before giving
    up, on ANY Groq-side failure (bad/missing key, rate limit, timeout, capacity, outage,
    malformed response). This is what turns "Groq is at capacity" from a user-facing error into
    an invisible retry, instead of just failing faster (which is all a timeout tweak alone can
    do - see app/config.py's history note on groq_read_timeout_seconds).

    `kwargs` (temperature, max_tokens, response_format, tools, tool_choice, ...) are sent to
    BOTH providers - they're standard OpenAI-compatible chat-completions fields.
    `groq_only_kwargs` (e.g. {"reasoning_effort": "low"} - see each call site's
    `_groq_kwargs_for`) is applied ONLY to the Groq attempt and deliberately never forwarded to
    OpenRouter: it's a Groq-specific extension to the schema, and sending an unrecognized field
    to a different provider risks the fallback attempt itself being rejected with a 400 - which
    would turn a would-be-successful fallback into a guaranteed failure. Better to lose one
    provider-specific tuning knob on the fallback path than to break the fallback entirely.

    On failure, re-raises whatever exception GROQ raised (not OpenRouter's) if the fallback
    isn't configured, isn't attempted, or also fails - so every existing call site's exception
    handling (agent.py's typed LLMError mapping via `except groq_sdk.RateLimitError` etc., or
    the simpler "log and return None" sites elsewhere) keeps working completely unchanged. This
    function only ever returns a completion; it never invents a new failure mode for callers to
    handle.
    """
    client = get_groq_client(api_key, heavy=heavy)
    try:
        return client.chat.completions.create(model=model, messages=messages,
                                                **kwargs, **(groq_only_kwargs or {}))
    except Exception as groq_error:
        fallback_client = _get_openrouter_client(heavy=heavy)
        if fallback_client is None:
            raise
        try:
            completion = fallback_client.chat.completions.create(model=model, messages=messages, **kwargs)
            print(f"[LLM] Groq failed ({type(groq_error).__name__}: {groq_error}) - "
                  f"OpenRouter fallback succeeded for model='{model}'.")
            return completion
        except Exception as fallback_error:
            print(f"[LLM] OpenRouter fallback ALSO failed for model='{model}' "
                  f"({type(fallback_error).__name__}: {fallback_error}). Raising the original "
                  f"Groq error so existing error-mapping still applies.")
            raise groq_error
