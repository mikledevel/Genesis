"""Suite-wide isolation: a developer's own .env (real OpenRouter / Gemini keys, custom provider order)
must never leak into the tests, and the provider cooldown state must never leak between tests."""
import pytest
from app.config import settings
from app.core import groq_client

_FIELDS = ("openrouter_api_key", "openrouter_model", "gemini_api_key", "gemini_model",
           "llm_provider_order", "llm_cooldown_seconds", "llm_auth_cooldown_seconds")


@pytest.fixture(autouse=True)
def _isolated_llm_settings():
    saved = {f: getattr(settings, f) for f in _FIELDS}
    settings.openrouter_api_key = ""
    settings.openrouter_model = ""
    settings.gemini_api_key = ""
    settings.gemini_model = "gemini-flash-latest"
    settings.llm_provider_order = "groq,openrouter,gemini"
    settings.llm_cooldown_seconds = 60
    settings.llm_auth_cooldown_seconds = 900
    groq_client.reset_breaker()
    yield
    for f, v in saved.items():
        setattr(settings, f, v)
    groq_client.reset_breaker()
