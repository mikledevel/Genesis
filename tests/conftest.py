"""Suite-wide isolation: a developer's own .env (real OpenRouter / Gemini keys, custom provider order)
must never leak into the tests, and the provider cooldown state must never leak between tests."""
import pytest
from app.config import settings
from app.core import groq_client
from app.core.engine.trainer import DatasetTrainer

_FIELDS = ("openrouter_api_key", "openrouter_model", "gemini_api_key", "gemini_model",
           "llm_provider_order", "llm_cooldown_seconds", "llm_auth_cooldown_seconds",
           "llm_missing_model_cooldown_seconds", "groq_fallback_models", "admin_emails",
           "resend_api_key", "email_from", "turnstile_site_key", "turnstile_secret_key", "smtp_host",
           "disposable_email_domains_extra", "stripe_subscriptions_enabled", "crypto_payments_enabled",
           "agent_tool_protocol", "kb_retrieval_mode", "kb_semantic_threshold", "kb_semantic_margin", "kb_relevance_gate")


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
    settings.llm_missing_model_cooldown_seconds = 3600
    settings.groq_fallback_models = ""        # tests opt in; keeps single-model call counts deterministic
    settings.admin_emails = ""                # nobody is an admin unless a test says so
    settings.resend_api_key = settings.email_from = settings.turnstile_site_key = settings.turnstile_secret_key = ""
    settings.smtp_host = settings.disposable_email_domains_extra = ""     # no real mail, no captcha unless a test turns it on
    settings.stripe_subscriptions_enabled = True
    settings.crypto_payments_enabled = False
    settings.agent_tool_protocol = "json"
    settings.kb_retrieval_mode = "lexical"
    settings.kb_relevance_gate = 0.16
    settings.kb_semantic_threshold = 0.6
    settings.kb_semantic_margin = 0.05
    groq_client.reset_breaker()
    from app.core.quota import quota_owner
    context_token = quota_owner.set(None)

    saved_budget = DatasetTrainer.DEFAULT_BUDGET
    DatasetTrainer.DEFAULT_BUDGET = "fast"      # the product default is 'balanced'; tests opt in to tuning
    yield
    DatasetTrainer.DEFAULT_BUDGET = saved_budget
    quota_owner.reset(context_token)
    for f, v in saved.items():
        setattr(settings, f, v)
    groq_client.reset_breaker()

@pytest.fixture(autouse=True)
def _isolated_request_limits():
    import app.main
    current = app.main.app.middleware_stack
    while current is not None:
        limiter = getattr(current, "_limiter", None)
        if limiter is not None:
            limiter._hits.clear()
            limiter._expires.clear()
        current = getattr(current, "app", None)



@pytest.fixture(autouse=True)
def _deterministic_sandbox(monkeypatch,request):
    if request.node.get_closest_marker("real_docker") is None:
        monkeypatch.setattr("app.core.codegen.docker_sandbox.docker_available",lambda:False)
