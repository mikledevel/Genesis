"""Shared imports, request models, helpers and service singletons for the route modules.

Everything here used to live at the top of app/main.py, in the same order (load_dotenv() still
runs before anything reads the environment). Route modules import what they use from here;
app/main.py only builds the FastAPI app, middleware and startup hooks and includes the routers.
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()  # must run before GenesisAgent reads GROQ_API_KEY from os.environ
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union

from app.core.auth import get_current_user_id, get_current_user_id_optional, get_api_key_from_request, hash_password, verify_password, create_token
from app.core.rate_limit import RateLimitMiddleware
from app.core.geo_block import GeoBlockMiddleware
from app.config import settings

# Error monitoring - optional, only activates if SENTRY_DSN is set (see app/config.py).
# Without it, unhandled errors just print() to the log as before, which means an operator
# only finds out about a production outage when a user complains. This does not replace the
# explicit success/failure logging already done for LLM calls (see GenesisAgent._log_llm_call)
# - it catches everything else: unhandled exceptions anywhere in a request.
if settings.sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    sentry_sdk.init(dsn=settings.sentry_dsn, integrations=[FastApiIntegration()],
                     traces_sample_rate=0.1, environment="production" if not settings.debug else "development")
# In debug/local dev, allow any origin so the frontend can be served from anywhere without
# fiddling with config. In production, ALLOWED_ORIGINS must be set to the real frontend
# domain(s) - "*" in production would let any website make authenticated requests against
# this API from a visitor's browser (their cookies/tokens, our CORS headers granting it).
_allowed_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
if settings.debug and not _allowed_origins:
    _allowed_origins = ["*"]
elif not settings.debug and not _allowed_origins:
    raise RuntimeError(
        "ALLOWED_ORIGINS is not set. In production (debug=False), you must set it to your "
        "real frontend domain(s), e.g. ALLOWED_ORIGINS=https://your-domain.com - leaving "
        "this unset would otherwise require defaulting to '*', which lets any website make "
        "authenticated requests against this API from a visitor's browser."
    )

# Refuse to boot with the placeholder JWT signing secret unless we're explicitly in
# debug/dev mode. Every token issued under "change-me" is forgeable by anyone who has
# read this file (i.e. anyone), so this must never reach a real deployment silently.
if not settings.debug and settings.secret_key == "change-me":
    raise RuntimeError(
        "settings.secret_key is still the placeholder 'change-me'. Set a real, random "
        "SECRET_KEY in the environment before running with debug=False. "
        "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
    )

_agents: Dict[str, Any] = {}
def get_agent(conversation_id: str, user_id: str):
    key = f"{user_id}:{conversation_id}"
    if key not in _agents:
        from app.core.agent import GenesisAgent
        _agents[key] = GenesisAgent()
    return _agents[key]

class MessageRequest(BaseModel):
    message: str; conversation_id: Optional[str] = None
    projects: Optional[List[Dict]] = None
    attachments: Optional[List[str]] = None   # ids returned by POST /api/agent/upload

class PublishRequest(BaseModel):
    model_id: str; title: str; description: str
    task: str; model_type: str
    metrics: Dict[str, float] = {}; price_per_call: float = 0.01

class RegisterRequest(BaseModel):
    email: str; password: str; name: str = ""

class LoginRequest(BaseModel):
    email: str; password: str

class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsRequest(BaseModel):
    """Body for POST /api/v1/chat/completions - the public developer API. Deliberately named
    and shaped like OpenAI's chat-completions request so existing tooling/snippets built for
    that shape need minimal changes to point at this platform instead."""
    model: Optional[str] = None
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7

class UseModelRequest(BaseModel):
    features: List[Union[float, str]] = []   # raw values; categorical models take strings

# ── Payments (Stripe - cards + USDC via the same Checkout flow) ──
class CheckoutRequest(BaseModel):
    amount_usd: float

# ── Public Developer API ──
# OpenAI-compatible-shaped chat completions endpoint for third-party integrations: point an
# existing OpenAI- or Groq-SDK-based client at this platform (base_url + a Genesis API key in
# place of the provider's own) and it works with minimal changes. Priced per token and billed
# against the caller's platform account balance - top up the same way as the web app (POST
# /api/payments/checkout, or POST /api/payments/dev-credit in a DEBUG=true environment). See
# app/config.py's public_api_* settings for pricing/limits and
# app/core/public_api_billing.py for the pre-charge/reconcile billing logic that makes the
# per-token pricing possible without knowing the real cost until after generation.
#
# Model list: the same non-retired ids GenesisAgent.MODEL_MAP resolves internally (see that
# map's comments for which Groq ids are currently live vs retired) - kept as an explicit
# allowlist here rather than accepting any string, so an unsupported/typo'd model name fails
# fast with a clear 400 instead of an opaque provider-side error several seconds later.
PUBLIC_API_SUPPORTED_MODELS = {
    "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "qwen/qwen3.6-27b",
}

# ── Knowledge Base (per-bot uploaded documents, retrieved at chat time - see
# app.core.knowledge_base and GenesisAgent._get_effective_system_prompt) ──

SUPPORTED_KB_EXTENSIONS = (".txt", ".md", ".csv")

def _require_bot_owner(agent_id: str, user_id: str):
    from app.core.agent_store import AgentStore
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")

# ── Telegram integration (connect a published bot to a real Telegram bot via a BotFather
# token) - see app.core.telegram_integration for the low-level Bot API client. Incoming
# messages are routed through the EXACT SAME GenesisAgent.process() pipeline the web chat
# uses (agent_override carries agent_id), so tool-scoping, knowledge base retrieval,
# predict_with_model, human handoff, and the Groq quota gate all behave identically over
# Telegram as they do in the web widget - nothing about the dispatch logic is duplicated. ──

import secrets as _secrets

# ── Bot Builder (no-code: describe -> AI designs+tests+fixes -> preview -> publish) ──

def _enforce_groq_quota(user_id: Optional[str]):
    """Raises HTTP 429 if this account is over its monthly Groq usage cap - see
    app.core.quota. Called directly by these REST endpoints because BotBuilder/ModelBuilder
    are invoked here without ever going through GenesisAgent.process(), which has its own
    version of this same gate for the chat-driven paths (train_model, create_bot, etc.) -
    without this, a bot-builder session (each build can be 5-15+ real Groq calls between the
    generate/test/fix loop) would completely bypass the cap."""
    from app.db.database import GenesisDB
    from app.core.quota import check_quota, quota_exceeded_message
    allowed, used, limit = check_quota(GenesisDB(), user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=quota_exceeded_message(used, limit))
