import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()  # must run before GenesisAgent reads GROQ_API_KEY from os.environ
from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.core.auth import get_current_user_id, get_current_user_id_optional, hash_password, verify_password, create_token
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

app = FastAPI(title="Genesis AI", version="1.0.0")
# Order matters: Starlette applies the LAST-added middleware outermost, so it must be added
# last to wrap everything else, including early-exit responses (like a 429 from the rate
# limiter) that never reach the route handler. CORS goes last so those responses still carry
# CORS headers - otherwise a rate-limited response looks like a CORS failure in the browser
# instead of the 429 it actually is.
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GeoBlockMiddleware)
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
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def _start_background_scheduler():
    # Runs scheduled bot skills (see app/core/scheduler.py) - "Level 3" autonomous checks
    # that fire on a timer rather than in response to a chat message. Safe to start
    # unconditionally: with zero scheduled_jobs rows, the poll loop just wakes up every 30s,
    # finds nothing due, and goes back to sleep.
    from app.core.scheduler import start_scheduler
    start_scheduler()

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

class PublishRequest(BaseModel):
    model_id: str; title: str; description: str
    task: str; model_type: str
    metrics: Dict[str, float] = {}; price_per_call: float = 0.01

class RegisterRequest(BaseModel):
    email: str; password: str; name: str = ""

class LoginRequest(BaseModel):
    email: str; password: str

# ── Auth ──
@app.post("/api/auth/register")
async def auth_register(req: RegisterRequest):
    from app.db.database import GenesisDB
    from app.core.email_sender import send_verification_email
    db = GenesisDB()
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if "@" not in req.email or "." not in req.email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    if db.get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = db.create_user(req.email, hash_password(req.password), req.name)
    token = create_token(user["id"])
    verify_token = db.create_auth_token(user["id"], "verify_email", ttl_seconds=86400)
    send_verification_email(user["email"], verify_token)
    return {"token": token, "user": user}

@app.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    from app.db.database import GenesisDB
    db = GenesisDB()
    user = db.get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_token(user["id"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"], "name": user["name"]}}

@app.get("/api/auth/me")
async def auth_me(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    user = GenesisDB().get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    user["email_verified"] = GenesisDB().is_email_verified(user_id)
    return {"user": user}

@app.post("/api/auth/verify-email/resend")
async def resend_verification_email(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.email_sender import send_verification_email
    db = GenesisDB()
    if db.is_email_verified(user_id):
        return {"status": "already_verified"}
    user = db.get_user_by_id(user_id)
    verify_token = db.create_auth_token(user_id, "verify_email", ttl_seconds=86400)
    send_verification_email(user["email"], verify_token)
    return {"status": "sent"}

@app.post("/api/auth/verify-email/confirm")
async def confirm_verification_email(data: dict):
    from app.db.database import GenesisDB
    db = GenesisDB()
    user_id = db.consume_auth_token(data.get("token", ""), "verify_email")
    if not user_id:
        raise HTTPException(status_code=400, detail="This verification link is invalid or has expired")
    db.mark_email_verified(user_id)
    return {"status": "verified"}

@app.post("/api/auth/password-reset/request")
async def request_password_reset(data: dict):
    from app.db.database import GenesisDB
    from app.core.email_sender import send_password_reset_email
    db = GenesisDB()
    email = (data.get("email") or "").lower().strip()
    user = db.get_user_by_email(email)
    # Always return the same response whether or not the email exists - returning a
    # different response for "no account with that email" would let anyone enumerate which
    # emails are registered on the platform just by hitting this endpoint repeatedly.
    if user:
        reset_token = db.create_auth_token(user["id"], "password_reset", ttl_seconds=3600)
        send_password_reset_email(user["email"], reset_token)
    return {"status": "if_account_exists_email_sent"}

@app.post("/api/auth/password-reset/confirm")
async def confirm_password_reset(data: dict):
    from app.db.database import GenesisDB
    db = GenesisDB()
    new_password = data.get("new_password", "")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    user_id = db.consume_auth_token(data.get("token", ""), "password_reset")
    if not user_id:
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired")
    db.update_password(user_id, hash_password(new_password))
    return {"status": "password_updated"}

# ── Pages ──
@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.get("/builder")
async def builder(): return FileResponse("static/builder/index.html")

# ── Agent ──
@app.get("/api/agent/conversations")
async def agent_conversations(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"conversations": GenesisDB().list_chats(user_id=user_id)}

@app.get("/api/agent/chat/{conversation_id}")
async def agent_chat(conversation_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    owner = db.get_chat_owner(conversation_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": db.get_messages(conversation_id)}

@app.post("/api/agent/message")
async def agent_message(req: MessageRequest, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.llm_errors import LLMError
    cid = req.conversation_id or "default"
    db = GenesisDB()
    owner = db.get_chat_owner(cid)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="This conversation belongs to another account")

    # If the user has a paid bot selected (per-user, stored in DB), a message to it costs
    # money. We only CHECK the balance here (no mutation) so an obviously-unaffordable
    # request fails fast without spending an LLM call. The actual charge happens AFTER
    # agent.process() succeeds, below - previously the charge happened here, before the LLM
    # was ever called, so a Groq outage or any other generation failure still cost the buyer
    # money for a message that was never actually generated.
    selected = db.get_selected_model(user_id)
    price = selected.get("price_per_call", 0) or 0
    bot_author = selected.get("author")
    price_cents = round(price * 100)
    billable = price > 0 and bot_author and bot_author != user_id
    if billable and db.get_balance(user_id) < price_cents:
        raise HTTPException(status_code=402, detail=(
            f"Insufficient balance - this bot costs ${price:.3f}/message, "
            f"you have ${db.get_balance(user_id)/100:.2f}. Add funds to continue."))

    agent = get_agent(cid, user_id)
    # If the currently-selected bot has a running A/B test, use this conversation's assigned
    # variant's system_prompt instead of the bot's normal stored prompt - sticky per
    # conversation_id (see get_or_assign_variant) so a single chat doesn't flip between arms.
    # Reuses the same agent_override mechanism built for external channels (Telegram): it
    # bypasses users.selected_model_json entirely for this one field rather than mutating it,
    # so it can't race with the user's own unrelated model selection.
    agent.agent_override = None
    agent_id = selected.get("agent_id")
    if agent_id:
        ab_test = db.get_running_ab_test_for_agent(agent_id)
        if ab_test and ab_test["variants"]:
            variant = db.get_or_assign_variant(ab_test["id"], ab_test["variants"], cid)
            agent.agent_override = {**selected, "system_prompt": variant["system_prompt"]}

    try:
        response = agent.process(req.message, cid, req.projects, user_id=user_id)
    except LLMError as e:
        # Generation genuinely failed - surface a real error instead of a fake 200, and
        # (critically) never reach the charge_for_usage call below, so nobody is billed for
        # a message the model never produced. See app.core.llm_errors for the status-code
        # mapping (config -> 503, rate limit -> 429, timeout -> 504, generic -> 502).
        raise HTTPException(status_code=e.http_status, detail=(
            "The AI provider failed to generate a response. Please try again in a moment. "
            f"({type(e).__name__})"))

    if billable:
        charge = db.charge_for_usage(
            buyer_user_id=user_id, amount_cents=price_cents,
            description=f"Message to {selected.get('agent_name', 'bot')}",
            author_user_id=bot_author, related_id=agent_id,
        )
        if not charge["ok"]:
            # Balance changed between our pre-check and now (e.g. a concurrent spend) - the
            # response was already generated, so we still return it rather than discard real
            # output, but we do NOT pretend the charge succeeded.
            print(f"[billing] charge_for_usage failed post-generation for user={user_id} agent={agent_id}: "
                  f"balance={charge.get('balance_cents')}")

    if agent_id:
        # Must run AFTER agent.process() - that's what actually creates the chats row
        # (via create_chat inside process()) for a brand-new conversation_id. Tagging before
        # that would silently update zero rows on the very first message of a chat.
        db.tag_chat_with_agent(cid, agent_id)  # links this conversation to the bot for
                                                # analytics - see GenesisDB.get_bot_analytics
    return {"response": response.message, "conversation_id": cid, "data": response.data, "message_id": response.message_id}

@app.post("/api/agent/chat/{conversation_id}/messages/{message_id}/rate")
async def rate_message(conversation_id: str, message_id: int, data: dict, user_id: str = Depends(get_current_user_id)):
    """Thumbs up(1)/down(-1)/clear(null) on an assistant message - the signal A/B test stats
    are built from (see GenesisDB.get_ab_test_stats)."""
    from app.db.database import GenesisDB
    db = GenesisDB()
    if db.get_chat_owner(conversation_id) != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")
    rating = data.get("rating")
    if rating not in (1, -1, None):
        raise HTTPException(status_code=400, detail="rating must be 1, -1, or null")
    ok = db.rate_message(message_id, conversation_id, rating)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found in this conversation")
    return {"status": "ok", "rating": rating}

@app.post("/api/agent/chat/{conversation_id}/rename")
async def rename_chat(conversation_id: str, data: Dict[str, Any], user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    if db.get_chat_owner(conversation_id) != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")
    db.rename_chat(conversation_id, data.get('title', conversation_id))
    return {"status": "renamed"}

@app.delete("/api/agent/chat/{conversation_id}")
async def delete_chat(conversation_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    if db.get_chat_owner(conversation_id) != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")
    db.delete_chat(conversation_id)
    return {"status": "deleted"}

# ── Marketplace ──
@app.get("/api/marketplace/listings")
async def marketplace_listings(task: Optional[str] = None):
    from app.core.marketplace import Marketplace
    return {"listings": Marketplace().search(task=task) if task else Marketplace().search()}

@app.post("/api/marketplace/publish")
async def marketplace_publish(req: PublishRequest, user_id: str = Depends(get_current_user_id)):
    from app.core.marketplace import Marketplace
    from app.core.engine.registry import ModelRegistry
    try:
        ModelRegistry().get_record(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found in Registry - train/register it first")
    lid = Marketplace().publish(author=user_id, model_id=req.model_id, title=req.title, description=req.description, task=req.task, model_type=req.model_type, metrics=req.metrics, price_per_call=req.price_per_call)
    return {"listing_id": lid}

@app.get("/api/marketplace/stats")
async def marketplace_stats():
    from app.core.marketplace import Marketplace
    return Marketplace().get_platform_stats()

class UseModelRequest(BaseModel):
    features: List[float] = []

@app.post("/api/marketplace/{listing_id}/use")
async def marketplace_use(listing_id: str, req: UseModelRequest, user_id: str = Depends(get_current_user_id)):
    from app.core.marketplace import Marketplace
    from app.core.engine.registry import ModelRegistry
    from app.db.database import GenesisDB
    mkt = Marketplace()
    listing = mkt.listings.get(listing_id)
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    if not listing["active"]:
        raise HTTPException(status_code=410, detail="This listing is no longer active")

    price_cents = round(listing["price_per_call"] * 100)
    db = GenesisDB()
    if price_cents > 0:
        result = db.charge_for_usage(
            buyer_user_id=user_id, amount_cents=price_cents,
            description=f"Used model: {listing['title']}",
            author_user_id=listing["author"], platform_fee_pct=Marketplace.COMMISSION * 100,
            related_id=listing_id,
        )
        if not result["ok"]:
            raise HTTPException(status_code=402, detail=(
                f"Insufficient balance - this costs ${listing['price_per_call']:.4f}, "
                f"you have ${result['balance_cents']/100:.2f}. Add funds to continue."))

    try:
        reg = ModelRegistry()
        model = reg.get_model(listing["model_id"])
        import numpy as np
        prediction = model.predict(np.array(req.features).reshape(1, -1))
        mkt.use_model(listing_id, user_id)  # update call count / earnings stats shown in listing
        return {"prediction": prediction.tolist(), "charged_cents": price_cents}
    except Exception as e:
        # The prediction itself failed after we already charged the buyer and credited the
        # author - reverse BOTH sides of that transaction, not just the buyer's refund,
        # or the author keeps an earning for a call that never produced a result.
        if price_cents > 0:
            db.add_transaction(user_id, "usage_refund", price_cents, f"Refund: {listing['title']} (prediction error)", related_id=listing_id)
            author_share = round(price_cents * (1 - Marketplace.COMMISSION))
            if author_share > 0:
                db.add_transaction(listing["author"], "usage_earning_reversal", -author_share,
                                   f"Reversal: {listing['title']} (prediction error)", related_id=listing_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed (refunded): {e}")

# ── Registry ──
@app.get("/api/registry/models")
async def registry_models(task: Optional[str] = None):
    from app.core.engine.registry import ModelRegistry
    return {"models": ModelRegistry().list_models(task=task)}

@app.get("/api/registry/models/{model_id}")
async def registry_model(model_id: str):
    from app.core.engine.registry import ModelRegistry
    try:
        return ModelRegistry().get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")

@app.get("/api/registry/models/{model_id}/history")
async def registry_model_history(model_id: str):
    """Every version in this model's lineage (newest first) - see ModelRegistry.register's
    family_id/version and retrain_model, which is what actually grows this list."""
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    try:
        rec = reg.get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    family_id = rec.get("family_id", rec.get("model_id"))
    return {"family_id": family_id, "versions": reg.get_family_history(family_id)}

@app.post("/api/registry/models/{model_id}/retrain")
async def registry_retrain_model(model_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Retrains an existing registered model on a new/updated dataset, producing a new
    VERSION in the same lineage (see GenesisAgent._do_retrain / ModelRegistry.register)."""
    from app.core.agent import GenesisAgent
    agent = GenesisAgent()
    agent.current_user_id = user_id
    dataset_path = data.get("dataset_path")
    resp = agent._do_retrain(model_id, dataset_path, data)
    if not resp.success:
        raise HTTPException(status_code=400, detail=resp.message)
    return {"message": resp.message, "data": resp.data}

@app.get("/api/registry/best")
async def registry_best(task: str = "binary_classification", metric: str = "accuracy"):
    from app.core.engine.registry import ModelRegistry
    return {"best": ModelRegistry().find_best(task, metric)}

@app.get("/api/registry/stats")
async def registry_stats():
    from app.core.engine.registry import ModelRegistry
    return ModelRegistry().get_stats()

# ── Payments (Stripe - cards + USDC via the same Checkout flow) ──
class CheckoutRequest(BaseModel):
    amount_usd: float

@app.post("/api/payments/checkout")
async def payments_checkout(req: CheckoutRequest, request: Request, user_id: str = Depends(get_current_user_id)):
    from app.core.payments import create_checkout_session, MIN_TOPUP_USD, MAX_TOPUP_USD
    if req.amount_usd < MIN_TOPUP_USD or req.amount_usd > MAX_TOPUP_USD:
        raise HTTPException(status_code=400, detail=f"Amount must be between ${MIN_TOPUP_USD} and ${MAX_TOPUP_USD}")
    base = str(request.base_url).rstrip("/")
    session = create_checkout_session(
        user_id, req.amount_usd,
        success_url=f"{base}/?topup=success",
        cancel_url=f"{base}/?topup=cancelled",
    )
    if not session:
        raise HTTPException(status_code=503, detail="Payments aren't configured yet (missing Stripe keys)")
    return session

@app.post("/api/payments/webhook")
async def payments_webhook(request: Request):
    from app.core.payments import verify_and_parse_webhook, handle_checkout_completed
    from app.db.database import GenesisDB
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event = verify_and_parse_webhook(payload, sig_header)
    if not event:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    result = handle_checkout_completed(event, GenesisDB())
    return {"received": True, "result": result}

@app.get("/api/payments/balance")
async def payments_balance(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"balance_cents": GenesisDB().get_balance(user_id)}

@app.get("/api/payments/transactions")
async def payments_transactions(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"transactions": GenesisDB().get_transactions(user_id)}

@app.post("/api/payments/dev-credit")
async def payments_dev_credit(user_id: str = Depends(get_current_user_id)):
    """DEV-ONLY: credits $10 test balance so payment/charging logic can be verified from
    the browser without a working Stripe account. Only works when settings.debug is True -
    set DEBUG=true in .env to enable, and turn it back off before this ever goes live."""
    if not settings.debug:
        raise HTTPException(status_code=403, detail="Dev endpoints are disabled (set DEBUG=true in .env to enable for local testing)")
    from app.db.database import GenesisDB
    db = GenesisDB()
    db.add_transaction(user_id, "topup", 1000, "Dev test credit (not a real payment)")
    return {"balance_cents": db.get_balance(user_id)}

# ── Memory ──
@app.get("/api/memory/facts")
async def memory_facts(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"facts": GenesisDB().get_facts(user_id)}

@app.get("/api/dashboard/activity")
async def dashboard_activity(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"activity": GenesisDB().get_activity_last_7_days(user_id=user_id)}

@app.get("/api/memory/persona")
async def memory_persona():
    from app.core.memory.graph import KnowledgeGraph
    from app.core.memory.persona import PersonaGenerator
    return {"persona": PersonaGenerator(KnowledgeGraph()).generate()}

@app.get("/api/memory/graph")
async def memory_graph():
    from app.core.memory.graph import KnowledgeGraph
    return KnowledgeGraph().to_dict()

# ── Datasets ──
@app.get("/api/datasets/search")
async def dataset_search(q: str):
    from app.core.engine.dataset_finder import DatasetFinder
    return {"query": q, "results": [{"title": d.title, "url": d.url, "source": d.source} for d in DatasetFinder().search(q)]}

# ── Projects ──
@app.get("/api/projects")
async def get_projects(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"projects": GenesisDB().get_projects(user_id=user_id)}

@app.post("/api/projects")
async def save_projects(data: Dict[str, Any], user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    for p in data.get("projects", []):
        db.add_project(p.get("name",""), p.get("desc",""), p.get("task","other"), p.get("icon","🤖"), user_id=user_id)
    return {"status": "saved"}

# ── API Keys ──
@app.get("/api/keys")
async def list_keys(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"keys": GenesisDB().list_api_keys(user_id)}

@app.post("/api/keys/create")
async def create_key(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    key = GenesisDB().create_api_key(user_id, data.get("user_name", ""), data.get("calls_limit", 100))
    return key

@app.delete("/api/keys/{api_key}")
async def delete_key(api_key: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_api_key(api_key, user_id):
        raise HTTPException(status_code=404, detail="API key not found, or it doesn't belong to you")
    return {"status": "deleted"}

@app.get("/api/usage")
async def usage_stats(api_key: str = None, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    # Previously unauthenticated and, with no api_key given, returned every user's usage log
    # platform-wide. Now requires login, and an explicit api_key must actually belong to the
    # caller - otherwise this route would let anyone read another user's usage history just
    # by supplying (or guessing) their key string.
    if api_key and db.get_api_key_owner(api_key) != user_id:
        raise HTTPException(status_code=403, detail="That API key doesn't belong to you")
    if not api_key:
        owned_keys = {k["api_key"] for k in db.list_api_keys(user_id)}
        all_usage = db.get_usage_stats(None)
        return {"usage": [u for u in all_usage if u.get("api_key") in owned_keys]}
    return {"usage": db.get_usage_stats(api_key)}

@app.get("/api/usage/groq")
async def groq_usage_stats(user_id: str = Depends(get_current_user_id)):
    """The person's own internal Groq API usage (chat routing, bot builder, codegen,
    pipeline suggestions) - see GenesisDB.log_groq_usage for why this exists and what it
    is/isn't. Scoped to the caller's own usage only, consistent with per-user isolation -
    there is intentionally no cross-user or platform-wide view exposed here. Also includes
    this account's monthly quota status (see app.core.quota) so a person can see WHY they
    might be getting blocked before it happens, not just discover it mid-conversation."""
    from app.db.database import GenesisDB
    from app.core.quota import check_quota
    db = GenesisDB()
    summary = db.get_groq_usage_summary(user_id)
    allowed, used_this_month, limit = check_quota(db, user_id)
    summary["quota"] = {"used_this_month": used_this_month, "limit": limit,
                        "unlimited": limit <= 0, "allowed": allowed}
    return summary

# ── Model Predict ──
@app.post("/api/models/{model_id}/predict")
async def model_predict(
    model_id: str,
    data: dict = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = None,  # deprecated: query-param fallback, logs/URLs can leak this - prefer X-API-Key header
):
    from app.db.database import GenesisDB
    db = GenesisDB()
    key = x_api_key or api_key
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key - send it as the X-API-Key header")
    key_info = db.validate_api_key(key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    try:
        record = reg.get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")

    # Authorization: an API key lets its owner call their OWN registered models freely.
    # A model with no "author" in metadata is a legacy/platform model with no owner to
    # protect, so it stays open (unchanged prior behavior). A model owned by someone ELSE
    # is only reachable this way if it has an active, priced marketplace listing - in which
    # case it's billed exactly like /api/marketplace/{listing_id}/use, so an API key can no
    # longer be used to get someone else's paid model for free by calling this route instead
    # of the metered marketplace endpoint. If it has no listing, it's private - denied.
    record_author = (record.get("metadata") or {}).get("author")
    caller_id = key_info.get("user_id")
    listing = None
    price_cents = 0
    if record_author and record_author != caller_id:
        from app.core.marketplace import Marketplace
        mkt = Marketplace()
        listing = next((l for l in mkt.listings.values()
                         if l.get("model_id") == model_id and l.get("active")), None)
        if not listing:
            raise HTTPException(status_code=403, detail="This model is private and you don't have access to it")
        price_cents = round(listing.get("price_per_call", 0) * 100)
        if price_cents > 0:
            if not caller_id:
                raise HTTPException(status_code=401, detail="This is a paid model - use an API key tied to a logged-in account")
            charge = db.charge_for_usage(
                buyer_user_id=caller_id, amount_cents=price_cents,
                description=f"Predict via API: {listing.get('title', model_id)}",
                author_user_id=record_author, platform_fee_pct=25.0, related_id=model_id)
            if not charge["ok"]:
                raise HTTPException(status_code=402, detail=(
                    f"Insufficient balance - this model costs ${price_cents/100:.4f}/call, "
                    f"you have ${charge['balance_cents']/100:.2f}. Add funds to continue."))

    if not db.use_api_key(key, model_id):
        raise HTTPException(status_code=429, detail="API call limit exceeded")
    try:
        model = reg.get_model(model_id)
        features = data.get("features", []) if data else []
        if features:
            import numpy as np
            prediction = model.predict(np.array(features).reshape(1, -1))
            if listing is not None:
                from app.core.marketplace import Marketplace
                Marketplace().use_model(listing["listing_id"], caller_id)
            return {"prediction": prediction.tolist(), "model_id": model_id}
        return {"message": "Model loaded", "model_id": model_id, "metrics": record.get("metrics", {})}
    except HTTPException:
        raise
    except Exception as e:
        # Mirrors /api/marketplace/{listing_id}/use: if we already charged for a paid model
        # and the prediction itself then failed, refund both sides rather than keep a charge
        # for a call that produced no result.
        if price_cents > 0 and listing is not None:
            db.add_transaction(caller_id, "usage_refund", price_cents,
                               f"Refund: {listing.get('title', model_id)} (prediction error)", related_id=model_id)
            author_share = round(price_cents * 0.75)
            if author_share > 0:
                db.add_transaction(record_author, "usage_earning_reversal", -author_share,
                                   f"Reversal: {listing.get('title', model_id)} (prediction error)", related_id=model_id)
        raise HTTPException(status_code=500, detail=str(e))

# ── Code Generation ──
@app.post("/api/codegen/generate")
async def generate_code(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.codegen.generator import CodeGenerator
    from app.db.database import GenesisDB
    gen = CodeGenerator(GenesisDB())
    description = data.get("description", "")
    code = gen.generate_from_description(description)
    path = gen.save_code(code, owner_user_id=user_id)
    return {"code_path": path, "code": code}

@app.get("/api/codegen/list")
async def list_generated(user_id: str = Depends(get_current_user_id)):
    import glob
    # Only ever return the caller's own generated files - see save_code's owner-tagged
    # filenames. Previously this globbed every file in the shared "generated/" directory
    # and returned all of it, unauthenticated, regardless of who generated what.
    prefix = f"user_{user_id}_"
    files = [f for f in glob.glob("generated/*.py") if os.path.basename(f).startswith(prefix)]
    return {"files": [{"name": os.path.basename(f), "path": f} for f in files]}

# ── Pipeline Builder ──
@app.get("/api/pipeline/templates")
async def pipeline_templates():
    return {
        "templates": [
            {"name": "Binary Classification", "blocks": ["load_data","profile","clean","train_xgboost","evaluate","save"]},
            {"name": "Regression", "blocks": ["load_data","profile","clean","train_xgboost_reg","evaluate_reg","save"]},
            {"name": "Quick Classify", "blocks": ["load_data","train_xgboost","evaluate"]},
        ]
    }

@app.post("/api/pipeline/build")
async def build_pipeline(data: dict):
    from app.core.codegen.generator import CodeGenerator
    gen = CodeGenerator(None)
    code = gen.generate_from_description(data.get("description", ""))
    path = gen.save_code(code)
    return {"code": code, "code_path": path}

@app.post("/api/pipeline/suggest")
async def suggest_pipeline_blocks(data: dict):
    """Used by the Pipeline Builder's 'build from prompt' box - asks the real LLM to pick
    blocks from the fixed catalog, instead of doing naive keyword matching in JS.

    BUGFIX: _ai_build_pipeline returns a (blocks, custom_defs) TUPLE on success, or None
    on failure (see its docstring) - the same contract GenesisAgent._do_suggest_pipeline
    already unpacks correctly. This route used to do
    `agent._ai_build_pipeline(description) or agent._keyword_build_pipeline(description)`,
    which - whenever the AI call actually succeeded - assigned that raw 2-tuple to `blocks`
    instead of the block-id list inside it. The very next line then called
    `PIPELINE_BLOCKS.get(b, b)` on `b = <that list>`, which raises "unhashable type: 'list'"
    since a list can't be a dict key - an unhandled 500 every time the AI path succeeded.
    It also called _ai_build_pipeline a SECOND time just to compute `used_ai` (a redundant
    real Groq call/cost), and never materialized any AI-requested custom blocks the way
    _do_suggest_pipeline does. Fixed by unpacking the tuple once, like the chat path does.
    """
    from app.core.agent import GenesisAgent
    from app.core.custom_blocks import CustomBlockStore
    description = data.get("description", "")
    if not description:
        return {"blocks": [], "explanation": "Опишите, что должен делать пайплайн."}
    agent = GenesisAgent()
    ai_result = agent._ai_build_pipeline(description)
    used_ai = ai_result is not None
    if ai_result:
        blocks, custom_defs = ai_result
    else:
        blocks, custom_defs = agent._keyword_build_pipeline(description), []

    # Materialize any AI-requested custom blocks into real CustomBlock rows and splice
    # their real ids into the pipeline in place of the "custom:xxx" placeholder - same
    # step _do_suggest_pipeline performs for the chat-driven path.
    store = CustomBlockStore()
    block_labels = dict(agent.PIPELINE_BLOCKS)
    ref_to_real_id = {}
    for cdef in custom_defs:
        ref = cdef.get("ref")
        name = cdef.get("name", ref)
        desc = cdef.get("description", "")
        if not ref or ref in ref_to_real_id:
            continue
        created = store.create(
            name=name,
            code=f"# TODO: implement '{name}'\n# {desc}\ndef run(data):\n    raise NotImplementedError({desc!r})",
            icon="🔧", category="custom",
        )
        ref_to_real_id[ref] = created["id"]
        block_labels[created["id"]] = f"{name} (custom) - {desc}"
    blocks = [ref_to_real_id.get(b, b) for b in blocks]

    explanation = "\n".join(f"{i+1}. {block_labels.get(b, b)}" for i, b in enumerate(blocks))
    return {"blocks": blocks, "explanation": explanation, "ai_generated": used_ai}

# ── Agent Store ──
@app.get("/api/agents")
async def list_agents(author: str = None):
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().list_agents(author=author)}

@app.get("/api/agents/featured")
async def featured_agents():
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().get_featured()}

@app.get("/api/agents/search")
async def search_agents(q: str):
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().search_agents(q)}

@app.get("/api/agents/mine")
async def my_agents(user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().list_agents(author=user_id, published_only=False)}

@app.get("/api/agents/tools")
async def available_tools():
    from app.core.agent_store import AgentStore
    return {"tools": AgentStore.AVAILABLE_TOOLS}

@app.get("/api/agents/models")
async def available_models():
    """Flat list of every catalog model (chat + audio), richest single source of truth.
    Frontend model-selector cards and the Create Agent / Bot Settings dropdowns should
    all read from this instead of hardcoding model data."""
    from app.core.models_catalog import MODEL_CATALOG, to_dict
    return {"models": [to_dict(spec) for spec in MODEL_CATALOG.values()]}

@app.get("/api/agents/models/by-category")
async def available_models_by_category():
    """Category metadata (id/label/icon) plus the full flat model list. The frontend
    filters client-side by category membership, since a model can belong to several
    categories at once and grouping would duplicate cards."""
    from app.core.models_catalog import MODEL_CATALOG, CATEGORIES, to_dict
    return {
        "categories": CATEGORIES,
        "models": [to_dict(spec) for spec in MODEL_CATALOG.values()],
    }

@app.post("/api/agents/models/auto")
async def auto_select_model(data: dict):
    """Auto mode: given a task hint and hard constraints, return the best-fit chat model."""
    from app.core.models_catalog import pick_auto_model, MODEL_CATALOG, to_dict
    model_id = pick_auto_model(
        task_hint=data.get("task_hint", ""),
        needs_vision=bool(data.get("needs_vision", False)),
        needs_json_schema=bool(data.get("needs_json_schema", False)),
        prioritize_speed=bool(data.get("prioritize_speed", False)),
        prioritize_cost=bool(data.get("prioritize_cost", False)),
    )
    return {"model_id": model_id, "model": to_dict(MODEL_CATALOG[model_id])}

# NOTE: the routes above (tools/models/models/by-category/models/auto) MUST stay registered
# before /api/agents/{agent_id} below - FastAPI matches path routes in registration order,
# and {agent_id} is a single-segment wildcard that would otherwise swallow /api/agents/models
# (treating "models" as an agent_id) before the more specific route below ever gets a chance.
# This was a pre-existing bug that silently broke /api/agents/models entirely; fixed here.
@app.get("/api/agents/{agent_id}")
async def get_agent_by_id(agent_id: str, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    from app.core.agent_store import AgentStore
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Not found")
    # Published agents are meant to be publicly browsable (that's the marketplace). An
    # unpublished/private agent - including its system_prompt - must only be visible to its
    # own author. Previously this endpoint had no auth dependency at all and returned any
    # agent, published or not, to anyone who knew or guessed its id. Returning 404 (not 403)
    # for a private agent owned by someone else avoids confirming that a given id exists.
    if not agent.get("published") and agent.get("author") != user_id:
        raise HTTPException(status_code=404, detail="Not found")
    return agent

# ── Knowledge Base (per-bot uploaded documents, retrieved at chat time - see
# app.core.knowledge_base and GenesisAgent._get_effective_system_prompt) ──

SUPPORTED_KB_EXTENSIONS = (".txt", ".md", ".csv")

@app.post("/api/agents/{agent_id}/knowledge")
async def upload_kb_document(agent_id: str, file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.core.knowledge_base import chunk_text
    from app.db.database import GenesisDB

    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")

    filename = file.filename or "document.txt"
    if not filename.lower().endswith(SUPPORTED_KB_EXTENSIONS):
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_KB_EXTENSIONS)} "
                   f"(PDF/DOCX not yet supported - convert to plain text first)")

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text")
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any content to index")

    db = GenesisDB()
    doc_id = db.add_kb_document(agent_id, user_id, filename, content, chunks)
    return {"id": doc_id, "filename": filename, "chunk_count": len(chunks)}

@app.get("/api/agents/{agent_id}/knowledge")
async def list_kb_documents(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    return {"documents": GenesisDB().list_kb_documents(agent_id, user_id)}

@app.delete("/api/agents/{agent_id}/knowledge/{document_id}")
async def delete_kb_document(agent_id: str, document_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_kb_document(document_id, user_id):
        raise HTTPException(status_code=404, detail="Document not found or not yours")
    return {"status": "deleted"}

@app.get("/api/agents/{agent_id}/analytics")
async def bot_analytics(agent_id: str, user_id: str = Depends(get_current_user_id)):
    """Aggregates what's already being collected (conversations, ratings) - see
    GenesisDB.get_bot_analytics for what is/isn't included and why."""
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    return GenesisDB().get_bot_analytics(agent_id)

@app.get("/api/agents/{agent_id}/handoffs")
async def bot_handoffs(agent_id: str, user_id: str = Depends(get_current_user_id)):
    """Recent human-handoff events for this bot - see app.core.handoff and
    GenesisAgent._do_request_handoff for how these get created."""
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    return {"handoffs": GenesisDB().get_handoff_events(agent_id)}

# ── Telegram integration (connect a published bot to a real Telegram bot via a BotFather
# token) - see app.core.telegram_integration for the low-level Bot API client. Incoming
# messages are routed through the EXACT SAME GenesisAgent.process() pipeline the web chat
# uses (agent_override carries agent_id), so tool-scoping, knowledge base retrieval,
# predict_with_model, human handoff, and the Groq quota gate all behave identically over
# Telegram as they do in the web widget - nothing about the dispatch logic is duplicated. ──

import secrets as _secrets

@app.post("/api/agents/{agent_id}/telegram/connect")
async def telegram_connect(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    from app.core import telegram_integration as tg

    bot = AgentStore().get_agent(agent_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if bot.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")

    token = (data.get("bot_token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="bot_token is required")

    try:
        identity = tg.get_me(token)
    except tg.TelegramAPIError as e:
        raise HTTPException(status_code=400, detail=f"Invalid bot token: {e.description}")

    db = GenesisDB()
    webhook_secret = _secrets.token_urlsafe(24)
    status, warning = "connected_no_webhook", None
    if settings.public_base_url:
        webhook_url = f"{settings.public_base_url.rstrip('/')}/api/telegram/webhook/{webhook_secret}"
        try:
            tg.set_webhook(token, webhook_url, webhook_secret)
            status = "connected"
        except tg.TelegramAPIError as e:
            warning = f"Token is valid but webhook setup failed: {e.description}"
    else:
        warning = ("No public server URL is configured (PUBLIC_BASE_URL), so Telegram can't "
                   "deliver live messages yet - use the built-in test chat below, or deploy "
                   "publicly and reconnect once you have a real URL.")

    imported = {}
    try:
        imported = tg.import_bot_settings(token, identity["id"])
    except tg.TelegramAPIError:
        pass  # import is best-effort - a fresh bot with nothing configured yet is normal, not an error

    link_id = db.upsert_telegram_link(agent_id, user_id, token, identity["id"],
                                      identity.get("username", ""), webhook_secret, status)
    db.log_telegram_event(link_id, "connected", f"@{identity.get('username', '')}")
    return {"status": status, "bot_username": identity.get("username"), "bot_id": identity["id"],
            "imported": imported, "warning": warning}

@app.get("/api/agents/{agent_id}/telegram")
async def telegram_status(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    link = GenesisDB().get_telegram_link(agent_id, user_id)
    if not link:
        return {"connected": False}
    return {"connected": True, **link}

@app.delete("/api/agents/{agent_id}/telegram")
async def telegram_disconnect(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core import telegram_integration as tg
    deleted = GenesisDB().delete_telegram_link(agent_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No Telegram connection found for this bot")
    try:
        tg.delete_webhook(deleted["bot_token"])
    except tg.TelegramAPIError:
        pass  # best-effort - the link is already gone from our side either way
    return {"status": "disconnected"}

@app.get("/api/agents/{agent_id}/telegram/events")
async def telegram_events(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    db = GenesisDB()
    link = db.get_telegram_link(agent_id, user_id)
    if not link:
        return {"events": []}
    return {"events": db.get_telegram_events(link["id"])}

@app.post("/api/agents/{agent_id}/telegram/test")
async def telegram_test_chat(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Runs a message through the EXACT SAME dispatch pipeline the real Telegram webhook
    uses, without touching Telegram at all - lets a bot owner verify their bot's behavior
    (including RAG/predict/handoff) before ever connecting a real token, and remains fully
    testable in environments (like this one) that can't reach api.telegram.org."""
    from app.core.agent_store import AgentStore
    from app.core.agent import GenesisAgent
    bot = AgentStore().get_agent(agent_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if bot.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    agent = GenesisAgent()
    agent.agent_override = {"model": bot.get("model", "openai/gpt-oss-120b"),
                            "system_prompt": bot.get("system_prompt", ""), "agent_id": agent_id}
    cid = f"telegram_test_{agent_id}_{user_id}"
    resp = agent.process(data.get("message", ""), conversation_id=cid, user_id=user_id)
    return {"response": resp.message, "data": resp.data}

@app.post("/api/telegram/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    """The real webhook Telegram calls on every incoming message. No auth dependency - the
    secret in the URL path IS the credential, additionally confirmed via the
    X-Telegram-Bot-Api-Secret-Token header Telegram sends on every call (defense in depth -
    URL paths can end up in logs/proxies more easily than a header)."""
    from app.db.database import GenesisDB
    db = GenesisDB()
    link = db.get_telegram_link_by_secret(secret)
    if not link:
        raise HTTPException(status_code=404, detail="Unknown webhook")
    header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if header_secret != secret:
        raise HTTPException(status_code=403, detail="Secret mismatch")

    update = await request.json()
    message = update.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    if not chat_id or not text:
        return {"ok": True}  # nothing to process (e.g. a sticker/photo) - ack so Telegram doesn't retry

    from app.core.agent_store import AgentStore
    from app.core.agent import GenesisAgent
    from app.core import telegram_integration as tg

    bot = AgentStore().get_agent(link["agent_id"])
    if not bot:
        return {"ok": True}

    db.log_telegram_event(link["id"], "message_in", text[:200])
    agent = GenesisAgent()
    agent.agent_override = {"model": bot.get("model", "openai/gpt-oss-120b"),
                            "system_prompt": bot.get("system_prompt", ""), "agent_id": link["agent_id"]}
    conversation_id = f"telegram_{link['agent_id']}_{chat_id}"
    # Attributed to the BOT OWNER's account, not the (unauthenticated, no GenesisAI account)
    # Telegram user - the owner's own Groq quota funds their bot's real traffic, same as any
    # other cost their published bot generates.
    resp = agent.process(text, conversation_id=conversation_id, user_id=link["user_id"])

    try:
        tg.send_message(link["bot_token"], chat_id, resp.message)
        db.log_telegram_event(link["id"], "message_out", resp.message[:200])
        db.update_telegram_link_status(link["id"], "connected")
    except tg.TelegramAPIError as e:
        db.log_telegram_event(link["id"], "error", str(e))
        db.update_telegram_link_status(link["id"], "error", str(e))
    return {"ok": True}

@app.post("/api/agents/create")
async def create_agent(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore, AgentConfig
    config = AgentConfig(
        name=data.get("name", "New Agent"),
        system_prompt=data.get("system_prompt", "You are a helpful assistant.")
    )
    config.description = data.get("description", "")
    config.model = data.get("model", "openai/gpt-oss-120b")
    config.temperature = data.get("temperature", 0.5)
    config.max_tokens = data.get("max_tokens", 1000)
    config.tools = data.get("tools", [])
    config.author = user_id
    store = AgentStore()
    store.create_agent(config)
    return config.to_dict()

@app.patch("/api/agents/{agent_id}")
async def update_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    if "linked_model_id" in data and data["linked_model_id"]:
        from app.core.engine.registry import ModelRegistry
        try:
            record = ModelRegistry().get_record(data["linked_model_id"])
        except KeyError:
            raise HTTPException(status_code=400, detail="linked_model_id does not exist in the registry")
        if record.get("metadata", {}).get("author", user_id) != user_id:
            # Older registry records (before ownership tracking existed on models) have no
            # owner_user_id at all - default to allowing rather than retroactively locking
            # a solo dev out of their own pre-existing models.
            raise HTTPException(status_code=403, detail="You can only link a bot to a model you own")
    if "handoff_webhook_url" in data and data["handoff_webhook_url"]:
        from app.core.handoff import is_plausible_webhook_url
        if not is_plausible_webhook_url(data["handoff_webhook_url"]):
            raise HTTPException(status_code=400, detail="handoff_webhook_url must be a valid http(s) URL")
    updated = AgentStore().update_agent(agent_id, data, owner_user_id=user_id)
    if not updated:
        raise HTTPException(status_code=403, detail="Not your bot, or it doesn't exist")
    return updated

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    if not AgentStore().delete_agent(agent_id, owner_user_id=user_id):
        raise HTTPException(status_code=403, detail="Not your bot, or it doesn't exist")
    return {"status": "deleted"}

@app.post("/api/agents/{agent_id}/publish")
async def publish_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    if not AgentStore().publish_agent(agent_id, data.get("price", 0), owner_user_id=user_id):
        raise HTTPException(status_code=403, detail="Not your bot, or it doesn't exist")
    return {"status": "published"}

@app.post("/api/agents/{agent_id}/clone")
async def clone_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    config = AgentStore().clone_agent(agent_id, user_id)
    if not config:
        raise HTTPException(status_code=404, detail="Not found")
    return config.to_dict()

@app.post("/api/agents/{agent_id}/review")
async def review_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    AgentStore().add_review(agent_id, user_id, data.get("rating", 5), data.get("comment", ""))
    return {"status": "reviewed"}

# ── Health ──

@app.post("/api/agent/select-model")
async def select_model(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    GenesisDB().set_selected_model(user_id, data)
    return {"status": "ok", "model": data.get("model", "")}

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

@app.post("/api/bot-builder/generate")
async def bot_builder_generate(data: dict, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    _enforce_groq_quota(user_id)
    from app.core.bot_builder import BotBuilder
    result = BotBuilder(user_id=user_id).build(
        description=data.get("description", ""),
        model=data.get("model"),
        max_iterations=data.get("max_iterations"),
    )
    return result

@app.post("/api/bot-builder/chat")
async def bot_builder_chat(data: dict, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    _enforce_groq_quota(user_id)
    from app.core.bot_builder import BotBuilder
    reply = BotBuilder(user_id=user_id).chat_with_draft(
        spec=data.get("spec", {}),
        message=data.get("message", ""),
        history=data.get("history", []),
        model=data.get("model"),
    )
    return {"response": reply}

@app.post("/api/bot-builder/revise")
async def bot_builder_revise(data: dict, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    _enforce_groq_quota(user_id)
    from app.core.bot_builder import BotBuilder
    bb = BotBuilder(user_id=user_id)
    spec = bb.revise(spec=data.get("spec", {}), feedback=data.get("feedback", ""), model=data.get("model"))
    test_results = bb.test_draft(spec, model=data.get("model"), skills=spec.get("skills_needed"))
    return {"spec": spec, "test_results": test_results, "success": all(r["ok"] for r in test_results)}

@app.post("/api/bot-builder/publish")
async def bot_builder_publish(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.bot_builder import BotBuilder
    config = BotBuilder(user_id=user_id).publish(
        spec=data.get("spec", {}),
        model=data.get("model"),
        author=user_id,
        price=data.get("price", 0.0),
    )
    return {"status": "published", "agent": config}

@app.post("/api/model-builder/generate")
async def model_builder_generate(data: dict, user_id: str = Depends(get_current_user_id)):
    _enforce_groq_quota(user_id)
    from app.core.model_builder import ModelBuilder
    dataset_path = data.get("dataset_path", "")
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Valid dataset_path is required - upload/select a dataset first")
    result = ModelBuilder().build(
        description=data.get("description", ""),
        dataset_path=dataset_path,
        target_column=data.get("target_column"),
        user_id=user_id,
    )
    return result

@app.post("/api/model-builder/publish")
async def model_builder_publish(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.model_builder import ModelBuilder
    saved_model_path = data.get("saved_model_path")
    if not saved_model_path or not os.path.exists(saved_model_path):
        raise HTTPException(status_code=400, detail="No trained model to publish - build one first")
    model_id = ModelBuilder().register(saved_model_path, data.get("metrics", {}), author_user_id=user_id)
    if not model_id:
        raise HTTPException(status_code=500, detail="Could not register the model")
    price = data.get("price_per_call", 0.0)
    if price >= 0:
        from app.core.marketplace import Marketplace
        Marketplace().publish(
            author=user_id, model_id=model_id,
            title=data.get("title", "Untitled Model"), description=data.get("description", ""),
            task=data.get("metrics", {}).get("task", "binary_classification"),
            model_type=data.get("metrics", {}).get("model_type", "unknown"),
            metrics={k: v for k, v in data.get("metrics", {}).items() if isinstance(v, (int, float))},
            price_per_call=price,
        )
    return {"status": "published", "model_id": model_id}



# ---- Custom Blocks ----
@app.get("/api/blocks")
async def list_blocks(category: str = None):
    from app.core.custom_blocks import CustomBlockStore
    return {"blocks": CustomBlockStore().list_blocks(category)}

@app.post("/api/blocks/create")
async def create_block(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.custom_blocks import CustomBlockStore
    block = CustomBlockStore().create(
        name=data.get("name","New Block"),
        code=data.get("code","# Python code"),
        icon=data.get("icon","🔧"),
        category=data.get("category","custom"),
        author=user_id,
    )
    return block

@app.delete("/api/blocks/{block_id}")
async def delete_block(block_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.custom_blocks import CustomBlockStore
    if not CustomBlockStore().delete_block(block_id, user_id):
        raise HTTPException(status_code=404, detail="Block not found, or it doesn't belong to you")
    return {"status": "deleted"}


# ---- Bot Skills (AI-generated, sandbox-tested network capabilities for a bot) ----
@app.post("/api/skills/create")
async def create_skill(data: dict, user_id: str = Depends(get_current_user_id)):
    """Generates, sandbox-tests, and (if the test passes) saves a new skill for a bot the
    caller owns. This can take several seconds and several real Groq calls (the
    generate -> test -> fix loop) - same cost shape as /api/bot-builder/generate, so it's
    gated by the same quota check."""
    _enforce_groq_quota(user_id)
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    from app.core.skills.skill_generator import SkillGenerator

    agent_id = data.get("agent_id")
    description = (data.get("description") or "").strip()
    allowed_domains = data.get("allowed_domains") or []
    if not agent_id or not description:
        raise HTTPException(status_code=400, detail="agent_id and description are required")
    if not allowed_domains or not isinstance(allowed_domains, list):
        raise HTTPException(status_code=400, detail=(
            "allowed_domains must be a non-empty list of domains this skill is allowed to "
            "contact, e.g. [\"ebay.com\"] - a skill with no declared domains can't reach the "
            "network at all, and one is required so the sandbox can enforce it."))

    agent = AgentStore().get_agent(agent_id)
    if not agent or agent.get("author") != user_id:
        raise HTTPException(status_code=404, detail="Bot not found, or it doesn't belong to you")

    declared_action_type = data.get("action_type", "read_only")
    if declared_action_type not in ("read_only", "outbound_action"):
        raise HTTPException(status_code=400, detail="action_type must be 'read_only' or 'outbound_action'")

    result = SkillGenerator(user_id=user_id).build(description, allowed_domains)
    from app.core.skills.skill_generator import _detect_write_http_method
    # Defense in depth: a skill that writes (POST/PUT/DELETE/PATCH) is ALWAYS treated as an
    # outbound_action, regardless of what the creator declared - see _detect_write_http_method's
    # docstring. This can only make the gate stricter than requested, never looser.
    action_type = "outbound_action" if (result["code"] and _detect_write_http_method(result["code"])) else declared_action_type

    db = GenesisDB()
    skill = db.create_skill(
        agent_id=agent_id, owner_user_id=user_id,
        name=data.get("name") or description[:60],
        description=description, code=result["code"] or "",
        allowed_domains=allowed_domains, input_schema=data.get("input_schema") or {},
        action_type=action_type)
    db.update_skill_test_result(skill["id"], passed=result["success"],
                                 log=json.dumps(result["log"])[:4000])
    skill = db.get_skill(skill["id"])
    del skill["code"]  # never return the generated source to the client - see get_skill's docstring note
    return {"skill": skill, "test_success": result["success"], "log": result["log"]}

@app.get("/api/skills")
async def list_my_skills(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"skills": GenesisDB().list_skills_for_owner(user_id)}

@app.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_skill(skill_id, user_id):
        raise HTTPException(status_code=404, detail="Skill not found, or it doesn't belong to you")
    return {"status": "deleted"}


# ---- Pending actions ("Level 4": outbound_action skills always stop here first) ----
@app.get("/api/pending-actions")
async def list_pending_actions(status: Optional[str] = None, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"actions": GenesisDB().list_pending_actions_for_owner(user_id, status=status)}

@app.post("/api/pending-actions/{action_id}/approve")
async def approve_pending_action(action_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.skills.skill_generator import SkillGenerator
    db = GenesisDB()
    action = db.decide_pending_action(action_id, user_id, approve=True)
    if not action:
        raise HTTPException(status_code=404, detail=(
            "Pending action not found, doesn't belong to you, or was already decided"))

    skill = db.get_skill(action["skill_id"])
    if not skill or not skill["active"]:
        db.record_pending_action_result(action_id, success=False, result=None,
                                         error="The skill this action depends on is no longer available.")
        raise HTTPException(status_code=409, detail="The underlying skill is no longer available")

    outcome = SkillGenerator().run_stored_skill(skill["code"], action["params"], skill["allowed_domains"])
    db.record_pending_action_result(action_id, success=outcome["success"],
                                     result=outcome.get("result"), error=outcome.get("error"))
    return {"action": db.get_pending_action(action_id)}

@app.post("/api/pending-actions/{action_id}/reject")
async def reject_pending_action(action_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    action = GenesisDB().decide_pending_action(action_id, user_id, approve=False)
    if not action:
        raise HTTPException(status_code=404, detail=(
            "Pending action not found, doesn't belong to you, or was already decided"))
    return {"action": action}


# ---- Scheduled jobs ("Level 3": autonomous, timer-driven skill runs) ----
@app.post("/api/scheduled-jobs/create")
async def create_scheduled_job(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    skill_id = data.get("skill_id")
    if not skill_id:
        raise HTTPException(status_code=400, detail="skill_id is required")
    skill = db.get_skill(skill_id)
    if not skill or skill["owner_user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Skill not found, or it doesn't belong to you")
    if not skill["active"] or not skill["test_passed"]:
        raise HTTPException(status_code=400, detail="Only an active, sandbox-tested skill can be scheduled")

    try:
        job = db.create_scheduled_job(
            skill_id=skill_id, agent_id=skill["agent_id"], owner_user_id=user_id,
            params=data.get("params") or {},
            interval_minutes=int(data.get("interval_minutes", 0)),
            notify_on_change=bool(data.get("notify_on_change", True)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job": job}

@app.get("/api/scheduled-jobs")
async def list_scheduled_jobs(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"jobs": GenesisDB().list_scheduled_jobs_for_owner(user_id)}

@app.post("/api/scheduled-jobs/{job_id}/pause")
async def pause_scheduled_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().set_scheduled_job_active(job_id, user_id, active=False):
        raise HTTPException(status_code=404, detail="Job not found, or it doesn't belong to you")
    return {"status": "paused"}

@app.post("/api/scheduled-jobs/{job_id}/resume")
async def resume_scheduled_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().set_scheduled_job_active(job_id, user_id, active=True):
        raise HTTPException(status_code=404, detail="Job not found, or it doesn't belong to you")
    return {"status": "resumed"}

@app.delete("/api/scheduled-jobs/{job_id}")
async def delete_scheduled_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_scheduled_job(job_id, user_id):
        raise HTTPException(status_code=404, detail="Job not found, or it doesn't belong to you")
    return {"status": "deleted"}


@app.get("/health")
async def health(): return {"status": "healthy", "version": "1.0.0"}

# ── Prompt Library (versioned) & A/B Testing ──

@app.post("/api/prompts")
async def create_prompt(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    name = (data.get("name") or "").strip()
    content = (data.get("content") or "").strip()
    if not name or not content:
        raise HTTPException(status_code=400, detail="name and content are required")
    return GenesisDB().create_prompt(user_id, name, content, data.get("description", ""), data.get("tags"))

@app.get("/api/prompts")
async def list_prompts(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"prompts": GenesisDB().list_prompts(user_id)}

@app.get("/api/prompts/{prompt_id}")
async def get_prompt(prompt_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    prompt = GenesisDB().get_prompt(prompt_id, user_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt

@app.put("/api/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Creates a NEW version rather than overwriting - see add_prompt_version."""
    from app.db.database import GenesisDB
    content = (data.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    result = GenesisDB().add_prompt_version(prompt_id, user_id, content, data.get("change_note", ""))
    if not result:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return result

@app.post("/api/prompts/{prompt_id}/revert/{version}")
async def revert_prompt(prompt_id: str, version: int, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    result = GenesisDB().revert_prompt(prompt_id, user_id, version)
    if not result:
        raise HTTPException(status_code=404, detail="Prompt or version not found")
    return result

@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_prompt(prompt_id, user_id):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "deleted"}

@app.post("/api/ab-tests")
async def create_ab_test(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    agent_id = data.get("agent_id")
    variants = data.get("variants") or []
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    if len(variants) < 2:
        raise HTTPException(status_code=400, detail="At least 2 variants are required for an A/B test")
    for v in variants:
        if not (v.get("label") and (v.get("system_prompt") or "").strip()):
            raise HTTPException(status_code=400, detail="Each variant needs a label and non-empty system_prompt")
    if db.get_running_ab_test_for_agent(agent_id):
        raise HTTPException(status_code=409, detail="This bot already has a running A/B test - stop it before starting a new one")
    return db.create_ab_test(user_id, agent_id, data.get("name", "Untitled test"), variants)

@app.get("/api/ab-tests")
async def list_ab_tests(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"tests": GenesisDB().list_ab_tests(user_id)}

@app.get("/api/ab-tests/{test_id}")
async def get_ab_test(test_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    tests = [t for t in db.list_ab_tests(user_id) if t["id"] == test_id]
    if not tests:
        raise HTTPException(status_code=404, detail="Test not found")
    return {**tests[0], "stats": db.get_ab_test_stats(test_id)}

@app.post("/api/ab-tests/{test_id}/stop")
async def stop_ab_test(test_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Stopping optionally promotes a winning variant's prompt to be the bot's live,
    permanent system_prompt (via AgentStore), so the test's result actually sticks."""
    from app.db.database import GenesisDB
    from app.core.agent_store import AgentStore
    db = GenesisDB()
    winner_variant_id = data.get("winner_variant_id")
    tests = [t for t in db.list_ab_tests(user_id) if t["id"] == test_id]
    if not tests:
        raise HTTPException(status_code=404, detail="Test not found")
    if winner_variant_id:
        stats = db.get_ab_test_stats(test_id)
        if winner_variant_id not in stats:
            raise HTTPException(status_code=400, detail="winner_variant_id is not a variant of this test")
        run = db.get_running_ab_test_for_agent(tests[0]["agent_id"])
        variant_prompt = None
        if run:
            match = next((v for v in run["variants"] if v["id"] == winner_variant_id), None)
            variant_prompt = match["system_prompt"] if match else None
        if variant_prompt:
            # update_agent itself enforces ownership (returns None if not owned by user_id) -
            # no separate author check needed here.
            AgentStore().update_agent(tests[0]["agent_id"], {"system_prompt": variant_prompt}, user_id)
    if not db.stop_ab_test(test_id, user_id, winner_variant_id):
        raise HTTPException(status_code=404, detail="Test not found")
    return {"status": "stopped", "winner_variant_id": winner_variant_id}

@app.delete("/api/ab-tests/{test_id}")
async def delete_ab_test(test_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_ab_test(test_id, user_id):
        raise HTTPException(status_code=404, detail="Test not found")
    return {"status": "deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
