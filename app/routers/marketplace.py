"""marketplace routes (split out of app/main.py)."""
from fastapi import APIRouter
from app.deps import (
    ChatCompletionsRequest,
    Depends,
    HTTPException,
    Optional,
    PUBLIC_API_SUPPORTED_MODELS,
    PublishRequest,
    UseModelRequest,
    get_api_key_from_request,
    get_current_user_id,
    os,
    settings
)

router = APIRouter()

# ── Marketplace ──
@router.get("/api/marketplace/listings")
async def marketplace_listings(task: Optional[str] = None):
    from app.core.marketplace import Marketplace
    return {"listings": Marketplace().search(task=task) if task else Marketplace().search()}

@router.post("/api/marketplace/publish")
async def marketplace_publish(req: PublishRequest, user_id: str = Depends(get_current_user_id)):
    from app.core.marketplace import Marketplace
    from app.core.engine.registry import ModelRegistry
    try:
        ModelRegistry().get_record(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found in Registry - train/register it first")
    lid = Marketplace().publish(author=user_id, model_id=req.model_id, title=req.title, description=req.description, task=req.task, model_type=req.model_type, metrics=req.metrics, price_per_call=req.price_per_call)
    return {"listing_id": lid}

@router.get("/api/marketplace/stats")
async def marketplace_stats():
    from app.core.marketplace import Marketplace
    return Marketplace().get_platform_stats()

@router.post("/api/marketplace/{listing_id}/use")
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
        from app.core.engine.predict_utils import build_model_input_from_list
        prediction = model.predict(build_model_input_from_list(reg.get_record(listing["model_id"]), req.features))
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

@router.post("/api/v1/chat/completions")
async def chat_completions_v1(req: ChatCompletionsRequest, api_key: str = Depends(get_api_key_from_request)):
    import time, uuid
    import groq as groq_sdk
    from app.db.database import GenesisDB
    from app.core.groq_client import call_llm
    from app.core.public_api_billing import estimate_worst_case_cost_cents, reconcile_charge, estimate_prompt_tokens

    db = GenesisDB()
    key_info = db.validate_api_key(api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    user_id = key_info["user_id"]

    if not req.messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty array")
    model = req.model or settings.public_api_default_model
    if model not in PUBLIC_API_SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=(
            f"Unsupported model '{model}'. Supported models: {sorted(PUBLIC_API_SUPPORTED_MODELS)}"))
    max_tokens = min(req.max_tokens or 1024, settings.public_api_max_tokens_cap)
    if max_tokens < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be at least 1")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # PRE-CHARGE the worst case before spending anything with the provider - see
    # public_api_billing.py's module docstring for why this has to happen before generation,
    # not after. Nothing is sent to Groq/OpenRouter at all if this fails.
    pre_charge_cents = estimate_worst_case_cost_cents(messages, max_tokens)
    charge = db.charge_for_usage(user_id, pre_charge_cents,
                                  f"Public API call ({model}, up to {max_tokens} tokens)", related_id=api_key)
    if not charge["ok"]:
        raise HTTPException(status_code=402, detail=(
            f"Insufficient balance: at current pricing this call could cost up to "
            f"{pre_charge_cents} cent(s), but your balance is {charge['balance_cents']} cent(s). "
            f"Top up at POST /api/payments/checkout."))

    try:
        completion = call_llm(
            os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key,
            model=model, messages=messages,
            temperature=req.temperature if req.temperature is not None else 0.7,
            max_tokens=max_tokens,
        )
    except Exception as e:
        # Nothing was generated at all - refund the FULL pre-charge, not a partial amount.
        db.add_transaction(user_id, "public_api_refund", pre_charge_cents,
                            f"Refund: public API call failed before generating a response "
                            f"({type(e).__name__})", related_id=api_key)
        status_code = 504 if isinstance(e, groq_sdk.APITimeoutError) else 502
        raise HTTPException(status_code=status_code,
                             detail=f"The AI provider failed to generate a response: {e}")

    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) or estimate_prompt_tokens(messages)
    completion_tokens = getattr(usage, "completion_tokens", None) or 0
    final_charge_cents = reconcile_charge(db, user_id, api_key, pre_charge_cents, prompt_tokens, completion_tokens)
    db.log_public_api_usage(api_key, user_id, model, prompt_tokens, completion_tokens, final_charge_cents)

    choice = completion.choices[0]
    return {
        "id": f"genesis-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": choice.message.content},
            "finish_reason": getattr(choice, "finish_reason", None) or "stop",
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                   "total_tokens": prompt_tokens + completion_tokens},
        # Non-standard extra field (ignored by any OpenAI-compatible client, which only reads
        # the fields above) - included because unlike OpenAI's own API, billing here is
        # prepaid-balance-based, so surfacing what THIS call actually cost and what's left is
        # directly useful to show a developer without a second API call.
        "genesis_billing": {"charged_cents": final_charge_cents, "balance_cents_remaining": db.get_balance(user_id)},
    }

@router.get("/api/v1/models")
async def list_models_v1(api_key: str = Depends(get_api_key_from_request)):
    """OpenAI-SDK-compatible model list (client.models.list()) - also doubles as a cheap way
    for a developer to confirm their API key works before wiring up real calls."""
    from app.db.database import GenesisDB
    if not GenesisDB().validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "genesis-ai"}
                                        for m in sorted(PUBLIC_API_SUPPORTED_MODELS)]}

@router.get("/api/v1/usage")
async def public_api_usage_v1(user_id: str = Depends(get_current_user_id)):
    """Spend/token summary for the LOGGED-IN platform account (not per-key) - reuses the
    normal JWT session auth, since this is meant for a developer checking their own dashboard,
    not for embedding in a server-to-server integration the way chat/completions is."""
    from app.db.database import GenesisDB
    return GenesisDB().get_public_api_usage_summary(user_id)
