"""marketplace routes (split out of app/main.py)."""
from fastapi import APIRouter, Header, Query
import os
from app.api.contracts import ChatCompletionsRequest, PublishRequest, UseModelRequest
from fastapi import Depends, HTTPException
from typing import Optional
from app.deps import PUBLIC_API_SUPPORTED_MODELS
from app.core.auth import get_api_key_from_request, get_current_user_id
from app.config import settings

router = APIRouter()

# ── Marketplace ──
@router.get("/api/marketplace/listings")
def marketplace_listings(task: Optional[str] = None,limit: int = Query(20,ge=1,le=200),offset: int = Query(0,ge=0)):
    from app.core.marketplace import Marketplace
    return {"listings":Marketplace().search(task=task,limit=limit,offset=offset),"limit":limit,"offset":offset}

LOCAL_LABEL = ("Trained on the owner's own computer, not by the platform. The scores below were measured by the platform on a hidden "
               "test set; the platform cannot check which rows the model was trained on.")


def listing_args(record: dict, user_id: str) -> dict:
    """Everything a listing says about a model, taken from the registry record (what the platform measured). Raises HTTPException if the
    caller is not the author or the platform rates the model as unreliable. Shared by both publish routes."""
    meta = record.get("metadata") or {}
    if meta.get("author") != user_id:
        raise HTTPException(status_code=403, detail="Only the author of a model can publish it.")
    trust = meta.get("trust") or {}
    if trust.get("level") == "low":
        why = "; ".join(str(r) for r in (trust.get("reasons") or [])[:3])
        raise HTTPException(status_code=400, detail="The platform's check rates this model as not reliable enough to sell"
                                                    + (f": {why}" if why else "") + ".")
    provenance = {"trained_on": meta.get("trained_on", "platform"), "built_via": meta.get("built_via"),
                  "verified": bool(meta.get("verified", False)), "trust": trust or None}
    if meta.get("trained_on") == "user_machine":
        provenance["label"] = LOCAL_LABEL
    measured = {k: v for k, v in (record.get("metrics") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return {"task": record.get("task", "unknown"), "model_type": record.get("model_type", "unknown"),
            "metrics": measured, "provenance": provenance}


@router.post("/api/marketplace/publish")
def marketplace_publish(req: PublishRequest, user_id: str = Depends(get_current_user_id)):
    """Only the author can list a model. The listing's task, algorithm and scores come from the registry record (what the platform
    measured), never from the request: `task`, `model_type` and `metrics` in the body are accepted for compatibility and ignored."""
    from app.core.marketplace import Marketplace
    from app.core.engine.registry import ModelRegistry
    try:
        record = ModelRegistry().get_record(req.model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_id}' not found in Registry - train/register it first")
    kw = listing_args(record, user_id)
    lid = Marketplace().publish(author=user_id, model_id=req.model_id, title=req.title, description=req.description,
                                price_per_call=req.price_per_call, **kw)
    return {"listing_id": lid, "provenance": kw["provenance"]}

@router.get("/api/marketplace/stats")
def marketplace_stats():
    from app.core.marketplace import Marketplace
    return Marketplace().get_platform_stats()

@router.post("/api/marketplace/{listing_id}/use")
def marketplace_use(listing_id: str, req: UseModelRequest, user_id: str = Depends(get_current_user_id),
                    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
    from app.core.marketplace import Marketplace
    from app.core.engine.registry import ModelRegistry
    from app.core.engine.predict_utils import build_model_input_from_list
    from app.core.inference import predict_model
    from app.core.money import price_cents
    from app.core.usage import reserved_usage, operation_key, fingerprint
    from app.db.database import GenesisDB
    listing = Marketplace().listings.get(listing_id)
    if not listing or not listing.get("active"):
        raise HTTPException(status_code=404, detail="Active listing not found")
    reg, db = ModelRegistry(), GenesisDB()
    try:
        prepared = build_model_input_from_list(reg.get_record(listing["model_id"]), req.features)
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=422, detail="Invalid model features") from error
    amount = 0 if listing["author"] == user_id else price_cents(listing["price_per_call"])
    oid = operation_key(user_id, "listing:" + listing_id, idempotency_key)
    with reserved_usage(db, user_id, amount, author_id=listing["author"], operation_id=oid, request_hash=fingerprint(req.model_dump()), related_id=listing_id) as operation:
        if operation["replayed"]:
            return operation["result"]
        try:
            prediction = predict_model(reg, listing["model_id"], prepared)
            result = {"prediction": prediction.tolist(), "charged_cents": amount}
            db.finish_usage(operation["id"], result=result)
            return result
        except Exception as error:
            raise HTTPException(status_code=502, detail="Prediction failed; the reservation was refunded") from error

@router.post("/api/v1/chat/completions")
def chat_completions_v1(req: ChatCompletionsRequest, api_key: str = Depends(get_api_key_from_request),
                        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
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

    from app.core.usage import reserved_usage, operation_key, fingerprint
    from app.core.public_api_billing import compute_cost_cents, prompt_token_upper_bound
    if prompt_token_upper_bound(messages) > 120000:
        raise HTTPException(status_code=413, detail="Prompt exceeds the supported input budget")
    pre_charge_cents = estimate_worst_case_cost_cents(messages, max_tokens)
    operation_id = operation_key(user_id, "public-chat", idempotency_key)
    with reserved_usage(db, user_id, pre_charge_cents, operation_id=operation_id, request_hash=fingerprint(req.model_dump())) as operation:
        if operation["replayed"]:
            return operation["result"]
        try:
            completion = call_llm(os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key,
                model=model, messages=messages, temperature=req.temperature if req.temperature is not None else 0.7,
                max_tokens=max_tokens)
            if not completion.choices:
                raise ValueError("No completion choices")
            usage = getattr(completion, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) or prompt_token_upper_bound(messages)
            completion_tokens = getattr(usage, "completion_tokens", None)
            if completion_tokens is None:
                completion_tokens = max_tokens
            final_charge_cents = compute_cost_cents(prompt_tokens, completion_tokens)
            if final_charge_cents > pre_charge_cents:
                raise ValueError("Provider usage exceeds the reserved budget")
            response = _completion_response(completion, model, prompt_tokens, completion_tokens, final_charge_cents,
                                            db.get_balance(user_id) + pre_charge_cents-final_charge_cents)
            db.finish_usage(operation["id"], result=response, actual_cents=final_charge_cents,
                public_usage=(api_key,getattr(completion,"model",None) or model,prompt_tokens,completion_tokens))
            return response
        except HTTPException:
            raise
        except Exception as error:
            status_code = 504 if isinstance(error, groq_sdk.APITimeoutError) else 502
            raise HTTPException(status_code=status_code, detail="The AI provider failed; the reservation was refunded") from error


def _completion_response(completion, model, prompt_tokens, completion_tokens, final_charge_cents, balance_cents):
    import time, uuid
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
        "genesis_billing": {"charged_cents": final_charge_cents, "balance_cents_remaining": balance_cents},
    }

@router.get("/api/v1/models")
def list_models_v1(api_key: str = Depends(get_api_key_from_request)):
    """OpenAI-SDK-compatible model list (client.models.list()) - also doubles as a cheap way
    for a developer to confirm their API key works before wiring up real calls."""
    from app.db.database import GenesisDB
    if not GenesisDB().validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "genesis-ai"}
                                        for m in sorted(PUBLIC_API_SUPPORTED_MODELS)]}

@router.get("/api/v1/usage")
def public_api_usage_v1(user_id: str = Depends(get_current_user_id)):
    """Spend/token summary for the LOGGED-IN platform account (not per-key) - reuses the
    normal JWT session auth, since this is meant for a developer checking their own dashboard,
    not for embedding in a server-to-server integration the way chat/completions is."""
    from app.db.database import GenesisDB
    return GenesisDB().get_public_api_usage_summary(user_id)
