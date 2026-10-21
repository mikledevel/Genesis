"""Plans (tiers): ONE data-driven catalog + the helpers that enforce it.

Only limits that the code really enforces are listed under "limits" - what is promised on the pricing page and what the
server checks are the same dict. Things that do not exist yet live under ROADMAP and are never sold.

Two separate money layers exist and must not be mixed up:
  * PLANS (this file)    - how much of the PLATFORM an account may use: monthly LLM tokens (the platform's own cost),
                           how many bots / models / API keys, upload size, LLM-written custom code.
  * BALANCE (cents)      - pay-per-use money inside the marketplace (paid bots/models, public API, agent browser).
                           Plans do not touch it; not even admins get free marketplace money.

Limit value None = unlimited. The free plan's token limit is settings.default_monthly_groq_token_limit (0 or less there
keeps its old meaning: unlimited). Subscriptions are assigned by an admin for now (no automatic payment collection yet).
"""
from typing import Any, Dict, List, Optional

PLAN_ORDER = ["free", "pro", "business"]

_CATALOG: Dict[str, Dict[str, Any]] = {
    "free": {
        "name": "Free", "price_usd_month": 0, "public": True,
        "tagline": "Try it on your own data",
        "limits": {"monthly_tokens": "settings", "max_bots": 2, "max_models": 3, "max_api_keys": 1,
                   "api_calls_per_key": 100, "max_dataset_mb": 5, "custom_code_models": False},
    },
    "pro": {
        "name": "Pro", "price_usd_month": 19, "public": True,
        "tagline": "For a shop or a small team that plans from its data",
        "limits": {"monthly_tokens": 5_000_000, "max_bots": 10, "max_models": 25, "max_api_keys": 5,
                   "api_calls_per_key": 5_000, "max_dataset_mb": 25, "custom_code_models": True},
    },
    "business": {   # by arrangement: not shown on the public pricing list until it has something Pro does not
        "name": "Business", "price_usd_month": 79, "public": False,
        "tagline": "Bigger limits, set up by arrangement",
        "limits": {"monthly_tokens": 25_000_000, "max_bots": 50, "max_models": 100, "max_api_keys": 20,
                   "api_calls_per_key": 50_000, "max_dataset_mb": 25, "custom_code_models": True},
    },
}

# Not built yet - never put these on a pricing page as if they existed.
ROADMAP = ["team seats", "forecast report export from the web app", "priority support", "white-label bots"]

# gpt-oss-120b on Groq, per 1M tokens (checked Oct 2026: $0.15 in / $0.60 out). Only for the admin cost ESTIMATE.
LLM_PRICE_IN_PER_M = 0.15
LLM_PRICE_OUT_PER_M = 0.60

FEATURE_LABELS = {"max_bots": "bots", "max_models": "saved models", "max_api_keys": "API keys"}


def normalize_plan(plan_id: Optional[str]) -> str:
    return plan_id if plan_id in _CATALOG else "free"


def estimate_llm_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return round(prompt_tokens / 1e6 * LLM_PRICE_IN_PER_M + completion_tokens / 1e6 * LLM_PRICE_OUT_PER_M, 4)


def plan_tokens(plan_id: str) -> int:
    """Monthly token limit of a plan; 0 or less means unlimited (the quota module's convention)."""
    v = get_plan(plan_id)["limits"]["monthly_tokens"]
    return 0 if v is None else int(v)


def get_plan(plan_id: str) -> Dict[str, Any]:
    """The plan with its limits resolved (the free token limit is read from settings at call time)."""
    from app.config import settings
    p = _CATALOG[normalize_plan(plan_id)]
    limits = dict(p["limits"])
    if limits["monthly_tokens"] == "settings":
        limits["monthly_tokens"] = settings.default_monthly_groq_token_limit
    if limits["monthly_tokens"] is not None and limits["monthly_tokens"] <= 0:
        limits["monthly_tokens"] = None
    return {"id": normalize_plan(plan_id), "name": p["name"], "price_usd_month": p["price_usd_month"],
            "public": p["public"], "tagline": p["tagline"], "limits": limits}


def public_catalog() -> List[Dict[str, Any]]:
    return [get_plan(pid) for pid in PLAN_ORDER if _CATALOG[pid]["public"]]


UNLIMITED_LIMITS = {"monthly_tokens": None, "max_bots": None, "max_models": None, "max_api_keys": None,
                    "api_calls_per_key": None, "max_dataset_mb": None, "custom_code_models": True}


def limits_for(db, user_id: Optional[str]) -> Dict[str, Any]:
    """Effective limits of an account. Admins (verified owner e-mails) and anonymous internal paths are unlimited; a
    per-user token override (users.monthly_groq_token_limit) wins over the plan's token limit."""
    if not user_id or user_id == "anonymous" or db.is_admin(user_id) is True:
        return dict(UNLIMITED_LIMITS)
    limits = dict(get_plan(db.get_user_plan(user_id))["limits"])
    override = db.get_user_groq_limit(user_id)
    if override is not None:
        limits["monthly_tokens"] = None if override <= 0 else override
    return limits


def usage_for(db, user_id: str) -> Dict[str, int]:
    return {"tokens_this_month": db.get_groq_usage_this_month(user_id), "bots": _count_bots(user_id),
            "models": _count_models(user_id), "api_keys": db.count_active_api_keys(user_id)}


def _count_bots(user_id: str) -> int:
    from app.core.agent_store import AgentStore
    return len(AgentStore().list_agents(author=user_id, published_only=False))


def _count_models(user_id: str) -> int:
    from app.core.engine.registry import ModelRegistry
    return len(ModelRegistry().list_models_by_author(user_id, limit=100000))


def _upgrade_hint(plan_id: str) -> str:
    i = PLAN_ORDER.index(normalize_plan(plan_id))
    nxt = next((p for p in PLAN_ORDER[i + 1:] if _CATALOG[p]["public"]), None)
    return f" Upgrade to {_CATALOG[nxt]['name']} for more." if nxt else " Contact us for a higher limit."


def gate_message(db, user_id: Optional[str], kind: str) -> Optional[str]:
    """None = allowed. Otherwise the sentence to show the user. kind: 'bot' | 'model' | 'api_key' | 'custom_code'."""
    if not user_id or user_id == "anonymous":
        return None
    limits = limits_for(db, user_id)
    plan_id = db.get_user_plan(user_id)
    name = get_plan(plan_id)["name"]
    if kind == "custom_code":
        if limits["custom_code_models"]:
            return None
        return (f"Custom code models (neural networks, clustering, anomaly detection written by the AI) are not part of the "
                f"{name} plan." + _upgrade_hint(plan_id))
    key = {"bot": "max_bots", "model": "max_models", "api_key": "max_api_keys"}[kind]
    limit = limits[key]
    if limit is None:
        return None
    used = {"bot": _count_bots, "model": _count_models, "api_key": db.count_active_api_keys}[kind](user_id)
    if used < limit:
        return None
    return f"The {name} plan allows {limit} {FEATURE_LABELS[key]} and you have {used}." + _upgrade_hint(plan_id)


def max_dataset_bytes(db, user_id: Optional[str]) -> Optional[int]:
    mb = limits_for(db, user_id)["max_dataset_mb"]
    return None if mb is None else int(mb * 1024 * 1024)


def api_key_calls(db, user_id: str, requested: Any) -> int:
    """Calls limit for a NEW API key: the requested value (default = the plan's), capped by the plan's maximum."""
    cap = limits_for(db, user_id)["api_calls_per_key"]
    default = cap if cap is not None else 100
    try:
        n = int(requested) if requested is not None else default
    except (TypeError, ValueError):
        n = default
    n = max(1, n)
    return n if cap is None else min(n, cap)
