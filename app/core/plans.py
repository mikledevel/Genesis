"""Plans (tiers): ONE data-driven catalog + the helpers that enforce it.

Only limits that the code really enforces are listed under "limits" - what is promised on the pricing page and what the
server checks are the same dict. Things that do not exist yet live under ROADMAP and are never sold.

Two separate money layers exist and must not be mixed up:
  * PLANS (this file)    - how much of the PLATFORM an account may use: monthly LLM tokens (the platform's own cost),
                           how many bots / models / API keys, upload size, LLM-written custom code.
  * BALANCE (cents)      - pay-per-use money inside the marketplace (paid bots/models, public API, agent browser).
                           Plans do not touch it; not even admins get free marketplace money.

"server_training": training on the PLATFORM's servers (train, forecast, retrain, create_model) is a paid feature because it
spends the platform's CPU. Training on the user's own computer (app/core/local_training.py) costs the platform nothing and is
free on every plan.

Counted resources (bots, models, API keys, scheduled jobs, A/B tests, saved prompts, knowledge-base documents and text per bot) are
limits on what a user HAS. Server training is limited by what a user DOES per calendar month (server_jobs_per_month, counted in
db.feature_usage; train / forecast / retrain / improve / create model each start one job). Plan-dependent behaviour that is not a
count is also here: kb_top_k (how many knowledge-base passages a bot gets per question; the BOT OWNER's plan decides, not the
chatter's) and improve_budget (how deep model improvement may search).

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
                   "api_calls_per_key": 100, "max_dataset_mb": 5, "custom_code_models": False,
                   "server_training": False, "server_jobs_per_month": 0, "improve_budget": None,
                   "kb_documents": 3, "kb_chars": 200_000, "kb_top_k": 3,
                   "max_scheduled_jobs": 1, "max_ab_tests": 2, "max_prompts": 10},
    },
    "pro": {
        "name": "Pro", "price_usd_month": 19, "public": True,
        "tagline": "For a shop or a small team that plans from its data",
        "limits": {"monthly_tokens": 5_000_000, "max_bots": 10, "max_models": 25, "max_api_keys": 5,
                   "api_calls_per_key": 5_000, "max_dataset_mb": 25, "custom_code_models": True,
                   "server_training": True, "server_jobs_per_month": 40, "improve_budget": "balanced",
                   "kb_documents": 25, "kb_chars": 5_000_000, "kb_top_k": 6,
                   "max_scheduled_jobs": 10, "max_ab_tests": 20, "max_prompts": 100},
    },
    "business": {
        "name": "Business", "price_usd_month": 79, "public": True,
        "tagline": "Higher limits for growing teams",
        "limits": {"monthly_tokens": 25_000_000, "max_bots": 50, "max_models": 100, "max_api_keys": 20,
                   "api_calls_per_key": 50_000, "max_dataset_mb": 25, "custom_code_models": True,
                   "server_training": True, "server_jobs_per_month": 300, "improve_budget": "thorough",
                   "kb_documents": 100, "kb_chars": 25_000_000, "kb_top_k": 8,
                   "max_scheduled_jobs": 50, "max_ab_tests": 100, "max_prompts": 500},
    },
}

# Not built yet - never put these on a pricing page as if they existed.
ROADMAP = ["team seats", "forecast report export from the web app", "priority support", "white-label bots"]

# gpt-oss-120b on Groq, per 1M tokens (checked Oct 2026: $0.15 in / $0.60 out). Only for the admin cost ESTIMATE.
LLM_PRICE_IN_PER_M = 0.15
LLM_PRICE_OUT_PER_M = 0.60

FEATURE_LABELS = {"max_bots": "bots", "max_models": "saved models", "max_api_keys": "API keys",
                  "max_scheduled_jobs": "active scheduled jobs", "max_ab_tests": "A/B tests", "max_prompts": "saved prompts"}

IMPROVE_BUDGETS = ["fast", "balanced", "thorough"]   # how deep model improvement may search (app/core/model_improver.py)


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
                    "api_calls_per_key": None, "max_dataset_mb": None, "custom_code_models": True,
                    "server_training": True, "server_jobs_per_month": None, "improve_budget": "thorough",
                    "kb_documents": None, "kb_chars": None, "kb_top_k": 8,
                    "max_scheduled_jobs": None, "max_ab_tests": None, "max_prompts": None}


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
            "models": _count_models(user_id), "api_keys": db.count_active_api_keys(user_id),
            "scheduled_jobs": _count_jobs(db, user_id), "ab_tests": len(db.list_ab_tests(user_id)),
            "prompts": len(db.list_prompts(user_id)), "server_jobs_this_month": db.count_feature_use_this_month(user_id, SERVER_JOB)}


def _count_bots(user_id: str) -> int:
    from app.core.agent_store import AgentStore
    return len(AgentStore().list_agents(author=user_id, published_only=False))


def _count_models(user_id: str) -> int:
    from app.core.engine.registry import ModelRegistry
    return len(ModelRegistry().list_models_by_author(user_id, limit=100000))


def _count_jobs(db, user_id: str) -> int:
    return len(db.list_scheduled_jobs_for_owner(user_id, active_only=True))


SERVER_JOB = "server_job"


def _upgrade_hint(plan_id: str) -> str:
    i = PLAN_ORDER.index(normalize_plan(plan_id))
    nxt = next((p for p in PLAN_ORDER[i + 1:] if _CATALOG[p]["public"]), None)
    return f" Upgrade to {_CATALOG[nxt]['name']} for a higher limit." if nxt else " Contact us for a higher limit."


def gate_message(db, user_id: Optional[str], kind: str) -> Optional[str]:
    """None = allowed. Otherwise the sentence to show the user.
    kind: 'bot' | 'model' | 'api_key' | 'scheduled_job' | 'ab_test' | 'prompt' | 'custom_code' | 'server_training'.
    ('server_training' only says whether the plan has it; the monthly job count is start_server_job's business.)"""
    if not user_id or user_id == "anonymous":
        return None
    limits = limits_for(db, user_id)
    plan_id = db.get_user_plan(user_id)
    name = get_plan(plan_id)["name"]
    if kind == "server_training":
        if limits["server_training"]:
            return None
        from app.core.local_training import local_instructions
        return (f"Training on the platform's servers is not part of the {name} plan." + _upgrade_hint(plan_id)
                + "\n\n" + local_instructions())
    if kind == "custom_code":
        if limits["custom_code_models"]:
            return None
        return (f"Custom code models (neural networks, clustering, anomaly detection written by the AI) are not part of the "
                f"{name} plan." + _upgrade_hint(plan_id))
    key = {"bot": "max_bots", "model": "max_models", "api_key": "max_api_keys", "scheduled_job": "max_scheduled_jobs",
           "ab_test": "max_ab_tests", "prompt": "max_prompts"}[kind]
    limit = limits[key]
    if limit is None:
        return None
    used = {"bot": _count_bots, "model": _count_models, "api_key": db.count_active_api_keys,
            "scheduled_job": lambda u: _count_jobs(db, u), "ab_test": lambda u: len(db.list_ab_tests(u)),
            "prompt": lambda u: len(db.list_prompts(u))}[kind](user_id)
    if used < limit:
        return None
    return f"The {name} plan allows {limit} {FEATURE_LABELS[key]} and you have {used}." + _upgrade_hint(plan_id)


def start_server_job(db, user_id: Optional[str]) -> Optional[str]:
    """Call when a job that spends the platform's CPU (train, forecast, retrain, improve, create model) is about to start.
    None = go ahead, and the job is counted for this calendar month. Otherwise the sentence to show. Owner/admin and internal
    callers are not limited and not counted."""
    if not user_id or user_id == "anonymous":
        return None
    limit = limits_for(db, user_id)["server_jobs_per_month"]
    if limit is None:
        return None
    used = db.count_feature_use_this_month(user_id, SERVER_JOB)
    if not db.claim_monthly_use(user_id, SERVER_JOB, limit):
        plan_id = db.get_user_plan(user_id)
        return (f"The {get_plan(plan_id)['name']} plan allows {limit} server training jobs per month and {used} have been used "
                f"(the count restarts on the 1st)." + _upgrade_hint(plan_id) + " Training on your own computer is free and not counted.")
    return None


def kb_gate_message(db, owner_id: Optional[str], agent_id: str, new_chars: int) -> Optional[str]:
    """Is one more document of `new_chars` characters allowed in this bot's knowledge base? Judged by the bot OWNER's plan."""
    if not owner_id or owner_id == "anonymous":
        return None
    limits = limits_for(db, owner_id)
    plan_id = db.get_user_plan(owner_id)
    name = get_plan(plan_id)["name"]
    use = db.get_kb_usage(agent_id, owner_id)
    if limits["kb_documents"] is not None and use["documents"] >= limits["kb_documents"]:
        return (f"The {name} plan allows {limits['kb_documents']} knowledge-base documents per bot and this bot has {use['documents']}. "
                "Delete one or merge documents." + _upgrade_hint(plan_id))
    cap = limits["kb_chars"]
    if cap is not None and use["chars"] + new_chars > cap:
        return (f"The {name} plan allows {cap:,} characters of knowledge-base text per bot; this bot has {use['chars']:,} and the new "
                f"document adds {new_chars:,}." + _upgrade_hint(plan_id))
    return None


def kb_max_upload_bytes(db, owner_id: Optional[str]) -> Optional[int]:
    """The most bytes worth reading from one upload (UTF-8 is at most 4 bytes per character); None = no cap."""
    cap = limits_for(db, owner_id)["kb_chars"]
    return None if cap is None else cap * 4


def kb_top_k(db, owner_id: Optional[str]) -> int:
    """How many knowledge-base passages the bot gets per question, by its OWNER's plan."""
    return int(limits_for(db, owner_id)["kb_top_k"])


def clamp_improve_budget(db, user_id: Optional[str], requested: Optional[str]):
    """(budget, note). The requested search depth, lowered to what the plan allows; note says so when it was lowered."""
    allowed = limits_for(db, user_id)["improve_budget"] or "fast"
    want = requested if requested in IMPROVE_BUDGETS else "balanced"
    if IMPROVE_BUDGETS.index(want) <= IMPROVE_BUDGETS.index(allowed):
        return want, ""
    plan_id = db.get_user_plan(user_id) if user_id and user_id != "anonymous" else "free"
    return allowed, f"The {get_plan(plan_id)['name']} plan searches up to '{allowed}' depth; '{want}' was lowered." + _upgrade_hint(plan_id)


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
