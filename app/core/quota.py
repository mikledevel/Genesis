"""
Per-user monthly Groq usage quota - the enforcement half of the visibility built by
groq_usage_log/get_groq_usage_summary. Visibility alone doesn't stop a single account from
running up unlimited real cost via repeated bot-builder sessions or codegen fix loops (each
bot build can be 5-15+ real Groq calls; a stuck fix loop compounds that). This adds an actual
cap, checked BEFORE a Groq call is made at every call site (not just logged after the fact),
so a call that would exceed quota never happens instead of being tracked as having happened
too late to matter.

There is deliberately no public/self-service endpoint for a user to raise their own limit -
the cap protects the platform's own shared GROQ_API_KEY, not the user's own preference, so
raising it is an operator action (directly via GenesisDB.set_user_groq_limit), not something
exposed over the API.
"""
from typing import Tuple, Optional


def check_quota(db, user_id: Optional[str]) -> Tuple[bool, int, int]:
    """Returns (allowed, used_this_month, limit). A None user_id (no logged-in session - e.g.
    some internal/anonymous code path) is always allowed, since quota is a per-account
    concept and there's no account to attribute or throttle. A limit of 0 or less means
    unlimited (explicit opt-out), also always allowed."""
    if not user_id:
        return True, 0, 0
    limit = db.get_user_groq_limit(user_id)
    if limit is None:
        from app.config import settings
        limit = settings.default_monthly_groq_token_limit
    if limit is None or limit <= 0:
        return True, 0, 0
    used = db.get_groq_usage_this_month(user_id)
    return used < limit, used, limit


def quota_exceeded_message(used: int, limit: int) -> str:
    return (
        f"Достигнут месячный лимит использования AI ({used:,}/{limit:,} токенов). "
        f"Лимит обновится в начале следующего месяца."
    )
