"""Pricing math for the public developer API (POST /api/v1/chat/completions).

Pulled out of app/main.py into pure functions so the (fiddly, money-related) rounding and
reconciliation logic can be unit-tested directly without a running app or a real database -
see tests/test_public_api_billing.py. See app/config.py's public_api_* settings for the
actual price and the reasoning behind it.

THE CORE PROBLEM this module solves: a chat completion's real cost isn't known until AFTER
generation (completion_tokens depends on what the model actually writes), but the platform
can't just generate first and hope the caller can pay - a real Groq/OpenRouter API call has
already been paid for by then even if the developer's balance turns out to be empty. The
pattern used here (the same one real metered LLM APIs use in one form or another):

    1. PRE-CHARGE a worst-case estimate (prompt tokens estimated from message length,
       completion tokens = the max_tokens the caller asked for) BEFORE calling the LLM at
       all, atomically via GenesisDB.charge_for_usage - if the developer can't afford the
       worst case, the call is rejected with 402 and no provider spend happens.
    2. Generate the real completion.
    3. RECONCILE: compute the real cost from the real token counts the provider returned,
       and true up the pre-charge to match - refund the difference (the normal case, since
       actual usage is almost always less than the worst-case max_tokens ceiling), or make
       one more small charge attempt if the estimate somehow undershot (rare - see
       reconcile_charge's docstring).
"""
import math

from app.config import settings


def compute_cost_cents(prompt_tokens: int, completion_tokens: int) -> int:
    """Real metered cost in whole cents for a completion with known token counts.

    Rounds UP to the nearest cent (ceil, not round) and enforces the per-request minimum -
    standard practice for metered billing (AWS, GCP, etc. all round usage up to the smallest
    billable unit in the provider's favor, never down), and avoids an integer-rounding
    loophole where short requests would otherwise cost $0.00.
    """
    raw_cents = (prompt_tokens * settings.public_api_input_price_cents_per_million
                 + completion_tokens * settings.public_api_output_price_cents_per_million) / 1_000_000
    return max(settings.public_api_min_charge_cents, math.ceil(raw_cents))


def estimate_prompt_tokens(messages: list) -> int:
    """A quick, conservative (rounds UP) token estimate for the pre-charge, before a real
    completion (and its real usage.prompt_tokens) exists.

    ~4 characters per token is the standard rough heuristic for English text. Deliberately
    biased toward OVERestimating: this number only affects the size of the up-front hold,
    which gets trued up against the real count once the completion comes back (see
    reconcile_charge) - an overestimate just means a slightly larger temporary hold and a
    slightly larger refund, while an underestimate could let a request through whose real
    cost the caller then can't actually cover.
    """
    total_chars = sum(len(str(m.get("content", ""))) for m in (messages or []))
    return max(1, math.ceil(total_chars / 4))


def estimate_worst_case_cost_cents(messages: list, max_tokens: int) -> int:
    """The amount pre-charged before the LLM call: prompt cost from the conservative
    estimate above, plus completion cost as if max_tokens were used in full (the actual
    worst case - a completion can't cost more than that, by construction, since the
    provider is asked to stop generating at max_tokens)."""
    return compute_cost_cents(estimate_prompt_tokens(messages), max_tokens)


def reconcile_charge(db, user_id: str, api_key: str, pre_charged_cents: int,
                      prompt_tokens: int, completion_tokens: int) -> int:
    """After the real completion is in hand: compute the real cost from the real token
    counts, and true up the pre-charge to match it exactly. Returns the FINAL amount the
    caller actually ends up paying (for the response's billing block).

    - real cost < pre-charge (the normal case - actual usage is almost always under the
      worst-case max_tokens ceiling): refund the difference. A refund is a plain credit
      (add_transaction), not a conditional debit, so it needs none of charge_for_usage's
      atomic-race protection - there's no "insufficient balance" failure mode for giving
      money back.
    - real cost > pre-charge (rare - only if estimate_prompt_tokens undershot the real
      prompt_tokens count, e.g. unusually token-dense input like source code or non-English
      text where the ~4-chars/token heuristic runs a bit low): attempt one more small charge
      for the shortfall. If THAT fails (the caller's balance dropped in the meantime), the
      platform eats the shortfall rather than trying to claw money back for a completion
      that has already been generated and can't be un-generated - a bounded, rare loss
      (capped at a few tokens' worth of cost) is the right trade-off here, not penalizing a
      caller after the fact for our own estimate being a little low.
    """
    real_cost_cents = compute_cost_cents(prompt_tokens, completion_tokens)
    if real_cost_cents < pre_charged_cents:
        refund = pre_charged_cents - real_cost_cents
        db.add_transaction(user_id, "public_api_refund", refund,
                            f"Refund: pre-charged {pre_charged_cents}c for an estimated worst case, "
                            f"actual usage cost {real_cost_cents}c", related_id=api_key)
        return real_cost_cents
    elif real_cost_cents > pre_charged_cents:
        shortfall = real_cost_cents - pre_charged_cents
        result = db.charge_for_usage(user_id, shortfall,
                                      f"Additional usage beyond the pre-charge estimate ({api_key})")
        return pre_charged_cents + (shortfall if result["ok"] else 0)
    return real_cost_cents
