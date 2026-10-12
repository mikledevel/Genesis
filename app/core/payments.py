"""
Stripe integration - prepaid balance top-ups (cards AND USDC/stablecoin, same Checkout
flow - Stripe surfaces both payment methods automatically once enabled in the dashboard).

Flow:
  1. Frontend calls POST /api/payments/checkout with an amount -> we create a Stripe
     Checkout Session and return its URL -> frontend redirects the browser there.
  2. Buyer pays (card or USDC) on Stripe's hosted page.
  3. Stripe calls our webhook (POST /api/payments/webhook) with a signed event once
     payment completes. We verify the signature, then credit the user's balance.

We never touch card/wallet details ourselves - Stripe's hosted Checkout page handles
that entirely, so there's no PCI-scope burden on this codebase.
"""
import os
from typing import Optional, Dict

from app.config import settings

MIN_TOPUP_USD = 5.0
MAX_TOPUP_USD = 500.0
PLATFORM_FEE_PCT = 25.0  # matches the commission already shown in the model-publish modal


def _get_stripe():
    """Import + configure the stripe SDK lazily, so the whole app doesn't fail to start
    just because Stripe keys aren't set yet (same pattern as the Groq key)."""
    import stripe
    key = settings.stripe_secret_key or os.environ.get("STRIPE_SECRET_KEY", "")
    if not key:
        return None
    stripe.api_key = key
    return stripe


def create_checkout_session(user_id: str, amount_usd: float, success_url: str, cancel_url: str) -> Optional[Dict]:
    """Create a Stripe Checkout Session for topping up a user's platform balance.
    Returns {"url": ..., "id": ...} or None if Stripe isn't configured or the call fails."""
    stripe = _get_stripe()
    if not stripe:
        return None
    if amount_usd < MIN_TOPUP_USD or amount_usd > MAX_TOPUP_USD:
        return None
    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card"],  # Stripe adds the "crypto"/USDC option automatically
                                              # once it's enabled in Dashboard -> Payment methods;
                                              # it doesn't need to be listed explicitly here.
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": "Genesis AI platform credit"},
                    "unit_amount": round(amount_usd * 100),
                },
                "quantity": 1,
            }],
            metadata={"user_id": user_id, "amount_usd": str(amount_usd)},
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"url": session.url, "id": session.id}
    except Exception as e:
        print(f"[payments] checkout session creation failed: {e}")
        return None


def verify_and_parse_webhook(payload: bytes, sig_header: str) -> Optional[Dict]:
    """Verify the Stripe webhook signature and return the parsed event, or None if the
    signature is invalid/missing. NEVER trust a webhook body without this check - anyone
    could POST a fake 'payment succeeded' event otherwise."""
    stripe = _get_stripe()
    if not stripe:
        return None
    webhook_secret = settings.stripe_webhook_secret or os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        print("[payments] STRIPE_WEBHOOK_SECRET not set - refusing to process webhook")
        return None
    try:
        import stripe as stripe_module
        event = stripe_module.Webhook.construct_event(payload, sig_header, webhook_secret)
        return event
    except (ValueError, Exception) as e:
        print(f"[payments] webhook signature verification failed: {e}")
        return None


def handle_checkout_completed(event, db) -> Optional[Dict]:
    """Process a checkout.session.completed event: credit the buyer's balance.
    Idempotent - safe to call multiple times for the same session (Stripe redelivers
    webhooks on transient failures).

    NOTE: `event` is a Stripe SDK object (from stripe.Webhook.construct_event), not a
    plain dict - it supports event["key"] and event.key, but NOT event.get("key")
    (Stripe's StripeObject intercepts .get as an attempted key lookup and raises
    AttributeError). Always use bracket/attribute access on Stripe objects, never .get()."""
    if event["type"] != "checkout.session.completed":
        return None
    session = event["data"]["object"]
    session_id = session["id"]
    if db.stripe_session_already_processed(session_id):
        return {"status": "already_processed", "session_id": session_id}
    metadata = session["metadata"] if session["metadata"] else {}
    user_id = metadata["user_id"] if "user_id" in metadata else None
    if not user_id:
        print(f"[payments] webhook session {session_id} has no user_id in metadata - skipping")
        return None
    amount_cents = session["amount_total"] if session["amount_total"] else 0
    db.add_transaction(user_id, "topup", amount_cents, "Stripe top-up", stripe_session_id=session_id)
    return {"status": "credited", "user_id": user_id, "amount_cents": amount_cents}
