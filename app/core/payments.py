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
import time
import uuid
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


def subscription_price_id(plan_id: str) -> str:
    field = {"pro": "stripe_price_pro_monthly", "business": "stripe_price_business_monthly"}.get(plan_id)
    return str(getattr(settings, field, "") or os.environ.get(field.upper(), "")) if field else ""


def plan_for_subscription_price(price_id: Optional[str]) -> Optional[str]:
    if not price_id:
        return None
    return next((plan for plan in ("pro", "business") if subscription_price_id(plan) == price_id), None)


def create_subscription_checkout(db, user_id: str, plan_id: str, success_url: str, cancel_url: str) -> Dict:
    """Create or resume one pending hosted Checkout session for a monthly plan."""
    stripe = _get_stripe()
    if not stripe:
        raise RuntimeError("Stripe payments are not configured")
    price_id = subscription_price_id(plan_id)
    if not price_id:
        raise RuntimeError(f"Stripe price for the {plan_id} plan is not configured")
    user = db.get_billing_user(user_id)
    if not user:
        raise ValueError("GenesisAI account was not found")

    customer_id = user["stripe_customer_id"]
    if not customer_id:
        customer = stripe.Customer.create(
            email=user["email"], name=user["name"] or None,
            metadata={"genesis_user_id": user_id},
            idempotency_key=f"genesis_customer_{user_id}",
        )
        customer_id = db.save_stripe_customer(user_id, customer["id"])

    attempt_id = uuid.uuid4().hex
    reservation = db.reserve_stripe_checkout(user_id, plan_id, attempt_id)
    if reservation["status"] == "subscription_exists":
        raise ValueError("An active subscription already exists. Use Manage billing to change or cancel it.")
    if reservation["status"] == "pending" and reservation["session_id"]:
        existing = stripe.checkout.Session.retrieve(reservation["session_id"])
        if existing["status"] == "open" and existing["url"]:
            return {"url": existing["url"], "id": existing["id"], "resumed": True}
        if existing["status"] == "complete":
            raise ValueError("Checkout completed; subscription confirmation is still processing.")
        db.clear_stripe_checkout(user_id, reservation["attempt_id"])
        reservation = db.reserve_stripe_checkout(user_id, plan_id, uuid.uuid4().hex)
        if reservation["status"] != "reserved":
            raise ValueError("A subscription checkout is already being prepared. Try again shortly.")

    attempt_id = reservation["attempt_id"]
    if reservation["status"] == "pending" and not reservation["session_id"]:
        # A previous request may have timed out after Stripe accepted it. Reusing the same
        # idempotency key makes Stripe return the original session instead of creating another.
        pass
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            customer=customer_id,
            client_reference_id=user_id,
            metadata={"genesis_user_id": user_id, "genesis_plan_id": plan_id},
            subscription_data={
                "metadata": {"genesis_user_id": user_id, "genesis_plan_id": plan_id},
                "billing_mode": {"type": "flexible"},
            },
            success_url=success_url,
            cancel_url=cancel_url,
            expires_at=int(time.time()) + 1800,
            submit_type="subscribe",
            idempotency_key=f"genesis_checkout_{attempt_id}",
        )
    except Exception:
        # Keep the reservation so a retry reuses the same Stripe idempotency key.
        raise
    db.save_stripe_checkout_session(user_id, attempt_id, session["id"])
    return {"url": session["url"], "id": session["id"], "resumed": False}


def create_customer_portal_session(db, user_id: str, return_url: str) -> Dict:
    stripe = _get_stripe()
    if not stripe:
        raise RuntimeError("Stripe payments are not configured")
    user = db.get_billing_user(user_id)
    if not user or not user["stripe_customer_id"]:
        raise ValueError("No Stripe customer is linked to this account yet.")
    params = {"customer": user["stripe_customer_id"], "return_url": return_url}
    if settings.stripe_portal_configuration_id:
        params["configuration"] = settings.stripe_portal_configuration_id
    session = stripe.billing_portal.Session.create(**params)
    return {"url": session["url"]}


def _value(obj, key, default=None):
    try:
        value = obj[key]
    except (KeyError, TypeError, AttributeError):
        value = default
    return default if value is None else value


def _stripe_id(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    return _value(value, "id") if value else None


def handle_subscription_event(event, db) -> Optional[Dict]:
    """Refresh the latest subscription from Stripe, then atomically update Genesis entitlements.

    Stripe events can arrive out of order. Fetching the current subscription prevents an old
    event payload from revoking or restoring access after a newer change has already happened.
    """
    event_type = _value(event, "type", "")
    event_id = _value(event, "id")
    obj = _value(_value(event, "data", {}), "object", {})
    session_user_id = None
    if event_type == "checkout.session.completed":
        if _value(obj, "mode") != "subscription":
            return None
        subscription_id = _stripe_id(_value(obj, "subscription"))
        session_user_id = _value(_value(obj, "metadata", {}), "genesis_user_id")
    elif event_type in {
        "customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted",
        "customer.subscription.paused", "customer.subscription.resumed",
    }:
        subscription_id = _stripe_id(_value(obj, "id"))
    else:
        return None

    if not event_id or not subscription_id:
        return None
    stripe = _get_stripe()
    if not stripe:
        raise RuntimeError("Stripe payments are not configured")
    subscription = stripe.Subscription.retrieve(subscription_id)
    customer_id = _stripe_id(_value(subscription, "customer"))
    metadata = _value(subscription, "metadata", {})
    user_id = _value(metadata, "genesis_user_id") or session_user_id
    if not user_id and customer_id:
        user_id = db.get_user_id_for_stripe_customer(customer_id)
    if not user_id or not customer_id:
        return None

    items = _value(_value(subscription, "items", {}), "data", []) or []
    price_id = _stripe_id(_value(items[0], "price")) if items else None
    status = str(_value(subscription, "status", "unknown"))
    period_end = _value(subscription, "current_period_end")
    result = db.apply_stripe_subscription_event(
        event_id=event_id,
        event_type=event_type,
        user_id=user_id,
        customer_id=customer_id,
        subscription_id=subscription_id,
        subscription_status=status,
        price_id=price_id,
        period_end=int(period_end) if period_end is not None else None,
        cancel_at_period_end=bool(_value(subscription, "cancel_at_period_end", False)),
        target_plan=plan_for_subscription_price(price_id),
    )
    return result


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
    if _value(session, "mode", "payment") != "payment":
        return None
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
