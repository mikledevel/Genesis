"""auth routes (split out of app/main.py)."""
from typing import Optional
from fastapi import APIRouter, Header, Request
from app.deps import (
    CheckoutRequest,
    Depends,
    HTTPException,
    LoginRequest,
    RegisterRequest,
    Request,
    create_token,
    get_current_user_id,
    hash_password,
    settings,
    verify_password
)

router = APIRouter()

# ── Auth ──
@router.post("/api/auth/register")
async def auth_register(req: RegisterRequest, request: Request):
    from app.db.database import GenesisDB
    from app.core.email_sender import send_verification_email
    from app.core import antibot
    from app.core.rate_limit import RateLimitMiddleware
    db = GenesisDB()
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not antibot.valid_email_format(req.email.strip()):
        raise HTTPException(status_code=400, detail="Invalid email address")
    if len(req.name) > 100:
        raise HTTPException(status_code=400, detail="Name is too long")
    # Bots first: they must not be able to probe which addresses exist, nor cost us a database lookup.
    antibot.check_signup(req.model_dump(), RateLimitMiddleware._client_ip(request), req.email.strip().lower(), "register")
    if db.get_user_by_email(req.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    if db.canonical_email_exists(antibot.canonical_email(req.email)):      # a+1@gmail.com is the mailbox a@gmail.com already has
        raise HTTPException(status_code=409, detail="An account with this email already exists")
    user = db.create_user(req.email, hash_password(req.password), req.name)
    token = create_token(user["id"])
    verify_token = db.create_auth_token(user["id"], "verify_email", ttl_seconds=86400)
    send_verification_email(user["email"], verify_token)
    return {"token": token, "user": user}

@router.post("/api/auth/login")
async def auth_login(req: LoginRequest):
    from app.db.database import GenesisDB
    db = GenesisDB()
    user = db.get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_token(user["id"])
    return {"token": token, "user": {"id": user["id"], "email": user["email"], "name": user["name"]}}

@router.get("/api/auth/me")
async def auth_me(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    user = GenesisDB().get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    user["email_verified"] = GenesisDB().is_email_verified(user_id)
    user["plan"] = GenesisDB().get_user_plan(user_id)
    user["is_admin"] = GenesisDB().is_admin(user_id)
    return {"user": user}

@router.get("/api/config/public")
async def public_config():
    """What the browser needs before login: the (public) Turnstile site key, empty when the captcha is off."""
    from app.core import antibot
    return {"turnstile_site_key": settings.turnstile_site_key if antibot.captcha_enabled() else ""}

@router.get("/api/plans")
async def public_plans():
    """The pricing list: only plans marked public, with the same limits the server enforces."""
    from app.core import plans
    return {"plans": plans.public_catalog()}

@router.get("/api/me/plan")
async def my_plan(user_id: str = Depends(get_current_user_id)):
    """The caller's plan, effective limits (null = unlimited) and current usage."""
    from app.db.database import GenesisDB
    from app.core import plans
    db = GenesisDB()
    admin = db.is_admin(user_id)
    return {"plan": plans.get_plan(db.get_user_plan(user_id)), "is_admin": admin,
            "effective_limits": plans.limits_for(db, user_id), "usage": plans.usage_for(db, user_id)}

@router.post("/api/auth/verify-email/resend")
async def resend_verification_email(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB as _DB
    if _DB().count_recent_auth_tokens(user_id, "verify_email", 60) >= settings.email_per_address_per_hour:
        raise HTTPException(status_code=429, detail="Too many verification e-mails were requested. Please try again in an hour.")
    from app.db.database import GenesisDB
    from app.core.email_sender import send_verification_email
    db = GenesisDB()
    if db.is_email_verified(user_id):
        return {"status": "already_verified"}
    user = db.get_user_by_id(user_id)
    verify_token = db.create_auth_token(user_id, "verify_email", ttl_seconds=86400)
    send_verification_email(user["email"], verify_token)
    return {"status": "sent"}

@router.post("/api/auth/verify-email/confirm")
async def confirm_verification_email(data: dict, authorization: Optional[str] = Header(None)):
    from app.db.database import GenesisDB
    from app.core.auth import decode_token
    db = GenesisDB()
    # An OWNER e-mail (settings.admin_emails) is verified only from a session of that SAME account: it takes both the
    # mailbox (the link) and the password (the session). Without this, someone who registered the owner's address first
    # (with a password of their own) would become admin the moment the real owner clicked the verification mail.
    # If that happened to you, the way in is "forgot password" (it voids the other person's password and sessions).
    pending = db.peek_auth_token(data.get("token", ""), "verify_email")
    if pending:
        owner = db.get_user_by_id(pending)
        if owner and (owner["email"] or "").lower() in db.admin_email_set():
            caller = decode_token(authorization.removeprefix("Bearer ").strip()) if authorization and authorization.startswith("Bearer ") else None
            if caller != pending:
                raise HTTPException(status_code=401, detail="This is an owner account: log in first, then open the verification link "
                                                            "in the same browser. If you did not create this account, use 'forgot password'.")
    user_id = db.consume_auth_token(data.get("token", ""), "verify_email")
    if not user_id:
        raise HTTPException(status_code=400, detail="This verification link is invalid or has expired")
    db.mark_email_verified(user_id)
    # Free starter balance - see settings.signup_credit_cents docstring for why this is tied
    # to verification rather than granted at raw registration. consume_auth_token above
    # already invalidates the token on first use (returns None on any reuse), so this whole
    # block - and therefore the credit grant - can only run once per account.
    if settings.signup_credit_cents > 0:
        db.add_transaction(user_id, "signup_bonus", settings.signup_credit_cents,
                           "Welcome bonus - thanks for verifying your email")
    return {"status": "verified", "credit_cents": settings.signup_credit_cents}

@router.post("/api/auth/password-reset/request")
async def request_password_reset(data: dict, request: Request):
    from app.db.database import GenesisDB
    from app.core.email_sender import send_password_reset_email
    from app.core import antibot
    from app.core.rate_limit import RateLimitMiddleware
    db = GenesisDB()
    # A bot can use this endpoint to e-mail-bomb any address and to burn the mail provider's daily quota.
    antibot.check_signup(data, RateLimitMiddleware._client_ip(request), None, "reset")
    email = (data.get("email") or "").lower().strip()
    user = db.get_user_by_email(email)
    # Always return the same response whether or not the email exists - returning a
    # different response for "no account with that email" would let anyone enumerate which
    # emails are registered on the platform just by hitting this endpoint repeatedly.
    # Per-address cap: past it we silently send nothing (the response stays identical).
    if user and db.count_recent_auth_tokens(user["id"], "password_reset", 60) < settings.email_per_address_per_hour:
        reset_token = db.create_auth_token(user["id"], "password_reset", ttl_seconds=3600)
        send_password_reset_email(user["email"], reset_token)
    return {"status": "if_account_exists_email_sent"}

@router.post("/api/auth/password-reset/confirm")
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
    db.invalidate_sessions(user_id)          # a reset means "someone else may have my session": void all existing tokens
    return {"status": "password_updated"}

@router.post("/api/payments/checkout")
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

@router.post("/api/payments/webhook")
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

@router.get("/api/payments/balance")
async def payments_balance(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"balance_cents": GenesisDB().get_balance(user_id)}

@router.get("/api/payments/transactions")
async def payments_transactions(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"transactions": GenesisDB().get_transactions(user_id)}

@router.post("/api/payments/dev-credit")
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

# ── API Keys ──
@router.get("/api/keys")
async def list_keys(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"keys": GenesisDB().list_api_keys(user_id)}

@router.post("/api/keys/create")
async def create_key(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core import plans
    db = GenesisDB()
    _limit = plans.gate_message(db, user_id, "api_key")
    if _limit:
        raise HTTPException(status_code=403, detail=_limit)
    return db.create_api_key(user_id, data.get("user_name", ""), plans.api_key_calls(db, user_id, data.get("calls_limit")))

@router.delete("/api/keys/{api_key}")
async def delete_key(api_key: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_api_key(api_key, user_id):
        raise HTTPException(status_code=404, detail="API key not found, or it doesn't belong to you")
    return {"status": "deleted"}

@router.get("/api/usage")
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

@router.get("/api/usage/groq")
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
