"""Admin API (owner only). Every route depends on require_admin: a valid, non-revoked session AND an e-mail that is in
settings.admin_emails AND verified. Every change is written to the admin_actions audit table.

Deliberately NOT here: reading anyone's data (chats, datasets, models), impersonation ("log in as"), password or e-mail
changes, deleting accounts, minting marketplace balance. The panel manages plans, limits and suspensions only.
"""
from fastapi import APIRouter, Query
from fastapi import Depends, HTTPException
from app.core.auth import require_admin

router = APIRouter()

_MAX_TOKEN_LIMIT = 10 ** 12


def _db():
    from app.db.database import GenesisDB
    return GenesisDB()


def _target(db, user_id: str) -> dict:
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="No such user")
    return user


@router.get("/api/admin/overview")
def admin_overview(admin_id: str = Depends(require_admin)):
    from app.core import antibot, email_sender
    data = _db().admin_overview()
    data["email"] = email_sender.status()
    data["captcha_enabled"] = antibot.captcha_enabled()
    return data


@router.post("/api/admin/email/test")
def admin_email_test(admin_id: str = Depends(require_admin)):
    """Sends one test message to the admin's own address through the configured provider and reports the real result:
    the quickest way to find out that the key, the sender domain and the recipient are right."""
    from app.core import email_sender
    db = _db()
    me = db.get_user_by_id(admin_id)
    ok, detail = email_sender.deliver(me["email"], "Genesis AI test e-mail",
                                      "This is a test message from your Genesis AI admin panel. If you can read it, e-mail delivery works.")
    st = email_sender.status()
    db.log_admin_action(admin_id, "email_test", admin_id, {"provider": st["provider"], "ok": ok})
    return {"ok": ok, "provider": st["provider"], "from": st["from"], "to": me["email"], "detail": detail}


@router.get("/api/admin/users")
def admin_users(q: str = Query("", max_length=100), limit: int = Query(50, ge=1, le=200),
                      offset: int = Query(0, ge=0), admin_id: str = Depends(require_admin)):
    return _db().admin_list_users(q, limit, offset)


@router.post("/api/admin/users/{user_id}/plan")
def admin_set_plan(user_id: str, data: dict, admin_id: str = Depends(require_admin)):
    from app.core import plans
    db = _db()
    user = _target(db, user_id)
    plan = data.get("plan")
    if plan not in plans.PLAN_ORDER:
        raise HTTPException(status_code=400, detail=f"plan must be one of: {', '.join(plans.PLAN_ORDER)}")
    before = db.get_user_plan(user_id)
    db.set_user_plan(user_id, plan)
    db.log_admin_action(admin_id, "set_plan", user_id, {"email": user["email"], "from": before, "to": plan})
    return {"status": "ok", "user_id": user_id, "plan": plan}


@router.post("/api/admin/users/{user_id}/token-limit")
def admin_set_token_limit(user_id: str, data: dict, admin_id: str = Depends(require_admin)):
    """limit: null = back to the plan's limit; 0 = unlimited; a positive integer = that many tokens per month."""
    db = _db()
    user = _target(db, user_id)
    limit = data.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 or limit > _MAX_TOKEN_LIMIT):
        raise HTTPException(status_code=400, detail="limit must be null, 0 (unlimited) or a positive integer")
    before = db.get_user_groq_limit(user_id)
    db.set_user_groq_limit(user_id, limit)
    db.log_admin_action(admin_id, "set_token_limit", user_id, {"email": user["email"], "from": before, "to": limit})
    return {"status": "ok", "user_id": user_id, "token_limit_override": limit}


@router.post("/api/admin/users/{user_id}/disable")
def admin_set_disabled(user_id: str, data: dict, admin_id: str = Depends(require_admin)):
    db = _db()
    user = _target(db, user_id)
    disabled = data.get("disabled")
    if not isinstance(disabled, bool):
        raise HTTPException(status_code=400, detail="disabled must be true or false")
    if disabled and (user_id == admin_id or db.is_admin(user_id)):
        raise HTTPException(status_code=400, detail="An admin account cannot be suspended here")
    db.set_user_disabled(user_id, disabled)
    db.log_admin_action(admin_id, "disable" if disabled else "enable", user_id, {"email": user["email"]})
    return {"status": "ok", "user_id": user_id, "disabled": disabled}


@router.get("/api/admin/audit")
def admin_audit(limit: int = Query(50, ge=1, le=200), admin_id: str = Depends(require_admin)):
    return {"actions": _db().list_admin_actions(limit)}
