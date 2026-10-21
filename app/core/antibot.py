"""Bot protection for the two forms bots abuse most: account registration (free tokens, signup credit, spam accounts) and the
password-reset request (e-mail bombing of a victim, burning the mail provider's daily quota).

Layers, cheapest first - and what each does NOT do:
  1. honeypot field   - a hidden input a human never fills. Catches naive form-filling scripts only.
  2. minimum fill time - the form must have been open a few seconds. A script that talks to the API directly simply omits it
                         (and can fake it), so this is a speed bump, not a wall.
  3. disposable e-mail - refuses well-known throw-away mail services. The list below is PARTIAL and always will be; extend it
                         with DISPOSABLE_EMAIL_DOMAINS_EXTRA. A determined person can still use a real mailbox.
  4. captcha           - Cloudflare Turnstile, verified SERVER-side. This is the real gate. It is on only when both
                         TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY are set; if Cloudflare cannot be reached the request is
                         refused (fail closed) rather than waved through.
  5. canonical e-mail  - a+1@gmail.com, a.1@gmail.com and a@gmail.com are one mailbox; one mailbox = one account (see
                         GenesisDB.create_user / canonical_email_exists). Stops farming free quotas with aliases.
Rate limiting (app/core/rate_limit.py) and e-mail verification before any signup credit sit on top of these.
"""
import logging
import re
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException

from app.config import settings

log = logging.getLogger(__name__)

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

GENERIC_REFUSAL = "We could not complete this request. Please reload the page and try again."

# Well-known throw-away mail services (partial, by design: see the module docstring).
DISPOSABLE_DOMAINS = frozenset({
    "mailinator.com", "guerrillamail.com", "guerrillamail.net", "guerrillamail.org", "guerrillamailblock.com", "sharklasers.com",
    "grr.la", "spam4.me", "10minutemail.com", "10minutemail.net", "20minutemail.com", "tempmail.com", "temp-mail.org",
    "temp-mail.io", "tempmailo.com", "tempail.com", "tempr.email", "throwawaymail.com", "yopmail.com", "yopmail.net",
    "yopmail.fr", "getnada.com", "nada.email", "trashmail.com", "trashmail.net", "trashmail.de", "dispostable.com",
    "maildrop.cc", "mailnesia.com", "mintemail.com", "mohmal.com", "fakeinbox.com", "emailondeck.com", "getairmail.com",
    "mytemp.email", "burnermail.io", "discard.email", "discardmail.com", "spambox.us", "spamgourmet.com", "mailcatch.com",
    "moakt.com", "tmail.ws", "tmpmail.org", "tmpmail.net", "inboxkitten.com", "mail.tm", "1secmail.com", "1secmail.net",
    "1secmail.org", "emailfake.com", "fakemail.net", "luxusmail.org", "owlymail.com", "anonbox.net", "mailforspam.com",
    "harakirimail.com", "jetable.org", "33mail.com", "linshiyouxiang.net", "crazymailing.com", "dropmail.me", "mailtemp.net",
})

_EMAIL_RE = re.compile(r"^[^@\s,;<>()\[\]\\]+@[^@\s,;<>()\[\]\\]+\.[A-Za-z]{2,}$")


def valid_email_format(email: str) -> bool:
    return isinstance(email, str) and len(email) <= 254 and bool(_EMAIL_RE.match(email)) and ".." not in email


def email_domain(email: str) -> str:
    return (email or "").rsplit("@", 1)[-1].strip().lower()


def _extra_domains() -> set:
    return {d.strip().lower() for d in (settings.disposable_email_domains_extra or "").split(",") if d.strip()}


def is_disposable(email: str) -> bool:
    if not settings.antibot_block_disposable:
        return False
    dom = email_domain(email)
    blocked = DISPOSABLE_DOMAINS | _extra_domains()
    return dom in blocked or any(dom.endswith("." + d) for d in blocked)


def canonical_email(email: str) -> str:
    """The mailbox an address really delivers to. Gmail ignores dots and everything after '+'; other providers mostly
    ignore '+tag'. Used only to refuse a second account on the same mailbox - the login e-mail stays exactly as typed."""
    email = (email or "").strip().lower()
    if "@" not in email:
        return email
    local, domain = email.rsplit("@", 1)
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return f"{local}@{domain}"


def captcha_enabled() -> bool:
    on = bool(settings.turnstile_site_key and settings.turnstile_secret_key)
    if bool(settings.turnstile_site_key) != bool(settings.turnstile_secret_key):
        log.warning("Turnstile is half-configured (only one of TURNSTILE_SITE_KEY / TURNSTILE_SECRET_KEY is set): captcha is OFF")
    return on


def verify_turnstile(token: Optional[str], ip: Optional[str] = None) -> Tuple[bool, str]:
    """-> (ok, reason). reason 'unreachable' means Cloudflare could not be asked (the caller fails closed)."""
    if not token or not isinstance(token, str) or len(token) > 4096:
        return False, "missing-input-response"
    data = {"secret": settings.turnstile_secret_key, "response": token}
    if ip:
        data["remoteip"] = ip
    try:
        r = httpx.post(TURNSTILE_VERIFY_URL, data=data, timeout=5.0)
        body = r.json()
    except Exception as e:
        log.error("Turnstile verification could not be completed: %s", type(e).__name__)
        return False, "unreachable"
    if body.get("success") is True:
        return True, ""
    return False, ",".join(body.get("error-codes") or []) or "rejected"


def check_signup(payload: dict, ip: Optional[str], email: Optional[str] = None, kind: str = "register") -> None:
    """Raises HTTPException when the request looks automated. payload carries the extra form fields:
    website (honeypot), form_elapsed_ms (how long the form was open in the browser), captcha_token."""
    if str(payload.get("website") or "").strip():
        log.info("antibot: honeypot filled (%s)", kind)
        raise HTTPException(status_code=400, detail=GENERIC_REFUSAL)
    elapsed = payload.get("form_elapsed_ms")
    if elapsed is not None:
        try:
            too_fast = float(elapsed) < settings.antibot_min_form_seconds * 1000
        except (TypeError, ValueError):
            too_fast = True
        if too_fast:
            log.info("antibot: form submitted too fast (%s)", kind)
            raise HTTPException(status_code=400, detail=GENERIC_REFUSAL)
    if kind == "register" and email and is_disposable(email):
        raise HTTPException(status_code=400, detail="Please use a permanent e-mail address: temporary-mail services are not accepted.")
    if captcha_enabled():
        ok, why = verify_turnstile(payload.get("captcha_token"), ip)
        if not ok:
            if why == "unreachable":
                raise HTTPException(status_code=503, detail="We could not verify that you are human right now. Please try again in a minute.")
            log.info("antibot: captcha refused (%s): %s", kind, why)
            raise HTTPException(status_code=400, detail="Please complete the human check and try again.")
