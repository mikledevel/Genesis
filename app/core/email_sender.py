"""Outbound e-mail: verification, password reset and the scheduler's notifications.

Provider, in order: Resend (RESEND_API_KEY) -> SMTP (SMTP_HOST) -> "dev" (nothing is sent: the message, including the real link,
is printed to the server log so local work and tests still run end to end).

Rules this module keeps:
  * it never raises on a delivery failure - a broken mail provider must not break registration or password reset for the person
    in front of the request; the failure is logged (status and the provider's message, never the key);
  * with a real provider the verification / reset mails go out from a background thread, so a slow provider neither stalls the
    server nor lets anyone learn from the response time whether an address has an account (see request_password_reset);
  * header injection is closed: the subject and recipient are flattened to one line, and the HTML body escapes every value;
  * the API key lives only in settings (.env) and is scrubbed from anything that is logged.
"""
import html as _html
import logging
import smtplib
import ssl
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple

import httpx

from app.config import settings

log = logging.getLogger(__name__)

RESEND_URL = "https://api.resend.com/emails"
_SYNC = False          # tests set True: send in the calling thread so results can be asserted


def email_provider() -> str:
    if settings.resend_api_key:
        return "resend"
    if settings.smtp_host:
        return "smtp"
    return "dev"


def from_address() -> str:
    return settings.email_from or settings.smtp_from


def status() -> dict:
    """What the admin panel shows. Never contains the key itself."""
    return {"provider": email_provider(), "from": from_address(), "key_set": bool(settings.resend_api_key)}


def _one_line(s: str, limit: int = 200) -> str:
    return " ".join(str(s).split())[:limit]


def _scrub(text: str) -> str:
    key = settings.resend_api_key
    return text.replace(key, "***") if key else text


def _send_resend(to: str, subject: str, text: str, html: Optional[str]) -> Tuple[bool, str]:
    payload = {"from": _one_line(from_address()), "to": [to], "subject": subject, "text": text}
    if html:
        payload["html"] = html
    headers = {"Authorization": f"Bearer {settings.resend_api_key}", "Content-Type": "application/json", "User-Agent": "genesis-ai/1.0"}
    detail = "unknown error"
    for attempt in range(3):
        delay = 0.6 * (attempt + 1)
        try:
            r = httpx.post(RESEND_URL, json=payload, headers=headers, timeout=10.0)
        except httpx.HTTPError as e:
            detail = f"network error ({type(e).__name__})"
        else:
            if r.status_code < 300:
                try:
                    return True, str(r.json().get("id", ""))
                except Exception:
                    return True, ""
            try:
                msg = r.json().get("message") or r.text[:200]
            except Exception:
                msg = r.text[:200]
            detail = _scrub(f"HTTP {r.status_code}: {msg}")
            if r.status_code not in (429, 500, 502, 503, 504):
                return False, detail                     # a 4xx (bad key, unverified domain, invalid address) will not fix itself
            ra = r.headers.get("retry-after", "")
            if ra.replace(".", "", 1).isdigit():
                delay = min(float(ra), 5.0)
        if attempt < 2:
            time.sleep(delay)
    return False, detail


def _send_smtp(to: str, subject: str, text: str, html: Optional[str]) -> Tuple[bool, str]:
    try:
        msg = MIMEMultipart("alternative")
        msg["From"], msg["To"], msg["Subject"] = _one_line(from_address()), to, subject
        msg.attach(MIMEText(text, "plain", "utf-8"))
        if html:
            msg.attach(MIMEText(html, "html", "utf-8"))
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
            server.starttls(context=context)
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(from_address(), to, msg.as_string())
        return True, ""
    except Exception as e:
        return False, f"smtp error ({type(e).__name__}): {e}"


def deliver(to_email: str, subject: str, body_text: str, html: Optional[str] = None) -> Tuple[bool, str]:
    """Send now, in this thread. -> (ok, detail). Never raises."""
    to_email, subject = _one_line(to_email, 254), _one_line(subject)
    provider = email_provider()
    if provider == "dev":
        print(f"[email:DEV MODE - not actually sent, no RESEND_API_KEY / SMTP_HOST configured] "
              f"To: {to_email} | Subject: {subject}\n{body_text}\n")
        return True, "dev mode: printed to the server log, nothing was sent"
    try:
        ok, detail = (_send_resend if provider == "resend" else _send_smtp)(to_email, subject, body_text, html)
    except Exception as e:                                # defensive: delivery must never take a request down
        ok, detail = False, f"unexpected error ({type(e).__name__})"
    if ok:
        log.info("email sent via %s to %s: %s", provider, to_email, detail or "ok")
    else:
        log.error("email FAILED via %s to %s: %s", provider, to_email, detail)
        print(f"[email] FAILED via {provider} to {to_email}: {detail}")
    return ok, detail


def send_email(to_email: str, subject: str, body_text: str, html: Optional[str] = None) -> bool:
    return deliver(to_email, subject, body_text, html)[0]


def _dispatch(*args) -> None:
    """Dev mode and tests: right here. A real provider: a background thread (the caller does not wait for the network)."""
    if email_provider() == "dev" or _SYNC:
        deliver(*args)
    else:
        threading.Thread(target=deliver, args=args, daemon=True).start()


def _html_mail(heading: str, paragraphs: list, link: str, button: str, footnote: str) -> str:
    e = _html.escape
    body = "".join(f'<p style="margin:0 0 14px;color:#374151;font-size:15px;line-height:1.5">{e(p)}</p>' for p in paragraphs)
    return (f'<!doctype html><html><body style="margin:0;background:#f3f4f6;padding:24px;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">'
            f'<div style="max-width:480px;margin:0 auto;background:#ffffff;border-radius:10px;padding:28px">'
            f'<h1 style="margin:0 0 16px;font-size:20px;color:#111827">{e(heading)}</h1>{body}'
            f'<p style="margin:22px 0"><a href="{e(link, quote=True)}" style="background:#2563eb;color:#ffffff;text-decoration:none;'
            f'padding:11px 20px;border-radius:8px;font-weight:600;display:inline-block">{e(button)}</a></p>'
            f'<p style="margin:0 0 6px;color:#6b7280;font-size:12px">If the button does not work, copy this link into your browser:<br>'
            f'<span style="word-break:break-all">{e(link)}</span></p>'
            f'<p style="margin:14px 0 0;color:#6b7280;font-size:12px">{e(footnote)}</p></div></body></html>')


def _base() -> str:
    return (settings.public_base_url or "http://127.0.0.1:8000").rstrip("/")


def send_verification_email(to_email: str, token: str):
    link = f"{_base()}/?verify_email={token}"
    text = (f"Welcome to Genesis AI!\n\nPlease verify your email by opening this link:\n{link}\n\n"
            f"This link expires in 24 hours. If you didn't create this account, you can ignore this email.")
    html = _html_mail("Welcome to Genesis AI", ["Please verify your email address to finish setting up your account."], link,
                      "Verify my email", "This link expires in 24 hours. If you didn't create this account, you can ignore this email.")
    _dispatch(to_email, "Verify your Genesis AI email", text, html)


def send_password_reset_email(to_email: str, token: str):
    link = f"{_base()}/?reset_password={token}"
    text = (f"We received a request to reset your Genesis AI password.\n\n"
            f"Open this link to choose a new password:\n{link}\n\n"
            f"This link expires in 1 hour. If you didn't request this, you can safely ignore this email - "
            f"your password will not be changed.")
    html = _html_mail("Reset your password", ["We received a request to reset your Genesis AI password."], link, "Choose a new password",
                      "This link expires in 1 hour. If you didn't request this, you can safely ignore this email: your password will not be changed.")
    _dispatch(to_email, "Reset your Genesis AI password", text, html)
