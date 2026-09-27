"""
Outbound email for password reset and email verification.

Follows the same "optional external dependency, degrade loudly not silently" pattern as
Groq/Stripe elsewhere in this codebase: if SMTP isn't configured (settings.smtp_host is
empty), we don't pretend to send an email - we print the content (including the actual
link) to the server log instead, so local development and testing still work end-to-end
without a real mail server. In production, set SMTP_HOST/SMTP_USER/SMTP_PASSWORD in .env and
real emails will actually be delivered.

This module deliberately never raises on a delivery failure - a broken mail server should
not break registration or password reset for the user in front of the request; it should
just be logged so an operator can notice and fix it. The caller decides what to tell the
user (see main.py: registration/reset-request endpoints always return success either way,
so as not to leak which emails are registered - see note on user enumeration below).
"""
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import settings


def send_email(to_email: str, subject: str, body_text: str) -> bool:
    if not settings.smtp_host:
        print(f"[email:DEV MODE - not actually sent, SMTP_HOST not configured] "
              f"To: {to_email} | Subject: {subject}\n{body_text}\n")
        return True
    try:
        msg = MIMEMultipart()
        msg["From"] = settings.smtp_from
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain"))
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls(context=context)
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_from, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[email] FAILED to send to {to_email}: {e}")
        return False


def send_verification_email(to_email: str, token: str):
    link = f"{settings.public_base_url or 'http://127.0.0.1:8000'}/?verify_email={token}"
    send_email(to_email, "Verify your Genesis AI email",
               f"Welcome to Genesis AI!\n\nPlease verify your email by opening this link:\n{link}\n\n"
               f"This link expires in 24 hours. If you didn't create this account, you can ignore this email.")


def send_password_reset_email(to_email: str, token: str):
    link = f"{settings.public_base_url or 'http://127.0.0.1:8000'}/?reset_password={token}"
    send_email(to_email, "Reset your Genesis AI password",
               f"We received a request to reset your Genesis AI password.\n\n"
               f"Open this link to choose a new password:\n{link}\n\n"
               f"This link expires in 1 hour. If you didn't request this, you can safely ignore this email - "
               f"your password will not be changed.")
