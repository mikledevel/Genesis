"""Разовые отчёты: согласованная цена, подтверждённая оплата и отдельная выдача."""
import copy
import secrets
import string
import time
import uuid
from urllib.parse import urlsplit

from app.config import settings
from app.db.document_store import DatabaseDocument


def payments_ready():
    import os
    return bool(settings.stripe_report_payments_enabled and settings.stripe_secret_key
                and settings.stripe_webhook_secret and (settings.database_url or settings.debug or not os.getenv("RENDER_EXTERNAL_URL"))
                and (settings.debug or settings.public_base_url.startswith("https://")))


def stripe_client(require_enabled=False):
    if require_enabled and not payments_ready():
        raise RuntimeError("Card payments for reports are not enabled")
    if not settings.stripe_secret_key:
        raise RuntimeError("Stripe is not configured")
    from stripe import StripeClient
    return StripeClient(settings.stripe_secret_key, stripe_version="2026-11-18.endive", max_network_retries=2)


def _value(obj, name, default=None):
    from app.core.payments import _value as read
    return read(obj, name, default)


def _audit(quote, action, actor):
    quote["audit"].append({"action": action, "actor": actor, "at": int(time.time())})


class ReportQuotes:
    def __init__(self, db, user_id):
        self.db, self.user_id = db, user_id
        self.document = DatabaseDocument("report-quotes:" + user_id, {"quotes": {}}, db)

    def get(self, quote_id):
        quote = self.document.read()["quotes"].get(quote_id)
        if not quote:
            raise KeyError("Quote not found")
        return copy.deepcopy(quote)

    def list(self):
        fields = ("id", "title", "amount_cents", "currency", "status", "created_at", "expires_at")
        return [{**{name: q[name] for name in fields}, "report_available":q["status"] == "ready"}
                for q in sorted(self.document.read()["quotes"].values(), key=lambda q:q["created_at"], reverse=True)]

    def issue(self, actor, title, amount_cents, currency, quote_id=None):
        if isinstance(amount_cents, bool) or not isinstance(amount_cents, int) or not 100 <= amount_cents <= 1_000_000:
            raise ValueError("Quote amount must be between 100 and 1000000 minor units")
        if currency not in ("usd", "eur", "gbp") or not isinstance(title, str) or not 1 <= len(title.strip()) <= 200:
            raise ValueError("Invalid quote title or currency")
        quote_id = quote_id or uuid.uuid4().hex
        with self.document.transaction() as state:
            quotes = state["quotes"]
            if quote_id in quotes:
                q = quotes[quote_id]
                if (q["title"],q["amount_cents"],q["currency"]) != (title.strip(),amount_cents,currency):
                    raise ValueError("Idempotency key belongs to a different quote")
                return copy.deepcopy(q)
            if len(quotes) >= 50:
                raise ValueError("The pilot supports at most 50 quotes per customer")
            now = int(time.time())
            q = {"id":quote_id, "title":title.strip(), "amount_cents":amount_cents, "currency":currency,
                 "status":"quoted", "created_at":now, "expires_at":now+7*86400, "audit":[]}
            _audit(q,"issued",actor); quotes[quote_id] = q
            return copy.deepcopy(q)

    def checkout(self, quote_id, base_url):
        client = stripe_client(require_enabled=True)
        if not self.db.is_email_verified(self.user_id):
            raise ValueError("Verify your email before payment")
        with self.document.transaction() as state:
            q = state["quotes"].get(quote_id)
            if not q:
                raise KeyError("Quote not found")
            now = int(time.time())
            if q["status"] != "quoted" or q["expires_at"] <= now:
                raise ValueError("This quote is paid, expired or unavailable")
            if not q.get("checkout_attempt"):
                q.update(checkout_attempt=uuid.uuid4().hex, checkout_deadline=now+3600, checkout_base=base_url,
                         integration="genesis_report_"+"".join(secrets.choice(string.ascii_lowercase) for _ in range(8)))
            if q["checkout_deadline"] <= now:
                raise ValueError("Checkout expired. Ask the organiser to review the quote before retrying.")
            quote = copy.deepcopy(q)
        metadata = {"purchase_type":"report", "genesis_user_id":self.user_id,
                    "quote_id":quote_id, "attempt_id":quote["checkout_attempt"]}
        params = {"mode":"payment", "client_reference_id":self.user_id, "metadata":metadata,
                  "payment_intent_data":{"metadata":metadata}, "integration_identifier":quote["integration"],
                  "line_items":[{"price_data":{"currency":quote["currency"], "unit_amount":quote["amount_cents"],
                                               "product_data":{"name":quote["title"]}},"quantity":1}],
                  "success_url":quote["checkout_base"]+"/?report_payment=returned",
                  "cancel_url":quote["checkout_base"]+"/?report_payment=cancelled",
                  "expires_at":quote["checkout_deadline"], "adaptive_pricing":{"enabled":False},
                  "automatic_tax":{"enabled":False}, "allow_promotion_codes":False}
        session = client.v1.checkout.sessions.create(params=params, options={"idempotency_key":"report-checkout-"+quote["checkout_attempt"]})
        url, session_id = _value(session,"url"), _value(session,"id")
        if not url or urlsplit(url).scheme != "https" or urlsplit(url).hostname != "checkout.stripe.com" or not session_id:
            raise ValueError("Payment confirmation is pending or the checkout URL is unavailable")
        with self.document.transaction() as state:
            q = state["quotes"][quote_id]
            if q.get("session_id") not in (None,session_id):
                raise ValueError("A different checkout already exists")
            q["session_id"] = session_id
        return {"id":session_id,"url":url}

    def mark_paid(self, quote_id, session):
        from app.core.payments import _stripe_id
        with self.document.transaction() as state:
            q = state["quotes"].get(quote_id)
            if not q:
                return {"status":"quote_not_found"}
            metadata = _value(session,"metadata",{})
            expected = {"purchase_type":"report", "genesis_user_id":self.user_id, "quote_id":quote_id,
                        "attempt_id":q.get("checkout_attempt")}
            if (_value(session,"mode") != "payment" or _value(session,"payment_status") != "paid"
                    or _value(session,"amount_total") != q["amount_cents"] or _value(session,"currency") != q["currency"]
                    or _value(session,"client_reference_id") != self.user_id
                    or not q.get("checkout_attempt") or any(_value(metadata,k) != v for k,v in expected.items())
                    or q.get("session_id") not in (None,_value(session,"id"))
                    or not _stripe_id(_value(session,"payment_intent"))):
                return {"status":"payment_not_confirmed"}
            if q["status"] != "quoted":
                return {"status":"already_processed"}
            charge = _value(_value(session,"payment_intent",{}),"latest_charge",{})
            refunded = bool(_value(charge,"refunded",False) and _value(charge,"amount_refunded") == q["amount_cents"])
            status = "refunded" if refunded else "paid"
            if refunded:
                q["refund_previous_status"] = "paid"
            q.update(status=status, session_id=_value(session,"id"), payment_intent=_stripe_id(_value(session,"payment_intent")), paid_at=int(time.time()))
            _audit(q,status,"stripe")
            return {"status":status,"quote_id":quote_id}

    def fulfill(self, quote_id, actor, job_id, snapshot):
        from app.core.forecast_reports import Snapshot
        snapshot = Snapshot.model_validate(snapshot).model_dump(mode="json")
        with self.document.transaction() as state:
            q = state["quotes"].get(quote_id)
            if not q:
                raise KeyError("Quote not found")
            if q["status"] == "ready" and q.get("source_job") == job_id:
                return
            if q["status"] != "paid":
                raise ValueError("Only a paid report can be delivered")
            q.update(status="ready", source_job=job_id, report_snapshot=snapshot, delivered_at=int(time.time()))
            _audit(q,"delivered",actor)

    def apply_refund(self, quote_id, refund, actor="stripe"):
        from app.core.payments import _stripe_id
        with self.document.transaction() as state:
            q = state["quotes"].get(quote_id)
            if not q:
                raise KeyError("Quote not found")
            if not q.get("payment_intent"):
                raise RuntimeError("Payment binding is not saved yet; retry this webhook")
            if (_stripe_id(_value(refund,"payment_intent")) != q["payment_intent"]
                    or _value(refund,"amount") != q["amount_cents"] or _value(refund,"currency") != q["currency"]):
                return {"status":"refund_not_confirmed"}
            refund_id, status = _value(refund,"id"), _value(refund,"status")
            if refund_id in q.get("retired_refunds",[]):
                return {"status":"old_refund_ignored"}
            if q.get("refund_id") not in (None, refund_id) and q["status"] in ("refunded","refund_pending"):
                return {"status":"different_refund_pending"}
            if status in ("succeeded","pending","requires_action"):
                if q["status"] not in ("refunded","refund_pending"):
                    q["refund_previous_status"] = q["status"]
                q.update(status="refunded" if status == "succeeded" else "refund_pending", refund_id=refund_id)
                _audit(q,q["status"],actor)
            elif status in ("failed","canceled"):
                if q.get("refund_id") in (None,refund_id):
                    if q["status"] in ("refund_pending","refunded"):
                        q["status"] = q.get("refund_previous_status","paid")
                    q.update(refund_id=refund_id, refund_attempt_failed=True)
                    _audit(q,"refund_failed",actor)
            return {"status":q["status"]}

    def refund(self, quote_id, actor):
        client = stripe_client()
        with self.document.transaction() as state:
            q = state["quotes"].get(quote_id)
            if not q:
                raise KeyError("Quote not found")
            if q["status"] == "refunded":
                return {"status":"refunded"}
            if q["status"] not in ("paid","ready","refund_pending"):
                raise ValueError("This quote has no confirmed refundable payment")
            if not q.get("refund_attempt") or q.get("refund_attempt_failed"):
                if q.get("refund_id"):
                    q.setdefault("retired_refunds",[]).append(q["refund_id"])
                q.update(refund_attempt=uuid.uuid4().hex, refund_attempt_failed=False, refund_id=None)
            quote = copy.deepcopy(q)
        refund = client.v1.refunds.create(params={"payment_intent":quote["payment_intent"],
            "metadata":{"purchase_type":"report","genesis_user_id":self.user_id,"quote_id":quote_id}},
            options={"idempotency_key":"report-refund-"+quote["refund_attempt"]})
        return self.apply_refund(quote_id, refund, actor)


def handle_report_event(event, db):
    event_type = _value(event,"type","")
    obj = _value(_value(event,"data",{}),"object",{})
    metadata = _value(obj,"metadata",{})
    if _value(metadata,"purchase_type") != "report":
        if event_type not in ("refund.created","refund.updated","refund.failed") or not _value(obj,"payment_intent"):
            return None
        from app.core.payments import _stripe_id
        intent = stripe_client().v1.payment_intents.retrieve(_stripe_id(_value(obj,"payment_intent")))
        metadata = _value(intent,"metadata",{})
        if _value(metadata,"purchase_type") != "report":
            return None
    user_id, quote_id = _value(metadata,"genesis_user_id"), _value(metadata,"quote_id")
    if not isinstance(user_id,str) or not isinstance(quote_id,str):
        return {"status":"invalid_report_reference"}
    store = ReportQuotes(db,user_id)
    client = stripe_client()
    if event_type in ("checkout.session.completed","checkout.session.async_payment_succeeded"):
        session = client.v1.checkout.sessions.retrieve(_value(obj,"id"), params={"expand":["line_items","payment_intent.latest_charge"]})
        return store.mark_paid(quote_id,session)
    if event_type == "charge.refunded":
        q = store.get(quote_id)
        if not q.get("payment_intent"):
            raise RuntimeError("Payment binding is not saved yet; retry this webhook")
        if _value(obj,"payment_intent") != q["payment_intent"]:
            return {"status":"refund_not_confirmed"}
        intent = client.v1.payment_intents.retrieve(q["payment_intent"], params={"expand":["latest_charge"]})
        charge = _value(intent,"latest_charge",{})
        if not _value(charge,"refunded",False):
            return {"status":"refund_not_confirmed"}
        refunds = client.v1.refunds.list(params={"payment_intent":q["payment_intent"],"limit":100})
        result = {"status":"refund_not_confirmed"}
        for refund in _value(refunds,"data",[]):
            result = store.apply_refund(quote_id,refund)
        return result
    if event_type in ("refund.created","refund.updated","refund.failed"):
        refund = client.v1.refunds.retrieve(_value(obj,"id"))
        return store.apply_refund(quote_id,refund)
    return {"status":"report_event_ignored"}
