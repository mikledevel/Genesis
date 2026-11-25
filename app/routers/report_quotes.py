from typing import Literal
from urllib.parse import urlsplit
from fastapi import APIRouter, Depends, HTTPException, Header, Request
from pydantic import BaseModel, ConfigDict, Field
from app.config import settings
from app.core.auth import get_current_user_id, require_admin
from app.core.report_quotes import ReportQuotes, payments_ready
from app.db.database import GenesisDB

router = APIRouter()


class QuoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1,max_length=200)
    title: str = Field(min_length=1,max_length=200)
    amount_cents: int = Field(ge=100,le=1_000_000,strict=True)
    currency: Literal["usd","eur","gbp"] = "usd"


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FulfillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(min_length=1,max_length=200)


def _error(error):
    if isinstance(error,KeyError):
        return HTTPException(status_code=404,detail="Quote or report not found")
    if isinstance(error,ValueError):
        return HTTPException(status_code=409,detail=str(error))
    if isinstance(error,RuntimeError):
        return HTTPException(status_code=503,detail="Card payments are not enabled or configured")
    return HTTPException(status_code=502,detail="The payment provider could not complete this request. Retry without creating a new quote.")


@router.get("/api/report-quotes")
def my_quotes(user_id: str = Depends(get_current_user_id)):
    return {"quotes":ReportQuotes(GenesisDB(),user_id).list(),"payments_enabled":payments_ready()}


@router.post("/api/report-quotes/{quote_id}/checkout")
def checkout(quote_id: str, body: EmptyRequest, request: Request, user_id: str = Depends(get_current_user_id)):
    base = (settings.public_base_url or "").rstrip("/")
    if settings.debug and not base:
        base = str(request.base_url).rstrip("/")
    parsed = urlsplit(base)
    if not parsed.hostname or parsed.username or parsed.password or (parsed.scheme != "https" and not settings.debug):
        raise HTTPException(status_code=503,detail="Configure the public HTTPS URL before accepting payments")
    try:
        return ReportQuotes(GenesisDB(),user_id).checkout(quote_id,base)
    except Exception as error:
        raise _error(error) from None


@router.get("/api/report-quotes/{quote_id}/report")
def quote_report(quote_id: str, lang: Literal["en","ru"] = "en", format: Literal["html","csv"] = "html",
                 user_id: str = Depends(get_current_user_id)):
    try:
        quote = ReportQuotes(GenesisDB(),user_id).get(quote_id)
    except KeyError:
        raise HTTPException(status_code=404,detail="Quote not found") from None
    if quote["status"] != "ready" or not quote.get("report_snapshot"):
        raise HTTPException(status_code=409,detail="The report has not been delivered or this payment was refunded")
    from app.api.forecast_download import forecast_download
    return forecast_download(quote["report_snapshot"],lang,format)


@router.post("/api/admin/report-quotes",status_code=201)
def issue_quote(body: QuoteRequest, admin_id: str = Depends(require_admin), idempotency_key: str | None = Header(None,alias="Idempotency-Key")):
    db = GenesisDB()
    state = db.get_session_state(body.user_id)
    if not state or state[1]:
        raise HTTPException(status_code=404,detail="Customer not found or inactive")
    from app.core.usage import operation_key
    try:
        q = ReportQuotes(db,body.user_id).issue(admin_id,body.title,body.amount_cents,body.currency,
                operation_key(body.user_id,"report-quote",idempotency_key))
    except ValueError as error:
        raise _error(error) from None
    return {"quote_id":q["id"],"status":q["status"]}


@router.get("/api/admin/report-quotes/{user_id}")
def customer_quotes(user_id: str, admin_id: str = Depends(require_admin)):
    db = GenesisDB()
    if not db.get_user_by_id(user_id):
        raise HTTPException(status_code=404,detail="Customer not found")
    return {"quotes":ReportQuotes(db,user_id).list()}


@router.post("/api/admin/report-quotes/{user_id}/{quote_id}/fulfill")
def deliver(user_id: str, quote_id: str, body: FulfillRequest, admin_id: str = Depends(require_admin)):
    db = GenesisDB()
    job = db.get_job(admin_id,body.job_id)
    if not job:
        raise HTTPException(status_code=404,detail="Your report job was not found")
    snapshot = ((job.get("result") or {}).get("data") or {}).get("report_snapshot")
    if job["kind"] != "forecast" or job["status"] != "succeeded" or not snapshot:
        raise HTTPException(status_code=409,detail="Choose your completed forecast with a saved report")
    try:
        ReportQuotes(db,user_id).fulfill(quote_id,admin_id,body.job_id,snapshot)
    except (KeyError,ValueError) as error:
        raise _error(error) from None
    return {"status":"ready"}


@router.post("/api/admin/report-quotes/{user_id}/{quote_id}/refund")
def refund(user_id: str, quote_id: str, body: EmptyRequest, admin_id: str = Depends(require_admin)):
    try:
        return ReportQuotes(GenesisDB(),user_id).refund(quote_id,admin_id)
    except Exception as error:
        raise _error(error) from None
