"""Authenticated invoice creation and on-chain confirmation."""
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from app.core.auth import get_current_user_id
from app.core.crypto_payments import crypto_ready, check_node, verify_transfer, check_invoice_automatically
from app.config import settings
from app.db.database import GenesisDB
from app.core import plans

router = APIRouter(prefix='/api/crypto', tags=['crypto payments'])


class InvoiceRequest(BaseModel):
    plan_id: Literal['pro','business']


class ConfirmationRequest(BaseModel):
    txid: str = Field(pattern=r'^[a-fA-F0-9]{64}$')


def require_crypto():
    if not crypto_ready():
        raise HTTPException(503, 'Crypto payments are not configured yet')


@router.get('/invoices')
def invoices(user_id: str = Depends(get_current_user_id)):
    return {'invoices':GenesisDB().list_crypto_invoices(user_id)}


@router.post('/invoices')
def create_invoice(body: InvoiceRequest, user_id: str = Depends(get_current_user_id)):
    require_crypto()
    db = GenesisDB()
    if not db.is_email_verified(user_id):
        raise HTTPException(403, 'Verify your email before creating a payment invoice')
    try:
        check_node()  # Do not invite a payment while the verification node is unreachable.
        invoice = db.create_crypto_invoice(user_id,body.plan_id,
                    plans.get_plan(body.plan_id)['price_usd_month'], settings.crypto_usdt_tron_address)
        return {'invoice':invoice}
    except ValueError as e:
        raise HTTPException(409,str(e)) from None
    except RuntimeError as e:
        raise HTTPException(503,str(e)) from None


@router.post('/invoices/{invoice_id}/confirm')
def confirm_invoice(invoice_id: str, body: ConfirmationRequest, user_id: str = Depends(get_current_user_id)):
    # Existing invoices remain claimable even when creation is disabled or the receiving wallet rotates.
    db = GenesisDB()
    invoice = db.get_crypto_invoice(invoice_id,user_id)
    if not invoice:
        raise HTTPException(404,'Invoice was not found')
    if invoice['paid_at']:
        return {'status':'paid','already_processed':True,'access':db.get_crypto_access(user_id)}
    if not db.begin_crypto_check(invoice_id,user_id,body.txid.lower()):
        raise HTTPException(429,'Wait ten seconds before checking this invoice again')
    try:
        proof = verify_transfer(invoice,body.txid)
        if proof is None:
            return {'status':'confirming','message':'Waiting for blockchain confirmation. Retry with the same transaction ID; do not pay again.'}
        return {**db.apply_crypto_payment(invoice_id,user_id,proof),'access':db.get_crypto_access(user_id)}
    except ValueError as e:
        raise HTTPException(409,str(e)) from None
    except RuntimeError as e:
        raise HTTPException(503,str(e)) from None
    except db._integrity_errors:
        raise HTTPException(409,'This transaction has already been assigned to another invoice') from None


@router.get('/invoices/{invoice_id}/qr')
def invoice_qr(invoice_id: str, user_id: str = Depends(get_current_user_id)):
    invoice = GenesisDB().get_crypto_invoice(invoice_id,user_id)
    if not invoice:
        raise HTTPException(404,'Invoice was not found')
    import qrcode
    from qrcode.image.svg import SvgPathImage
    image = qrcode.make(invoice['recipient'],image_factory=SvgPathImage,box_size=6,border=4)
    return Response(image.to_string(),media_type='image/svg+xml',headers={'Cache-Control':'no-store'})


@router.post('/invoices/{invoice_id}/check')
def automatic_invoice_check(invoice_id: str, user_id: str = Depends(get_current_user_id)):
    db = GenesisDB()
    invoice = db.get_crypto_invoice(invoice_id,user_id)
    if not invoice:
        raise HTTPException(404,'Invoice was not found')
    try:
        return check_invoice_automatically(db,invoice)
    except RuntimeError as error:
        raise HTTPException(503,str(error)) from None
    except ValueError as error:
        raise HTTPException(409,str(error)) from None
    except db._integrity_errors:
        raise HTTPException(409,'Payment needs review; contact support with your invoice ID') from None
