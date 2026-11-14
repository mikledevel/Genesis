import copy
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.core import crypto_payments as crypto
from app.db.database import GenesisDB

ADDRESS = 'TLjsykFxDXnM4DH1cBT8nYo6MzY81MYPmT'
TXID = 'a'*64


@pytest.fixture
def billing(tmp_path):
    db = GenesisDB(str(tmp_path/'crypto.sqlite'))
    uid = db.create_user('buyer@example.test','hash','Buyer')['id']
    db.mark_email_verified(uid)
    invoice = db.create_crypto_invoice(uid,'pro',19,ADDRESS)
    return db,uid,invoice


def receipt_for(invoice):
    return {'id':TXID,'receipt':{'result':'SUCCESS'},'blockNumber':87800000,
            'blockTimeStamp':(invoice['created_at']+2)*1000,
            'log':[{'address':crypto.tron_address_hex(crypto.USDT_CONTRACT),
                    'topics':[crypto.TRANSFER_TOPIC,'0'*24+'1'*40,'0'*24+crypto.tron_address_hex(ADDRESS)],
                    'data':f"{invoice['amount_units']:064x}"}]}


def proof_for(invoice):
    return {'txid':TXID,'recipient':ADDRESS,'amount_units':invoice['amount_units'],
            'timestamp_ms':(invoice['created_at']+2)*1000,'block_number':87800000}


def test_known_address_checksum_and_invalid_address():
    assert len(crypto.tron_address_hex(ADDRESS)) == 40
    with pytest.raises(ValueError):
        crypto.tron_address_hex(ADDRESS[:-1]+'1')


def test_invoice_amount_is_unique_and_cannot_be_reused_after_expiry(billing):
    db,uid,invoice = billing
    assert invoice['amount'] == '18.999999'
    assert db.create_crypto_invoice(uid,'pro',19,ADDRESS)['id'] == invoice['id']
    other = db.create_user('other@example.test','hash','Other')['id']
    second = db.create_crypto_invoice(other,'pro',19,ADDRESS)
    assert second['amount_units'] != invoice['amount_units']
    with db._connect() as c:
        c.execute('UPDATE crypto_invoices SET expires_at=? WHERE id=?',(int(time.time())-1,invoice['id']))
    assert db.create_crypto_invoice(uid,'pro',19,ADDRESS)['amount_units'] not in (invoice['amount_units'],second['amount_units'])


def test_other_plan_cannot_be_bought_while_invoice_pending(billing):
    db,uid,_ = billing
    with pytest.raises(ValueError,match='unpaid invoice'):
        db.create_crypto_invoice(uid,'business',79,ADDRESS)


def test_confirmed_official_token_transfer_is_accepted(billing,monkeypatch):
    _,_,invoice = billing
    monkeypatch.setattr(crypto,'_rpc',lambda method,payload:receipt_for(invoice))
    assert crypto.verify_transfer(invoice,TXID) == proof_for(invoice)


@pytest.mark.parametrize('bad',['token','recipient','amount','timestamp_before','timestamp_after','failed','txid','topic','missing_height','trx','malformed_amount'])
def test_wrong_or_unrelated_transfers_never_activate_access(billing,monkeypatch,bad):
    db,uid,invoice = billing
    r = receipt_for(invoice)
    if bad == 'token': r['log'][0]['address']='f'*40
    if bad == 'recipient': r['log'][0]['topics'][2]='0'*64
    if bad == 'amount': r['log'][0]['data']=f"{19_000_000:064x}"
    if bad == 'timestamp_before': r['blockTimeStamp']=(invoice['created_at']-1)*1000
    if bad == 'timestamp_after': r['blockTimeStamp']=(invoice['expires_at']+1)*1000
    if bad == 'failed': r['receipt']['result']='OUT_OF_ENERGY'
    if bad == 'txid': r['id']='b'*64
    if bad == 'topic': r['log'][0]['topics'][0]='f'*64
    if bad == 'missing_height': del r['blockNumber']
    if bad == 'trx': r['log']=[]
    if bad == 'malformed_amount': r['log'][0]['data']='not-hex'
    monkeypatch.setattr(crypto,'_rpc',lambda method,payload:r)
    with pytest.raises((ValueError,RuntimeError)):
        crypto.verify_transfer(invoice,TXID)
    assert db.get_user_plan(uid) == 'free'


def test_unsolidified_receipt_stays_pending(billing,monkeypatch):
    db,uid,invoice = billing
    monkeypatch.setattr(crypto,'_rpc',lambda method,payload:{})
    assert crypto.verify_transfer(invoice,TXID) is None
    assert db.get_user_plan(uid) == 'free'


def test_grant_is_atomic_idempotent_renews_and_expires(billing,monkeypatch):
    db,uid,invoice = billing
    with ThreadPoolExecutor(max_workers=4) as workers:
        results=list(workers.map(lambda _:db.apply_crypto_payment(invoice['id'],uid,proof_for(invoice)),range(8)))
    assert sum(not x['already_processed'] for x in results) == 1
    first_until=db.get_crypto_access(uid)['access_until']
    assert db.get_user_plan(uid)=='pro'
    assert db.get_balance(uid)==0
    renewal=db.create_crypto_invoice(uid,'pro',19,ADDRESS)
    proof=proof_for(renewal);proof['txid']='b'*64
    db.apply_crypto_payment(renewal['id'],uid,proof)
    assert db.get_crypto_access(uid)['access_until']==first_until+30*86400
    with db._connect() as c:
        c.execute('UPDATE users SET crypto_access_until=? WHERE id=?',(int(time.time())-1,uid))
    assert db.get_user_plan(uid)=='free'


def test_invoice_ownership_and_txid_replay(billing):
    db,uid,invoice=billing
    other=db.create_user('intruder@example.test','hash')['id']
    assert db.get_crypto_invoice(invoice['id'],other) is None
    with pytest.raises(ValueError):
        db.apply_crypto_payment(invoice['id'],other,proof_for(invoice))
    db.apply_crypto_payment(invoice['id'],uid,proof_for(invoice))
    second=db.create_crypto_invoice(other,'pro',19,ADDRESS)
    with pytest.raises(db._integrity_errors):
        db.apply_crypto_payment(second['id'],other,proof_for(second))
    assert db.get_user_plan(other)=='free'


def test_api_requires_verified_email_and_owns_invoice(tmp_path,monkeypatch):
    from app.main import app
    from app.routers import crypto as routes
    from app.core.auth import create_token
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings,'debug',True)
    monkeypatch.setattr(settings,'crypto_payments_enabled',True)
    monkeypatch.setattr(settings,'crypto_usdt_tron_address',ADDRESS)
    monkeypatch.setattr(routes,'check_node',lambda:87800000)
    db=GenesisDB();uid=db.create_user('route@example.test','hash')['id']
    client=TestClient(app);headers={'Authorization':'Bearer '+create_token(uid)}
    assert client.post('/api/crypto/invoices',json={'plan_id':'pro'}).status_code==401
    assert client.post('/api/crypto/invoices',json={'plan_id':'pro'},headers=headers).status_code==403
    db.mark_email_verified(uid)
    r=client.post('/api/crypto/invoices',json={'plan_id':'pro'},headers=headers)
    assert r.status_code==200,r.text
    invoice=r.json()['invoice']
    other=db.create_user('other-route@example.test','hash')['id']
    assert client.post('/api/crypto/invoices/'+invoice['id']+'/confirm',json={'txid':TXID},headers={'Authorization':'Bearer '+create_token(other)}).status_code==404
    monkeypatch.setattr(routes,'verify_transfer',lambda i,t:proof_for(i))
    paid=client.post('/api/crypto/invoices/'+invoice['id']+'/confirm',json={'txid':TXID},headers=headers)
    assert paid.status_code==200,paid.text
    assert paid.json()['status']=='paid'
    assert client.get('/api/me/plan',headers=headers).json()['plan']['id']=='pro'
