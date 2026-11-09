import asyncio
import pytest
from fastapi.testclient import TestClient
from app.core import crypto_payments as crypto
from app.config import settings
from app.db.database import GenesisDB
from tests.test_crypto_payments import ADDRESS, TXID, proof_for, receipt_for


@pytest.fixture
def invoice(tmp_path):
    db=GenesisDB(str(tmp_path/'auto.sqlite'))
    uid=db.create_user('automatic@example.test','hash')['id']
    return db,uid,db.create_crypto_invoice(uid,'pro',19,ADDRESS)


@pytest.fixture(autouse=True)
def clear_history_cache():
    crypto._DISCOVERY_CACHE.clear()
    yield
    crypto._DISCOVERY_CACHE.clear()


def indexed_transfer(i):
    return {'transaction_id':TXID,'to':ADDRESS,'type':'Transfer','value':str(i['amount_units']),
            'block_timestamp':(i['created_at']+2)*1000,
            'token_info':{'address':crypto.USDT_CONTRACT,'decimals':6}}


def test_discovers_on_later_page_and_verifies_receipt(invoice,monkeypatch):
    db,uid,i=invoice
    calls=[]
    def page(address,params):
        calls.append(dict(params))
        return {'data':[indexed_transfer(i)] if params.get('fingerprint') else [],
                'meta':{} if params.get('fingerprint') else {'fingerprint':'second'}}
    monkeypatch.setattr(crypto,'_transfer_page',page)
    monkeypatch.setattr(crypto,'_rpc',lambda method,payload:receipt_for(i))
    result=crypto.check_invoice_automatically(db,i)
    assert result['status']=='paid' and db.get_user_plan(uid)=='pro'
    assert calls[1]['fingerprint']=='second'
    assert calls[0]['only_confirmed']=='true' and calls[0]['contract_address']==crypto.USDT_CONTRACT
    assert crypto.check_invoice_automatically(db,i)['status']=='paid'
    assert len(calls)==2


@pytest.mark.parametrize('wrong',['token','amount','recipient','type','time'])
def test_index_rows_cannot_authorize_unrelated_payments(invoice,monkeypatch,wrong):
    db,uid,i=invoice; row=indexed_transfer(i)
    if wrong=='token':row['token_info']['address']='fake'
    if wrong=='amount':row['value']='19000000'
    if wrong=='recipient':row['to']='other'
    if wrong=='type':row['type']='Approval'
    if wrong=='time':row['block_timestamp']=(i['created_at']-1)*1000
    monkeypatch.setattr(crypto,'_transfer_page',lambda a,p:{'data':[row],'meta':{}})
    def forbidden(*args):raise AssertionError('invalid index row must not be considered a payment')
    monkeypatch.setattr(crypto,'verify_transfer',forbidden)
    assert crypto.check_invoice_automatically(db,i)['status']=='pending'
    assert db.get_user_plan(uid)=='free'


def test_index_confirmation_alone_is_never_sufficient(invoice,monkeypatch):
    db,uid,i=invoice
    monkeypatch.setattr(crypto,'_transfer_page',lambda a,p:{'data':[indexed_transfer(i)],'meta':{}})
    monkeypatch.setattr(crypto,'verify_transfer',lambda i,t:None)
    assert crypto.check_invoice_automatically(db,i)['status']=='pending'
    assert db.get_user_plan(uid)=='free'


def test_broken_pagination_is_not_cached_as_empty_history(invoice,monkeypatch):
    db,uid,i=invoice
    monkeypatch.setattr(crypto,'_transfer_page',lambda a,p:{'data':[],'meta':{'fingerprint':'loop'}})
    with pytest.raises(RuntimeError,match='too busy'):
        crypto.check_invoice_automatically(db,i)
    assert crypto._DISCOVERY_CACHE=={} and db.get_user_plan(uid)=='free'


def test_background_monitor_credits_without_customer_check(invoice,monkeypatch):
    db,uid,i=invoice
    monkeypatch.setattr(crypto,'crypto_ready',lambda:True)
    monkeypatch.setattr(crypto,'discover_transfer',lambda inv:proof_for(inv))
    crypto.scan_pending_payments(db)
    assert db.get_user_plan(uid)=='pro'
    until=db.get_crypto_access(uid)['access_until']
    crypto.scan_pending_payments(db)
    assert db.get_crypto_access(uid)['access_until']==until


def test_provider_failure_never_grants_access(invoice,monkeypatch):
    db,uid,i=invoice
    monkeypatch.setattr(crypto,'crypto_ready',lambda:True)
    def unavailable(inv):raise RuntimeError('offline')
    monkeypatch.setattr(crypto,'discover_transfer',unavailable)
    crypto.scan_pending_payments(db)
    assert db.get_user_plan(uid)=='free'
    assert db.get_crypto_invoice(i['id'],uid)['paid_at'] is None


def test_qr_and_automatic_check_are_owner_only(tmp_path,monkeypatch):
    from app.main import app
    from app.core.auth import create_token
    monkeypatch.chdir(tmp_path)
    db=GenesisDB();uid=db.create_user('automatic-route@example.test','hash')['id']
    i=db.create_crypto_invoice(uid,'pro',19,ADDRESS)
    client=TestClient(app);headers={'Authorization':'Bearer '+create_token(uid)}
    path='/api/crypto/invoices/'+i['id']
    assert client.get(path+'/qr').status_code==401
    image=client.get(path+'/qr',headers=headers)
    assert image.status_code==200 and image.headers['content-type'].startswith('image/svg+xml')
    assert '<svg' in image.text and '<script' not in image.text
    other=db.create_user('other-auto@example.test','hash')['id']
    other_headers={'Authorization':'Bearer '+create_token(other)}
    assert client.get(path+'/qr',headers=other_headers).status_code==404
    assert client.post(path+'/check',headers=other_headers).status_code==404
    monkeypatch.setattr(crypto,'discover_transfer',lambda inv:proof_for(inv))
    result=client.post(path+'/check',headers=headers)
    assert result.status_code==200 and result.json()['status']=='paid'
    assert client.post(path+'/check',headers=headers).json()['access']['access_until']==result.json()['access']['access_until']


def test_monitor_starts_once_and_can_be_cancelled(monkeypatch):
    from app.core import crypto_monitor as monitor
    monkeypatch.setattr(settings,'crypto_payments_enabled',True)
    async def waiting():await asyncio.Event().wait()
    monkeypatch.setattr(monitor,'_monitor_loop',waiting)
    async def check():
        first=monitor.start_crypto_monitor()
        assert monitor.start_crypto_monitor() is first
        monitor.stop_crypto_monitor()
        with pytest.raises(asyncio.CancelledError):await first
        assert monitor._task is None
    asyncio.run(check())
