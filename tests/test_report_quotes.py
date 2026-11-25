from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import pytest
import app.main
from app.db.database import GenesisDB


@pytest.fixture
def quotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.core.report_quotes import ReportQuotes
    db = GenesisDB()
    buyer = db.create_user("buyer@example.test", "unused")["id"]
    db.mark_email_verified(buyer)
    store = ReportQuotes(db, buyer)
    quote = store.issue("operator", "Forecast report and review", 9900, "usd", "one")
    return db, buyer, store, quote


def test_checkout_has_server_price_and_stable_retry_parameters(quotes):
    _, _, store, quote = quotes
    client = Mock()
    client.v1.checkout.sessions.create.return_value = {"id": "cs_quote", "url": "https://checkout.stripe.com/c/pay/test"}
    with patch("app.core.report_quotes.stripe_client", return_value=client):
        one = store.checkout(quote["id"], "https://example.test")
        two = store.checkout(quote["id"], "https://example.test")
    assert one == two
    calls = client.v1.checkout.sessions.create.call_args_list
    assert calls[0] == calls[1]
    params = calls[0].kwargs["params"]
    assert params["mode"] == "payment" and params["line_items"][0]["price_data"]["unit_amount"] == 9900
    assert "payment_method_types" not in params
    assert "idempotency_key" in calls[0].kwargs["options"]


def paid_session(store, quote):
    q = store.get(quote["id"])
    return {"id":"cs_quote", "mode":"payment", "payment_status":"paid", "amount_total":9900,
            "currency":"usd", "client_reference_id":store.user_id, "payment_intent":"pi_quote",
            "metadata":{"purchase_type":"report", "genesis_user_id":store.user_id,
                        "quote_id":quote["id"], "attempt_id":q["checkout_attempt"]}}


def test_webhooks_are_atomic_and_do_not_grant_pro_or_wallet_credit(quotes):
    from app.core.report_quotes import handle_report_event
    db, buyer, store, quote = quotes
    client = Mock(); client.v1.checkout.sessions.create.return_value = {"id":"cs_quote", "url":"https://checkout.stripe.com/c/pay/test"}
    with patch("app.core.report_quotes.stripe_client", return_value=client):
        store.checkout(quote["id"], "https://example.test")
        session = paid_session(store, quote)
        client.v1.checkout.sessions.retrieve.return_value = session
        event = {"id":"evt_paid", "type":"checkout.session.completed", "data":{"object":session}}
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _:handle_report_event(event, db), range(8)))
    assert store.get(quote["id"])["status"] == "paid"
    assert sum(r["status"] == "paid" for r in results) == 1
    assert db.get_user_plan(buyer) == "free" and db.get_balance(buyer) == 0


def test_unpaid_and_wrong_amount_cannot_unlock_report(quotes):
    from app.core.report_quotes import handle_report_event
    db, _, store, quote = quotes
    client = Mock(); client.v1.checkout.sessions.create.return_value = {"id":"cs_quote", "url":"https://checkout.stripe.com/c/pay/test"}
    with patch("app.core.report_quotes.stripe_client", return_value=client):
        store.checkout(quote["id"], "https://example.test")
        session = paid_session(store, quote)
        event = {"id":"evt_bad", "type":"checkout.session.completed", "data":{"object":session}}
        for changes in ({"payment_status":"unpaid"}, {"amount_total":1}, {"currency":"eur"}):
            client.v1.checkout.sessions.retrieve.return_value = {**session, **changes}
            assert handle_report_event(event, db)["status"] != "paid"
    assert store.get(quote["id"])["status"] == "quoted"


def test_report_delivery_survives_job_removal_and_refund_blocks_download(quotes):
    from app.core.report_quotes import handle_report_event
    from app.core.forecast_reports import create_snapshot
    from tests.test_forecast_exports import example_result
    db, _, store, quote = quotes
    client = Mock(); client.v1.checkout.sessions.create.return_value = {"id":"cs_quote", "url":"https://checkout.stripe.com/c/pay/test"}
    with patch("app.core.report_quotes.stripe_client", return_value=client):
        store.checkout(quote["id"], "https://example.test")
        session = paid_session(store, quote); client.v1.checkout.sessions.retrieve.return_value = session
        handle_report_event({"type":"checkout.session.async_payment_succeeded","data":{"object":session}}, db)
        snapshot = create_snapshot(example_result(),"model",1,"sum")
        store.fulfill(quote["id"], "operator", "job", snapshot)
        assert store.get(quote["id"])["report_snapshot"] == snapshot
        client.v1.refunds.create.return_value = {"id":"re_quote", "status":"succeeded", "amount":9900, "payment_intent":"pi_quote", "currency":"usd"}
        store.refund(quote["id"], "operator")
        assert store.get(quote["id"])["status"] == "refunded"
        assert store.get(quote["id"])["audit"][-1]["action"] == "refunded"


def test_customer_cannot_issue_quotes_or_change_price_and_provider_stays_disabled(quotes, monkeypatch):
    from app.core.auth import create_token
    from fastapi.testclient import TestClient
    from app.config import settings
    db, buyer, store, quote = quotes
    monkeypatch.setattr(settings, "stripe_report_payments_enabled", False)
    client = TestClient(app.main.app)
    headers = {"Authorization":"Bearer "+create_token(buyer)}
    assert client.get("/api/report-quotes",headers=headers).json()["quotes"][0]["amount_cents"] == 9900
    assert client.post("/api/admin/report-quotes",headers=headers,json={"user_id":buyer,"title":"Forged","amount_cents":100}).status_code == 403
    assert client.post(f"/api/report-quotes/{quote['id']}/checkout",headers=headers,json={"amount_cents":1}).status_code == 422
    assert client.post(f"/api/report-quotes/{quote['id']}/checkout",headers=headers,json={}).status_code == 503
    other = db.create_user("other@example.test","unused")["id"]
    assert client.get(f"/api/report-quotes/{quote['id']}/report",headers={"Authorization":"Bearer "+create_token(other)}).status_code == 404


def test_refund_before_payment_webhook_does_not_resurrect_paid_order(quotes):
    from app.core.report_quotes import handle_report_event
    db, _, store, quote = quotes
    client = Mock(); client.v1.checkout.sessions.create.return_value = {"id":"cs_quote","url":"https://checkout.stripe.com/c/pay/test"}
    with patch("app.core.report_quotes.stripe_client",return_value=client):
        store.checkout(quote["id"],"https://example.test")
        session = paid_session(store,quote)
        session["payment_intent"]={"id":"pi_quote","latest_charge":{"refunded":True,"amount_refunded":9900}}
        client.v1.checkout.sessions.retrieve.return_value=session
        result=handle_report_event({"type":"checkout.session.completed","data":{"object":session}},db)
    assert result["status"] == "refunded" and store.get(quote["id"])["status"] == "refunded"


def test_signed_webhook_and_operator_delivery_to_free_customer(quotes, monkeypatch):
    import hashlib, hmac, json, time
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.core.auth import create_token
    from app.core.forecast_reports import create_snapshot
    from tests.test_forecast_exports import example_result
    db, buyer, store, quote = quotes
    operator=db.create_user('operator@example.test','unused')['id'];db.mark_email_verified(operator)
    monkeypatch.setattr(settings,'admin_emails','operator@example.test')
    monkeypatch.setattr(settings,'stripe_secret_key','rk_test_isolated_placeholder')
    monkeypatch.setattr(settings,'stripe_webhook_secret','whsec_isolated_placeholder')
    provider=Mock();provider.v1.checkout.sessions.create.return_value={'id':'cs_quote','url':'https://checkout.stripe.com/c/pay/test'}
    client=TestClient(app.main.app)
    with patch('app.core.report_quotes.stripe_client',return_value=provider):
        store.checkout(quote['id'],'https://example.test')
        session=paid_session(store,quote);provider.v1.checkout.sessions.retrieve.return_value=session
        event={'id':'evt_signed','object':'event','api_version':'2026-11-18.endive','livemode':False,
               'type':'checkout.session.completed','data':{'object':session}}
        payload=json.dumps(event).encode();ts=str(int(time.time()))
        signature=hmac.new(settings.stripe_webhook_secret.encode(),ts.encode()+b'.'+payload,hashlib.sha256).hexdigest()
        assert client.post('/api/payments/webhook',content=payload,headers={'Content-Type':'application/json','Stripe-Signature':'wrong'}).status_code==400
        assert store.get(quote['id'])['status']=='quoted'
        response=client.post('/api/payments/webhook',content=payload,headers={'Content-Type':'application/json','Stripe-Signature':f't={ts},v1={signature}'})
        assert response.status_code==200 and store.get(quote['id'])['status']=='paid'
    snapshot=create_snapshot(example_result(),'model',1,'sum')
    job=db.enqueue_job(operator,'forecast',{'path':'operator.csv'});db.claim_job()
    db.finish_job(job,result={'data':{'report_snapshot':snapshot}})
    admin_headers={'Authorization':'Bearer '+create_token(operator)}
    delivered=client.post(f"/api/admin/report-quotes/{buyer}/{quote['id']}/fulfill",headers=admin_headers,json={'job_id':job})
    assert delivered.status_code==200,delivered.text
    with db.write_transaction() as conn:conn.execute('DELETE FROM background_jobs WHERE id=?',(job,))
    buyer_headers={'Authorization':'Bearer '+create_token(buyer)}
    report=client.get(f"/api/report-quotes/{quote['id']}/report?lang=ru",headers=buyer_headers)
    assert report.status_code==200 and 'не откалиброван' in report.text
    assert db.get_balance(buyer)==0 and db.get_user_plan(buyer)=='free'


def test_refund_arriving_during_payment_confirmation_requests_retry(quotes):
    from app.core.report_quotes import handle_report_event
    db, _, store, quote = quotes
    client=Mock(); client.v1.checkout.sessions.create.return_value={'id':'cs_quote','url':'https://checkout.stripe.com/c/pay/test'}
    with patch('app.core.report_quotes.stripe_client',return_value=client):
        store.checkout(quote['id'],'https://example.test')
        stale=paid_session(store,quote)
        early={'type':'charge.refunded','data':{'object':{'payment_intent':'pi_quote','metadata':stale['metadata']}}}
        with pytest.raises(RuntimeError,match='retry this webhook'):
            handle_report_event(early,db)
        store.mark_paid(quote['id'],stale)
        client.v1.payment_intents.retrieve.return_value={'latest_charge':{'refunded':True}}
        client.v1.refunds.list.return_value={'data':[{'id':'re_early','payment_intent':'pi_quote','amount':9900,'currency':'usd','status':'succeeded'}]}
        assert handle_report_event(early,db)['status']=='refunded'
    assert store.get(quote['id'])['status']=='refunded'


def test_failed_refund_restores_delivery_and_allows_new_attempt(quotes):
    _, _, store, quote=quotes
    client=Mock();client.v1.checkout.sessions.create.return_value={'id':'cs_quote','url':'https://checkout.stripe.com/c/pay/test'}
    refund={'id':'re_one','payment_intent':'pi_quote','amount':9900,'currency':'usd','status':'succeeded'}
    with patch('app.core.report_quotes.stripe_client',return_value=client):
        store.checkout(quote['id'],'https://example.test');store.mark_paid(quote['id'],paid_session(store,quote))
        client.v1.refunds.create.return_value=refund
        store.refund(quote['id'],'operator')
        store.apply_refund(quote['id'],{**refund,'status':'requires_action'})
        assert store.get(quote['id'])['status']=='refund_pending'
        store.apply_refund(quote['id'],{**refund,'status':'failed'})
        assert store.get(quote['id'])['status']=='paid'
        client.v1.refunds.create.return_value={**refund,'id':'re_two','status':'pending'}
        store.refund(quote['id'],'operator');store.refund(quote['id'],'operator')
    keys=[call.kwargs['options']['idempotency_key'] for call in client.v1.refunds.create.call_args_list]
    assert keys[0]!=keys[1] and keys[1]==keys[2]


def test_dashboard_refund_uses_payment_intent_binding_and_old_failure_is_ignored(quotes):
    from app.core.report_quotes import handle_report_event
    db,_,store,quote=quotes
    client=Mock();client.v1.checkout.sessions.create.return_value={'id':'cs_quote','url':'https://checkout.stripe.com/c/pay/test'}
    with patch('app.core.report_quotes.stripe_client',return_value=client):
        store.checkout(quote['id'],'https://example.test');session=paid_session(store,quote);store.mark_paid(quote['id'],session)
        refund={'id':'re_dashboard','payment_intent':'pi_quote','amount':9900,'currency':'usd','status':'succeeded'}
        client.v1.payment_intents.retrieve.return_value={'metadata':session['metadata']}
        client.v1.refunds.retrieve.return_value=refund
        event={'type':'refund.created','data':{'object':refund}}
        assert handle_report_event(event,db)['status']=='refunded'
        store.apply_refund(quote['id'],{**refund,'status':'failed'})
        client.v1.refunds.create.return_value={**refund,'id':'re_new','status':'pending'}
        store.refund(quote['id'],'operator')
        assert store.apply_refund(quote['id'],{**refund,'status':'failed'})['status']=='old_refund_ignored'
        assert store.get(quote['id'])['status']=='refund_pending'


def test_admin_issues_one_quote_per_idempotency_key(quotes,monkeypatch):
    from app.config import settings
    from app.core.auth import create_token
    from fastapi.testclient import TestClient
    db,buyer,_,_=quotes
    operator=db.create_user('quote-operator@example.test','unused')['id'];db.mark_email_verified(operator)
    monkeypatch.setattr(settings,'admin_emails','quote-operator@example.test')
    client=TestClient(app.main.app);headers={'Authorization':'Bearer '+create_token(operator),'Idempotency-Key':'agreed-report-one'}
    body={'user_id':buyer,'title':'Agreed report','amount_cents':12345,'currency':'eur'}
    first=client.post('/api/admin/report-quotes',headers=headers,json=body)
    second=client.post('/api/admin/report-quotes',headers=headers,json=body)
    assert first.status_code==201 and second.json()==first.json()
    assert client.post('/api/admin/report-quotes',headers=headers,json={**body,'amount_cents':12346}).status_code==409
    quotes_response=client.get('/api/admin/report-quotes/'+buyer,headers=headers)
    assert len(quotes_response.json()['quotes'])==2
