from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core import payments
from app.db.database import GenesisDB


class _FakeStripe:
    def __init__(self):
        self.customer_creates = []
        self.checkout_creates = []
        self.subscriptions = {}
        self.checkout = SimpleNamespace(Session=SimpleNamespace(
            create=self._create_checkout, retrieve=self._retrieve_checkout))
        self.Customer = SimpleNamespace(create=self._create_customer)
        self.Subscription = SimpleNamespace(retrieve=self._retrieve_subscription)
        self.billing_portal = SimpleNamespace(Session=SimpleNamespace(create=lambda **kw: {"url": "https://billing.test"}))

    def _create_customer(self, **kwargs):
        self.customer_creates.append(kwargs)
        return {"id": "cus_genesis_test"}

    def _create_checkout(self, **kwargs):
        self.checkout_creates.append(kwargs)
        return {"id": "cs_genesis_test", "url": "https://checkout.test/session"}

    def _retrieve_checkout(self, session_id):
        return {"id": session_id, "status": "open", "url": "https://checkout.test/session"}

    def _retrieve_subscription(self, subscription_id):
        return self.subscriptions[subscription_id]


@pytest.fixture
def billing_settings(monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_placeholder")
    monkeypatch.setattr(settings, "stripe_price_pro_monthly", "price_pro_test")
    monkeypatch.setattr(settings, "stripe_price_business_monthly", "price_business_test")
    monkeypatch.setattr(settings, "stripe_portal_configuration_id", "")


def _new_db(tmp_path):
    db = GenesisDB(str(tmp_path / "billing.sqlite"))
    user = db.create_user("billing@example.test", "unused-hash", "Billing user")
    db.mark_email_verified(user["id"])
    return db, user["id"]


def test_checkout_creates_one_customer_and_reuses_an_open_session(tmp_path, monkeypatch, billing_settings):
    db, user_id = _new_db(tmp_path)
    fake = _FakeStripe()
    monkeypatch.setattr(payments, "_get_stripe", lambda: fake)

    first = payments.create_subscription_checkout(db, user_id, "pro", "https://site.test/success", "https://site.test/cancel")
    second = payments.create_subscription_checkout(db, user_id, "pro", "https://site.test/success", "https://site.test/cancel")

    assert first["url"] == second["url"]
    assert len(fake.customer_creates) == 1
    assert len(fake.checkout_creates) == 1
    params = fake.checkout_creates[0]
    assert params["mode"] == "subscription"
    assert params["line_items"] == [{"price": "price_pro_test", "quantity": 1}]
    assert params["subscription_data"]["metadata"]["genesis_user_id"] == user_id
    assert params["subscription_data"]["billing_mode"]["type"] == "flexible"
    assert db.get_billing_user(user_id)["stripe_customer_id"] == "cus_genesis_test"


def test_checkout_rejects_duplicate_active_subscription(tmp_path, monkeypatch, billing_settings):
    db, user_id = _new_db(tmp_path)
    db.apply_stripe_subscription_event("evt_active", "customer.subscription.created", user_id,
                                       "cus_genesis_test", "sub_active", "active", "price_pro_test",
                                       2_000_000_000, False, "pro")
    fake = _FakeStripe()
    monkeypatch.setattr(payments, "_get_stripe", lambda: fake)

    with pytest.raises(ValueError, match="active subscription"):
        payments.create_subscription_checkout(db, user_id, "business", "https://site.test/s", "https://site.test/c")
    assert fake.checkout_creates == []


def test_subscription_webhook_uses_latest_stripe_state_and_is_idempotent(tmp_path, monkeypatch, billing_settings):
    db, user_id = _new_db(tmp_path)
    fake = _FakeStripe()
    fake.subscriptions["sub_1"] = {
        "id": "sub_1", "customer": "cus_genesis_test", "metadata": {"genesis_user_id": user_id},
        "status": "active", "current_period_end": 2_000_000_000, "cancel_at_period_end": False,
        "items": {"data": [{"price": {"id": "price_pro_test"}}]},
    }
    monkeypatch.setattr(payments, "_get_stripe", lambda: fake)
    event = {"id": "evt_subscription_1", "type": "customer.subscription.updated",
             "data": {"object": {"id": "sub_1", "customer": "cus_genesis_test"}}}

    result = payments.handle_subscription_event(event, db)
    duplicate = payments.handle_subscription_event(event, db)
    assert result["plan"] == "pro"
    assert duplicate == {"status": "already_processed"}
    assert db.get_user_plan(user_id) == "pro"

    fake.subscriptions["sub_1"]["status"] = "canceled"
    event["id"] = "evt_subscription_2"
    event["type"] = "customer.subscription.deleted"
    assert payments.handle_subscription_event(event, db)["plan"] == "free"
    assert db.get_user_plan(user_id) == "free"


def test_subscription_checkout_completion_never_credits_topup_balance():
    class _DB:
        def add_transaction(self, *args, **kwargs):
            raise AssertionError("subscription checkout must not credit prepaid balance")

    event = {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_sub", "mode": "subscription", "metadata": {"genesis_user_id": "user_1"},
        "amount_total": 1900,
    }}}
    assert payments.handle_checkout_completed(event, _DB()) is None


def test_subscription_checkout_route_requires_a_verified_user_and_returns_hosted_url(tmp_path, monkeypatch, billing_settings):
    from app.main import app
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "debug", True)
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test")
    from app import deps
    from app.core.auth import create_token
    from app.main import app
    deps._agents.clear()
    db = GenesisDB()
    user = db.create_user("route@example.test", "unused-hash", "Route user")
    db.mark_email_verified(user["id"])
    fake = _FakeStripe()
    monkeypatch.setattr(payments, "_get_stripe", lambda: fake)

    response = TestClient(app).post("/api/subscriptions/checkout", json={"plan_id": "pro"},
                                    headers={"Authorization": "Bearer " + create_token(user["id"])})
    assert response.status_code == 200, response.text
    assert response.json()["url"] == "https://checkout.test/session"



def test_render_refuses_checkout_without_persistent_database(tmp_path, monkeypatch, billing_settings):
    from app.main import app
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RENDER_EXTERNAL_URL', 'https://genesis.example.test')
    monkeypatch.setattr(settings, 'database_url', '')
    monkeypatch.setattr(settings, 'stripe_webhook_secret', 'whsec_test')
    from app.main import app
    from app.core.auth import create_token
    db = GenesisDB()
    user = db.create_user('ephemeral@example.test', 'hash', 'Ephemeral user')
    db.mark_email_verified(user['id'])
    def unexpected_checkout(*args, **kwargs):
        raise AssertionError('must not accept payment with ephemeral billing storage')
    monkeypatch.setattr(payments, 'create_subscription_checkout', unexpected_checkout)
    client = TestClient(app)
    response = client.post('/api/subscriptions/checkout', json={'plan_id': 'pro'},
                           headers={'Authorization': 'Bearer ' + create_token(user['id'])})
    assert response.status_code == 503
    assert 'storage' in response.json()['detail']
    assert client.get('/api/config/public').json()['stripe_subscriptions_enabled'] is False
