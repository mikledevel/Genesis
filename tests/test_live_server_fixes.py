"""Regression tests for two bugs found by running the real server over HTTP (not via mocks):
1. A message without conversation_id used the shared id "default": the first account to use it owned
   it and every other account got a permanent 403.
2. POST /api/models/{id}/predict returned 500 (with a raw exception text) for malformed `features`,
   and had already spent one call of the API key's quota on that request."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.main import app
    from app import deps
    deps._agents.clear()
    return TestClient(app)


def _register(client, email):
    r = client.post("/api/auth/register", json={"email": email, "password": "secret12", "name": "T"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def _fake_agent(monkeypatch):
    seen = []

    class FakeAgent:
        agent_override = None

        def process(self, message, cid, projects, user_id=None, **kw):
            seen.append((user_id, cid))
            return types.SimpleNamespace(message="ok", data={}, message_id=1)

    from app.routers import chat_agent
    monkeypatch.setattr(chat_agent, "get_agent", lambda cid, uid: FakeAgent())
    return seen


def test_two_accounts_can_both_message_without_a_conversation_id(client, monkeypatch):
    seen = _fake_agent(monkeypatch)
    h1, h2 = _register(client, "a@x.io"), _register(client, "b@x.io")
    r1 = client.post("/api/agent/message", json={"message": "hi"}, headers=h1)
    r2 = client.post("/api/agent/message", json={"message": "hi"}, headers=h2)
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text            # used to be 403
    assert r1.json()["conversation_id"] != r2.json()["conversation_id"]
    assert len({cid for _, cid in seen}) == 2


def test_the_default_thread_stays_stable_for_one_account(client, monkeypatch):
    _fake_agent(monkeypatch)
    h = _register(client, "c@x.io")
    a = client.post("/api/agent/message", json={"message": "1"}, headers=h).json()["conversation_id"]
    b = client.post("/api/agent/message", json={"message": "2"}, headers=h).json()["conversation_id"]
    assert a == b


def _setup_model(monkeypatch, owner, n_features=2):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(np.array([[0.0] * n_features, [1.0] * n_features]), [0.0, 1.0])
    record = {"features": ["a", "b"][:n_features], "metadata": {"author": owner}, "metrics": {}}

    class FakeRegistry:
        def get_record(self, model_id):
            return record

        def get_model(self, model_id):
            return model

    import app.core.engine.registry as reg
    monkeypatch.setattr(reg, "ModelRegistry", FakeRegistry)


def _key(client, headers, limit=100):
    return client.post("/api/keys/create", json={"calls_limit": limit}, headers=headers).json()["api_key"]


def _used(client, headers):
    return client.get("/api/keys", headers=headers).json()["keys"][0]["calls_used"]


def test_valid_predict_still_works_and_costs_one_call(client, monkeypatch):
    h = _register(client, "d@x.io")
    _setup_model(monkeypatch, client.get("/api/auth/me", headers=h).json()["user"]["id"])
    k = _key(client, h)
    r = client.post("/api/models/m1/predict", json={"features": [1, 1]}, headers={"X-API-Key": k})
    assert r.status_code == 200, r.text
    assert len(r.json()["prediction"]) == 1
    assert _used(client, h) == 1


@pytest.mark.parametrize("features", [[1], ["x", "y"], [1, 2, 3]])
def test_malformed_features_are_422_and_do_not_spend_quota(client, monkeypatch, features):
    h = _register(client, "e@x.io")
    _setup_model(monkeypatch, client.get("/api/auth/me", headers=h).json()["user"]["id"])
    k = _key(client, h)
    r = client.post("/api/models/m1/predict", json={"features": features}, headers={"X-API-Key": k})
    assert r.status_code == 422, r.text               # used to be 500
    assert r.json()["detail"].startswith("Invalid model features")
    assert _used(client, h) == 0                      # used to be 1
