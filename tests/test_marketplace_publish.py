"""Marketplace listing of a model: only its author can list it, the numbers on the listing are the ones the platform measured
(never the ones in the request), a model the platform rates as unreliable is not sold, and a model trained on the owner's own
computer carries a label saying so."""
import io
import os
import pickle

import pandas as pd

from tests.test_local_models import _prepare, _register, _register_model, _train_locally, client  # noqa: F401


def _local_model(client, tmp_path, headers):
    p = _prepare(client, headers).json()
    blob = _train_locally(tmp_path, client.get(p["download_url"], headers=headers).content)
    r = _register_model(client, headers, p["prepare_id"], blob)
    assert r.status_code == 200, r.text
    return r.json()


def _publish(client, headers, model_id, **over):
    body = {"model_id": model_id, "title": "Churn", "description": "d", "task": "regression", "model_type": "FAKE",
            "metrics": {"accuracy_test": 0.999}, "price_per_call": 0.02}
    body.update(over)
    return client.post("/api/marketplace/publish", json=body, headers=headers)


def test_a_local_model_is_listed_with_measured_scores_and_a_label(client, tmp_path):
    _uid, h = _register(client, "seller@x.io")
    m = _local_model(client, tmp_path, h)
    r = _publish(client, h, m["model_id"])
    assert r.status_code == 200, r.text
    assert r.json()["provenance"]["trained_on"] == "user_machine" and "not by the platform" in r.json()["provenance"]["label"]
    listing = next(x for x in client.get("/api/marketplace/listings").json()["listings"] if x["model_id"] == m["model_id"])
    assert listing["task"] == m["task"] == "binary_classification" and listing["model_type"] == m["model_type"] != "FAKE"
    assert listing["metrics"]["accuracy_test"] == m["metrics"]["accuracy_test"] != 0.999          # the request's numbers are ignored
    assert listing["provenance"]["verified"] is True and listing["provenance"]["trust"]["level"] in ("high", "medium")
    assert listing["price_per_call"] == 0.02


def test_only_the_author_can_publish(client, tmp_path):
    _a, ha = _register(client, "a@x.io")
    _b, hb = _register(client, "b@x.io")
    m = _local_model(client, tmp_path, ha)
    r = _publish(client, hb, m["model_id"])
    assert r.status_code == 403 and "Only the author" in r.json()["detail"]
    assert client.get("/api/marketplace/listings").json()["listings"] == []
    assert _publish(client, ha, "no_such_model").status_code == 404
    assert client.post("/api/marketplace/publish", json={"model_id": "x", "title": "t", "description": "d", "task": "t",
                                                         "model_type": "m"}).status_code in (401, 403)


def test_an_unreliable_model_is_not_sold(client):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    _uid, h = _register(client, "seller@x.io")
    p = _prepare(client, h).json()
    train = pd.read_csv(io.BytesIO(client.get(p["download_url"], headers=h).content))
    pipe = Pipeline([("c", ColumnTransformer([("o", OneHotEncoder(handle_unknown="ignore"), ["plan"])], remainder="drop")),
                     ("m", DummyClassifier(strategy="most_frequent"))]).fit(train.drop(columns=["churn"]), train["churn"])
    reg = _register_model(client, h, p["prepare_id"], pickle.dumps(pipe)).json()
    r = _publish(client, h, reg["model_id"])
    assert r.status_code == 400 and "not reliable enough to sell" in r.json()["detail"]
    assert client.get("/api/marketplace/listings").json()["listings"] == []


def test_the_page_shows_provenance_and_does_not_claim_a_refused_publish_succeeded():
    page = open(os.path.join(os.path.dirname(__file__), "..", "static", "index.html"), encoding="utf-8-sig").read()
    assert "function provBadge" in page and page.count("provBadge(") >= 3          # the definition + model card + landing card
    assert "m.provenance && m.provenance.label" in page                             # the detail modal shows the full label
    assert "Trained on owner's computer" in page
    assert "if (!r.ok) { toast(out.detail || 'Could not publish model', 'error'); return; }" in page
