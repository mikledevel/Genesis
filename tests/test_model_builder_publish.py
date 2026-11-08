"""POST /api/model-builder/publish used to take the model file path and the scores from the request body. Now the server keeps what the
build produced (app/core/build_records.py) and publish reads it back: nobody can register another user's model file, point it at an
arbitrary server file, or advertise made-up numbers; only a verified build can be sold, an unreliable one cannot."""
import os
import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tests.test_local_models import _register, client  # noqa: F401


def _model_file(tmp_path, name="m.pkl"):
    X = np.random.default_rng(0).normal(size=(60, 2))
    d = tmp_path / "built"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_bytes(pickle.dumps(LogisticRegression().fit(X, (X[:, 0] > 0).astype(int))))
    return str(p)


def _result(path, verified=True, trust="high"):
    v = {"metrics": {"accuracy_test": 0.9, "trust": {"level": trust, "reasons": ["too good to be true"] if trust == "low" else []},
                     "data_checks": []},
         "contract": "numeric_vector", "task": "binary_classification", "model_type": "LogisticRegression",
         "features": ["a", "b"], "target": "y"}
    return {"success": True, "code": "x", "metrics": {"accuracy_test": 0.9, "task": "binary_classification"}, "log": "", "iterations": 1,
            "saved_path": None, "saved_model_path": path, "verified": verified, "verification": v if verified else None}


def _build(client, monkeypatch, tmp_path, headers, uid, **kw):
    from app.db.database import GenesisDB
    GenesisDB().set_user_plan(uid, "pro")                        # /generate is a server-training route
    monkeypatch.setattr("app.core.model_builder.ModelBuilder.build", lambda self, **_k: _result(_model_file(tmp_path, kw.pop("name", "m.pkl")), **kw))
    (tmp_path / "datasets").mkdir(exist_ok=True)                 # a dataset path must be inside the datasets folder (app/core/paths.py), like a real upload
    (tmp_path / "datasets" / "d.csv").write_text("a,b,y\n1,2,0\n")
    r = client.post("/api/model-builder/generate",
                    json={"description": "x", "target_column": "y", "dataset_path": str(tmp_path / "datasets" / "d.csv")}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _publish(client, headers, **body):
    return client.post("/api/model-builder/publish", json={"title": "T", "description": "d", **body}, headers=headers)


def _listings(client):
    return client.get("/api/marketplace/listings").json()["listings"]


def test_publish_uses_the_servers_record_not_the_request(client, monkeypatch, tmp_path):
    uid, h = _register(client, "a@x.io")
    res = _build(client, monkeypatch, tmp_path, h, uid)
    assert res["build_id"]
    r = _publish(client, h, build_id=res["build_id"], metrics={"accuracy_test": 0.999, "task": "regression"}, price_per_call=0.03)
    assert r.status_code == 200 and r.json()["status"] == "published"
    (lst,) = _listings(client)
    assert lst["metrics"]["accuracy_test"] == 0.9 and lst["task"] == "binary_classification" and lst["price_per_call"] == 0.03
    assert lst["provenance"]["verified"] is True and lst["provenance"]["trained_on"] == "platform" and "label" not in lst["provenance"]
    from app.core.engine.registry import ModelRegistry
    assert ModelRegistry().get_record(r.json()["model_id"])["metadata"]["author"] == uid


def test_an_older_client_may_send_the_path_generate_returned(client, monkeypatch, tmp_path):
    uid, h = _register(client, "a@x.io")
    res = _build(client, monkeypatch, tmp_path, h, uid)
    r = _publish(client, h, saved_model_path=res["saved_model_path"])
    assert r.status_code == 200 and len(_listings(client)) == 1


def test_nobody_can_publish_a_file_they_did_not_build(client, monkeypatch, tmp_path):
    a, ha = _register(client, "a@x.io")
    b, hb = _register(client, "b@x.io")
    res = _build(client, monkeypatch, tmp_path, ha, a)
    assert _publish(client, hb, build_id=res["build_id"]).status_code == 400                   # another user's build
    assert _publish(client, hb, saved_model_path=res["saved_model_path"]).status_code == 400   # another user's file
    other = _model_file(tmp_path, "someone_elses.pkl")
    assert _publish(client, ha, saved_model_path=other).status_code == 400                     # a real file the server holds, not a build of theirs
    assert _publish(client, ha, saved_model_path="/etc/passwd").status_code == 400
    assert _publish(client, ha, build_id="../../x").status_code == 400
    assert _listings(client) == []


def test_only_a_verified_build_can_be_sold_but_any_can_be_saved(client, monkeypatch, tmp_path):
    uid, h = _register(client, "a@x.io")
    res = _build(client, monkeypatch, tmp_path, h, uid, verified=False)
    r = _publish(client, h, build_id=res["build_id"], price_per_call=0.02)
    assert r.status_code == 400 and "verified" in r.json()["detail"] and _listings(client) == []
    r = _publish(client, h, build_id=res["build_id"], price_per_call=-1)
    assert r.status_code == 200 and r.json()["status"] == "saved" and r.json()["model_id"] and _listings(client) == []


def test_an_unreliable_build_is_refused_before_anything_is_registered(client, monkeypatch, tmp_path):
    from app.core.engine.registry import ModelRegistry
    uid, h = _register(client, "a@x.io")
    res = _build(client, monkeypatch, tmp_path, h, uid, trust="low")
    before = len(ModelRegistry().list_models())
    r = _publish(client, h, build_id=res["build_id"])
    assert r.status_code == 400 and "too good to be true" in r.json()["detail"]
    assert len(ModelRegistry().list_models()) == before and _listings(client) == []


def test_publishing_twice_does_not_register_twice(client, monkeypatch, tmp_path):
    from app.core.engine.registry import ModelRegistry
    uid, h = _register(client, "a@x.io")
    res = _build(client, monkeypatch, tmp_path, h, uid)
    m1 = _publish(client, h, build_id=res["build_id"], price_per_call=-1).json()["model_id"]
    n = len(ModelRegistry().list_models())
    m2 = _publish(client, h, build_id=res["build_id"], price_per_call=0.01).json()["model_id"]
    assert m1 == m2 and len(ModelRegistry().list_models()) == n and len(_listings(client)) == 1


def test_a_bad_price_is_a_400(client, monkeypatch, tmp_path):
    uid, h = _register(client, "a@x.io")
    res = _build(client, monkeypatch, tmp_path, h, uid)
    assert _publish(client, h, build_id=res["build_id"], price_per_call="free").status_code == 400
