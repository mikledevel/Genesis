"""Bring a locally trained model back: prepare (platform keeps a hidden test set) -> train on the user's machine -> register
(platform scores it itself). What must hold: the hidden rows never leave the server, a hostile pickle runs nothing, another
user's prepared id is invisible, the score and trust verdict are the platform's, and it all works on the Free plan."""
import io
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_IP = 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from app.main import app
    from app import deps
    deps._agents.clear()
    c = TestClient(app)
    global _IP
    _IP += 1
    c.headers["x-forwarded-for"] = f"10.8.{_IP // 250}.{_IP % 250}"
    return c


def _register(client, email):
    r = client.post("/api/auth/register", json={"email": email, "password": "secret12", "name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["user"]["id"], {"Authorization": "Bearer " + r.json()["token"]}


def _csv(n=300, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "plus", "pro"], n)
    spend = rng.normal(50, 15, n)
    y = ((spend + (plan == "basic") * 12 + rng.normal(0, 8, n)) > 58).astype(int)
    df = pd.DataFrame({"customer_id": range(n), "age": rng.integers(18, 70, n), "plan": plan, "spend": spend, "churn": y})
    return df.to_csv(index=False).encode()


def _upload(client, headers, data=None, name="customers.csv"):
    r = client.post("/api/agent/upload", files={"file": (name, data or _csv(), "text/csv")}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _prepare(client, headers, target="churn", data=None):
    return client.post("/api/local-training/prepare", json={"upload_id": _upload(client, headers, data), "target": target}, headers=headers)


def _train_locally(tmp_path, train_csv_bytes):
    work = tmp_path / "laptop"
    work.mkdir(exist_ok=True)
    (work / "train.csv").write_bytes(train_csv_bytes)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "train_local.py"), "train.csv", "--target", "churn",
                        "--budget", "fast", "--out", "out"], cwd=work, capture_output=True, text=True, timeout=240, env=env)
    assert r.returncode == 0, r.stderr + r.stdout
    return (work / "out" / "model.pkl").read_bytes()


def _register_model(client, headers, prepare_id, blob):
    return client.post("/api/local-training/register", data={"prepare_id": prepare_id},
                       files={"model": ("model.pkl", blob, "application/octet-stream")}, headers=headers)


# ------------------------------------------------------------------ the whole flow, on the Free plan
def test_full_flow_on_the_free_plan(client, tmp_path):
    uid, h = _register(client, "free@x.io")
    prep = _prepare(client, h)
    assert prep.status_code == 200, prep.text
    p = prep.json()
    assert p["target"] == "churn" and p["task"] == "binary_classification"
    assert p["rows_train"] + p["rows_hidden"] == 300 and p["rows_hidden"] >= 50
    assert "customer_id" in p["dropped"]                                    # ID column removed by the platform
    assert p["command"] == "python3 train_local.py train.csv --target churn"

    got = client.get(p["download_url"], headers=h)
    assert got.status_code == 200 and "text/csv" in got.headers["content-type"]
    train = pd.read_csv(io.BytesIO(got.content))
    assert len(train) == p["rows_train"] and "customer_id" not in train.columns

    blob = _train_locally(tmp_path, got.content)
    r = _register_model(client, h, p["prepare_id"], blob)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["model_id"] and out["metrics"]["verified"] is True and out["metrics"]["n_test"] == p["rows_hidden"]
    assert 0.5 < out["metrics"]["accuracy_test"] <= 1 and out["trust"]["level"] in ("high", "medium", "low")
    assert "not by the platform" in out["note"]

    rec = client.get(f"/api/registry/models/{out['model_id']}").json()
    meta = rec.get("metadata", rec)
    assert meta["built_via"] == "local_training" and meta["trained_on"] == "user_machine" and meta["verified"] is True
    assert meta["author"] == uid and meta["input_format"]


# ------------------------------------------------------------------ the hidden rows stay hidden
def test_only_train_csv_is_ever_served(client):
    _uid, h = _register(client, "a@x.io")
    pid = _prepare(client, h).json()["prepare_id"]
    for name in ("holdout.csv", "meta.json", "../train.csv"):
        assert client.get(f"/api/local-training/prepared/{pid}/{name}", headers=h).status_code in (404, 405)
    assert client.get(f"/api/local-training/prepared/{pid}/train.csv").status_code in (401, 403)     # login required


def test_another_users_prepared_id_is_invisible(client, tmp_path):
    _a, ha = _register(client, "a@x.io")
    _b, hb = _register(client, "b@x.io")
    p = _prepare(client, ha).json()
    assert client.get(p["download_url"], headers=hb).status_code == 404
    train = client.get(p["download_url"], headers=ha).content
    assert _register_model(client, hb, p["prepare_id"], _train_locally(tmp_path, train)).status_code == 404


def test_prepare_validates_its_input(client):
    _uid, h = _register(client, "a@x.io")
    r = _prepare(client, h, target="nope")
    assert r.status_code == 400 and "There is no column 'nope'" in r.json()["detail"]
    r = client.post("/api/local-training/prepare", json={"upload_id": "deadbeef", "target": "churn"}, headers=h)
    assert r.status_code == 404
    r = client.post("/api/local-training/prepare", json={"upload_id": "../../x", "target": "churn"}, headers=h)
    assert r.status_code == 400
    assert client.post("/api/local-training/prepare", json={"upload_id": "x", "target": "y"}).status_code in (401, 403)
    assert _prepare(client, h, target="CHURN").json()["target"] == "churn"           # case-insensitive, resolved to the real name


def test_a_prepared_dataset_expires(client, tmp_path):
    from app.core import local_models
    _uid, h = _register(client, "a@x.io")
    p = _prepare(client, h).json()
    import json
    metas = list((tmp_path / "datasets" / "local_training").glob("*/*/meta.json"))
    assert len(metas) == 1
    m = json.loads(metas[0].read_text()); m["created"] -= local_models.TTL_SECONDS + 10
    metas[0].write_text(json.dumps(m))
    assert client.get(p["download_url"], headers=h).status_code == 404


# ------------------------------------------------------------------ the uploaded file is untrusted
class _Evil:
    def __reduce__(self):
        return (os.system, ("touch pwned_by_pickle",))


def test_a_hostile_pickle_runs_nothing(client, tmp_path):
    _uid, h = _register(client, "a@x.io")
    pid = _prepare(client, h).json()["prepare_id"]
    r = _register_model(client, h, pid, pickle.dumps(_Evil()))
    assert r.status_code == 400 and "nothing in it was run" in r.json()["detail"]
    assert not (tmp_path / "pwned_by_pickle").exists() and not os.path.exists("pwned_by_pickle")


def test_garbage_empty_and_oversized_files_are_refused(client):
    from app.core import local_models
    _uid, h = _register(client, "a@x.io")
    pid = _prepare(client, h).json()["prepare_id"]
    assert _register_model(client, h, pid, b"this is not a pickle").status_code == 400
    assert _register_model(client, h, pid, b"").status_code == 400
    assert _register_model(client, h, pid, b"\0" * (local_models.MAX_MODEL_BYTES + 5)).status_code == 413
    assert _register_model(client, h, "../../etc", b"x").status_code == 404


def test_a_model_that_cannot_predict_the_columns_is_explained(client):
    from sklearn.linear_model import LogisticRegression
    _uid, h = _register(client, "a@x.io")
    pid = _prepare(client, h).json()["prepare_id"]
    m = LogisticRegression().fit(np.random.default_rng(0).normal(size=(50, 7)), np.arange(50) % 2)   # 7 unrelated numeric inputs
    r = _register_model(client, h, pid, pickle.dumps(m))
    assert r.status_code == 400 and "could not score" in r.json()["detail"]


def test_a_useless_model_gets_a_low_verdict_not_a_pass(client):
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    _uid, h = _register(client, "a@x.io")
    p = _prepare(client, h).json()
    train = pd.read_csv(io.BytesIO(client.get(p["download_url"], headers=h).content))
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    pipe = Pipeline([("c", ColumnTransformer([("o", OneHotEncoder(handle_unknown="ignore"), ["plan"])], remainder="drop")),
                     ("m", DummyClassifier(strategy="most_frequent"))]).fit(train.drop(columns=["churn"]), train["churn"])
    r = _register_model(client, h, p["prepare_id"], pickle.dumps(pipe))
    assert r.status_code == 200, r.text
    assert r.json()["trust"]["level"] == "low" and any("наивн" in x for x in r.json()["trust"]["reasons"])


def test_saved_model_slots_still_count(client, tmp_path, monkeypatch):
    from app.core import plans
    _uid, h = _register(client, "a@x.io")
    p = _prepare(client, h).json()
    blob = _train_locally(tmp_path, client.get(p["download_url"], headers=h).content)
    monkeypatch.setattr(plans, "_count_models", lambda user_id: 3)
    r = _register_model(client, h, p["prepare_id"], blob)
    assert r.status_code == 403 and "Free plan allows 3 saved models" in r.json()["detail"]
