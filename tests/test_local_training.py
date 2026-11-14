"""Server-side training is a paid feature (Pro / Business / owner); training on the user's OWN computer is free.

What must hold:
  * Free is refused on every server-training entry point (agent train / forecast / retrain / create_model and the
    /api/model-builder/generate route) and the refusal shows the local alternative; Pro, Business and admins pass.
  * The platform only ever hands out text and a zip. It never runs anything on a user's machine, and the command it
    prints is shell-safe even for hostile file or column names.
  * The zip really works on its own: unpacked in an empty folder, with the repo NOT importable, it trains a model,
    writes model.pkl + metrics.json, and predicts on new rows.
"""
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import app.main  # noqa: E402,F401  (must be imported before chdir: it mounts ./static)
from fastapi.testclient import TestClient  # noqa: E402

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
    c.headers["x-forwarded-for"] = f"10.7.{_IP // 250}.{_IP % 250}"
    return c


def _db():
    from app.db.database import GenesisDB
    return GenesisDB()


def _register(client, email, password="secret12"):
    r = client.post("/api/auth/register", json={"email": email, "password": password, "name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["user"]["id"], {"Authorization": "Bearer " + r.json()["token"]}


def _agent(uid):
    from app.core.agent import GenesisAgent
    a = GenesisAgent()
    a.current_user_id = uid
    return a


# ------------------------------------------------------------------ the plan gate
def test_server_training_is_a_paid_feature_in_the_catalog():
    from app.core import plans
    assert plans.get_plan("free")["limits"]["server_training"] is False
    assert plans.get_plan("pro")["limits"]["server_training"] is True
    assert plans.get_plan("business")["limits"]["server_training"] is True
    assert plans.UNLIMITED_LIMITS["server_training"] is True
    assert plans.get_plan("nonsense")["limits"]["server_training"] is False      # unknown plan = Free


def test_free_is_refused_and_pointed_to_local_training(client):
    from app.core import plans
    uid, _h = _register(client, "free@x.io")
    msg = plans.gate_message(_db(), uid, "server_training")
    assert "not part of the Free plan" in msg and "Upgrade to Pro" in msg
    assert "/api/local-training/bundle" in msg and "train_local.py" in msg
    _db().set_user_plan(uid, "pro")
    assert plans.gate_message(_db(), uid, "server_training") is None
    _db().set_user_plan(uid, "business")
    assert plans.gate_message(_db(), uid, "server_training") is None
    assert plans.gate_message(_db(), "anonymous", "server_training") is None


def test_an_admin_is_never_refused(client, monkeypatch):
    from app.core import plans
    from app.config import settings
    monkeypatch.setattr(settings, "admin_emails", "owner@genesis.test")
    uid, h = _register(client, "owner@genesis.test")
    tok = _db().create_auth_token(uid, "verify_email", 3600)
    assert client.post("/api/auth/verify-email/confirm", json={"token": tok}, headers=h).status_code == 200
    assert plans.gate_message(_db(), uid, "server_training") is None


@pytest.mark.parametrize("call", [
    lambda a: a._do_train("train a model on x.csv"),
    lambda a: a._do_forecast("forecast sales"),
    lambda a: a._do_retrain("some-model", None),
    lambda a: a._do_create_model("build a churn model"),
])
def test_every_server_training_entry_point_refuses_free_and_lets_pro_through(client, call):
    uid, _h = _register(client, "u@x.io")
    res = call(_agent(uid))
    assert res.success is False and "Free plan" in res.message and "train_local.py" in res.message
    _db().set_user_plan(uid, "pro")
    res = call(_agent(uid))
    assert "Free plan" not in res.message            # it got past the plan gate (and fails later for lack of a dataset)


def test_the_model_builder_route_is_gated(client):
    uid, h = _register(client, "u@x.io")
    r = client.post("/api/model-builder/generate", json={"dataset_path": "nope.csv"}, headers=h)
    assert r.status_code == 403 and "not part of the Free plan" in r.json()["detail"]
    _db().set_user_plan(uid, "pro")
    r = client.post("/api/model-builder/generate", json={"dataset_path": "nope.csv"}, headers=h)
    assert r.status_code == 400                      # past the plan gate: the missing dataset is what stops it now


# ------------------------------------------------------------------ the command the user runs
def test_train_local_works_on_the_free_plan_and_runs_nothing(client, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("the platform must never execute a command")
    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(os, "system", _boom)
    uid, _h = _register(client, "free@x.io")
    res = _agent(uid)._do_train_local("train it myself", {"path": "datasets/churn.csv", "target_column": "churn", "budget": "fast"})
    assert res.success is True and res.data["bundle_url"] == "/api/local-training/bundle"
    assert "python3 train_local.py churn.csv --target churn --budget fast" in res.message
    assert "never run anything on your computer" in res.message


def test_train_local_is_a_tool_the_model_can_pick():
    from app.core.agent import GenesisAgent
    assert "train_local" in GenesisAgent.SYSTEM_PROMPT.split("Tools:")[1].split("\n")[0]
    assert "never" in GenesisAgent.SYSTEM_PROMPT.lower() and "run commands on the user's computer" in GenesisAgent.SYSTEM_PROMPT


@pytest.mark.parametrize("dataset,target", [
    ("a b; rm -rf ~.csv", "x$(id)"),
    ("data.csv\nrm -rf /", "t`whoami`"),
    ("it's.csv", "col'; echo hi #"),
    ("$(touch pwned).csv", "a && b"),
])
def test_the_command_is_shell_safe_for_hostile_names(dataset, target):
    import shlex
    from app.core.local_training import local_command
    cmd = local_command(dataset, target)
    parts = shlex.split(cmd)                          # what a shell would see: exactly one argument per value
    assert parts[:2] == ["python3", "train_local.py"] and parts[3] == "--target" and len(parts) == 5
    assert "\n" not in cmd and "\x00" not in cmd
    import re
    expected = re.sub(r"[\x00-\x1f\x7f]", "", os.path.basename(dataset)).strip() or "your_data.csv"
    assert parts[2] == expected                       # a name that ends in "/" has no file part: the neutral fallback
    assert parts[4] == re.sub(r"[\x00-\x1f\x7f]", "", target).strip()


def test_command_without_a_target_asks_for_one():
    from app.core.local_training import local_command
    assert local_command("a.csv") == "python3 train_local.py a.csv --target <column to predict>"
    assert local_command(None) .startswith("python3 train_local.py your_data.csv")


def test_command_endpoint_is_text_only(client):
    r = client.get("/api/local-training/command", params={"dataset": "sales.csv", "target": "revenue", "budget": "thorough"})
    assert r.status_code == 200
    assert r.json()["command"] == "python3 train_local.py sales.csv --target revenue --budget thorough"


# ------------------------------------------------------------------ the zip
def test_bundle_is_public_small_and_contains_only_the_trainer(client):
    r = client.get("/api/local-training/bundle")       # no login: it is open code
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert "genesis-local-trainer.zip" in r.headers["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    top = "genesis-local-trainer/"
    assert names == {top + n for n in (
        "train_local.py", "README.txt", "requirements-local.txt", "app/__init__.py", "app/core/__init__.py",
        "app/core/engine/__init__.py", "app/core/engine/trainer.py", "app/core/engine/transformers.py",
        "app/core/engine/predict_utils.py")}
    blob = b"".join(z.read(n) for n in names).lower()
    for secret in (b"groq_api_key", b"stripe", b"jwt_secret", b"resend", b"sk-"):
        assert secret not in blob
    assert len(r.content) < 200_000


def test_bundle_is_rebuilt_when_a_source_file_changes(tmp_path, monkeypatch):
    from app.core import local_training as lt
    first = lt.build_bundle()
    assert lt.build_bundle() is first                  # cached
    monkeypatch.setitem(lt._cache, "sig", None)
    assert lt.build_bundle() == first                  # same content when the sources did not change


def _synthetic(path, n=240):
    rng = np.random.default_rng(0)
    age = rng.integers(18, 70, n)
    plan = rng.choice(["basic", "plus", "pro"], n)
    spend = rng.normal(50, 15, n)
    y = ((spend + (plan == "basic") * 12 + rng.normal(0, 8, n)) > 58).astype(int)
    pd.DataFrame({"customer_id": range(n), "age": age, "plan": plan, "spend": spend, "churn": y}).to_csv(path, index=False)


def test_the_unpacked_bundle_trains_and_predicts_on_its_own(client, tmp_path):
    work = tmp_path / "user_machine"
    work.mkdir()
    zipfile.ZipFile(io.BytesIO(client.get("/api/local-training/bundle").content)).extractall(work)
    folder = work / "genesis-local-trainer"
    data = work / "customers.csv"
    _synthetic(data)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "GROQ_API_KEY")}
    run = lambda *a: subprocess.run([sys.executable, "-I", str(folder / "train_local.py"), *a], cwd=work,   # -I: repo is NOT importable
                                    capture_output=True, text=True, timeout=240, env=env)
    out = work / "my_model"
    r = run(str(data), "--target", "churn", "--budget", "fast", "--out", str(out))
    assert r.returncode == 0, r.stderr + r.stdout
    assert (out / "model.pkl").exists() and not (out / "_tmp").exists()
    summary = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert summary["target"] == "churn" and summary["task"] == "binary_classification"
    assert "customer_id" not in summary["feature_columns"]            # an ID column is dropped by the trainer
    new = work / "new_rows.csv"
    pd.read_csv(data).drop(columns=["churn"]).head(7).to_csv(new, index=False)
    r = run(str(new), "--predict", "--model", str(out / "model.pkl"))
    assert r.returncode == 0, r.stderr + r.stdout
    preds = pd.read_csv(work / "predictions.csv")
    assert len(preds) == 7 and "prediction" in preds.columns and set(preds["prediction"]) <= {0, 1}


def test_the_script_explains_a_wrong_target_instead_of_crashing(tmp_path):
    data = tmp_path / "d.csv"
    _synthetic(data, 60)
    root = os.path.join(os.path.dirname(__file__), "..")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "train_local.py"), str(data), "--target", "nope"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "There is no column 'nope'" in (r.stderr + r.stdout) and "churn" in (r.stderr + r.stdout)
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "train_local.py"), str(data)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "--target" in (r.stderr + r.stdout)
