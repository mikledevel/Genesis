"""GenesisAgent._do_improve_model: the glue between the agent, the registry and app/core/model_improver.py, with a fake registry and profiler
(the real GenesisAgent is used). Also: _do_retrain refuses a path outside the allowed folders, like train_model does.

Needs the complete project (app/core/engine, app/db): nothing of the platform is stubbed."""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from app.core.agent import GenesisAgent  # noqa: E402

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

NUM = ["a", "b", "c", "d"]


def _frame(n=700, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=NUM)
    df["kind"] = rng.choice(["x", "y", "z"], n)
    logit = 1.5 * df.a - 1.0 * df.b * df.c + 0.8 * (df.kind == "x") + 0.5 * df.d
    df["y"] = (logit + rng.normal(scale=0.6, size=n) > 0).astype(int)
    return df


def _weak_model(df):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), ["kind"])])
    m = Pipeline([("prep", pre), ("model", RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0))])
    return m.fit(df.drop(columns="y"), df["y"])


class FakeRegistry:
    def __init__(self, model, author="u1"):
        self.model = model
        self.record = {"model_id": "m1", "family_id": "fam", "task": "binary_classification", "target": "y", "model_type": "random_forest",
                       "features": NUM + ["kind"], "metadata": {"author": author, "input_schema": {"a": "float"}, "input_format": "table"}}
        self.registered = []

    def get_record(self, model_id):
        if model_id != "m1":
            raise KeyError(model_id)
        return self.record

    def get_model(self, model_id):
        return self.model

    def register(self, **kw):
        self.registered.append(kw)
        return "m2"

    def get_family_history(self, family_id):
        return [{"version": 2}, {"version": 1}]


class FakeProfiler:
    def __init__(self, df):
        self.df = None
        self._df = df

    def run_full_profile(self, path):
        self.df = self._df


@pytest.fixture
def env(monkeypatch):
    """The real GenesisAgent and the real plan gates over a real (temporary) database; only the registry and the profiler are fakes."""
    from app.db.database import GenesisDB
    from tests.pro_user import make_pro_user
    workdir = tempfile.mkdtemp()
    monkeypatch.chdir(workdir)
    os.makedirs("datasets")
    df = _frame()
    df.to_csv("datasets/data.csv", index=False)
    db = GenesisDB()
    make_pro_user(db, "u1")                                  # a paying customer: server training is a Pro feature
    reg = FakeRegistry(_weak_model(df))
    agent = GenesisAgent(profiler=FakeProfiler(df), trainer=MagicMock(), registry=reg, db=db)
    agent.current_user_id = "u1"
    return agent, reg


def test_an_improvement_is_registered_as_a_new_version_with_lineage_and_the_old_metadata(env):
    agent, reg = env
    r = agent._do_improve_model("m1", "datasets/data.csv", {"budget": "fast"})
    assert r.success and r.data["improved"] is True, r.message
    assert "improved" in r.message and "v2" in r.message and "m2" in r.message
    (kw,) = reg.registered
    assert kw["family_id"] == "fam" and kw["parent_model_id"] == "m1" and kw["target"] == "y" and kw["features"] == NUM + ["kind"]
    assert kw["metadata"]["improved_from"] == "m1" and kw["metadata"]["input_schema"] == {"a": "float"}     # predict-time metadata survives
    assert kw["metadata"]["improve_report"]["status"] == "improved"
    assert kw["metrics"]["balanced_accuracy_test"] > kw["metrics"]["test_before_improvement"]
    assert kw["metrics"]["source"] == "model_improver"


def test_when_nothing_helps_nothing_is_registered_and_the_answer_says_so(env, monkeypatch):
    agent, reg = env
    from app.core import model_improver as MI
    monkeypatch.setattr(MI, "improve_model", lambda *a, **k: MI.ImproveResult(
        model=a[0], improved=False, report={"status": "unchanged", "reason": "nothing beat the reference on validation", "metric": "balanced_accuracy",
                                            "tried": [{"name": "x", "validation": 0.5}], "diagnosis": ["a rare class"]}))
    r = agent._do_improve_model("m1", "datasets/data.csv", {})
    assert r.success and r.data["improved"] is False and "NOT changed" in r.message and "nothing beat" in r.message
    assert reg.registered == []


def test_only_the_models_author_may_improve_it(env):
    from tests.pro_user import make_pro_user
    agent, reg = env
    make_pro_user(agent.db, "someone-else")                  # a paying customer too: the refusal must come from ownership, not from the plan
    agent.current_user_id = "someone-else"
    r = agent._do_improve_model("m1", "datasets/data.csv", {})
    assert not r.success and "permission" in r.message and reg.registered == []


def test_a_path_outside_the_allowed_folders_is_refused_for_improve_and_for_retrain(env):
    agent, reg = env
    for fn in (agent._do_improve_model, agent._do_retrain):
        r = fn("m1", "/etc/passwd", {})
        assert not r.success and "Access denied" in r.message, (fn.__name__, r.message)
    assert reg.registered == []


def test_missing_model_and_missing_path(env):
    agent, reg = env
    assert "model_id" in agent._do_improve_model("", "datasets/data.csv", {}).message
    assert "path" in agent._do_improve_model("m1", None, {}).message
    assert "not found" in agent._do_improve_model("nope", "datasets/data.csv", {}).message


def test_a_free_account_is_refused_by_the_real_plan_gate(env):
    from tests.pro_user import make_pro_user
    agent, reg = env
    make_pro_user(agent.db, "free1", plan="free")
    agent.current_user_id = "free1"
    r = agent._do_improve_model("m1", "datasets/data.csv", {})
    assert not r.success and "not part of the Free plan" in r.message and reg.registered == []


def _stub_improve(monkeypatch, seen):
    from app.core import model_improver as MI

    def fake(model, df, target, task, **kw):
        seen.append(kw)
        return MI.ImproveResult(model=model, improved=False, report={"status": "unchanged", "reason": "stub", "metric": "balanced_accuracy", "tried": []})
    monkeypatch.setattr(MI, "improve_model", fake)


def test_the_monthly_server_job_allowance_counts_each_start_and_then_refuses(env, monkeypatch):
    from app.core import plans
    agent, reg = env
    _stub_improve(monkeypatch, [])
    monkeypatch.setitem(plans._CATALOG["pro"]["limits"], "server_jobs_per_month", 2)
    assert agent._do_improve_model("m1", "datasets/data.csv", {}).success
    assert agent._do_improve_model("m1", "datasets/data.csv", {}).success
    r = agent._do_improve_model("m1", "datasets/data.csv", {})
    assert not r.success and "2 server training jobs per month" in r.message and "free" in r.message
    assert agent.db.count_feature_use_this_month("u1", plans.SERVER_JOB) == 2          # the refused start is not counted


def test_a_job_refused_by_another_gate_is_not_counted(env, monkeypatch):
    from app.core import plans
    agent, reg = env
    monkeypatch.setitem(plans._CATALOG["pro"]["limits"], "max_models", 0)              # no saved-model slots: the 'model' gate refuses
    r = agent._do_improve_model("m1", "datasets/data.csv", {})
    assert not r.success
    assert agent.db.count_feature_use_this_month("u1", plans.SERVER_JOB) == 0


def test_the_search_depth_is_lowered_to_what_the_plan_allows_and_the_user_is_told(env, monkeypatch):
    agent, reg = env
    seen = []
    _stub_improve(monkeypatch, seen)
    r = agent._do_improve_model("m1", "datasets/data.csv", {"budget": "thorough"})
    assert seen[-1]["budget"] == "balanced" and "lowered" in r.message and "higher limit" in r.message
    agent._do_improve_model("m1", "datasets/data.csv", {"budget": "fast"})
    assert seen[-1]["budget"] == "fast"


def test_the_tool_is_listed_in_the_prompt_and_dispatched():
    src = open(os.path.join(os.path.dirname(__file__), "..", "app", "core", "agent.py"), encoding="utf-8-sig").read()
    assert "train_local, improve_model" in src and "improve_model: the user wants to IMPROVE" in src
    assert "self._do_improve_model(a.get(\"model_id\", \"\")" in src
