"""improve_model through the REAL stack: DatasetTrainer trains a model, ModelRegistry stores it, GenesisAgent._do_improve_model improves it.
Nothing is faked (only the dataset is synthetic). Isolated in a temporary working directory."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.agent import GenesisAgent  # noqa: E402
from app.core.engine.predict_utils import build_model_input  # noqa: E402
from app.core.engine.registry import ModelRegistry  # noqa: E402
from app.core.engine.trainer import DatasetTrainer  # noqa: E402
from tests.pro_user import make_pro_user  # noqa: E402


def _frame(seed, n=900, flip=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"age": rng.randint(20, 65, n), "salary": rng.randint(30000, 120000, n), "city": rng.choice(["a", "b", "c"], n)})
    logit = (df.age - 40) / 10 - (df.salary - 70000) / 30000 + (df.city == "a") * 0.8
    df["churn"] = (logit + rng.normal(scale=1.0, size=n) > 0).astype(int)
    if flip:
        df["churn"] = 1 - df["churn"]                      # the relationship changed: the old model is now confidently wrong
    return df


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    work = tmp_path_factory.mktemp("improve_real")
    old = os.getcwd()
    os.chdir(work)
    try:
        os.makedirs("datasets")
        _frame(0).to_csv("datasets/d1.csv", index=False)
        _frame(1, flip=True).to_csv("datasets/d2_flipped.csv", index=False)
        registry = ModelRegistry(storage_dir=os.path.join(str(work), "registry"))
        agent = GenesisAgent(registry=registry, trainer=DatasetTrainer(model_dir=os.path.join(str(work), "models")))
        agent.current_user_id = "u1"
        make_pro_user(agent.db, "u1")
        r = agent._do_train("train datasets/d1.csv", {"target": "churn"})
        assert r.success, r.message
        yield agent, registry, r.data["model_id"]
    finally:
        os.chdir(old)


def test_the_model_the_trainer_stores_is_something_the_improver_can_work_with(stack):
    from app.core import model_improver as MI
    agent, registry, mid = stack
    prep, est_name, est = MI._unwrap(registry.get_model(mid))          # raises Unsupported if the stored model had a shape it cannot handle
    assert est_name == "model" and prep is not None


def test_new_data_with_a_changed_pattern_yields_a_registered_better_version(stack):
    from app.core import plans
    agent, registry, mid = stack
    before_pred = registry.get_model(mid).predict(_frame(5).drop(columns="churn")).copy()
    resp = agent._do_improve_model(mid, "datasets/d2_flipped.csv", {"new_data": True, "budget": "fast"})
    assert resp.success and resp.data["improved"] is True, resp.message
    new_id = resp.data["model_id"]
    old, new = registry.get_record(mid), registry.get_record(new_id)
    assert new["version"] == 2 and new["family_id"] == mid and new["parent_model_id"] == mid
    assert new["task"] == old["task"] and new["target"] == old["target"] and new["features"] == old["features"]
    assert new["metadata"]["improved_from"] == mid and new["metadata"]["input_schema"] == old["metadata"]["input_schema"]
    assert new["metadata"]["input_format"] == old["metadata"]["input_format"]
    # the same metric keys the trainer writes, so the version sorts and displays like any other
    for k in ("accuracy_test", "f1_test", "balanced_accuracy_test", "baseline_accuracy_test", "roc_auc_test", "n_test"):
        assert k in new["metrics"], k
    assert new["metrics"]["source"] == "model_improver" and new["metrics"]["accuracy_test"] > new["metrics"]["baseline_accuracy_test"]
    assert new["metrics"]["test_before_improvement"] < 0.5 < new["metrics"]["balanced_accuracy_test"]
    # it takes part in find_best (which only considers records that have the metric). Which of the two wins is NOT asserted: the numbers
    # of two versions come from different data and different test rows; "test_before_improvement" is the like-for-like comparison.
    assert registry.find_best(new["task"], "accuracy_test")["model_id"] in (mid, new_id)
    # the old version is untouched and still answers exactly as before
    assert (registry.get_model(mid).predict(_frame(5).drop(columns="churn")) == before_pred).all()
    # a user's raw values go through the same input builder every predict path uses
    row = build_model_input(new, {"age": 30, "salary": 50000, "city": "a"})
    assert registry.get_model(new_id).predict(row).shape == (1,)
    assert [h["version"] for h in registry.get_family_history(mid)] == [2, 1]
    assert agent.db.count_feature_use_this_month("u1", plans.SERVER_JOB) >= 1                # the run was counted as a server job


def test_when_the_model_cannot_be_improved_no_version_is_registered(stack):
    agent, registry, mid = stack
    family = len(registry.get_family_history(mid))
    resp = agent._do_improve_model(mid, "datasets/d1.csv", {"budget": "fast"})              # the data the model was trained on: nothing to learn
    assert resp.success
    if resp.data["improved"]:
        assert resp.data["report"]["test_after"] > resp.data["report"]["test_before"]
    else:
        assert "NOT changed" in resp.message and len(registry.get_family_history(mid)) == family
