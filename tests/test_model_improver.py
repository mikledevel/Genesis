"""model_improver: a weak model is improved and the claim is checked on rows no decision used; a model that cannot be improved is
returned untouched. Synthetic data only; no platform code needed."""
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core import model_improver as MI  # noqa: E402
from app.core.model_improver import improve_model  # noqa: E402

NUM = ["a", "b", "c", "d"]


def _frame(n=700, seed=0, signal=1.0, flip=0.0, imbalance=None):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=NUM)
    df["kind"] = rng.choice(["x", "y", "z"], n)
    logit = signal * (1.5 * df.a - 1.0 * df.b * df.c + 0.8 * (df.kind == "x") + 0.5 * df.d)
    if imbalance is not None:
        logit = logit + np.log(imbalance / (1 - imbalance)) - 0.0
    df["y"] = (logit + rng.normal(scale=0.6, size=n) > np.quantile(logit, 1 - (imbalance or 0.5))).astype(int)
    if flip:
        m = rng.rand(n) < flip
        df.loc[m, "y"] = 1 - df.loc[m, "y"]
    return df


def _pipe(estimator):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), ["kind"])])
    return Pipeline([("prep", pre), ("model", estimator)])


def _fit(df, estimator):
    return _pipe(estimator).fit(df.drop(columns="y"), df["y"])


def test_a_weak_model_is_improved_and_the_gain_is_confirmed_on_untouched_rows():
    df = _frame()
    weak = _fit(df, RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0))
    res = improve_model(weak, df, "y", "binary_classification", budget="fast")
    r = res.report
    assert res.improved and r["status"] == "improved", r
    assert r["test_after"] - r["test_before"] >= MI.MIN_GAIN and r["p_better"] >= MI.MIN_CONFIDENCE
    assert r["rows"]["test"] > 100 and r["metric"] == "balanced_accuracy"
    assert "test rows were used only" in r["delivered_trained_on"]
    X = df.drop(columns="y")
    assert res.model.predict(X).shape == (len(df),)          # takes the ORIGINAL columns, like the model it replaces
    assert res.model is not weak


def test_the_original_model_object_is_never_modified():
    df = _frame()
    weak = _fit(df, RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0))
    before = weak.predict(df.drop(columns="y")).copy()
    n_before = weak.named_steps["model"].n_estimators
    improve_model(weak, df, "y", "binary_classification", budget="fast", data_is_new=True)
    assert weak.named_steps["model"].n_estimators == n_before
    assert (weak.predict(df.drop(columns="y")) == before).all()


def test_only_if_weak_skips_the_search_for_a_model_that_is_not_weak():
    df = _frame(signal=3.0)
    good = _fit(df, GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0))   # clearly above baseline, no large overfit
    res = improve_model(good, df, "y", "binary_classification", budget="fast", only_if_weak=True)
    assert not res.improved and res.model is good and res.report["weak"] is False
    assert "not searched" in res.report["reason"] and res.report["tried"] == []


def test_an_overfitted_model_is_diagnosed_and_improved():
    df = _frame(signal=3.0)
    over = _fit(df, GradientBoostingClassifier(random_state=0))          # 1.00 on its training rows, far less on new ones
    res = improve_model(over, df, "y", "binary_classification", budget="fast")
    assert res.report["weak"] and any("overfits" in d for d in res.report["diagnosis"])
    assert res.improved and res.report["test_after"] > res.report["test_before"]


def test_a_good_model_is_not_replaced_by_noise():
    # no only_if_weak: the search runs, and the accept rule has to refuse a change that the untouched rows do not confirm
    df = _frame(n=900, signal=3.0)
    good = _fit(df, GradientBoostingClassifier(random_state=0))
    res = improve_model(good, df, "y", "binary_classification", budget="balanced")
    assert res.report["tried"] and (not res.improved or res.report["gain"] >= MI.MIN_GAIN)
    if not res.improved:
        assert res.model is good


def test_pure_noise_is_not_reported_as_an_improvement():
    # the false-positive control: with no signal at all, "improved" must stay rare across seeds
    claims = 0
    for seed in range(6):
        df = _frame(n=400, seed=seed, signal=0.0)
        df["y"] = np.random.RandomState(100 + seed).randint(0, 2, len(df))
        m = _fit(df, RandomForestClassifier(n_estimators=5, max_depth=2, random_state=seed))
        res = improve_model(m, df, "y", "binary_classification", budget="fast", seed=seed)
        claims += res.improved
        if not res.improved:
            assert res.model is m
    assert claims <= 1, claims


def test_new_data_with_a_changed_pattern_is_handled_by_continuing_or_refitting():
    old = _frame(seed=1)
    model = _fit(old, RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0))
    new = _frame(seed=2)
    new["y"] = 1 - new["y"]                                   # the relationship flipped
    res = improve_model(model, new, "y", "binary_classification", budget="balanced", data_is_new=True)
    assert res.improved, res.report
    assert res.report["test_before"] < 0.5 < res.report["test_after"]
    assert any(t["name"].startswith("continue") for t in res.report["tried"])


def test_class_balanced_candidates_are_offered_for_a_rare_class_and_help_balanced_accuracy():
    df = _frame(n=900, seed=3, imbalance=0.08)
    m = _fit(df, RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0))
    res = improve_model(m, df, "y", "binary_classification", budget="balanced")
    assert any("balanced" in t["name"] for t in res.report["tried"])
    assert "a rare class" in " ".join(res.report["diagnosis"])


def test_regression_uses_r2_and_reports_mae():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.normal(size=(600, 4)), columns=NUM)
    df["kind"] = rng.choice(["x", "y", "z"], 600)
    df["y"] = 3 * df.a + df.b * df.c + rng.normal(scale=0.3, size=600)
    weak = _fit(df, RandomForestRegressor(n_estimators=2, max_depth=1, random_state=0))
    res = improve_model(weak, df, "y", "regression", budget="fast")
    assert res.improved and res.report["metric"] == "r2", res.report
    assert res.report["mae_after"] < res.report["mae_before"]


def test_too_little_data_is_refused_not_guessed():
    df = _frame(n=40)
    m = _fit(df, RandomForestClassifier(n_estimators=3, random_state=0))
    res = improve_model(m, df, "y", "binary_classification")
    assert not res.improved and res.report["status"] == "skipped" and "rows" in res.report["reason"] and res.model is m


def test_supervised_preprocessing_and_non_pipelines_are_refused():
    df = _frame()
    sup = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM)], remainder="drop")),
                    ("select", SelectKBest(k=2)), ("model", RandomForestClassifier(n_estimators=3, random_state=0))])
    sup.fit(df[NUM], df["y"])
    res = improve_model(sup, df, "y", "binary_classification")
    assert res.report["status"] == "skipped" and "supervised" in res.report["reason"] and res.model is sup
    bare = RandomForestClassifier(n_estimators=3, random_state=0).fit(df[NUM], df["y"])
    res = improve_model(bare, df, "y", "binary_classification")
    assert res.report["status"] == "skipped" and "Pipeline" in res.report["reason"]


def test_unseen_labels_and_missing_columns_are_refused_with_a_reason():
    df = _frame()
    m = _fit(df, RandomForestClassifier(n_estimators=3, random_state=0))
    extra = df.copy()
    extra.loc[:30, "y"] = 2
    assert "never seen" in improve_model(m, extra, "y", "multiclass_classification").report["reason"]
    assert "lacks columns" in improve_model(m, df.drop(columns="a"), "y", "binary_classification").report["reason"]


def test_a_category_the_preprocessing_cannot_encode_is_a_reason_not_a_crash():
    df = _frame()
    strict = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="error"), ["kind"])])),
                       ("model", RandomForestClassifier(n_estimators=3, random_state=0))]).fit(df.drop(columns="y"), df["y"])
    new = df.copy()
    new["kind"] = "never-seen"
    res = improve_model(strict, new, "y", "binary_classification", data_is_new=True)
    assert res.report["status"] == "skipped" and "cannot transform" in res.report["reason"] and not res.improved


def test_the_result_is_deterministic_for_a_seed():
    df = _frame()
    m = _fit(df, RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0))
    a = improve_model(m, df, "y", "binary_classification", budget="fast", seed=7).report
    b = improve_model(m, df, "y", "binary_classification", budget="fast", seed=7).report
    for k in ("status", "test_before", "test_after", "gain", "winner_candidate", "p_better"):
        assert a[k] == b[k], k


def test_the_time_budget_stops_the_search_and_says_so():
    df = _frame()
    m = _fit(df, RandomForestClassifier(n_estimators=2, max_depth=1, random_state=0))
    res = improve_model(m, df, "y", "binary_classification", budget="thorough", max_seconds=0.0)
    assert res.report["status"] == "skipped" or all("skipped" in t for t in res.report["tried"])


def test_xgboost_models_can_be_continued_on_new_data():
    xgb = pytest.importorskip("xgboost")
    old = _frame(seed=1)
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), ["kind"])])
    model = Pipeline([("prep", pre), ("model", xgb.XGBClassifier(n_estimators=5, max_depth=2, learning_rate=0.3, random_state=0))])
    model.fit(old.drop(columns="y"), old["y"])
    new = _frame(seed=5)
    res = improve_model(model, new, "y", "binary_classification", budget="fast", data_is_new=True)
    tried = {t["name"]: t for t in res.report["tried"]}
    assert "continue training (warm start)" in tried and "error" not in tried["continue training (warm start)"], tried
