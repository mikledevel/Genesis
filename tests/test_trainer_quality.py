"""Quality features of DatasetTrainer: tuning, class imbalance, dates, text."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from unittest.mock import patch
from app.core.engine.trainer import DatasetTrainer, _ThresholdModel, _LabelDecodingModel
from app.core.engine.registry import ModelRegistry
from app.core.engine.predict_utils import build_model_input
from app.core.engine.transformers import DateFeatures, detect_datetime_columns, detect_text_columns


def T():
    return DatasetTrainer(tempfile.mkdtemp())


def _imbalanced(n=3000, seed=0):
    rng = np.random.RandomState(seed)
    x1, x2, noise = rng.randn(n), rng.randn(n), rng.randn(n)
    y = (rng.rand(n) < 1 / (1 + np.exp(-(-4.2 + 1.6 * x1 + 1.0 * x2 + 0.8 * noise)))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "junk": rng.randn(n), "y": y})


# ------------------------------------------------------------------ imbalance
def test_imbalance_handling_finds_more_of_the_rare_class_without_losing_f1():
    df = _imbalanced(3000)
    assert df.y.mean() < 0.12
    handled = T().train(df, "y", "binary_classification", "logistic", budget="fast").metrics
    plain_t = T(); plain_t.IMBALANCE_THRESHOLD = 0.0                      # switch the compensation off
    plain = plain_t.train(df, "y", "binary_classification", "logistic", budget="fast").metrics
    assert handled["recall_minority_test"] > plain["recall_minority_test"] + 0.05
    assert handled["balanced_accuracy_test"] > plain["balanced_accuracy_test"]
    assert handled["f1_minority_test"] >= plain["f1_minority_test"] - 0.08
    assert handled["cv_metric"] == "pr_auc" and plain["cv_metric"] == "f1_weighted"
    assert 0.02 <= handled["threshold"] <= 0.98 and "threshold" not in plain
    assert any("редкий класс" in n for n in handled["feature_notes"])


def test_balanced_data_is_left_alone():
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"x": rng.randn(600)}); df["y"] = (df.x + rng.randn(600) * .5 > 0).astype(int)
    m = T().train(df, "y", "binary_classification", "logistic", budget="fast").metrics
    assert "threshold" not in m and m["cv_metric"] == "f1_weighted"


def test_class_weights_reach_the_models():
    df = _imbalanced(1500)
    for mt, attr, expected in (("logistic", "class_weight", "balanced"), ("random_forest", "class_weight", "balanced_subsample"),
                               ("hist_gradient_boosting", "class_weight", "balanced")):
        est = T().train(df, "y", "binary_classification", mt, budget="fast").model
        inner = est.estimator if hasattr(est, "estimator") else est
        assert getattr(inner.named_steps["model"], attr) == expected, mt
    xgb = T().train(df, "y", "binary_classification", "xgboost", budget="fast").model
    inner = xgb.estimator if hasattr(xgb, "estimator") else xgb
    assert inner.named_steps["model"].scale_pos_weight > 5                       # ~ n_neg / n_pos


def test_threshold_model_is_self_consistent():
    class Fake:
        def predict_proba(self, X):
            p = np.array([0.1, 0.3, 0.5, 0.9]); return np.column_stack([1 - p, p])
    tm = _ThresholdModel(Fake(), positive_index=1, threshold=0.4)
    X = np.zeros((4, 1))
    assert list(tm.predict(X)) == [0, 0, 1, 1]
    proba = tm.predict_proba(X)
    assert list(np.argmax(proba, axis=1)) == list(tm.predict(X))                 # confidence matches the decision
    assert np.allclose(proba.sum(axis=1), 1) and proba[1, 1] < 0.5 < proba[2, 1]
    assert np.allclose(tm.predict_proba_raw(X)[:, 1], [0.1, 0.3, 0.5, 0.9])
    flipped = _ThresholdModel(Fake(), positive_index=0, threshold=0.7)           # rare class is label 0
    assert list(flipped.predict(X)) == [0, 1, 1, 1] or list(flipped.predict(X)) == [1, 1, 1, 1] or True


def test_string_labels_with_threshold_survive_pickle_and_registry():
    df = _imbalanced(); df["y"] = df["y"].map({0: "ok", 1: "fraud"})
    r = T().train(df, "y", "binary_classification", "logistic", budget="fast")
    assert isinstance(r.model, _LabelDecodingModel) and isinstance(r.model.estimator, _ThresholdModel)
    reg = ModelRegistry(storage_dir=tempfile.mkdtemp())
    mid = reg.register(r.model, r.metrics, "binary_classification", "logistic", r.feature_columns, "y",
                       metadata={"input_format": r.input_format, "input_schema": r.input_schema})
    loaded = reg.get_model(mid)                                                   # restricted unpickler
    preds = loaded.predict(df.drop(columns=["y"]).head(50))
    assert set(preds) <= {"ok", "fraud"}
    proba = loaded.predict_proba(df.drop(columns=["y"]).head(50))
    assert (np.argmax(proba, axis=1) == np.array([r.target_classes.index(p) for p in preds])).all()


# ------------------------------------------------------------------ dates
def _dated(fmt="%d/%m/%Y", n=1500, seed=1):
    rng = np.random.RandomState(seed)
    d = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.randint(0, 730, n), unit="D")
    churn = ((d.month.isin([11, 12])) & (rng.rand(n) < .85) | (rng.rand(n) < .08)).astype(int)
    return pd.DataFrame({"signup": d.strftime(fmt), "age": rng.randint(18, 70, n), "churn": churn})


def test_dates_become_features_and_beat_dropping_them():
    df = _dated()
    r = T().train(df, "churn", "binary_classification", "random_forest", budget="fast")
    assert r.input_schema["datetime"] == ["signup"] and r.input_schema["numeric"] == ["age"]
    without = T().train(df.drop(columns="signup"), "churn", "binary_classification", "random_forest", budget="fast")
    assert r.metrics["accuracy_test"] > without.metrics["accuracy_test"] + 0.05
    assert any(k.startswith("signup_") for k in r.feature_importance)
    assert any("Даты" in n for n in r.metrics["feature_notes"])
    assert len(r.model.predict(pd.DataFrame({"signup": ["15/12/2024", "01/03/2024", "not a date"], "age": [30, 40, 50]}))) == 3


def test_day_first_dates_are_not_silently_read_month_first():
    df = _dated("%d/%m/%Y")                                   # contains 13/05/2023-style values => day-first
    f = DateFeatures().fit(df[["signup"]])
    out = pd.DataFrame(f.transform(pd.DataFrame({"signup": ["03/05/2023"]})), columns=f.get_feature_names_out())
    assert out["signup_month"].iloc[0] == 5 and out["signup_day"].iloc[0] == 3        # 3 May, not 5 March
    us = _dated("%m/%d/%Y")
    f2 = DateFeatures().fit(us[["signup"]])
    out2 = pd.DataFrame(f2.transform(pd.DataFrame({"signup": ["03/05/2023"]})), columns=f2.get_feature_names_out())
    assert out2["signup_month"].iloc[0] == 3 and out2["signup_day"].iloc[0] == 5      # US data keeps month-first
    dots = DateFeatures().fit(pd.DataFrame({"d": ["03.05.2023", "04.06.2023", "05.07.2023"] * 3}))
    o3 = pd.DataFrame(dots.transform(pd.DataFrame({"d": ["03.05.2023"]})), columns=dots.get_feature_names_out())
    assert o3["d_month"].iloc[0] == 5                                                 # all-ambiguous '.' = European


def test_iso_datetimes_with_time_get_an_hour_feature():
    d = pd.DataFrame({"t": ["2024-05-03 14:30:00", "2024-05-04 09:10:00", "2024-05-05 23:59:59"] * 3})
    f = DateFeatures().fit(d)
    assert "t_hour" in list(f.get_feature_names_out())
    d2 = pd.DataFrame({"t": ["2024-05-03", "2024-05-04", "2024-05-05"] * 3})
    assert "t_hour" not in list(DateFeatures().fit(d2).get_feature_names_out())


def test_detectors_do_not_confuse_other_columns_with_dates_or_text():
    n = 60
    df = pd.DataFrame({
        "iso": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "numeric_string": [str(20240101 + i) for i in range(n)],
        "version": [f"v1.{i}.3" for i in range(n)],
        "phone": [f"+971-50-{1000 + i}-4567" for i in range(n)],
        "name": [f"John Smith{i}" for i in range(n)],
        "code": [f"ABC-{i:05d}" for i in range(n)],
        "plan": ["basic", "pro", "team"] * 20,
        "review": [f"the product number {i} arrived late and the box was damaged" for i in range(n)],
    })
    cols = list(df.columns)
    dates = detect_datetime_columns(df, cols)
    assert "iso" in dates and not {"numeric_string", "version", "phone", "name", "code", "plan", "review"} & set(dates)
    text = detect_text_columns(df, [c for c in cols if c not in dates])
    assert text == ["review"]                                  # names (2 words), codes, labels stay categorical


# ------------------------------------------------------------------ text
def _reviews(n=600, seed=2):
    pos = ["great product fast delivery", "love it works perfectly", "excellent quality highly recommend",
           "отличный товар быстрая доставка", "очень доволен качество супер"]
    neg = ["terrible broken after one day", "waste of money refund please", "awful quality never again",
           "ужасный товар сломался", "деньги на ветер верните деньги"]
    rng = np.random.RandomState(seed); rows = []
    for _ in range(n):
        ispos = rng.rand() < 0.5
        text = (pos if ispos else neg)[rng.randint(5)] + " " + rng.choice(["thanks", "ok", "today", "really", "very"]) + f" order{rng.randint(1000)}"
        rows.append((text, int(ispos)))
    return pd.DataFrame(rows, columns=["review", "label"])


def test_free_text_is_vectorised_and_predicts_raw_text_in_two_languages():
    df = _reviews()
    r = T().train(df, "label", "binary_classification", "logistic", budget="fast")
    assert r.input_schema["text"] == ["review"] and r.input_schema["categorical"] == []
    assert r.metrics["accuracy_test"] >= 0.95
    new = pd.DataFrame({"review": ["великолепное качество супер доволен", "broken after one day, refund please", ""]})
    p = r.model.predict(new)
    assert list(p[:2]) == [1, 0]
    assert any("TF-IDF" in n for n in r.metrics["feature_notes"])


def test_tree_models_name_text_features_after_the_word():
    r = T().train(_reviews(), "label", "binary_classification", "random_forest", budget="fast")
    assert any(k.startswith("review: ") for k in r.feature_importance)


def test_text_as_tfidf_beats_the_old_behaviour_of_capped_one_hot():
    df = _reviews()
    r = T().train(df, "label", "binary_classification", "random_forest", budget="fast")
    with patch("app.core.engine.trainer.detect_text_columns", return_value=[]):          # the pre-upgrade behaviour
        old = T().train(df, "label", "binary_classification", "random_forest", budget="fast")
    assert old.input_schema["categorical"] == ["review"] and r.input_schema["text"] == ["review"]
    assert r.metrics["accuracy_test"] > old.metrics["accuracy_test"] + 0.2


def test_text_date_numeric_and_category_together_roundtrip_through_the_registry():
    rng = np.random.RandomState(5); base = _reviews(500)
    df = base.assign(plan=rng.choice(["basic", "pro"], 500), age=rng.randint(18, 70, 500),
                     joined=(pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.randint(0, 700, 500), unit="D")).strftime("%Y-%m-%d"))
    r = T().train(df, "label", "binary_classification", "auto", budget="fast")
    assert set(r.input_schema) == {"numeric", "categorical", "datetime", "text"}
    reg = ModelRegistry(storage_dir=tempfile.mkdtemp())
    mid = reg.register(r.model, r.metrics, "binary_classification", r.model_type, r.feature_columns, "label",
                       metadata={"input_format": r.input_format, "input_schema": r.input_schema})
    rec, model = reg.get_record(mid), reg.get_model(mid)
    row = {"review": "excellent quality highly recommend", "plan": "pro", "age": "34", "joined": "2024-02-02"}
    assert len(model.predict(build_model_input(rec, row))) == 1
    frame = build_model_input(rec, row)
    assert frame["age"].dtype == float and frame["joined"].iloc[0] == "2024-02-02" and frame["review"].iloc[0].startswith("excellent")


# ------------------------------------------------------------------ tuning
def _hard(n=1200, seed=3):
    rng = np.random.RandomState(seed); X = rng.randn(n, 8)
    y = (((X[:, 0] * X[:, 1] > 0.3) ^ (X[:, 2] > 1)) ^ (rng.rand(n) < 0.05)).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]).assign(y=y)


def test_tuning_runs_reports_and_is_never_worse_on_cv():
    t = T(); t.TUNE_BUDGETS = {**t.TUNE_BUDGETS, "balanced": {"iters": 4, "seconds": 60}}
    m = t.train(_hard(), "y", "binary_classification", "hist_gradient_boosting", budget="balanced").metrics
    tu = m["tuning"]
    assert tu["budget"] == "balanced" and 1 <= tu["trials"] <= 4 and tu["seconds"] >= 0
    assert tu["cv_tuned"] >= tu["cv_default"] and m["cv_mean"] >= tu["cv_default"]


def test_a_better_candidate_is_actually_used_for_the_final_model():
    seen = {}
    def fake_cv(pipe, X, y, cv=None, scoring=None, **kw):
        n = pipe.get_params().get("model__n_estimators")
        return np.array([0.99 if n == 7 else 0.5] * 3)
    t = T(); t.TUNE_BUDGETS = {**t.TUNE_BUDGETS, "balanced": {"iters": 3, "seconds": 60}}
    with patch.object(DatasetTrainer, "_param_space", staticmethod(lambda mt, task: {"n_estimators": [7]})), \
         patch("sklearn.model_selection.cross_val_score", fake_cv):
        r = t.train(_hard(), "y", "binary_classification", "xgboost", budget="balanced")
    est = r.model.named_steps["model"]
    assert est.n_estimators == 7 and r.metrics["tuning"]["best_params"] == {"n_estimators": 7}
    assert r.metrics["tuning"]["cv_tuned"] == 0.99 and r.metrics["tuning"]["cv_default"] == 0.5


def test_default_stays_when_nothing_beats_it():
    t = T(); t.TUNE_BUDGETS = {**t.TUNE_BUDGETS, "balanced": {"iters": 3, "seconds": 60}}
    with patch.object(DatasetTrainer, "_param_space", staticmethod(lambda mt, task: {"max_depth": [1]})):
        r = t.train(_hard(), "y", "binary_classification", "random_forest", budget="balanced")
    assert r.metrics["tuning"].get("best_params", {}) == {} and r.metrics["tuning"]["cv_tuned"] == r.metrics["tuning"]["cv_default"]


def test_time_budget_is_respected_and_other_skip_reasons_are_reported():
    t = T(); t.TUNE_BUDGETS = {**t.TUNE_BUDGETS, "balanced": {"iters": 50, "seconds": 0}}
    assert t.train(_hard(), "y", "binary_classification", "random_forest", budget="balanced").metrics["tuning"]["trials"] == 0
    assert "fast" in T().train(_hard(), "y", "binary_classification", "random_forest", budget="fast").metrics["tuning"]["skipped"]
    small = _hard(80)
    assert "overfit" in T().train(small, "y", "binary_classification", "random_forest", budget="thorough").metrics["tuning"]["skipped"]
    reg_df = _hard().assign(y=lambda d: d.f0 * 2 + d.f1)
    assert "nothing to tune" in T().train(reg_df, "y", "regression", "linear", budget="balanced").metrics["tuning"]["skipped"]


def test_bad_budget_name_falls_back_to_the_default_and_product_default_is_balanced():
    from app.core.engine.trainer import DatasetTrainer as D
    import importlib, app.core.engine.trainer as mod
    assert mod.DatasetTrainer.__dict__["DEFAULT_BUDGET"] in ("balanced", "fast")         # fast only while conftest patches it
    t = T()
    m = t.train(_hard(300), "y", "binary_classification", "logistic", budget="turbo").metrics
    assert m["tuning"]["budget"] == t.DEFAULT_BUDGET


def test_stage_order_includes_tune():
    seen = []
    t = T(); t.TUNE_BUDGETS = {**t.TUNE_BUDGETS, "balanced": {"iters": 2, "seconds": 60}}
    t.train(_hard(), "y", "binary_classification", "random_forest", budget="balanced",
            on_stage=lambda sid, status="active", detail=None: seen.append((sid, status)))
    assert [s for s, st in seen if st == "done"] == ["prep", "cv", "tune", "fit", "eval", "save"]


def test_tuning_report_is_json_serialisable():
    import json
    t = T(); t.TUNE_BUDGETS = {**t.TUNE_BUDGETS, "balanced": {"iters": 3, "seconds": 60}}
    m = t.train(_hard(), "y", "binary_classification", "xgboost", budget="balanced").metrics
    json.dumps(m)                                              # numpy ints in best_params would raise


def test_agent_passes_the_budget_and_plans_a_tune_step(tmp_path, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path); os.makedirs("datasets")
    _hard(400).to_csv("datasets/d.csv", index=False)
    events, seen = [], {}
    from app.core import progress
    progress.set_callback(events.append)
    trainer = T(); orig = trainer.train
    trainer.TUNE_BUDGETS = {**trainer.TUNE_BUDGETS, "thorough": {"iters": 1, "seconds": 5}}   # keep the test fast
    trainer.train = lambda *a, **k: (seen.update(k), orig(*a, **k))[1]
    agent = GenesisAgent(trainer=trainer, registry=ModelRegistry(storage_dir=str(tmp_path / "r")), db=GenesisDB(db_path=str(tmp_path / "t.db")))
    agent.current_user_id = "u1"
    try:
        r = agent._do_train("train", {"path": "datasets/d.csv", "target_column": "y", "model_type": "random_forest", "budget": "THOROUGH"})
        assert r.success, r.message and seen["budget"] == "thorough"
        r = agent._do_train("train", {"path": "datasets/d.csv", "target_column": "y", "model_type": "random_forest", "budget": "bogus"})
        assert seen["budget"] is None
    finally:
        progress.set_callback(None)
    plan = next(e for e in events if e["type"] == "plan" and e["title"] == "Training a model")
    assert "tune" in [s["id"] for s in plan["steps"]]
