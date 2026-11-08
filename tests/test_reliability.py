"""Reliability: target ranking, honest handling of ID / leakage / duplicate traps, trust verdict."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.transformers import looks_like_id


from tests.pro_user import make_pro_user  # noqa: E402


def T():
    return DatasetTrainer(tempfile.mkdtemp())


def profile(df, name="d.csv"):
    p = os.path.join(tempfile.mkdtemp(), name); df.to_csv(p, index=False)
    pr = DatasetProfiler(); pr.run_full_profile(p); return pr


def weak_bc():
    return load_breast_cancer(as_frame=True).frame[["mean radius", "mean texture", "mean smoothness", "target"]]


# --------------------------------------------------------------------------- profiler: which column?
@pytest.mark.parametrize("pos", ["last", "first", "middle"])
def test_named_target_is_found_wherever_it_sits(pos):
    df = load_breast_cancer(as_frame=True).frame
    cols = [c for c in df.columns if c != "target"]
    order = {"last": cols + ["target"], "first": ["target"] + cols, "middle": cols[:10] + ["target"] + cols[10:]}[pos]
    s = profile(df[order]).get_summary()["suggested_ml_task"]
    assert s["target_column"] == "target" and s["task"] == "binary_classification" and not s["needs_confirmation"]


def test_real_world_names_english_and_russian():
    rng = np.random.RandomState(0)
    for name, vals in (("churn", rng.randint(0, 2, 300)), ("Survived", rng.randint(0, 2, 300)), ("отток", rng.randint(0, 2, 300)),
                       ("price", rng.rand(300) * 1000), ("цена", rng.rand(300) * 1000)):
        df = pd.DataFrame({name: vals, "a": rng.randn(300), "b": rng.randn(300), "c": rng.randn(300)})
        assert profile(df).get_summary()["suggested_ml_task"]["target_column"] == name, name


def test_nothing_stands_out_means_ask_not_guess():
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"a": rng.randn(300), "b": rng.randn(300), "z9": rng.randn(300), "w": rng.randn(300)})
    s = profile(df).get_summary()["suggested_ml_task"]
    assert s["needs_confirmation"] and len(s["candidates"]) >= 2 and {"column", "task", "reason", "score"} <= set(s["candidates"][0])


def test_ids_dates_text_constants_and_empty_columns_are_never_candidates():
    n = 120; rng = np.random.RandomState(2)
    df = pd.DataFrame({"customer_id": np.arange(n), "signup_date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
                       "comment": [f"the order number {i} arrived late and was damaged" for i in range(n)],
                       "const": 1, "mostly_empty": [np.nan] * 100 + list(range(20)), "email": [f"u{i}@x.com" for i in range(n)],
                       "age": rng.randint(18, 70, n), "plan": rng.choice(["a", "b"], n)})
    cands = [c["column"] for c in profile(df).get_summary()["suggested_ml_task"]["candidates"]]
    assert not {"customer_id", "signup_date", "comment", "const", "mostly_empty", "email"} & set(cands)
    assert "plan" in cands


def test_an_explicit_target_always_gets_ITS_OWN_task():
    """Regression for a real bug: the task used to be derived from the guessed (last numeric) column."""
    rng = np.random.RandomState(3)
    df = pd.DataFrame({"churn": rng.randint(0, 2, 300), "age": rng.randint(18, 70, 300), "avg_value": rng.rand(300) * 500})
    pr = profile(df)
    assert pr.suggest_task("avg_value")[0] == "regression"
    task, target, _ = pr.suggest_task("churn")
    assert task == "binary_classification" and target == "churn" and pr.get_summary()["suggested_ml_task"]["needs_confirmation"] is False


def test_excel_files_can_be_profiled(tmp_path):
    p = tmp_path / "d.xlsx"; pd.DataFrame({"a": range(60), "churn": [0, 1] * 30}).to_excel(p, index=False)
    pr = DatasetProfiler(); pr.run_full_profile(str(p))
    assert pr.get_summary()["file_info"]["rows"] == 60


def test_openpyxl_is_a_declared_dependency():
    root = os.path.join(os.path.dirname(__file__), "..")
    assert "openpyxl" in open(os.path.join(root, "requirements.txt")).read()
    assert "openpyxl" in open(os.path.join(root, "requirements.lock.txt")).read()


# --------------------------------------------------------------------------- agent flows
@pytest.fixture
def agent(tmp_path, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.core.engine.registry import ModelRegistry
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path); os.makedirs("datasets")
    a = GenesisAgent(trainer=T(), registry=ModelRegistry(storage_dir=str(tmp_path / "r")), db=GenesisDB(db_path=str(tmp_path / "t.db")))
    a.current_user_id = "u1"; make_pro_user(a.db, "u1"); return a


def _churn_file(order=("churn", "age", "avg_value")):
    rng = np.random.RandomState(4); n = 400
    age = rng.randint(18, 70, n); avg = rng.rand(n) * 500
    churn = ((age < 30) ^ (rng.rand(n) < 0.15)).astype(int)
    pd.DataFrame({"churn": churn, "age": age, "avg_value": avg})[list(order)].to_csv("datasets/c.csv", index=False)


def test_on_a_file_path_is_not_mistaken_for_a_column_name(agent):
    _churn_file()
    r = agent._do_train("обучи модель на datasets/c.csv", {})
    assert r.success, r.message
    assert r.data["model_type"] and "datasets" not in r.data["feature_columns"] and r.data["task"] == "binary_classification"


def test_explicit_target_not_last_gets_the_right_task_end_to_end(agent):
    _churn_file(("churn", "age", "avg_value"))                     # last numeric column is avg_value (continuous)
    r = agent._do_train("train", {"path": "datasets/c.csv", "target_column": "churn"})
    assert r.success and r.data["task"] == "binary_classification" and "avg_value" in r.data["feature_columns"]


def test_target_matching_is_case_insensitive_and_unknown_names_are_reported(agent):
    _churn_file()
    assert agent._do_train("train", {"path": "datasets/c.csv", "target_column": "CHURN"}).success
    bad = agent._do_train("train", {"path": "datasets/c.csv", "target_column": "revenue"})
    assert not bad.success and "no column" in bad.message.lower() and "churn" in bad.message and "age" in bad.message


def test_ambiguous_file_asks_which_column_instead_of_training(agent):
    rng = np.random.RandomState(5)
    pd.DataFrame({"a": rng.randn(200), "b": rng.randn(200), "z9": rng.randn(200), "w": rng.randn(200)}).to_csv("datasets/amb.csv", index=False)
    r = agent._do_train("train a model on datasets/amb.csv", {"path": "datasets/amb.csv"})
    assert r.success and r.data["needs_target"] and r.data["path"] == "datasets/amb.csv" and len(r.data["target_candidates"]) >= 2
    assert "Which column" in r.message and agent.registry.list_models(limit=10) == []              # nothing was trained
    r2 = agent._do_train("train", {"path": "datasets/amb.csv", "target_column": "z9"})
    assert r2.success and r2.data["task"] == "regression" and len(agent.registry.list_models(limit=10)) == 1


def test_analyze_says_when_it_is_not_sure(agent):
    rng = np.random.RandomState(6)
    pd.DataFrame({"a": rng.randn(200), "b": rng.randn(200), "z9": rng.randn(200)}).to_csv("datasets/amb.csv", index=False)
    assert "not sure" in agent._do_analyze("datasets/amb.csv").message
    _churn_file()
    assert "not sure" not in agent._do_analyze("datasets/c.csv").message


def test_keep_columns_reaches_the_trainer(agent):
    rng = np.random.RandomState(7); n = 400; y = rng.randint(0, 2, n)
    pd.DataFrame({"x": rng.randn(n), "outcome_code": y * 10 + rng.rand(n), "churn": y}).to_csv("datasets/l.csv", index=False)
    r = agent._do_train("train", {"path": "datasets/l.csv", "target_column": "churn"})
    assert "outcome_code" not in r.data["feature_columns"]
    r = agent._do_train("train", {"path": "datasets/l.csv", "target_column": "churn", "keep_columns": ["outcome_code"]})
    assert "outcome_code" in r.data["feature_columns"]


# --------------------------------------------------------------------------- trainer: data traps
def test_leaked_column_is_excluded_and_the_score_stays_honest():
    clean = T().train(weak_bc(), "target", "binary_classification", "logistic", budget="fast").metrics
    leak = weak_bc().assign(outcome_code=lambda d: d.target * 10 + np.random.RandomState(0).rand(len(d)))
    r = T().train(leak, "target", "binary_classification", "logistic", budget="fast")
    assert "outcome_code" not in r.feature_columns
    assert abs(r.metrics["accuracy_test"] - clean["accuracy_test"]) < 0.03                 # not the fake 1.0
    kinds = [c["kind"] for c in r.metrics["data_checks"]]
    assert kinds == ["leak"] and "утечку" in r.metrics["data_checks"][0]["message"] and "оставь столбец outcome_code" in r.metrics["data_checks"][0]["message"]


def test_keep_columns_overrides_the_exclusion():
    leak = weak_bc().assign(outcome_code=lambda d: d.target * 10 + np.random.RandomState(0).rand(len(d)))
    r = T().train(leak, "target", "binary_classification", "logistic", budget="fast", keep_columns=["outcome_code"])
    assert "outcome_code" in r.feature_columns and r.metrics["accuracy_test"] > 0.99


def test_categorical_and_regression_leaks_are_found_too():
    rng = np.random.RandomState(1); n = 600
    y = rng.randint(0, 2, n)
    cat = pd.DataFrame({"x": rng.randn(n), "status_text": np.where(y == 1, "closed_lost", "closed_won"), "y": y})
    r = T().train(cat, "y", "binary_classification", "logistic", budget="fast")
    assert "status_text" not in r.feature_columns
    x = rng.randn(n); reg = pd.DataFrame({"x": x, "noise": rng.randn(n), "price_in_cents": (x * 3 + 1) * 100, "price": x * 3 + 1 + rng.randn(n) * 0.0})
    r = T().train(reg, "price", "regression", "linear", budget="fast")
    assert "price_in_cents" not in r.feature_columns or "x" not in r.feature_columns   # at least the duplicate-by-scale is caught


def test_a_lone_perfect_feature_is_kept_but_trust_is_low():
    rng = np.random.RandomState(2); y = rng.randint(0, 2, 400)
    x = np.where(y == 1, 1 + rng.rand(400), -1 - rng.rand(400))               # perfectly separable, with a gap
    r = T().train(pd.DataFrame({"x": x, "y": y}), "y", "binary_classification", "logistic", budget="fast")
    assert r.feature_columns == ["x"] and r.metrics["trust"]["level"] == "low"
    assert any("Единственный признак" in t for t in r.metrics["trust"]["reasons"])


@pytest.mark.parametrize("loader,task", [(load_iris, "multiclass_classification"), (load_breast_cancer, "binary_classification"),
                                         (load_diabetes, "regression")])
def test_no_false_positives_on_honest_strong_features(loader, task):
    r = T().train(loader(as_frame=True).frame, "target", task, "logistic" if "class" in task else "linear", budget="fast")
    assert not r.metrics.get("data_checks")


def test_id_columns_are_dropped_but_measurements_are_not():
    srt = weak_bc().sort_values("target").reset_index(drop=True).assign(row_id=lambda d: np.arange(len(d)))
    r = T().train(srt, "target", "binary_classification", "logistic", budget="fast")
    assert "row_id" not in r.feature_columns and r.metrics["data_checks"][0]["kind"] == "id"
    rng = np.random.RandomState(3); n = 300
    meas = pd.DataFrame({"avg_order_value": rng.rand(n) * 1000, "k": rng.randn(n), "y": rng.randint(0, 2, n)})   # unique floats = a measurement
    assert "avg_order_value" in T().train(meas, "y", "binary_classification", "logistic", budget="fast").feature_columns


def test_looks_like_id_rules():
    n = 120
    assert looks_like_id(pd.Series(np.arange(n)), "x")                                         # counting up
    assert looks_like_id(pd.Series(np.random.RandomState(0).permutation(n) + 1000), "customer_id")   # name says id
    assert looks_like_id(pd.Series([f"u{i}" for i in range(n)]), "token")                      # unique strings
    assert not looks_like_id(pd.Series(np.random.RandomState(0).rand(n)), "score")             # unique float measurement
    assert not looks_like_id(pd.Series(np.random.RandomState(0).permutation(n) + 1000), "age_days")   # shuffled ints, innocent name
    assert not looks_like_id(pd.Series(np.arange(10)), "id")                                   # too few rows to say


def test_surprising_duplicates_are_removed_natural_ones_are_kept():
    dup = pd.concat([weak_bc(), weak_bc()]).reset_index(drop=True)
    clean = T().train(weak_bc(), "target", "binary_classification", "logistic", budget="fast").metrics
    r = T().train(dup, "target", "binary_classification", "logistic", budget="fast")
    assert r.metrics["data_checks"][0]["kind"] == "duplicates" and r.metrics["data_checks"][0]["rows"] == 569
    assert abs(r.metrics["accuracy_test"] - clean["accuracy_test"]) < 0.01 and r.training_info["n_rows"] == 569
    rng = np.random.RandomState(4)                                                   # yes/no survey: repeats are normal
    survey = pd.DataFrame({f"q{i}": rng.randint(0, 2, 500) for i in range(4)}).assign(y=lambda d: (d.q0 | d.q1).astype(int) ^ (rng.rand(500) < .1))
    assert not T().train(survey, "y", "binary_classification", "logistic", budget="fast").metrics.get("data_checks")


# --------------------------------------------------------------------------- trust verdict + intervals
def test_wilson_interval_matches_known_values():
    lo, hi = DatasetTrainer._wilson(95, 100)
    assert abs(lo - 0.8884) < 0.002 and abs(hi - 0.9784) < 0.002
    assert DatasetTrainer._wilson(0, 0) == [0.0, 1.0]


def test_metrics_carry_intervals_that_contain_the_point_estimate():
    m = T().train(weak_bc(), "target", "binary_classification", "logistic", budget="fast").metrics
    lo, hi = m["accuracy_test_ci"]; assert lo <= m["accuracy_test"] <= hi and m["n_test"] == 114
    reg = T().train(load_diabetes(as_frame=True).frame, "target", "regression", "linear", budget="fast").metrics
    lo, hi = reg["r2_test_ci"]; assert lo < reg["r2_test"] < hi


def test_trust_levels_follow_the_evidence():
    big = load_breast_cancer(as_frame=True).frame
    assert T().train(big, "target", "binary_classification", "logistic", budget="fast").metrics["trust"]["level"] == "high"
    small = T().train(load_iris(as_frame=True).frame, "target", "multiclass_classification", "logistic", budget="fast").metrics["trust"]
    assert small["level"] == "medium" and any("Тестовая выборка мала" in r for r in small["reasons"])
    rng = np.random.RandomState(5)
    noise = pd.DataFrame({"a": rng.randn(600), "b": rng.randn(600), "y": rng.randint(0, 2, 600)})
    t = T().train(noise, "y", "binary_classification", "logistic", budget="fast").metrics["trust"]
    assert t["level"] == "low" and any("наивного" in r for r in t["reasons"])


def test_rare_class_with_few_test_examples_is_flagged():
    rng = np.random.RandomState(6); n = 400; x = rng.randn(n); y = (rng.rand(n) < 0.04).astype(int)
    m = T().train(pd.DataFrame({"x": x + y * 1.5, "z": rng.randn(n), "y": y}), "y", "binary_classification", "logistic", budget="fast").metrics
    assert any("редкого класса" in r for r in m["trust"]["reasons"]) and m["trust"]["level"] in ("medium", "low")
