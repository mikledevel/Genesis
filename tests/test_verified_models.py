"""create_model: the platform prepares the data, hides a test set, and measures the model itself.
The LLM is replaced by fixed scripts; they are REALLY executed in the sandbox (dev subprocess fallback here)."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from unittest.mock import patch
from app.core.codegen.ai_generator import AICodeGenerator
from app.core.engine import verification as V
from app.core.engine.registry import ModelRegistry
from app.core.engine.predict_utils import build_model_input
from app.core.model_builder import ModelBuilder

PREAMBLE = '''
import pandas as pd, numpy as np, pickle, json, os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
{checks}
df = pd.read_csv("train.csv")
target = "{target}"
X, y = df.drop(columns=[target]), df[target]
num = X.select_dtypes("number").columns.tolist(); cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("n", SimpleImputer(strategy="median"), num),
                         ("c", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="missing")),
                                         ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
'''
TAIL = '''
os.makedirs("models", exist_ok=True)
pickle.dump(pipe, open("models/best_model.pkl", "wb"))
print("METRICS_JSON:", json.dumps({{"accuracy": {claimed}, "task": "binary_classification", "model_type": "logistic", "target": target, "features": list(X.columns)}}))
print("PIPELINE_SUCCESS")
'''


def honest(target="churn", checks="", claimed="round(float(cv), 4)"):
    return (PREAMBLE + '''
pipe = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))])
cv = cross_val_score(pipe, X, y, cv=3).mean()
pipe.fit(X, y)
''' + TAIL).format(target=target, checks=checks, claimed=claimed)


def liar(target="churn"):
    return (PREAMBLE + '''
pipe = Pipeline([("pre", pre), ("m", DummyClassifier(strategy="most_frequent"))])
pipe.fit(X, y)
''' + TAIL).format(target=target, checks="", claimed="0.999")


def data(n=400, seed=0, extra=None):
    rng = np.random.RandomState(seed)
    plan = rng.choice(["basic", "pro"], n); age = rng.randint(18, 70, n)
    churn = ((plan == "basic") ^ (rng.rand(n) < 0.12)).astype(int)
    df = pd.DataFrame({"age": age, "plan": plan, "churn": churn})
    for k, v in (extra or {}).items():
        df[k] = v(df) if callable(v) else v
    return df


@pytest.fixture
def csv(tmp_path):
    def make(df, name="d.csv"):
        p = tmp_path / name; df.to_csv(p, index=False); return str(p)
    return make


def build(script, path, **kw):
    with patch.object(AICodeGenerator, "generate", return_value=script):
        return AICodeGenerator().build("predict churn", dataset_path=path, target_column="churn", **kw)


# --------------------------------------------------------------------------- data plan
def test_hidden_test_set_never_overlaps_training_and_is_reproducible():
    df = data(500).assign(rid=lambda d: np.random.RandomState(1).permutation(500).astype(float))
    p1, p2 = V.prepare_data(df, "churn"), V.prepare_data(df, "churn")
    assert len(p1.holdout) == 100 and len(p1.train) == 400
    assert set(p1.train["rid"]).isdisjoint(set(p1.holdout["rid"]))
    assert p1.holdout.equals(p2.holdout) and p1.verifiable
    assert abs(p1.holdout["churn"].mean() - df["churn"].mean()) < 0.03            # stratified


def test_plan_removes_traps_before_the_script_ever_sees_them():
    rng = np.random.RandomState(2); base = data(500)
    df = base.assign(row_id=np.arange(500), outcome_code=lambda d: d.churn * 10 + rng.rand(500))
    plan = V.prepare_data(df, "churn")
    assert "row_id" not in plan.train.columns and "outcome_code" not in plan.train.columns and "outcome_code" not in plan.holdout.columns
    assert [c["kind"] for c in plan.checks] == ["id", "leak"]
    kept = V.prepare_data(df, "churn", keep_columns=["outcome_code"])
    assert "outcome_code" in kept.train.columns
    dup = pd.concat([weak := pd.DataFrame({"a": np.random.RandomState(3).randn(300), "b": np.random.RandomState(4).randn(300), "churn": np.random.RandomState(5).randint(0, 2, 300)})] * 2).reset_index(drop=True)
    assert V.prepare_data(dup, "churn").checks[0]["kind"] == "duplicates" and V.prepare_data(dup, "churn").n_rows == 300


def test_tiny_data_cannot_be_verified():
    plan = V.prepare_data(data(8), "churn")
    assert not plan.verifiable


def test_infer_task():
    assert V.infer_task(pd.Series([0, 1] * 20)) == "binary_classification"
    assert V.infer_task(pd.Series(["a", "b", "c"] * 20)) == "multiclass_classification"
    assert V.infer_task(pd.Series(np.random.RandomState(0).rand(100))) == "regression"


# --------------------------------------------------------------------------- the sandbox only gets train.csv
def test_honest_script_is_verified_on_rows_it_never_saw(csv):
    path = csv(data(400))
    check = 'assert not os.path.exists("datasets") and os.path.exists("train.csv"); assert len(pd.read_csv("train.csv")) == 320'
    r = build(honest(checks=check), path)
    assert r["success"], r["log"]
    assert r["verified"] and r["verification"]["contract"] == "raw_dataframe"
    m = r["verification"]["metrics"]
    assert m["verified"] is True and m["n_test"] == 80 and m["accuracy_test"] > 0.8
    assert m["accuracy_test_ci"][0] <= m["accuracy_test"] <= m["accuracy_test_ci"][1]
    assert m["trust"]["level"] in ("medium", "high") and "baseline_accuracy_test" in m
    assert r["metrics"]["features"] == ["age", "plan"] and r["metrics"]["accuracy_test"] == m["accuracy_test"]    # script's own keys survive


def test_the_old_behaviour_a_host_path_in_the_script_really_failed():
    """Documents the bug this fixes: a relative dataset path does not exist inside the sandbox working directory."""
    gen = AICodeGenerator(); gen._sandbox_backend = "subprocess"
    wd = tempfile.mkdtemp()
    r = gen.execute('import pandas as pd\npd.read_csv("datasets/whatever.csv")\n', wd)
    assert not r["success"] and "FileNotFoundError" in r["stderr"]


def test_a_script_that_lies_about_its_score_is_caught(csv):
    r = build(liar(), csv(data(400)))
    assert r["success"] and r["verified"]
    m = r["verification"]["metrics"]
    assert m["self_reported"]["accuracy"] == 0.999 and m["accuracy_test"] < 0.7
    trust = m["trust"]
    assert trust["level"] == "low" and any("завышена" in t for t in trust["reasons"]) and any("наивного" in t for t in trust["reasons"])


def test_leaked_and_id_columns_are_not_even_in_train_csv(csv):
    rng = np.random.RandomState(7)
    df = data(500).assign(row_id=np.arange(500), outcome_code=lambda d: d.churn * 10 + rng.rand(500))
    check = 'assert "row_id" not in pd.read_csv("train.csv").columns and "outcome_code" not in pd.read_csv("train.csv").columns'
    r = build(honest(checks=check), csv(df))
    assert r["success"], r["log"]
    assert [c["kind"] for c in r["verification"]["metrics"]["data_checks"]] == ["id", "leak"]
    assert 0.8 < r["verification"]["metrics"]["accuracy_test"] < 0.95                     # honest, not the fake 1.0


# --------------------------------------------------------------------------- failures are fed back to the LLM
def test_encoded_labels_are_rejected_and_the_loop_recovers(csv):
    encoded = (PREAMBLE + '''
from sklearn.preprocessing import LabelEncoder
y2 = LabelEncoder().fit_transform(y)
class_names = ["a", "b"]
pipe = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))]); pipe.fit(X, y.map({{0: "no", 1: "yes"}}))
cv = 0.9
''' + TAIL).format(target="churn", checks="", claimed="0.9")
    with patch.object(AICodeGenerator, "generate", return_value=encoded), patch.object(AICodeGenerator, "fix", return_value=honest()) as fix:
        r = AICodeGenerator().build("predict churn", dataset_path=csv(data(400)), target_column="churn")
    assert fix.called and "ORIGINAL target labels" in " ".join(str(a) for a in fix.call_args.args)
    assert r["success"] and r["iterations"] == 2 and r["verified"]


def test_a_model_that_needs_manually_encoded_input_is_rejected_with_a_helpful_error(csv):
    manual = '''
import pandas as pd, pickle, json, os
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("train.csv"); y = df["churn"]
X = pd.get_dummies(df.drop(columns=["churn"]), dtype=float)
m = LogisticRegression(max_iter=500).fit(X.values, y)
os.makedirs("models", exist_ok=True); pickle.dump(m, open("models/best_model.pkl", "wb"))
print("METRICS_JSON:", json.dumps({"accuracy": 0.9, "task": "binary_classification", "model_type": "logistic", "target": "churn", "features": ["age", "plan"]}))
print("PIPELINE_SUCCESS")
'''
    with patch.object(AICodeGenerator, "generate", return_value=manual), patch.object(AICodeGenerator, "fix", return_value=honest()) as fix:
        r = AICodeGenerator().build("predict churn", dataset_path=csv(data(400)), target_column="churn")
    assert fix.called
    error_text = " ".join(str(a) for a in fix.call_args.args)
    assert "features" in error_text and "ColumnTransformer" in error_text
    assert r["success"] and r["iterations"] == 2


def test_features_that_are_not_columns_of_train_csv_are_reported(csv):
    ghost = honest().replace('"features": list(X.columns)', '"features": ["age", "plan", "ghost"]')
    with patch.object(AICodeGenerator, "generate", return_value=ghost), patch.object(AICodeGenerator, "fix", return_value=honest()) as fix:
        r = AICodeGenerator().build("x", dataset_path=csv(data(400)), target_column="churn")
    assert "ghost" in " ".join(str(a) for a in fix.call_args.args) and r["success"]


def test_numeric_only_legacy_models_are_still_accepted(csv):
    legacy = '''
import pandas as pd, pickle, json, os
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("train.csv"); X = df[["age"]]; y = df["churn"]
class_ok = LogisticRegression().fit(X.values, y)
os.makedirs("models", exist_ok=True); pickle.dump(class_ok, open("models/best_model.pkl", "wb"))
print("METRICS_JSON:", json.dumps({"accuracy": 0.5, "task": "binary_classification", "model_type": "logistic", "target": "churn", "features": ["age"]}))
print("PIPELINE_SUCCESS")
'''
    r = build(legacy, csv(data(400)))
    assert r["success"] and r["verified"] and r["verification"]["contract"] in ("raw_dataframe", "numeric_vector")


def test_unreadable_dataset_falls_back_to_unverified_mode_honestly(tmp_path):
    p = tmp_path / "broken.csv"; p.write_text("a,b\n1,2\n")                   # 1 row: not verifiable
    r = build(honest().replace("train.csv", str(p)), str(p))
    assert r["verified"] is False or r["verification"] is None


# --------------------------------------------------------------------------- registration + agent
def test_registered_verified_model_takes_raw_rows(csv, tmp_path):
    r = build(honest(), csv(data(400)))
    reg = ModelRegistry(storage_dir=str(tmp_path / "reg"))
    mid = ModelBuilder().register(r["saved_model_path"], r["metrics"], "u1", verification=r["verification"], registry=reg)
    rec = reg.get_record(mid)
    assert rec["metadata"]["verified"] and rec["metadata"]["input_format"] == "raw_dataframe"
    assert rec["metadata"]["input_schema"] == {"numeric": ["age"], "categorical": ["plan"], "datetime": [], "text": []}
    assert rec["features"] == ["age", "plan"] and rec["task"] == "binary_classification" and rec["metadata"]["trust"]["level"]
    pred = reg.get_model(mid).predict(build_model_input(rec, {"age": "30", "plan": "basic"}))
    assert len(pred) == 1


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path); os.makedirs("datasets")
    a = GenesisAgent(registry=ModelRegistry(storage_dir=str(tmp_path / "r")), db=GenesisDB(db_path=str(tmp_path / "t.db")))
    u = a.db.create_user("u1@test.io", "x", "u1"); a.db.set_user_plan(u["id"], "pro")   # custom code is a Pro feature
    a.current_user_id = u["id"]; a.groq_key = ""      # no network: the planner falls back to its rules
    return a


def test_agent_create_model_reports_verified_numbers_and_registers(agent):
    data(400).to_csv("datasets/c.csv", index=False)
    agent.current_dataset = "datasets/c.csv"
    with patch.object(AICodeGenerator, "generate", return_value=honest()):
        r = agent._do_create_model("build me a neural network for churn", {"target_column": "churn"})
    assert r.success and r.data["verified"] and r.data["model_id"]
    assert "verified" in r.message and "never saw" in r.message and "Reliability" in r.message and "95% CI" in r.message
    m = r.data["metrics"]
    assert m["trust"]["level"] in ("medium", "high") and "accuracy_test_ci" in m and m["n_test"] == 80
    rec = agent.registry.get_record(r.data["model_id"])
    assert rec["metadata"]["input_format"] == "raw_dataframe" and rec["features"] == ["age", "plan"]


def test_agent_create_model_asks_for_the_target_instead_of_guessing(agent):
    rng = np.random.RandomState(9)
    pd.DataFrame({"a": rng.randn(200), "b": rng.randn(200), "z9": rng.randn(200), "w": rng.randn(200)}).to_csv("datasets/amb.csv", index=False)
    agent.current_dataset = "datasets/amb.csv"
    with patch.object(AICodeGenerator, "generate", return_value=honest()) as gen:
        r = agent._do_create_model("build me a neural network", {})
    assert r.success and r.data["needs_target"] and not gen.called                  # no LLM call, no sandbox run
    data(300).to_csv("datasets/c.csv", index=False); agent.current_dataset = "datasets/c.csv"
    with patch.object(AICodeGenerator, "generate", return_value=honest()) as gen:
        bad = agent._do_create_model("build a neural network", {"target_column": "revenue"})
    assert not bad.success and "no column" in bad.message.lower() and not gen.called


def test_agent_plan_has_data_and_verify_steps(agent):
    from app.core import progress
    events = []; progress.set_callback(events.append)
    data(400).to_csv("datasets/c.csv", index=False); agent.current_dataset = "datasets/c.csv"
    try:
        with patch.object(AICodeGenerator, "generate", return_value=honest()):
            agent._do_create_model("build a neural network", {"target_column": "churn"})
    finally:
        progress.set_callback(None)
    plan = next(e for e in events if e["type"] == "plan" and e["title"] == "Building a model with AI")
    assert [s["id"] for s in plan["steps"]] == ["data", "build", "verify", "register"]
    done = [e["id"] for e in events if e["type"] == "step" and e["status"] == "done"]
    assert "data" in done and "verify" in done and "register" in done


def test_prompts_state_the_contract():
    g = AICodeGenerator.GENERATE_PROMPT
    assert "train.csv" in g and "HIDDEN TEST SET" in g and "ORIGINAL target labels" in g and "INSIDE the pipeline" in g or "INSIDE the Pipeline" in g
    assert "train.csv" in AICodeGenerator.FIX_PROMPT
