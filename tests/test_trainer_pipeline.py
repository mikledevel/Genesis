"""Leak-free pipeline trainer, raw-input prediction, chat history and optional tool chaining."""
import os, sys, tempfile, types
from unittest.mock import patch
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry
from app.core.engine.predict_utils import build_model_input, build_model_input_from_list


def _df(n=400, seed=0):
    rng = np.random.RandomState(seed)
    plan = rng.choice(["basic", "pro", "team"], n)
    age = rng.randint(18, 70, n).astype(float)
    age[rng.rand(n) < 0.1] = np.nan                      # missing numerics
    churn = ((plan == "basic") & (rng.rand(n) < 0.7) | (rng.rand(n) < 0.1)).astype(int)
    return pd.DataFrame({"age": age, "plan": plan, "churn": churn})


@pytest.fixture
def t():
    return DatasetTrainer(model_dir=tempfile.mkdtemp())


def test_model_takes_raw_columns_and_handles_missing_and_unseen_category(t):
    r = t.train(_df(), "churn", "binary_classification", "random_forest")
    assert r.feature_columns == ["age", "plan"]
    assert r.input_schema == {"numeric": ["age"], "categorical": ["plan"]}
    new = pd.DataFrame({"age": [np.nan, 40.0], "plan": ["enterprise", "pro"]})   # NaN + unseen category
    assert len(r.model.predict(new)) == 2


def test_preprocessing_is_fitted_on_train_only(t):
    """A category present only in the TEST part must not exist in the fitted encoder."""
    df = _df(300)
    df.loc[df.index[-1], "plan"] = "only_in_test"
    with patch("sklearn.model_selection.train_test_split",
               side_effect=lambda X, y, **kw: (X.iloc[:-1], X.iloc[-1:], y.iloc[:-1], y.iloc[-1:])):
        r = t.train(df, "churn", "binary_classification", "random_forest")
    enc = r.model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    assert "only_in_test" not in list(enc.categories_[0])


def test_cv_runs_on_training_part_only(t):
    seen = {}
    orig = DatasetTrainer._cv_score
    def spy(self, pipeline, X, y, task, cv_folds):
        seen["n"] = len(X); return orig(self, pipeline, X, y, task, cv_folds)
    with patch.object(DatasetTrainer, "_cv_score", spy):
        t.train(_df(400), "churn", "binary_classification", "logistic")
    assert seen["n"] == 320     # 80% of 400, never the full frame


def test_metrics_include_baseline_and_auc(t):
    m = t.train(_df(), "churn", "binary_classification", "auto").metrics
    assert {"baseline_accuracy_test", "roc_auc_test", "cv_metric", "automl_chosen"} <= set(m)


def test_non_contiguous_numeric_labels_do_not_crash_xgboost(t):
    df = _df(); df["churn"] = df["churn"].map({0: 1, 1: 2})
    r = t.train(df, "churn", "binary_classification", "xgboost")
    assert set(r.model.predict(df.drop(columns=["churn"]))) <= {1, 2}


def test_warns_when_not_better_than_baseline(t):
    df = pd.DataFrame({"x": np.random.RandomState(1).randn(400),
                       "churn": np.random.RandomState(5).randint(0, 2, 400)})        # target is pure noise
    assert any("наивного" in w for w in t.train(df, "churn", "binary_classification", "logistic").metrics.get("warnings", []))


def test_registry_roundtrip_through_safe_pickle_and_predict_helpers(t):
    r = t.train(_df(), "churn", "binary_classification", "auto")
    reg = ModelRegistry(storage_dir=tempfile.mkdtemp())
    mid = reg.register(r.model, r.metrics, "binary_classification", r.model_type, r.feature_columns, "churn",
                       metadata={"input_format": r.input_format, "input_schema": r.input_schema})
    rec, model = reg.get_record(mid), reg.get_model(mid)          # loads via the restricted unpickler
    assert len(model.predict(build_model_input(rec, {"age": "35", "plan": "pro"}))) == 1
    assert len(model.predict(build_model_input_from_list(rec, [35, "basic"]))) == 1
    with pytest.raises(KeyError):
        build_model_input(rec, {"age": 35})
    with pytest.raises(ValueError):
        build_model_input(rec, {"age": "not a number", "plan": "pro"})


def test_legacy_record_still_uses_numeric_vector():
    rec = {"features": ["a", "b"], "metadata": {}}
    assert build_model_input(rec, {"a": "1", "b": 2}).shape == (1, 2)


# ------------------------------------------------------------------ agent behaviour
@pytest.fixture
def agent(tmp_path):
    from app.core.agent import GenesisAgent
    from app.db.database import GenesisDB
    return GenesisAgent(registry=ModelRegistry(storage_dir=str(tmp_path / "reg")), db=GenesisDB(db_path=str(tmp_path / "t.db")))


def test_llm_receives_previous_turns_of_the_same_conversation(agent):
    calls = []
    def fake(msg, history=None):
        calls.append((msg, history)); return {"tool": None, "message": f"reply{len(calls)}"}
    with patch.object(agent, "_call_llm", side_effect=fake):
        agent.process("первое сообщение", "c1", user_id="u1")
        agent.process("второе", "c1", user_id="u1")
    assert calls[0][1] == []
    assert [h["content"] for h in calls[1][1]] == ["первое сообщение", "reply1"]


def test_fact_extraction_no_longer_turns_verbs_into_names(agent):
    agent._extract_facts("я хочу обучить модель", "u1")
    agent._extract_facts("Меня зовут Миша", "u1")
    facts = {f["predicate"]: f["object"] for f in agent.db.get_facts("u1")}
    assert facts.get("name") == "Миша" and facts.get("goal") == "обучить модель"


def test_single_step_by_default_and_chaining_when_enabled(agent):
    from app.config import settings
    from app.core.agent import AgentResponse
    def script():
        return [{"tool": "analyze_dataset", "args": {"path": "a.csv"}, "message": "m"},
                {"tool": "train_model", "args": {"path": "a.csv"}, "message": "m"},
                {"tool": None, "message": "готово"}]
    ran = []
    def fake_dispatch(t, *a, **k):
        ran.append(t); return AgentResponse(message=f"ran {t}")
    seq = script()
    with patch.object(agent, "_dispatch_tool", side_effect=fake_dispatch), \
         patch.object(agent, "_call_llm", side_effect=lambda *a, **k: seq.pop(0)):
        agent.process("проанализируй и обучи", "c2", user_id="u1")
    assert ran == ["analyze_dataset"] and len(seq) == 2            # default: single step, one LLM call
    ran.clear(); seq = script()
    with patch.object(settings, "agent_max_steps", 3), patch.object(agent, "_dispatch_tool", side_effect=fake_dispatch), \
         patch.object(agent, "_call_llm", side_effect=lambda *a, **k: seq.pop(0)):
        r = agent.process("проанализируй и обучи", "c3", user_id="u1")
    assert ran == ["analyze_dataset", "train_model"] and seq == []
    assert r.message == "ran analyze_dataset\n\nran train_model\n\nготово"


def test_bot_predict_accepts_categorical_strings(agent, t):
    from app.core.agent_store import AgentStore
    from app.core.bot_builder import BotBuilder
    r = t.train(_df(), "churn", "binary_classification", "random_forest")
    mid = agent.registry.register(r.model, r.metrics, "binary_classification", "random_forest", r.feature_columns, "churn",
                                  metadata={"author": "owner", "input_format": r.input_format, "input_schema": r.input_schema})
    bot = BotBuilder(user_id="owner").publish({"name": "T", "system_prompt": "x"}, author="owner", publish=False)
    AgentStore().update_agent(bot["id"], {"linked_model_id": mid}, "owner")
    resp = agent._do_bot_predict(bot["id"], {"features": {"age": 30, "plan": "basic"}})
    assert resp.success, resp.message
    assert "confidence" in resp.data
