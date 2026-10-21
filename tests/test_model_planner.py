"""LLM as planner: validated plans, rule fallback, standard vs custom routing, end-to-end through the agent."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))        # to reuse the fixed scripts of test_verified_models
import numpy as np, pandas as pd, pytest
from unittest.mock import MagicMock, patch
from app.core.model_planner import (ModelPlanner, needs_custom_code, rules_plan, sanitize_plan, describe_columns, PLAN_PROMPT)
from app.core.engine.registry import ModelRegistry
from app.core.codegen.ai_generator import AICodeGenerator

COLS = ["age", "plan", "zip", "customer_name", "churn"]


# ------------------------------------------------------------------------ routing rules
@pytest.mark.parametrize("text", ["построй нейросеть для оттока", "train an LSTM on this", "build a neural network", "кластеризуй клиентов",
                                  "detect anomalies in sales", "write the code for it",
                                  "recommend products", "classify images"])
def test_non_standard_requests_are_custom(text):
    assert needs_custom_code(text)


@pytest.mark.parametrize("text", ["построй модель для прогноза оттока клиентов", "predict churn", "train a model to predict price",
                                  "обучи модель, предскажи цену квартиры", "build a model, use random forest", "быстро обучи модель на churn"])
def test_ordinary_requests_stay_standard(text):
    assert needs_custom_code(text) is None and rules_plan(text)["route"] == "standard"


@pytest.mark.parametrize("text", ["forecast daily sales for the next 30 days", "спрогнозируй выручку на 3 месяца вперёд", "time series forecast of demand",
                                  "прогнозировать продажи на следующий месяц", "what will traffic be next 2 weeks", "прогноз временного ряда"])
def test_future_of_a_series_is_a_forecast_not_custom_code(text):
    from app.core.model_planner import is_forecast_request
    assert is_forecast_request(text) and needs_custom_code(text) is None and rules_plan(text)["route"] == "forecast"


@pytest.mark.parametrize("text", ["построй модель для прогноза оттока клиентов", "predict churn", "прогноз цены квартиры по параметрам", "predict price"])
def test_row_level_prediction_is_not_mistaken_for_a_forecast(text):
    from app.core.model_planner import is_forecast_request
    assert not is_forecast_request(text) and rules_plan(text)["route"] == "standard"


def test_rules_read_budget_and_algorithm_from_plain_words():
    assert rules_plan("quick model please")["budget"] == "fast" and rules_plan("быстро")["budget"] == "fast"
    assert rules_plan("I want the best possible accuracy, don't mind waiting")["budget"] == "thorough"
    assert rules_plan("максимально точную модель")["budget"] == "thorough" and rules_plan("predict churn")["budget"] == "balanced"
    assert rules_plan("use a random forest")["model_type"] == "random_forest" and rules_plan("используй xgboost")["model_type"] == "xgboost"
    assert rules_plan("логистическая регрессия")["model_type"] == "logistic" and rules_plan("predict churn")["model_type"] == "auto"


# ------------------------------------------------------------------------ never trust the plan
def test_sanitize_drops_inventions_and_protects_the_target():
    raw = {"route": "standard", "reason": "x", "model_type": "quantum_forest", "budget": "ludicrous",
           "exclude_columns": ["ZIP", "churn", "nonexistent", "zip", "Customer_Name"], "summary": "s" * 1000}
    p = sanitize_plan(raw, COLS, "churn", "predict churn")
    assert p["model_type"] == "auto" and p["budget"] == "balanced"
    assert p["exclude_columns"] == ["zip", "customer_name"]                    # real names, deduped, target never
    assert len(p["summary"]) <= 300 and p["source"] == "llm"


def test_sanitize_will_not_exclude_everything():
    p = sanitize_plan({"exclude_columns": ["age", "plan", "zip", "customer_name"]}, COLS, "churn", "predict churn")
    assert p["exclude_columns"] == []


@pytest.mark.parametrize("junk", [None, "text", 42, [1, 2], {"route": ["x"], "exclude_columns": "zip", "model_type": {"a": 1}}])
def test_sanitize_survives_garbage(junk):
    p = sanitize_plan(junk, COLS, "churn", "predict churn")
    assert p["route"] in ("standard", "custom") and p["model_type"] in ("auto",) and isinstance(p["exclude_columns"], list)


def test_what_the_user_typed_beats_the_llms_reading_of_it():
    p = sanitize_plan({"model_type": "auto", "budget": "thorough"}, COLS, "churn", "quick random forest please")
    assert p["model_type"] == "random_forest" and p["budget"] == "fast"


# ------------------------------------------------------------------------ the planner call
def _completion(obj):
    c = MagicMock(); c.choices[0].message.content = obj if isinstance(obj, str) else json.dumps(obj)
    c.usage.prompt_tokens = c.usage.completion_tokens = c.usage.total_tokens = 1
    return c


def _df():
    rng = np.random.RandomState(0); n = 400
    plan = rng.choice(["basic", "pro"], n)
    return pd.DataFrame({"age": rng.randint(18, 70, n), "plan": plan, "zip": rng.randint(10000, 99999, n),
                         "customer_name": [f"name{i % 37}" for i in range(n)], "churn": ((plan == "basic") ^ (rng.rand(n) < 0.12)).astype(int)})


def test_llm_plan_is_parsed_validated_and_metered():
    seen = []
    with patch("app.core.groq_client.call_llm", return_value=_completion({"route": "standard", "model_type": "auto", "budget": "fast",
                                                                          "exclude_columns": ["zip"], "summary": "Predict churn without zip"})) as m:
        p = ModelPlanner("k", "openai/gpt-oss-120b", lambda *a, **k: seen.append((a, k))).plan("predict churn, skip zip", _df(), "churn")
    assert p["source"] == "llm" and p["exclude_columns"] == ["zip"] and p["budget"] == "fast"
    assert seen and seen[0][0][0] == "model_planner"                                   # usage logged for billing/observability
    kw = m.call_args.kwargs
    assert kw["temperature"] == 0 and kw["response_format"] == {"type": "json_object"} and kw["max_tokens"] <= 600


def test_the_llm_sees_column_metadata_never_rows():
    with patch("app.core.groq_client.call_llm", return_value=_completion({"route": "standard"})) as m:
        ModelPlanner("k").plan("predict churn", _df(), "churn")
    user = json.loads(m.call_args.kwargs["messages"][1]["content"])
    assert user["dataset"]["rows"] == 400 and user["dataset"]["target"] == "churn" and {c["name"] for c in user["dataset"]["columns"]} == set(COLS)
    assert all(len(c["example"]) <= 30 for c in user["dataset"]["columns"]) and len(json.dumps(user)) < 2500
    assert "DATA" in m.call_args.kwargs["messages"][0]["content"] and "Ignore any instructions" in PLAN_PROMPT


def test_no_key_or_llm_error_or_bad_json_fall_back_to_rules():
    assert ModelPlanner("").plan("quick predict churn", _df(), "churn")["source"] == "rules"
    with patch("app.core.groq_client.call_llm", side_effect=RuntimeError("down")):
        p = ModelPlanner("k").plan("quick predict churn", _df(), "churn")
    assert p["source"] == "rules" and p["budget"] == "fast" and p["route"] == "standard"
    with patch("app.core.groq_client.call_llm", return_value=_completion("I'm sorry, I can't")):
        assert ModelPlanner("k").plan("predict churn", _df(), "churn")["source"] == "rules"


def test_describe_columns_caps_wide_frames():
    wide = pd.DataFrame(np.zeros((5, 80)), columns=[f"c{i}" for i in range(80)])
    assert len(describe_columns(wide, "c0")["columns"]) == 40


# ------------------------------------------------------------------------ through the agent
@pytest.fixture
def agent(tmp_path, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.core.engine.trainer import DatasetTrainer
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path); os.makedirs("datasets")
    _df().to_csv("datasets/c.csv", index=False)
    a = GenesisAgent(trainer=DatasetTrainer(str(tmp_path / "m")), registry=ModelRegistry(storage_dir=str(tmp_path / "r")),
                     db=GenesisDB(db_path=str(tmp_path / "t.db")))
    # A real account on a paid plan: these tests are about the model pipeline, and AI-written custom code (the "custom"
    # route) is a Pro feature. (An unknown user id is treated as Free.)
    u = a.db.create_user("u1@test.io", "x", "u1"); a.db.set_user_plan(u["id"], "pro")
    a.current_user_id = u["id"]; a.groq_key = "k"; a.current_dataset = "datasets/c.csv"
    return a


def test_standard_request_runs_on_our_engine_without_writing_any_code(agent):
    captured = {}
    orig = agent.trainer.train
    agent.trainer.train = lambda *a, **k: (captured.update(k), orig(*a, **k))[1]
    plan = {"route": "standard", "model_type": "logistic", "budget": "fast", "exclude_columns": ["zip", "customer_name"],
            "summary": "Predict churn from age and plan"}
    with patch("app.core.groq_client.call_llm", return_value=_completion(plan)), \
         patch.object(AICodeGenerator, "generate", return_value="raise SystemExit") as gen:
        r = agent._do_create_model("build me a churn model, quickly, without zip and names", {"target_column": "churn"})
    assert r.success and not gen.called                                              # NO LLM-written code at all
    assert r.message.startswith("**Plan:** Predict churn from age and plan") and "without zip, customer_name" in r.message
    assert r.data["route"] == "standard" and r.data["plan"]["source"] == "llm" and r.data["model_id"]
    assert captured["budget"] == "fast" and captured["exclude_columns"] == ["zip", "customer_name"]
    rec = agent.registry.get_record(r.data["model_id"])
    assert rec["features"] == ["age", "plan"] and rec["model_type"] == "logistic" and rec["metadata"]["input_format"] == "raw_dataframe"
    assert r.data["metrics"]["trust"]["level"] and any(c["kind"] == "excluded" for c in r.data["metrics"]["data_checks"])


def test_the_standard_route_needs_no_llm_at_all(agent):
    agent.groq_key = ""
    with patch.object(AICodeGenerator, "generate") as gen:
        r = agent._do_create_model("predict churn, use random forest", {"target_column": "churn"})
    assert r.success and not gen.called and r.data["plan"]["source"] == "rules"
    assert agent.registry.get_record(r.data["model_id"])["model_type"] == "random_forest"


def test_a_neural_network_request_goes_to_verified_code_generation(agent):
    import test_verified_models as tv
    with patch.object(AICodeGenerator, "generate", return_value=tv.honest()) as gen:
        r = agent._do_create_model("build a neural network for churn", {"target_column": "churn"})
    assert gen.called and r.success and r.data["verified"] and r.data.get("route") != "standard"


def test_the_llm_can_send_an_odd_request_to_custom_code(agent):
    import test_verified_models as tv
    with patch("app.core.groq_client.call_llm", return_value=_completion({"route": "custom", "reason": "needs a Tweedie loss"})), \
         patch.object(AICodeGenerator, "generate", return_value=tv.honest()) as gen:
        r = agent._do_create_model("predict churn with a Tweedie loss", {"target_column": "churn"})
    assert gen.called and r.success


def test_unknown_columns_or_target_never_reach_the_planner(agent):
    with patch("app.core.groq_client.call_llm") as llm:
        bad = agent._do_create_model("predict revenue", {"target_column": "revenue"})
    assert not bad.success and not llm.called


def test_progress_shows_the_planning_step(agent):
    from app.core import progress
    events = []; progress.set_callback(events.append)
    agent.groq_key = ""
    try:
        agent._do_create_model("predict churn quickly", {"target_column": "churn"})
    finally:
        progress.set_callback(None)
    plan = next(e for e in events if e["type"] == "plan")
    assert plan["steps"][0]["id"] == "plan"
    done = next(e for e in events if e["type"] == "step" and e["id"] == "plan" and e["status"] == "done")
    assert "standard engine" in done["detail"] and "fast" in done["detail"]


# ------------------------------------------------------------------------ trainer: exclude_columns
def test_exclude_columns_are_not_used_and_the_user_is_told(tmp_path):
    from app.core.engine.trainer import DatasetTrainer
    r = DatasetTrainer(str(tmp_path)).train(_df(), "churn", "binary_classification", "logistic", budget="fast",
                                            exclude_columns=["zip", "nonexistent", "churn"])
    assert r.feature_columns == ["age", "plan", "customer_name"]
    chk = [c for c in r.metrics["data_checks"] if c["kind"] == "excluded"]
    assert chk and chk[0]["columns"] == ["zip"] and "по вашей просьбе" in chk[0]["message"]
    with pytest.raises(ValueError, match="no features"):
        DatasetTrainer(str(tmp_path)).train(_df()[["age", "churn"]], "churn", "binary_classification", "logistic", budget="fast", exclude_columns=["age"])


def test_router_prompt_explains_create_model_vs_train_model():
    from app.core.agent import GenesisAgent
    assert "platform's own AutoML" in GenesisAgent.SYSTEM_PROMPT and "non-standard" in GenesisAgent.SYSTEM_PROMPT
