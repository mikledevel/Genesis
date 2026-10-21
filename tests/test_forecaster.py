"""Time-series forecasting: honest chronological evaluation, baselines, intervals, agent flow, safe pickling."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from unittest.mock import patch
from app.core.engine import forecaster as F
from app.core.engine.forecaster import Forecaster, prepare_series, infer_freq, season_for, units_to_steps, make_spec, features_at
from app.core.engine.registry import ModelRegistry


def series(n=400, freq="D", season=7, trend=0.2, amp=15, noise=3, seed=0, start="2022-01-01"):
    rng = np.random.RandomState(seed); d = pd.date_range(start, periods=n, freq=freq)
    y = 100 + trend * np.arange(n) + amp * np.sin(2 * np.pi * np.arange(n) / season) + rng.randn(n) * noise
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "sales": y})


def run(df, H=14, **kw):
    return Forecaster(tempfile.mkdtemp()).run(df, "date", "sales", horizon=H, **kw)


# ------------------------------------------------------------------ the series itself
@pytest.mark.parametrize("freq,prefix,season", [("D", "D", 7), ("W", "W", 52), ("MS", "MS", 12), ("h", "h", 24), ("QS", "QS", 4), ("QE", "QE", 4)])
def test_frequency_and_season_are_inferred(freq, prefix, season):
    idx = pd.date_range("2020-01-01", periods=60, freq=freq)
    f, notes = infer_freq(idx)                                   # pandas may anchor it (W-SUN, QS-OCT): same meaning
    assert f.startswith(prefix) and not notes and season_for(f) == season


def test_irregular_dates_get_a_median_step_and_a_note():
    d = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-11"])
    f, notes = infer_freq(pd.DatetimeIndex(d))
    assert f == "D" and notes and "медианному" in notes[0]


def test_gaps_are_interpolated_duplicates_aggregated_and_both_reported():
    df = series(100)
    df = df.drop(index=[10, 11, 50]).reset_index(drop=True)
    df = pd.concat([df, df.iloc[[5, 6]]]).reset_index(drop=True)                      # duplicate dates
    s, info = prepare_series(df, "date", "sales", "sum")
    assert len(s) == 100 and info["gaps"] == 3 and info["duplicates"] == 2 and not s.isna().any()
    assert any("пропущенных" in n for n in info["notes"]) and any("несколько строк" in n for n in info["notes"])


def test_mean_aggregation_for_duplicate_dates():
    df = pd.DataFrame({"date": ["2024-01-01"] * 2 + [f"2024-01-{i:02d}" for i in range(2, 31)], "sales": [10, 30] + [20] * 29})
    s, _ = prepare_series(df, "date", "sales", "mean")
    assert s.iloc[0] == 20 and len(s) == 30


def test_unreadable_dates_are_a_clear_error():
    with pytest.raises(ValueError, match="no readable dates"):
        prepare_series(pd.DataFrame({"date": ["x", "y"], "sales": [1, 2]}), "date", "sales")


@pytest.mark.parametrize("n,unit,freq,expected", [(3, "месяца", "D", 91), (2, "weeks", "D", 14), (14, "days", "D", 14), (3, "months", "MS", 3),
                                                  (1, "year", "MS", 12), (2, "quarters", "QS-JAN", 2), (5, "days", "MS", 1), (7, "steps", "D", 7), (2, "нед", "W-SUN", 2)])
def test_units_to_steps(n, unit, freq, expected):
    assert units_to_steps(n, unit, freq) == expected


# ------------------------------------------------------------------ chronological honesty
def test_features_are_computed_from_the_past_only():
    spec = make_spec(300, 7, "D", "level")
    vals = np.arange(100, dtype=float); idx = pd.date_range("2024-01-01", periods=100, freq="D")
    a = features_at(spec, vals[:50], idx[50], 50)
    vals2 = vals.copy(); vals2[50:] = 9999                                              # the future changes...
    assert np.array_equal(a, features_at(spec, vals2[:50], idx[50], 50))                # ...the features do not


def test_holdout_comes_after_training_and_is_never_used_for_selection():
    seen = {}
    orig = F.backtest
    def spy(est, spec, vals, index, start, end, H, max_origins=60):
        seen.setdefault("calls", []).append((start, end)); return orig(est, spec, vals, index, start, end, H, max_origins)
    with patch.object(F, "backtest", spy):
        run(series(400), 14)
    calls = seen["calls"]; n = 400; test_start = n - int(0.2 * n)
    selection, test = [c for c in calls if c[1] <= test_start], [c for c in calls if c[0] == test_start]
    assert selection and test and all(c[1] <= test_start for c in selection)         # validation ends where the holdout starts
    assert test[0][1] == n


# ------------------------------------------------------------------ quality
def test_it_beats_the_seasonal_naive_rule_on_a_trend_with_weekly_seasonality():
    r = run(series(400)); m = r["metrics"]
    assert m["mae_test"] < m["baseline_mae_test"] * 0.8 and m["improvement_vs_baseline_pct"] > 20
    assert m["mase_test"] < 1 and m["improvement_vs_best_simple_pct"] > 10 and m["chosen"] != "seasonal_naive"
    assert m["n_origins"] >= 20 and set(m["candidates"]) >= {"seasonal_naive", "ridge_level"}


def test_the_forecast_continues_the_pattern_after_the_last_date():
    df = series(400); r = run(df, 14); fc = r["forecast"]
    assert len(fc) == 14 and fc["date"].iloc[0] == pd.Timestamp("2022-01-01") + pd.Timedelta(days=400)
    truth = 100 + 0.2 * np.arange(400, 414) + 15 * np.sin(2 * np.pi * np.arange(400, 414) / 7)
    assert np.mean(np.abs(fc["forecast"].to_numpy() - truth)) < 6                      # close to the true noiseless signal
    assert (fc["lower"] < fc["forecast"]).all() and (fc["upper"] > fc["forecast"]).all()


@pytest.mark.parametrize("kw", [dict(n=72, freq="MS", season=12, trend=0.8, amp=12, noise=2, seed=2),
                                dict(n=208, freq="W", season=52, trend=0.1, amp=10, noise=2, seed=3)])
def test_monthly_and_weekly_series_work(kw):
    H = 12 if kw["freq"] == "MS" else 8
    m = run(series(**kw), H)["metrics"]
    assert m["mae_test"] < m["baseline_mae_test"] and m["season"] == (12 if kw["freq"] == "MS" else 52)


def test_pure_noise_is_reported_as_unpredictable_not_as_a_success():
    m = run(series(400, trend=0, amp=0, noise=10, seed=1))["metrics"]
    assert m["improvement_vs_best_simple_pct"] <= 5
    assert m["trust"]["level"] in ("low", "medium") and any("простых" in t or "закономерностей" in t for t in m["trust"]["reasons"])


def test_prediction_intervals_are_checked_out_of_sample():
    m = run(series(400))["metrics"]
    assert m["coverage_80_test"] is not None and 0.3 < m["coverage_80_test"] <= 1.0


def test_a_short_series_is_refused_clearly_and_a_borderline_one_is_flagged():
    with pytest.raises(ValueError, match="needs at least"):
        run(series(15), 7)
    m = run(series(40), 7)["metrics"]
    assert m["trust"]["level"] in ("medium", "low") and any("короткий" in t for t in m["trust"]["reasons"])


def test_metrics_are_json_serialisable_and_complete():
    import json
    m = run(series(300))["metrics"]; json.dumps(m)
    for k in ("mae_test", "rmse_test", "smape_test", "mase_test", "baseline_mae_test", "naive_mae_test", "ma_mae_test", "mean_mae_test",
              "coverage_80_test", "horizon", "n_points", "frequency", "trust", "chosen", "candidates"):
        assert k in m


def test_progress_stages_are_reported_in_order():
    seen = []
    run(series(300), 7, on_stage=lambda sid, st="active", d=None: seen.append((sid, st)))
    assert [s for s, st in seen if st == "done"] == ["prep", "select", "eval", "fit", "save"]


# ------------------------------------------------------------------ artifact + registry
def test_model_survives_the_restricted_unpickler_and_predicts_n_steps(tmp_path):
    r = run(series(300), 10)
    reg = ModelRegistry(storage_dir=str(tmp_path / "reg"))
    mid = reg.register(r["model"], r["metrics"], "forecasting", r["model_type"], ["steps"], "sales", metadata={"input_format": "forecast"})
    loaded = reg.get_model(mid)                                                        # safe_load: no custom code can run
    out = loaded.predict(np.array([[5.0]]))
    assert len(out) == 5 and np.allclose(out, r["model"].forecast(5)["forecast"].to_numpy())
    fc = loaded.forecast(30); assert len(fc) == 30 and fc["date"].is_monotonic_increasing


def test_the_artifact_holds_no_pandas_objects():
    import pickle, pickletools
    m = run(series(300), 7)["model"]
    blob = pickle.dumps(m)
    assert b"pandas" not in blob                                                         # would be blocked by safe_pickle


# ------------------------------------------------------------------ the agent
@pytest.fixture
def agent(tmp_path, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.core.engine.trainer import DatasetTrainer
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path); os.makedirs("datasets")
    a = GenesisAgent(trainer=DatasetTrainer(str(tmp_path / "m")), registry=ModelRegistry(storage_dir=str(tmp_path / "r")), db=GenesisDB(db_path=str(tmp_path / "t.db")))
    a.current_user_id = "u1"; a.groq_key = ""; return a


def test_agent_forecasts_and_registers_with_a_chart_payload(agent):
    series(400).assign(id=np.arange(400)).to_csv("datasets/s.csv", index=False)
    r = agent._do_forecast("прогноз продаж на 2 недели вперёд", {"path": "datasets/s.csv"})
    assert r.success, r.message
    assert "Forecast of «sales»" in r.message and "next 14 days" in r.message and "Reliability" in r.message and "| Date | Forecast | 80% range |" in r.message
    d = r.data["forecast"]
    assert d["freq"] == "D" and len(d["future"]["values"]) == 14 == len(d["future"]["lower"]) and len(d["history"]["dates"]) == len(d["history"]["values"]) >= 60
    rec = agent.registry.get_record(r.data["model_id"])
    assert rec["task"] == "forecasting" and rec["metadata"]["input_format"] == "forecast" and rec["metadata"]["date_column"] == "date" and rec["target"] == "sales"
    assert agent.registry.get_model(r.data["model_id"]).predict([[3]]).shape == (3,)


def test_agent_horizon_units_follow_the_series_frequency(agent):
    series(72, "MS", 12, 0.8, 12, 2, 2).to_csv("datasets/m.csv", index=False)
    r = agent._do_forecast("forecast", {"path": "datasets/m.csv", "horizon": 6, "horizon_unit": "month"})
    assert r.success and len(r.data["forecast"]["future"]["values"]) == 6 and r.data["forecast"]["freq"].startswith("MS")


def test_agent_needs_a_date_column_and_a_real_target(agent):
    pd.DataFrame({"a": np.arange(100), "b": np.random.RandomState(0).rand(100)}).to_csv("datasets/nodate.csv", index=False)
    r = agent._do_forecast("forecast sales", {"path": "datasets/nodate.csv"})
    assert not r.success and "date column" in r.message
    series(300).to_csv("datasets/s.csv", index=False)
    bad = agent._do_forecast("forecast", {"path": "datasets/s.csv", "target_column": "revenue"})
    assert not bad.success and "no column" in bad.message.lower()
    bad = agent._do_forecast("forecast", {"path": "datasets/s.csv", "date_column": "when"})
    assert not bad.success and "no column" in bad.message.lower()


def test_agent_asks_when_several_numeric_columns_could_be_the_series(agent):
    df = series(300).rename(columns={"sales": "a"}).assign(b=lambda d: d.a * 0.5 + np.random.RandomState(1).randn(300))
    df.to_csv("datasets/two.csv", index=False)
    r = agent._do_forecast("forecast it", {"path": "datasets/two.csv"})
    assert r.success and r.data["needs_target"] and {c["column"] for c in r.data["target_candidates"]} == {"a", "b"}
    assert "Forecast column {column} in {path}, date column date" == r.data["ask_template"]
    assert agent.registry.list_models(limit=5) == []
    named = df.rename(columns={"a": "revenue"})
    named.to_csv("datasets/two.csv", index=False)
    assert agent._do_forecast("forecast it", {"path": "datasets/two.csv"}).success         # a name like "revenue" settles it


def test_create_model_sends_forecast_requests_to_the_forecaster_only_when_there_is_a_date(agent):
    series(300).to_csv("datasets/s.csv", index=False)
    agent.current_dataset = "datasets/s.csv"
    r = agent._do_create_model("forecast sales for the next 2 weeks", {})
    assert r.success and r.data["route"] == "forecast"
    pd.DataFrame({"x": np.random.RandomState(0).randn(200), "y": np.random.RandomState(1).randint(0, 2, 200)}).to_csv("datasets/flat.csv", index=False)
    agent.current_dataset = "datasets/flat.csv"
    r2 = agent._do_create_model("forecast the next 5 days", {"target_column": "y"})
    assert r2.success and r2.data.get("route") != "forecast"                             # no date column -> the ordinary engine


def test_agent_dispatch_and_prompt_know_the_forecast_tool(agent):
    from app.core.agent import GenesisAgent
    assert "forecast_model" in GenesisAgent.SYSTEM_PROMPT and "NOT a forecast" in GenesisAgent.SYSTEM_PROMPT
    series(300).to_csv("datasets/s.csv", index=False)
    with patch.object(agent, "_call_llm", return_value={"tool": "forecast_model", "args": {"path": "datasets/s.csv", "horizon": 7, "horizon_unit": "day"}, "message": "ok"}):
        resp = agent.process("forecast", "c1", user_id="u1")
    assert resp.success and resp.data["route"] == "forecast"


def test_agent_progress_plan_for_forecasts(agent):
    from app.core import progress
    ev = []; progress.set_callback(ev.append)
    series(300).to_csv("datasets/s.csv", index=False)
    try:
        agent._do_forecast("forecast", {"path": "datasets/s.csv"})
    finally:
        progress.set_callback(None)
    plan = next(e for e in ev if e["type"] == "plan" and e["title"] == "Forecasting")
    assert [s["id"] for s in plan["steps"]] == ["prep", "select", "eval", "fit", "save", "register"]
    assert {"prep", "select", "eval", "fit", "save", "register"} <= {e["id"] for e in ev if e["type"] == "step" and e["status"] == "done"}
