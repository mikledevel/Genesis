"""Model selection: a learned model is shipped only if it clearly beats the best SIMPLE method.
Found on real data: a model chosen on a short validation window was 2.6x worse than "as yesterday" on the holdout."""
import datetime
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.forecaster import Forecaster, SIMPLE_MODELS, SELECTION_MARGIN, _name_of  # noqa: E402


def _df(y, start="2025-01-01", freq="D"):
    return pd.DataFrame({"date": pd.date_range(start, periods=len(y), freq=freq), "v": np.asarray(y, dtype=float)})


def _run(y, tmp_path, H=14, freq="D"):
    return Forecaster(model_dir=str(tmp_path)).run(_df(y, freq=freq), "date", "v", horizon=H)


def test_simple_methods_are_selectable_models_with_stable_names():
    assert set(SIMPLE_MODELS) == {"seasonal_naive", "last_value", "moving_average", "mean_level"}
    assert [_name_of(*SIMPLE_MODELS[k]) for k in SIMPLE_MODELS] == list(SIMPLE_MODELS)
    assert _name_of("hgb", "diff") == "hgb_diff" and SELECTION_MARGIN > 0


def test_a_random_walk_ships_as_the_last_value_not_as_a_learned_model(tmp_path):
    y = 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 300))
    r = _run(y, tmp_path)
    m = r["metrics"]
    assert m["chosen"] in SIMPLE_MODELS
    assert {"last_value", "moving_average", "mean_level"} <= set(m["candidates"])      # all simple methods were scored
    assert m["mae_test"] <= m["naive_mae_test"] * 1.05            # never clearly worse than "as yesterday"


def test_pure_noise_is_a_simple_method_and_never_rated_high(tmp_path):
    r = _run(100 + np.random.default_rng(4).normal(0, 10, 300), tmp_path)
    assert r["metrics"]["chosen"] in SIMPLE_MODELS and r["metrics"]["trust"]["level"] != "high"


def test_a_structured_series_still_picks_a_learned_model(tmp_path):
    i = np.arange(300)
    y = 100 + 0.2 * i + 15 * np.sin(2 * math.pi * i / 7) + np.random.default_rng(1).normal(0, 3, 300)
    m = _run(y, tmp_path)["metrics"]
    assert m["chosen"] not in SIMPLE_MODELS and m["improvement_vs_best_simple_pct"] > 5


@pytest.mark.parametrize("name", ["last_value", "moving_average", "mean_level"])
def test_each_simple_model_forecasts_what_it_says_and_survives_pickling(name, tmp_path):
    from app.core.engine import forecaster as fm
    y = 100 + np.random.default_rng(5).normal(0, 5, 120)
    df = _df(y)
    s, info = fm.prepare_series(df, "date", "v")
    vals, index = s.to_numpy(), s.index
    kind, mode = SIMPLE_MODELS[name]
    est, spec = fm.fit_candidate(kind, mode, vals, index, len(vals), info["season"], info["freq"])
    if getattr(spec, "simple", None) == "mean":
        spec.simple_param = float(np.mean(vals))
    model = fm._ForecastModel(est, spec, vals[-(spec.max_back + 1):], index[-1].isoformat(), len(vals),
                              np.zeros(5), np.zeros(5), "v", "date", info["freq"], name)
    model2 = pickle.loads(pickle.dumps(model))
    f = model2.forecast(5)["forecast"].to_numpy()
    expect = {"last_value": vals[-1], "moving_average": float(np.mean(vals[-max(3, spec.season_lag):])), "mean_level": float(np.mean(vals))}[name]
    assert np.allclose(f, expect) and len(f) == 5


def test_the_saved_model_of_a_simple_selection_reproduces_its_own_forecast(tmp_path):
    y = 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 300))
    r = _run(y, tmp_path)
    again = pickle.load(open(r["model_path"], "rb")).forecast(r["horizon"])["forecast"].to_numpy()
    assert np.allclose(again, r["forecast"]["forecast"].to_numpy())
