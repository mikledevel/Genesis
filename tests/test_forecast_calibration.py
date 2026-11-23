import numpy as np
import pandas as pd
import pytest
from app.core.engine.forecaster import Forecaster, interval_offsets


def test_split_interval_uses_finite_sample_absolute_residual_rank():
    errors = np.arange(1., 11.)[:, None]
    lower, upper = interval_offsets(errors, method="split")
    assert lower.tolist() == [-9.] and upper.tolist() == [9.]
    with pytest.raises(ValueError):
        interval_offsets(np.array([[np.nan]]), method="split")
    with pytest.raises(ValueError):
        interval_offsets(np.ones((2, 1)), method="split")


def test_split_protocol_keeps_selection_calibration_and_test_separate(tmp_path):
    rng = np.random.default_rng(10)
    t = np.arange(300)
    data = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=300),
                         "sales": 100 + .2*t + 10*np.sin(2*np.pi*t/7) + rng.normal(0,3,300)})
    result = Forecaster(tmp_path).run(data, "date", "sales", 14, interval_method="split")
    protocol = result["metrics"]["interval_protocol"]
    assert protocol["selection_end"] == protocol["calibration_start"] < protocol["calibration_end"]
    assert protocol["calibration_end"] == protocol["test_start"] < protocol["test_end"] == len(data)
    assert protocol["calibration_origins"] >= 4
    assert "coverage_guarantee" in protocol and protocol["coverage_guarantee"] is False


def test_invalid_numeric_data_is_not_silently_forecast_as_zero():
    from app.core.engine.forecaster import prepare_series
    data = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=30), "sales": ["unknown"]*30})
    with pytest.raises(ValueError, match="numeric"):
        prepare_series(data, "date", "sales")


def test_sparse_series_is_explicitly_flagged(tmp_path):
    rng = np.random.default_rng(20)
    data = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=200),
                         "sales": rng.binomial(1,.1,200)*rng.integers(1,20,200)})
    result = Forecaster(tmp_path).run(data, "date", "sales", 14)
    assert result["info"]["zero_share"] > .4
    assert result["metrics"]["trust"]["level"] == "low"
    assert any("нул" in item for item in result["metrics"]["trust"]["reasons"])
