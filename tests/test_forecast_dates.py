"""Regression: day-first dates (05.03.2025 = 5 March) must not be read month-first in the forecaster.
Before the fix a 120-day daily file in DD.MM.YYYY became a 338-day series with 218 interpolated (invented) points."""
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.forecaster import prepare_series  # noqa: E402


def _daily(fmt, days=120, start=datetime.date(2025, 3, 1)):
    dates = [(start + datetime.timedelta(days=k)).strftime(fmt) for k in range(days)]
    return pd.DataFrame({"d": dates, "y": np.arange(days, dtype=float)})


def test_dotted_day_first_dates_keep_their_real_order_and_length():
    s, info = prepare_series(_daily("%d.%m.%Y"), "d", "y")
    assert len(s) == 120 and info["gaps"] == 0 and info["freq"] == "D"
    assert s.index[0] == pd.Timestamp("2025-03-01") and s.index[-1] == pd.Timestamp("2025-06-28")
    assert (np.diff(s.to_numpy()) == 1).all()


def test_slashed_month_first_dates_still_work():
    s, info = prepare_series(_daily("%m/%d/%Y"), "d", "y")          # 03/25/2025: second part > 12 -> month-first
    assert len(s) == 120 and info["gaps"] == 0
    assert s.index[0] == pd.Timestamp("2025-03-01") and s.index[-1] == pd.Timestamp("2025-06-28")


def test_slashed_day_first_dates_are_detected_from_the_data():
    s, info = prepare_series(_daily("%d/%m/%Y"), "d", "y")          # 25/03/2025: first part > 12 -> day-first
    assert len(s) == 120 and info["gaps"] == 0 and s.index[0] == pd.Timestamp("2025-03-01")


def test_iso_dates_are_unaffected():
    s, info = prepare_series(_daily("%Y-%m-%d"), "d", "y")
    assert len(s) == 120 and info["gaps"] == 0 and s.index[0] == pd.Timestamp("2025-03-01")


def test_monthly_dotted_series_is_read_day_first_by_the_russian_convention():
    df = pd.DataFrame({"d": [f"01.{m:02d}.2024" for m in range(1, 13)] + [f"01.{m:02d}.2025" for m in range(1, 13)],
                       "y": np.arange(24, dtype=float)})
    s, info = prepare_series(df, "d", "y")
    assert len(s) == 24 and info["freq"].startswith("M") and s.index[0] == pd.Timestamp("2024-01-01")
