"""The client report: honest verdict, no external resources, user text escaped, clear errors."""
import math
import os
import re
import sys
import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.report import build_forecast_report, detect_columns, verdict_sentence  # noqa: E402


def _series(kind="season", n=200, seed=1, col="sales", date="date"):
    r = np.random.default_rng(seed)
    i = np.arange(n)
    y = {"season": 100 + 15 * np.sin(2 * math.pi * i / 7) + r.normal(0, 3, n),
         "noise": 100 + r.normal(0, 10, n)}[kind]
    d0 = datetime.date(2025, 1, 1)
    return pd.DataFrame({date: [d0 + datetime.timedelta(days=int(k)) for k in i], col: y})


def test_report_has_every_section_and_no_external_resources():
    res = build_forecast_report(_series(), horizon=14, client="ACME Shop")
    h = res["html"]
    for needle in ("ACME Shop", "How much to trust this", "Likely range", "The evidence", "Limits", "<svg"):
        assert needle in h
    assert not re.search(r"(src|href)\s*=\s*[\"']https?://", h)       # nothing remote
    assert "<script" not in h.lower() and "@import" not in h and "url(" not in h
    assert res["summary"]["trust"] in ("high", "medium", "low")
    assert res["summary"]["horizon"] == 14


def test_noise_is_reported_as_low_reliability_in_plain_words():
    res = build_forecast_report(_series("noise"), horizon=14)
    assert res["summary"]["trust"] == "low"
    assert "Low reliability" in res["html"] and "Do not rely on the numbers below" in res["html"]


def test_coverage_is_reported_as_measured_not_promised():
    res = build_forecast_report(_series(), horizon=14)
    cov = res["summary"]["coverage_80_test"]
    assert cov is not None and f"{cov:.0%}" in res["html"]


def test_user_supplied_text_is_escaped():
    df = _series(col="<img src=x onerror=alert(1)>")
    res = build_forecast_report(df, target="<img src=x onerror=alert(1)>", client="<script>alert(2)</script>", title="T<b>")
    h = res["html"]
    assert "<img src=x" not in h and "<script>alert" not in h and "T<b>" not in h
    assert "&lt;script&gt;alert(2)&lt;/script&gt;" in h


def test_russian_report_uses_russian_text():
    h = build_forecast_report(_series(), horizon=7, lang="ru")["html"]
    assert 'lang="ru"' in h and "Насколько этому можно верить" in h and "Ограничения" in h


def test_columns_are_guessed_and_the_guess_is_disclosed():
    df = _series(col="revenue", date="order_date")
    d, t, notes = detect_columns(df)
    assert (d, t) == ("order_date", "revenue") and any("chosen automatically" in n for n in notes)
    assert "chosen automatically" in build_forecast_report(df, horizon=7)["html"]


def test_explicit_columns_win_and_unknown_ones_fail_clearly():
    df = _series()
    df["other"] = 1.0
    assert detect_columns(df, "date", "other")[:2] == ("date", "other")
    with pytest.raises(ValueError, match="not found"):
        detect_columns(df, target="nope")
    with pytest.raises(ValueError, match="date column"):
        detect_columns(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))


def test_too_short_series_is_a_clear_error_not_a_crash():
    with pytest.raises(ValueError, match="only"):
        build_forecast_report(_series(n=15), horizon=14)


def test_bad_language_is_rejected():
    with pytest.raises(ValueError, match="lang"):
        build_forecast_report(_series(), lang="xx")


def test_russian_report_has_no_english_notes_and_gaps_are_disclosed_in_both_languages():
    df = _series(col="revenue", date="order_date")
    df = df.drop(df.index[60:75])                      # 15 missing days -> interpolated
    ru = build_forecast_report(df, horizon=7, lang="ru")["html"]
    en = build_forecast_report(df, horizon=7, lang="en")["html"]
    assert "выбрана автоматически" in ru and "chosen automatically" not in ru and "линейной интерполяцией" in ru
    assert "linear interpolation" in en and "not real data" in en
    assert not re.search(r"[А-Яа-я]", re.sub(r"<style>.*?</style>", "", en, flags=re.S))      # English report: no Russian leaks


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_a_forecast_that_equals_the_baseline_is_never_said_to_beat_it(lang):
    same = verdict_sentence("medium", 0.0, lang)
    assert same != verdict_sentence("medium", 12.0, lang)
    assert "beat" not in same and "обошла" not in same
    assert verdict_sentence("high", 15.0, lang) != same            # a real gain keeps the normal wording
    assert verdict_sentence("low", 0.0, lang) == verdict_sentence("low", -30.0, lang)   # low stays low


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_report_text_never_contradicts_its_own_numbers(seed):
    r = np.random.default_rng(seed)
    i = np.arange(96)
    y = 1000 + 300 * np.sin(2 * math.pi * i / 12) + r.normal(0, 80, 96)
    df = pd.DataFrame({"month": pd.date_range("2018-01-01", periods=96, freq="MS"), "sales": y})
    res = build_forecast_report(df, horizon=12)
    imp = res["summary"]["improvement_vs_best_simple_pct"]
    if imp is not None and imp < 1.0 and res["summary"]["trust"] != "low":
        assert "beat the simple baselines" not in res["html"]


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_a_chosen_simple_method_is_named_and_never_said_to_beat_itself(lang):
    for chosen in ("seasonal_naive", "last_value", "moving_average"):
        text = verdict_sentence("medium", 8.0, lang, chosen=chosen)       # even with a positive holdout gain
        assert "beat" not in text and "обошла" not in text
    assert verdict_sentence("medium", 8.0, "en", chosen="last_value").count("last known value") == 1
    assert verdict_sentence("medium", 8.0, "ru", chosen="moving_average").count("скользящее среднее") == 1
    assert verdict_sentence("medium", 8.0, lang, chosen="ridge_level") == verdict_sentence("medium", 8.0, lang)   # a learned model: normal wording
