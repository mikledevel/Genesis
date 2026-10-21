"""Time-series forecasting engine ("how many will I sell next month?").

Why a separate engine: a forecast is not a row-wise prediction. The ONLY honest evaluation is
chronological - train on the past, score on a later stretch - and a random split (what the tabular
trainer does) would leak the future. So this engine
  * puts the series on a regular grid (frequency inferred, duplicate dates aggregated, gaps interpolated and reported),
  * builds lag / rolling / calendar features and forecasts RECURSIVELY for a whole horizon,
  * picks the model by ROLLING-ORIGIN backtests on a validation stretch carved from the END of the training part
    (candidates: ridge, gradient boosting, random forest - each in "level" or "difference from the seasonal lag" mode -
    plus the seasonal-naive rule; if the simplest rule wins it is the model),
  * reports the holdout metrics (MAE / RMSE / sMAPE / MASE) next to the seasonal-naive and last-value baselines,
    measured the same way, plus an 80% prediction interval whose COVERAGE is checked out of sample,
  * refits the chosen model on all data for the actual forecast and attaches a trust verdict.
The saved artifact (_ForecastModel) stores only floats / arrays / strings so it unpickles through
app/core/safe_pickle.py; predict([[n]]) returns the next n forecast values.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.engine.transformers import _infer_dayfirst, _parse

log = logging.getLogger(__name__)


def _noop(*_a, **_k):
    return None


# ---------------------------------------------------------------------------- frequency and series
def _offset(freq: str):
    from pandas.tseries.frequencies import to_offset
    try:
        return to_offset(freq)
    except ValueError:                       # older pandas spells hourly 'H'
        return to_offset(freq.upper())


def season_for(freq: str) -> int:
    from pandas.tseries import offsets as o
    off = _offset(freq)
    if isinstance(off, o.Hour): return 24
    if isinstance(off, o.Day): return 7
    if isinstance(off, o.Week): return 52
    if isinstance(off, (o.MonthEnd, o.MonthBegin)): return 12
    if isinstance(off, (o.QuarterEnd, o.QuarterBegin)): return 4
    return 1


def infer_freq(index: "pd.DatetimeIndex") -> Tuple[str, List[str]]:
    idx = index.sort_values().unique()
    notes: List[str] = []
    if len(idx) >= 3:
        try:
            f = pd.infer_freq(idx)
        except (TypeError, ValueError):
            f = None
        if f:
            return f, notes
    deltas = np.diff(idx.values).astype("timedelta64[s]").astype(float) / 86400.0
    med = float(np.median(deltas)) if len(deltas) else 1.0
    day1 = float((idx.day == 1).mean())
    f = "h" if med < 0.5 else "D" if med < 3 else "W" if med < 14 else ("MS" if day1 > 0.5 else "ME") if med < 60 \
        else ("QS" if day1 > 0.5 else "QE") if med < 200 else "YS"
    notes.append(f"Регулярный шаг по датам не найден (есть пропуски), принята частота «{f}» по медианному интервалу ({med:.1f} дн.).")
    return f, notes


def prepare_series(df: "pd.DataFrame", date_col: str, target: str, agg: str = "sum"):
    """-> (series on a regular DatetimeIndex, info). Duplicate dates are aggregated, gaps are interpolated."""
    d = df[[date_col, target]].copy()
    # DD.MM.YYYY vs MM/DD/YYYY is decided from the data. Without this, 05.03.2025 became 3 May and a 120-day
    # daily file turned into a 338-day series with 218 INVENTED (interpolated) points.
    d[date_col] = _parse(d[date_col], _infer_dayfirst(d[date_col]))
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        raise ValueError(f"Column «{date_col}» contains no readable dates")
    notes: List[str] = []
    dup = int(d.duplicated(date_col).sum())
    s = d.groupby(date_col)[target].agg(agg).sort_index()
    if dup:
        notes.append(f"На одну дату приходилось несколько строк ({dup} лишних): значения объединены функцией «{agg}».")
    freq, fnotes = infer_freq(s.index)
    notes += fnotes
    rs = s.resample(freq)
    counts, grid = rs.count(), getattr(rs, agg if agg in ("sum", "mean", "last", "max", "min") else "sum")()
    grid = grid.where(counts > 0)
    gaps = int(grid.isna().sum())
    if gaps:
        grid = grid.interpolate("linear", limit_direction="both")
        notes.append(f"{gaps} пропущенных периодов из {len(grid)} заполнены линейной интерполяцией.")
    return grid.astype(float), {"freq": freq, "season": season_for(freq), "gaps": gaps, "duplicates": dup, "agg": agg,
                                "notes": notes, "n": int(len(grid))}


def units_to_steps(n: int, unit: str, freq: str) -> int:
    """"3 months" on daily data -> ~90 steps; on monthly data -> 3. A unit finer than the data's frequency means 1 step."""
    from pandas.tseries import offsets as o
    off = _offset(freq)
    unit = (unit or "").lower()
    kind = ("day" if unit.startswith(("d", "дн", "ден", "сут")) else "week" if unit.startswith(("w", "нед")) else
            "month" if unit.startswith(("m", "мес")) else "quarter" if unit.startswith(("q", "кварт")) else
            "year" if unit.startswith(("y", "год", "лет")) else "step" if unit.startswith(("p", "ш", "s")) else "")
    if not kind or kind == "step":
        return max(1, int(n))
    hours = {"day": 24, "week": 168, "month": 730, "quarter": 2190, "year": 8760}[kind]
    per = (1 if isinstance(off, o.Hour) else 24 if isinstance(off, o.Day) else 168 if isinstance(off, o.Week) else
           730 if isinstance(off, (o.MonthEnd, o.MonthBegin)) else 2190 if isinstance(off, (o.QuarterEnd, o.QuarterBegin)) else 8760)
    return max(1, int(round(n * hours / per)))


# ---------------------------------------------------------------------------- features
class _Spec:
    """Which lags / windows / calendar features, and whether the target is the level or the difference from the
    seasonal lag. A plain class of floats/lists/strings (pickles safely; allow-listed)."""
    def __init__(self, lags, windows, season, freq, mode, hourly):
        self.lags, self.windows, self.season, self.freq, self.mode, self.hourly = list(lags), list(windows), int(season), freq, mode, bool(hourly)
        self.season_lag = self.season if self.season > 1 else 1
        self.max_back = max(max(self.lags), max(self.windows), self.season_lag)
        self.use_trend = mode == "level"


def make_spec(n_train: int, season: int, freq: str, mode: str) -> _Spec:
    cap = max(2, n_train // 4)
    lags = {1, 2, 3}
    if season > 1:
        lags.add(season)
        if 2 * season <= cap:
            lags.add(2 * season)
    lags = sorted(l for l in lags if l <= cap) or [1]
    windows = sorted({w for w in (3, max(3, season)) if w <= cap}) or [min(3, cap)]
    return _Spec(lags, windows, season, freq, mode, hourly=season == 24)


def features_at(spec: _Spec, prefix: np.ndarray, date: "pd.Timestamp", t: int) -> np.ndarray:
    f = [prefix[-L] for L in spec.lags]
    for w in spec.windows:
        tail = prefix[-w:]
        f += [float(tail.mean()), float(tail.std()) if len(tail) > 1 else 0.0]
    # CYCLICAL calendar encoding. Raw month / day-of-year / week numbers reset at New Year, so a linear model that sees
    # "week 1" only in the first weeks of the training data confuses the calendar with the trend and breaks the first
    # time it forecasts January of the next year (found by the audit: holdout MAE 14 vs 2.6 in validation).
    def cyc(v, period):
        a = 2 * np.pi * v / period
        return [np.sin(a), np.cos(a)]
    f += cyc(date.dayofweek, 7) + cyc(date.month - 1, 12) + cyc(date.dayofyear - 1, 365.25) + cyc(date.day - 1, 31)
    f.append(float(date.dayofweek >= 5))
    if spec.hourly:
        f += cyc(date.hour, 24)
    if spec.use_trend:
        f.append(t)
    return np.asarray(f, dtype=float)


def _make_estimator(kind: str):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if kind == "hgb":
        return HistGradientBoostingRegressor(random_state=42, max_iter=200, learning_rate=0.08)
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1)   # n_jobs=1: single-row predicts in a recursion are slower with a worker pool


CANDIDATES = [("ridge", "level"), ("hgb", "level"), ("hgb", "diff"), ("rf", "diff")]
# The simple methods are real, selectable models (before, only seasonal-naive could be selected; "as yesterday", the moving
# average and the whole-history mean were just reported as baselines, so a model that loses to them could still be shipped).
SIMPLE_MODELS = {"seasonal_naive": ("naive", "level"), "last_value": ("simple", "last"),
                 "moving_average": ("simple", "ma"), "mean_level": ("simple", "mean")}
SIMPLE_RU = {"seasonal_naive": "как в прошлом сезоне", "last_value": "как вчера (последнее значение)",
             "moving_average": "скользящее среднее", "mean_level": "общее среднее"}
# A learned model is chosen only if it beats the best simple method on the validation backtest by this margin. The minimum
# over several candidates on a short window is optimistic (winner's curse); on ties the simpler method wins.
SELECTION_MARGIN = 0.05


def _name_of(kind: str, mode: str) -> str:
    if kind == "naive":
        return "seasonal_naive"
    if kind == "simple":
        return {v[1]: k for k, v in SIMPLE_MODELS.items() if v[0] == "simple"}[mode]
    return f"{kind}_{mode}"
MAX_TRAIN_ROWS = 20000


def fit_candidate(kind: str, mode: str, vals: np.ndarray, index: "pd.DatetimeIndex", end: int, season: int, freq: str):
    """Fit on vals[:end]. -> (estimator, spec). kind == 'naive' needs no fitting."""
    spec = make_spec(end, season, freq, mode)
    if kind == "naive":
        return None, spec
    if kind == "simple":
        spec.simple, spec.simple_param = mode, None           # 'last' | 'ma' | 'mean'; param is set only on the final refit
        return None, spec
    start = max(spec.max_back, end - MAX_TRAIN_ROWS)
    X = np.vstack([features_at(spec, vals[:t], index[t], t) for t in range(start, end)])
    y = np.asarray([vals[t] - (vals[t - spec.season_lag] if mode == "diff" else 0.0) for t in range(start, end)])
    est = _make_estimator(kind)
    est.fit(X, y)
    return est, spec


def recursive_forecast(est, spec: _Spec, history: np.ndarray, last_date: "pd.Timestamp", steps: int, t_start: int) -> np.ndarray:
    """Forecast `steps` periods after `history`, feeding each prediction back in. est is None => seasonal naive."""
    simple = getattr(spec, "simple", None)
    if est is None and simple in ("last", "ma", "mean"):
        h = np.asarray(history, dtype=float)
        if simple == "last":
            level = float(h[-1])
        elif simple == "ma":
            level = float(np.mean(h[-max(3, spec.season_lag):]))
        else:                                                  # whole-history mean; the saved model carries it (it keeps only a tail)
            level = spec.simple_param if getattr(spec, "simple_param", None) is not None else float(np.mean(h))
        return np.full(steps, level)
    dates = pd.date_range(start=last_date, periods=steps + 1, freq=spec.freq)[1:]
    ext = list(history[-(spec.max_back + 1):]) if est is not None else list(history[-max(spec.season_lag, 1):])
    out = []
    for i in range(steps):
        if est is None:
            val = ext[-spec.season_lag] if len(ext) >= spec.season_lag else ext[-1]
        else:
            x = features_at(spec, np.asarray(ext[-(spec.max_back):]), dates[i], t_start + i)
            val = float(est.predict(x.reshape(1, -1))[0]) + (ext[-spec.season_lag] if spec.mode == "diff" else 0.0)
        ext.append(val)
        out.append(val)
    return np.asarray(out)


def backtest(est, spec: _Spec, vals: np.ndarray, index: "pd.DatetimeIndex", start: int, end: int, H: int, max_origins: int = 60):
    """Rolling origins in [start, end-H]: forecast H steps from each, using only data before the origin.
    -> (errors[origins, H], predictions, actuals, seasonal-naive, last-value, moving-average and whole-history-mean baselines)."""
    origins = list(range(start, end - H + 1))
    if len(origins) > max_origins:
        origins = [origins[int(i)] for i in np.linspace(0, len(origins) - 1, max_origins)]
    errs, preds, acts, sn, lv, ma, mu = [], [], [], [], [], [], []
    for o in origins:
        p = recursive_forecast(est, spec, vals[:o], index[o - 1], H, o)
        a = vals[o:o + H]
        base = np.asarray([vals[o - spec.season_lag + (h % spec.season_lag)] for h in range(H)])
        w = max(3, spec.season_lag)
        errs.append(a - p); preds.append(p); acts.append(a); sn.append(base); lv.append(np.full(H, vals[o - 1]))
        ma.append(np.full(H, float(np.mean(vals[max(0, o - w):o]))))
        mu.append(np.full(H, float(np.mean(vals[:o]))))      # the whole-history mean: the optimum for a series with no structure
    return np.array(errs), np.array(preds), np.array(acts), np.array(sn), np.array(lv), np.array(ma), np.array(mu)


def interval_offsets(errs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """80% interval offsets per horizon step from backtest errors (empirical quantiles; normal approximation if few origins)."""
    if errs.shape[0] >= 8:
        return np.percentile(errs, 10, axis=0), np.percentile(errs, 90, axis=0)
    sd = np.std(errs, axis=0) if errs.shape[0] >= 3 else np.full(errs.shape[1], np.std(errs) if errs.size else 0.0)
    return -1.2816 * sd, 1.2816 * sd


# ---------------------------------------------------------------------------- the saved artifact
class _ForecastModel:
    """Picklable forecaster. forecast(n) -> DataFrame; predict([[n]]) -> the next n values (so the generic predict paths work).
    Allow-listed in app/core/safe_pickle.py; stores no pandas objects."""
    def __init__(self, estimator, spec, history_tail, last_date_iso, n_total, lo_off, hi_off, target, date_col, freq, kind):
        self.estimator, self.spec = estimator, spec
        self.history_tail = np.asarray(history_tail, dtype=float)
        self.last_date, self.n_total = last_date_iso, int(n_total)
        self.lo_off, self.hi_off = np.asarray(lo_off, dtype=float), np.asarray(hi_off, dtype=float)
        self.target, self.date_col, self.freq, self.kind = target, date_col, freq, kind

    def forecast(self, steps: int) -> "pd.DataFrame":
        steps = int(max(1, min(steps, 5000)))
        last = pd.Timestamp(self.last_date)
        vals = recursive_forecast(self.estimator, self.spec, self.history_tail, last, steps, self.n_total)
        dates = pd.date_range(start=last, periods=steps + 1, freq=self.freq)[1:]
        h = np.minimum(np.arange(steps), len(self.lo_off) - 1)
        return pd.DataFrame({"date": dates, "forecast": vals, "lower": vals + self.lo_off[h], "upper": vals + self.hi_off[h]})

    def predict(self, X):
        n = int(float(np.asarray(X).reshape(-1)[0]))
        return self.forecast(n)["forecast"].to_numpy()


# ---------------------------------------------------------------------------- orchestration
class Forecaster:
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir); self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df: "pd.DataFrame", date_col: str, target: str, horizon: Optional[int] = None, agg: str = "sum",
            on_stage=None) -> Dict[str, Any]:
        from sklearn.metrics import mean_absolute_error
        st = on_stage or _noop
        st("prep", "active", "putting the series on a regular time grid")
        s, info = prepare_series(df, date_col, target, agg)
        vals, index, n, season, freq = s.to_numpy(), s.index, len(s), info["season"], info["freq"]
        H = int(horizon) if horizon else (season if season > 1 else 12)
        H = max(1, min(H, max(1, n // 4)))
        need = max(24, 2 * max(season, 3) + H)
        if n < need:
            raise ValueError(f"The series has only {n} points - a forecast of {H} steps at this frequency needs at least {need}.")
        test_n = min(max(H, int(0.2 * n)), n // 3)
        train_end = n - test_n
        st("prep", "done", f"{n} {freq} points, season {season}, horizon {H}; last {test_n} kept as an untouched holdout")

        # model selection: rolling-origin backtest on the END of the training part (never on the holdout)
        st("select", "active", "backtesting candidate models")
        val_n = max(H, int(0.15 * train_end)); val_start = train_end - val_n
        scores: Dict[str, Optional[float]] = {}
        best_key, best_mae, val_errs = ("naive", "level"), None, None
        floor = make_spec(val_start, season, freq, "diff").max_back + 10
        if val_start >= floor:
            learned: Dict[str, Tuple[float, np.ndarray, Tuple[str, str]]] = {}
            simple_val: Dict[str, Tuple[float, np.ndarray]] = {}
            for kind, mode in [("naive", "level")] + CANDIDATES:
                name = _name_of(kind, mode)
                try:
                    est, spec = fit_candidate(kind, mode, vals, index, val_start, season, freq)
                    errs, _p, acts_v, sn_v, lv_v, ma_v, mu_v = backtest(est, spec, vals, index, val_start, train_end, H)
                    mae = float(np.mean(np.abs(errs)))
                    scores[name] = round(mae, 4)
                    if kind == "naive":                        # the four simple methods share one backtest (they need no fitting)
                        for nm, base in (("seasonal_naive", sn_v), ("last_value", lv_v), ("moving_average", ma_v), ("mean_level", mu_v)):
                            e = acts_v - base
                            simple_val[nm] = (float(np.mean(np.abs(e))), e)
                            scores[nm] = round(simple_val[nm][0], 4)
                    else:
                        learned[name] = (mae, errs, (kind, mode))
                except Exception as e:
                    log.warning("forecast candidate %s failed: %s", name, e); scores[name] = None
            if simple_val:
                top_simple = min(simple_val, key=lambda k: simple_val[k][0])
                s_mae, s_errs = simple_val[top_simple]
                top_learned = min(learned, key=lambda k: learned[k][0]) if learned else None
                if top_learned is not None and learned[top_learned][0] < s_mae * (1 - SELECTION_MARGIN):
                    best_key, best_mae, val_errs = learned[top_learned][2], learned[top_learned][0], learned[top_learned][1]
                else:
                    best_key, best_mae, val_errs = SIMPLE_MODELS[top_simple], s_mae, s_errs
            st("select", "done", "best: " + _name_of(*best_key) + (f" (backtest MAE {best_mae:.4g})" if best_mae is not None else ""))
        else:
            best_key = ("hgb", "diff"); scores["note"] = "series too short for model selection - used a default"
            st("select", "skipped", "series too short to compare models - using gradient boosting on seasonal differences")
        kind, mode = best_key
        model_name = _name_of(kind, mode)

        # honest evaluation: fit on the training part only, forecast the holdout from rolling origins
        st("eval", "active", "scoring on the untouched holdout")
        est, spec = fit_candidate(kind, mode, vals, index, train_end, season, freq)
        errs, preds, acts, sn, lv, ma, mu = backtest(est, spec, vals, index, train_end, n, H)
        mae = float(np.mean(np.abs(errs))); rmse = float(np.sqrt(np.mean(errs ** 2)))
        denom = float(np.mean(np.abs(vals[spec.season_lag:train_end] - vals[:train_end - spec.season_lag]))) or 1e-9
        smape = float(np.mean(2 * np.abs(errs) / np.clip(np.abs(acts) + np.abs(preds), 1e-9, None)) * 100)
        base_mae, naive_mae, ma_mae, mean_mae = (float(np.mean(np.abs(acts - sn))), float(np.mean(np.abs(acts - lv))),
                                                 float(np.mean(np.abs(acts - ma))), float(np.mean(np.abs(acts - mu))))
        best_simple = min(base_mae, naive_mae, ma_mae, mean_mae)
        coverage = None
        if val_errs is not None:
            lo, hi = interval_offsets(val_errs)
            coverage = float(np.mean((acts >= preds + lo) & (acts <= preds + hi)))
        metrics: Dict[str, Any] = {
            "mae_test": round(mae, 4), "rmse_test": round(rmse, 4), "smape_test": round(smape, 2), "mase_test": round(mae / denom, 3),
            "baseline_mae_test": round(base_mae, 4), "naive_mae_test": round(naive_mae, 4), "ma_mae_test": round(ma_mae, 4), "mean_mae_test": round(mean_mae, 4),
            "improvement_vs_baseline_pct": round((1 - mae / base_mae) * 100, 1) if base_mae else 0.0,
            "improvement_vs_best_simple_pct": round((1 - mae / best_simple) * 100, 1) if best_simple else 0.0,
            "coverage_80_test": round(coverage, 3) if coverage is not None else None,
            "horizon": H, "n_origins": int(errs.shape[0]), "n_points": n, "frequency": freq, "season": season,
            "cv_metric": "mae_backtest", "cv_mean": round(best_mae, 4) if best_mae is not None else None,
            "candidates": scores, "chosen": model_name}
        st("eval", "done", f"MAE {mae:.4g} vs seasonal-naive {base_mae:.4g}" + (f", interval coverage {coverage:.0%}" if coverage is not None else ""))

        # final model: refit on ALL data; interval offsets from every backtest error we have
        st("fit", "active", "refitting on the full series for the real forecast")
        est2, spec2 = fit_candidate(kind, mode, vals, index, n, season, freq)
        if getattr(spec2, "simple", None) == "mean":
            spec2.simple_param = float(np.mean(vals))
        pooled = errs if val_errs is None else np.vstack([val_errs, errs])
        lo_off, hi_off = interval_offsets(pooled)
        model = _ForecastModel(est2, spec2, vals[-(spec2.max_back + 1):], index[-1].isoformat(), n, lo_off, hi_off, target, date_col, freq, model_name)
        fc = model.forecast(H)
        st("fit", "done", f"forecast of {H} periods ready")

        warns, trust = self._assess(metrics, info, n, season, H)
        metrics["trust"] = trust
        if warns:
            metrics["warnings"] = warns
        if info["notes"]:
            metrics["data_checks"] = [{"kind": "series", "message": m} for m in info["notes"]]
        st("save", "active", "writing the model file")
        path = str(self.model_dir / f"forecast_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        st("save", "done")
        keep = min(n, max(60, 3 * H))
        return {"model": model, "model_path": path, "metrics": _clean(metrics), "forecast": fc, "model_type": f"forecast_{model_name}",
                "info": info, "history": s.iloc[-keep:], "horizon": H, "target": target, "date_col": date_col}

    @staticmethod
    def _assess(metrics: Dict, info: Dict, n: int, season: int, H: int):
        concerns: List[Tuple[int, str]] = []
        warns: List[str] = []
        gain = metrics["improvement_vs_best_simple_pct"]
        simple = metrics["chosen"] in SIMPLE_MODELS
        if not simple and gain <= 1:
            msg = ("Модель не лучше самых простых прогнозов («как в прошлом сезоне», «как вчера», скользящее и общее среднее): "
                   "в этом ряду нет закономерностей, которые можно предсказать лучше.")
            concerns.append((2, msg)); warns.append(msg)
        elif not simple and gain < 5:
            concerns.append((1, f"Выигрыш у самого простого прогноза всего {gain}% - закономерностей в ряду мало."))
        if simple:
            concerns.append((2 if metrics["chosen"] == "mean_level" else 1,
                             f"Лучшим оказалось простое правило «{SIMPLE_RU[metrics['chosen']]}»: в ряду нет закономерностей, которые модель могла бы поймать."
                             + (" Прогноз - просто среднее значение, ряд не предсказуем." if metrics["chosen"] == "mean_level" else "")))
        if metrics["chosen"] == "seasonal_naive":
            if metrics["ma_mae_test"] < metrics["baseline_mae_test"] * 0.95:
                concerns.append((2, "Даже скользящее среднее прогнозирует лучше, чем правило «как в прошлом сезоне»: сезонности в ряду, похоже, нет."))
        if n < max(50, 4 * season):
            concerns.append((1, f"Ряд короткий ({n} точек, сезон {season}): закономерности оценены по нескольким циклам, прогноз неустойчив."))
        if info["gaps"] / max(n, 1) > 0.05:
            concerns.append((1, f"Много пропусков ({info['gaps']} из {n}) - они заполнены интерполяцией."))
        if metrics["n_origins"] < 5:
            concerns.append((1, f"Для проверки доступно всего {metrics['n_origins']} независимых прогнозов - оценка неустойчива."))
        cov = metrics.get("coverage_80_test")
        if cov is not None and not (0.6 <= cov <= 0.95):
            concerns.append((1, f"Интервал «80%» на новых данных покрыл {cov:.0%} значений - границы ненадёжны."))
        worst = max([c[0] for c in concerns], default=0)
        return warns, {"level": {0: "high", 1: "medium", 2: "low"}[worst], "reasons": [t for _, t in sorted(concerns, key=lambda c: -c[0])]}


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
