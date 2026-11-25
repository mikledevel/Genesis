"""Локальный диагностический прогон. Это не сравнение с внешними SaaS."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.engine.forecaster import Forecaster


def make_series(kind, seed, n):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 15 * np.sin(2 * np.pi * t / 7)
    if kind == "seasonal":
        y = 100 + seasonal + rng.normal(0, 3, n)
    elif kind == "trend":
        y = 100 + 0.4 * t + seasonal + rng.normal(0, 3, n)
    elif kind == "noise":
        y = 100 + rng.normal(0, 15, n)
    elif kind == "intermittent":
        y = rng.binomial(1, 0.2, n) * rng.integers(1, 30, n)
    elif kind == "level_shift":
        y = 100 + seasonal + rng.normal(0, 3, n)
        y[-14:] += 60
    else:
        raise ValueError("Unknown synthetic case")
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "sales": y})


def measure(kind, seed, n, horizon, interval_method="empirical"):
    frame = make_series(kind, seed, n + horizon)
    train, actual = frame.iloc[:-horizon], frame.sales.to_numpy()[-horizon:]
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    row = {"case": kind, "seed": seed, "train_points": n, "horizon": horizon,
           "data_sha256": digest, "status": "failed"}
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="genesis-competitive-model-") as model_dir:
            result = Forecaster(model_dir=model_dir).run(train, "date", "sales", horizon=horizon, interval_method=interval_method)
        fc = result["forecast"]
        if len(fc) != horizon:
            raise ValueError("Unexpected forecast horizon")
        pred, lo, hi = (fc[c].to_numpy() for c in ("forecast", "lower", "upper"))
        if not all(np.isfinite(a).all() for a in (pred, lo, hi)) or np.any(lo > hi):
            raise ValueError("Invalid forecast values or intervals")
        history = train.sales.to_numpy()
        simple = {
            "seasonal_naive": history[-7:][np.arange(horizon) % 7],
            "last_value": np.full(horizon, history[-1]),
            "trailing_mean_7": np.full(horizon, np.mean(history[-7:])),
        }
        mae = float(np.mean(np.abs(actual - pred)))
        baseline = {name: float(np.mean(np.abs(actual - values))) for name, values in simple.items()}
        denominator = float(np.sum(np.abs(actual)))
        row.update(status="succeeded", model=result["model_type"], outer_mae=mae,
                   outer_wape=100 * float(np.sum(np.abs(actual - pred))) / denominator if denominator else None,
                   outer_coverage_80=float(np.mean((actual >= lo) & (actual <= hi))),
                   mean_interval_width=float(np.mean(hi - lo)),
                   outer_interval_score=float(np.mean((hi-lo) + 10*np.maximum(lo-actual,0) + 10*np.maximum(actual-hi,0))),
                   interval_method=interval_method,
                   internal_coverage_80=result["metrics"].get("coverage_80_test"),
                   internal_trust=result["metrics"].get("trust"),
                   **{name + "_mae": value for name, value in baseline.items()})
    except Exception as error:
        row.update(error_type=type(error).__name__, error=str(error))
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--interval-method", choices=("empirical", "split"), default="empirical")
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20 or not 100 <= args.points <= 1000:
        parser.error("Use 1..20 seeds and 100..1000 training points")
    rows = []
    for kind in ("seasonal", "trend", "noise", "intermittent", "level_shift"):
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            row = measure(kind, seed, args.points, 14, args.interval_method)
            rows.append(row)
            print(f"{kind} seed={seed}: {row['status']} in {row['seconds']}s", flush=True)
    payload = {"scope": "synthetic local diagnostic, not a competitor benchmark",
               "split": "last 14 future points withheld from the entire engine",
               "platform": platform.platform(), "python": platform.python_version(),
               "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")},
               "cases": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "forecast-baseline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (args.out / "forecast-baseline.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, lineterminator="\n", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return int(any(row["status"] != "succeeded" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
