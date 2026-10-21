"""Run the forecaster on four REAL public series (jbrownlee/Datasets on GitHub) with the last H points hidden.

Prints the engine's MAE next to seasonal-naive and last-value, the share of hidden points inside the interval, the
chosen method and the engine's trust level. One window per series: an illustration, not statistics.
    SECRET_KEY=test DEBUG=true GROQ_API_KEY=gsk_placeholder python3 scripts/public_series_check.py
First run (Oct 17, 2026): airline passengers MAE 15.1 vs seasonal-naive 47.8 (interval hit 42%); monthly car sales:
chose seasonal-naive (equal), hit 58%; female births: seasonal-naive, trust low, hit 100%; min temperatures: chosen
rf_diff MAE 3.0 vs last-value 1.16 (worse than the simplest method), trust low, hit 79%.
Check each dataset's licence/attribution before using it in anything you publish or sell.
"""
import os, sys, tempfile, urllib.request, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
DATA = os.path.join(tempfile.gettempdir(), "genesis_public_series")
os.makedirs(DATA, exist_ok=True)
BASE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
import numpy as np, pandas as pd
from app.core.engine.forecaster import Forecaster

CASES = [  # file, date col, target, horizon, season length for the naive baseline, keep last N rows
    ("airline-passengers", "Month", "Passengers", 12, 12, None),
    ("monthly-car-sales", "Month", "Sales", 12, 12, None),
    ("daily-total-female-births", "Date", "Births", 14, 7, None),
    ("daily-min-temperatures", "Date", "Temp", 14, 7, 730),
]
for name, dc, tc, H, m, keep in CASES:
    path = os.path.join(DATA, name + ".csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{BASE}/{name}.csv", path)
    df = pd.read_csv(path)
    if keep: df = df.iloc[-keep:].reset_index(drop=True)
    train, truth = df.iloc[:-H], df[tc].to_numpy(float)[-H:]
    out = Forecaster(model_dir=tempfile.mkdtemp()).run(train, dc, tc, horizon=H)
    fc = out["forecast"]; p = fc["forecast"].to_numpy(float); lo = fc["lower"].to_numpy(float); hi = fc["upper"].to_numpy(float)
    y = train[tc].to_numpy(float)
    snaive = np.array([y[len(y) - m + (i % m)] for i in range(H)])
    last = np.full(H, y[-1])
    mae = lambda a: float(np.mean(np.abs(truth - a)))
    tr = out["metrics"]["trust"]
    print(f"{name:27s} n_train={len(train):4d} H={H:2d} | MAE model={mae(p):9.2f} seasonal-naive={mae(snaive):9.2f} last-value={mae(last):9.2f} | "
          f"interval hit={np.mean((truth>=lo)&(truth<=hi)):.0%} | chosen={out['metrics']['chosen']} | trust={tr['level'] if isinstance(tr, dict) else tr}", flush=True)
