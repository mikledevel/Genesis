"""Out-of-sample calibration check for app/core/engine/forecaster.py (synthetic series, known future).

For each series type: hide the last 14 points, forecast them, then report
  coverage80      share of the hidden true values inside the "80%" interval (nominal 0.80)
  bias            mean(actual - forecast); positive = the forecast is too low
  point_position  where the point forecast sits inside its interval (0.5 = centered)
Run (about 6-8 minutes, so run it in the background):
    SECRET_KEY=test DEBUG=true GROQ_API_KEY=gsk_placeholder python3 scripts/forecast_calibration_check.py

Result of the first run (Oct 17, 2026, 8-12 series per row): coverage 66-75% instead of 80%; on series with
an upward trend the point forecast sits low (position 0.21-0.30, bias > 0); noise is correctly rated "low".
"""
import sys, warnings, tempfile, math, datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from app.core.engine.forecaster import Forecaster

def make(kind, seed, n=300):
    r = np.random.default_rng(seed); i = np.arange(n)
    season = 15 * np.sin(2 * math.pi * i / 7)
    if kind == "trend+season":   y = 100 + 0.2 * i + season + r.normal(0, 3, n)
    elif kind == "season_only":  y = 100 + season + r.normal(0, 3, n)
    elif kind == "strong_trend": y = 50 + 0.8 * i + season + r.normal(0, 4, n)
    elif kind == "noise":        y = 100 + r.normal(0, 10, n)
    return y

H = 14
res = {}
for kind in ["season_only", "trend+season", "strong_trend", "noise"]:
    cov, bias, lowpos, mape, picks = [], [], [], [], {}
    for seed in range(8):
        y = make(kind, seed, 300 + H)
        d0 = datetime.date(2025, 1, 1)
        df = pd.DataFrame({"date": [d0 + datetime.timedelta(days=int(k)) for k in range(len(y))], "sales": y})
        train, truth = df.iloc[:-H], y[-H:]
        out = Forecaster(model_dir=tempfile.mkdtemp()).run(train, "date", "sales", horizon=H)
        fc = out["forecast"]
        cols = list(fc.columns); 
        pc = [c for c in cols if "forecast" in c.lower() or c.lower() in ("yhat", "prediction", "value")][0]
        lo = [c for c in cols if "lower" in c.lower() or c.lower().endswith("_lo") or "low" in c.lower()][0]
        hi = [c for c in cols if "upper" in c.lower() or c.lower().endswith("_hi") or "high" in c.lower()][0]
        p, l, h = fc[pc].to_numpy(), fc[lo].to_numpy(), fc[hi].to_numpy()
        cov.append(np.mean((truth >= l) & (truth <= h)))
        bias.append(np.mean(truth - p))
        lowpos.append(np.mean((p - l) / np.maximum(h - l, 1e-9)))
        mape.append(np.mean(np.abs(truth - p) / np.abs(truth)) * 100)
        m = out["model_type"]; picks[m] = picks.get(m, 0) + 1
    res[kind] = (np.mean(cov), np.mean(bias), np.mean(lowpos), np.mean(mape), picks, out["metrics"]["trust"] if isinstance(out["metrics"]["trust"], str) else out["metrics"]["trust"].get("level", out["metrics"]["trust"]))
    c, b, lp, mp, pk, tr = res[kind]
    print(f"{kind:13s} coverage80={c:.0%} bias(actual-forecast)={b:+.2f} point_position_in_interval={lp:.2f} (0.5=centered) MAPE={mp:.1f}% models={pk} last_trust={tr}", flush=True)
