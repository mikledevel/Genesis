"""CSV or Excel (.xlsx, first sheet) -> one-page forecast report (HTML). Open it in a browser; Print -> Save as PDF gives a clean PDF.

    python3 scripts/make_report.py sales.csv --client "ACME Shop" --out acme.html
    python3 scripts/make_report.py sales.csv --date order_date --target revenue --horizon 28 --lang ru

Columns are guessed if you do not pass --date / --target; the report lists what was guessed, check it.
Everything runs locally; the file is not sent anywhere. The report has no scripts and no remote resources.
"""
import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402


def read_table(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".xlsx", ".xlsm")):             # marketplace and 1C exports usually arrive as Excel: first sheet is read
        return pd.read_excel(path)
    if low.endswith(".xls"):
        raise SystemExit("Old .xls files are not supported: save the sheet as .xlsx or CSV and run again.")
    last = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise SystemExit(f"Could not decode {path}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to a .csv (; or , or tab; utf-8 or cp1251) or .xlsx file")
    ap.add_argument("--date"), ap.add_argument("--target")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--agg", default="sum", choices=["sum", "mean"])
    ap.add_argument("--client", default=""), ap.add_argument("--title", default="")
    ap.add_argument("--lang", default="en", choices=["en", "ru"])
    ap.add_argument("--out")
    a = ap.parse_args()
    from app.core.report import build_forecast_report
    try:
        res = build_forecast_report(read_table(a.csv), a.date, a.target, a.horizon, a.agg, a.client, a.title, a.lang)
    except ValueError as e:
        print(f"Cannot build the report: {e}", file=sys.stderr)
        return 2
    s = res["summary"]
    out = a.out or f"report_{s['target']}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(res["html"])
    print(f"Wrote {out}  |  target={s['target']} date={s['date_col']} horizon={s['horizon']} trust={s['trust']} "
          f"vs-best-simple={s['improvement_vs_best_simple_pct']}% coverage={s['coverage_80_test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
