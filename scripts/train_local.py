"""Train a model on YOUR computer with the same engine the platform uses. Nothing is uploaded anywhere.

    python3 train_local.py sales.csv --target churn
    python3 train_local.py sales.xlsx --target revenue --budget thorough --out my_model
    python3 train_local.py new_customers.csv --predict --model my_model/model.pkl

Training writes into --out (default: ./genesis_model):
    model.pkl      the fitted model; it takes a table with the ORIGINAL columns (raw categories, dates, text)
    metrics.json   task, algorithm, scores on rows the model never saw, warnings (leak / duplicates / small data)
With --predict the script loads model.pkl and writes predictions.csv next to your file.

Needs Python 3.10+ and:  pip install pandas numpy scikit-learn xgboost openpyxl
Only open model.pkl files you trained yourself: a pickle file can run code when it is loaded.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..")):          # bundle: engine next to this file; repo: one level up
    if os.path.isdir(os.path.join(_p, "app", "core", "engine")):
        sys.path.insert(0, os.path.abspath(_p))
        break

import pandas as pd  # noqa: E402


def read_table(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(path)
    if low.endswith(".xls"):
        raise SystemExit("Old .xls files are not supported: save the sheet as .xlsx or CSV and run again.")
    if low.endswith(".json"):
        return pd.read_json(path)
    if low.endswith(".parquet"):
        return pd.read_parquet(path)
    last = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise SystemExit(f"Could not decode {path}: {last}")


def infer_task(s: pd.Series) -> str:
    """Same rule the platform uses (engine/verification.py)."""
    u = s.nunique(dropna=True)
    if u <= 2:
        return "binary_classification"
    if pd.api.types.is_numeric_dtype(s):
        return "multiclass_classification" if (u / max(len(s), 1) < 0.05 and u <= 20) else "regression"
    return "multiclass_classification" if u <= 20 else "regression"


def _resolve_column(name: str, columns) -> str:
    cols = [str(c) for c in columns]
    if name in cols:
        return name
    low = {c.lower().strip(): c for c in cols}
    if name.lower().strip() in low:
        return low[name.lower().strip()]
    raise SystemExit(f"There is no column '{name}'. Columns: {', '.join(cols[:60])}")


def cmd_train(args) -> int:
    from app.core.engine.trainer import DatasetTrainer
    df = read_table(args.data)
    df.columns = df.columns.astype(str)
    if not args.target:
        raise SystemExit("Say which column to predict: --target <column>. Columns: " + ", ".join(map(str, df.columns[:60])))
    target = _resolve_column(args.target, df.columns)
    if len(df.dropna(subset=[target])) < 20:
        raise SystemExit("Fewer than 20 rows have a value in the target column: too little to learn from.")
    task = args.task if args.task != "auto" else infer_task(df[target].dropna())
    exclude = [_resolve_column(c, df.columns) for c in (args.exclude or [])]
    keep = [_resolve_column(c, df.columns) for c in (args.keep or [])]

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    print(f"Data: {len(df)} rows x {len(df.columns)} columns · target: {target} · task: {task} · "
          f"algorithm: {args.algorithm} · budget: {args.budget}")
    print("Training on this computer (a few seconds to a few minutes)...")

    def on_stage(step, status, detail=""):
        if status == "done" and detail:
            print(f"  [{step}] {detail}")

    trainer = DatasetTrainer(model_dir=os.path.join(out, "_tmp"))
    res = trainer.train(df=df, target_col=target, task=task, model_type=args.algorithm, budget=args.budget,
                        on_stage=on_stage, keep_columns=keep or None, exclude_columns=exclude or None)

    model_pkl = os.path.join(out, "model.pkl")
    shutil.copyfile(res.model_path, model_pkl)
    shutil.rmtree(os.path.join(out, "_tmp"), ignore_errors=True)
    summary = {"target": target, "task": res.task, "model_type": res.model_type, "metrics": res.metrics,
               "feature_columns": res.feature_columns, "input_schema": res.input_schema,
               "target_classes": res.target_classes, "feature_importance": res.feature_importance,
               "training_info": res.training_info}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDone. Best algorithm: {res.model_type}")
    shown = {k: v for k, v in res.metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for k, v in list(shown.items())[:12]:
        print(f"  {k}: {round(v, 4)}")
    for w in (res.metrics.get("warnings") or [])[:6]:
        print(f"  ! {w}")
    print(f"\nSaved: {model_pkl}\n       {os.path.join(out, 'metrics.json')}")
    print(f"Predict on new rows:  python3 train_local.py new_rows.csv --predict --model {os.path.relpath(model_pkl)}")
    return 0


def cmd_predict(args) -> int:
    import pickle
    if not args.model or not os.path.exists(args.model):
        raise SystemExit("Give the trained model: --model path/to/model.pkl")
    df = read_table(args.data)
    df.columns = df.columns.astype(str)
    with open(args.model, "rb") as f:
        model = pickle.load(f)          # your own file, trained on this machine (see the warning at the top)
    need = getattr(model, "feature_names_in_", None)
    if need is not None:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit("The file is missing columns the model was trained on: " + ", ".join(map(str, missing)))
        X = df[list(need)]
    else:
        X = df
    out = df.copy()
    out["prediction"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            out["probability"] = proba.max(axis=1) if proba.ndim == 2 else proba
        except Exception:
            pass
    dest = os.path.join(os.path.dirname(os.path.abspath(args.data)), "predictions.csv")
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} predictions to {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="path to a .csv (; or , or tab; utf-8 or cp1251), .xlsx, .json or .parquet file")
    ap.add_argument("--target", help="the column to predict")
    ap.add_argument("--task", default="auto", choices=["auto", "binary_classification", "multiclass_classification", "regression"])
    ap.add_argument("--algorithm", default="auto",
                    choices=["auto", "xgboost", "random_forest", "hist_gradient_boosting", "logistic", "linear"],
                    help="auto compares them all and keeps the best")
    ap.add_argument("--model", help="with --predict: path to the model.pkl produced by training")
    ap.add_argument("--budget", default="balanced", choices=["fast", "balanced", "thorough"])
    ap.add_argument("--exclude", nargs="*", help="columns the model must not use")
    ap.add_argument("--keep", nargs="*", help="columns to keep even if they look like IDs or leaks")
    ap.add_argument("--out", default="genesis_model")
    ap.add_argument("--predict", action="store_true", help="predict with an existing model instead of training")
    args = ap.parse_args()
    if args.predict:
        return cmd_predict(args)
    return cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
