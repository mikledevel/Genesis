"""Platform-side verification of LLM-written models (the `create_model` path).

Why this exists: a script written by an LLM used to (a) never actually SEE the dataset - the sandbox
only mounts its own work directory, so the host path in the prompt did not exist inside it - and (b)
report its OWN metrics, which the platform accepted as truth. Now the platform
  1. loads the data itself, removes surprising duplicates / ID columns / probable target leakage,
  2. splits off a HIDDEN TEST SET the script never receives (the script gets only `train.csv`),
  3. loads the model the script saved (restricted unpickler) and SCORES it on the hidden set itself,
  4. compares its numbers with what the script claimed and says so if the claim was inflated,
  5. attaches a trust verdict, exactly like the built-in trainer does.
The contract the script must meet: predict() accepts a DataFrame with the ORIGINAL feature columns and
returns the ORIGINAL target labels (a legacy numeric-vector model is still accepted if all features
are numeric). Isolation caveat: with the Docker sandbox the script cannot read anything but train.csv;
with the dev-only subprocess fallback a malicious script could read other files - that fallback is
documented as not for untrusted users.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.engine.transformers import looks_like_id


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    loaders = {".csv": pd.read_csv, ".json": pd.read_json, ".parquet": pd.read_parquet, ".xlsx": pd.read_excel}
    if ext not in loaders:
        raise ValueError(f"Unsupported format: {ext}")
    df = loaders[ext](path)
    df.columns = df.columns.astype(str)
    return df


def infer_task(s: pd.Series) -> str:
    u = s.nunique(dropna=True)
    numeric = pd.api.types.is_numeric_dtype(s)
    if u <= 2:
        return "binary_classification"
    if numeric:
        return "multiclass_classification" if (u / max(len(s), 1) < 0.05 and u <= 20) else "regression"
    return "multiclass_classification" if u <= 20 else "regression"


@dataclass
class DataPlan:
    train: pd.DataFrame
    holdout: pd.DataFrame
    target: Optional[str]
    task: Optional[str]
    n_rows: int
    checks: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)

    @property
    def verifiable(self) -> bool:
        return len(self.holdout) >= 5


def prepare_data(df: pd.DataFrame, target: Optional[str] = None, test_size: float = 0.2, seed: int = 42,
                 keep_columns: Optional[List[str]] = None) -> DataPlan:
    from sklearn.model_selection import train_test_split
    df = df.copy()
    keep = set(keep_columns or [])
    checks: List[Dict[str, Any]] = []
    dropped: List[str] = []
    if target and target in df.columns:
        df = df.dropna(subset=[target])
    else:
        target = None

    # 1. duplicates - only when a repeat is SURPRISING (same rule as the built-in trainer)
    if len(df) >= 50:
        combos_log10 = sum(np.log10(max(2, df[c].nunique(dropna=True))) for c in df.columns)
        dups = int(df.duplicated().sum()) if combos_log10 >= np.log10(len(df)) + 2 else 0
        if dups >= max(2, 0.005 * len(df)):
            df = df.drop_duplicates().reset_index(drop=True)
            checks.append({"kind": "duplicates", "rows": dups, "message":
                           f"Удалено {dups} дублирующихся строк: иначе одинаковые строки попали бы и в обучение, и в тест."})

    # 2. identifier columns
    feats = [c for c in df.columns if c != target]
    ids = [c for c in feats if c not in keep and looks_like_id(df[c], c)]
    if ids and len(feats) > len(ids):
        df = df.drop(columns=ids)
        dropped += ids
        checks.append({"kind": "id", "columns": ids, "message":
                       f"Исключены ID-столбцы (уникальны почти в каждой строке): {', '.join(ids)}."})

    task = infer_task(df[target]) if target else None
    clf = task is not None and "classification" in task
    if len(df) < 10:
        return DataPlan(train=df, holdout=df.iloc[0:0], target=target, task=task, n_rows=len(df), checks=checks, dropped=dropped)
    strat = df[target] if (clf and df[target].value_counts().min() >= 2) else None
    train, holdout = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
    train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)

    # 3. leakage scan on the TRAINING rows only (needs the target)
    if target and len(train) >= 60:
        from app.core.engine.trainer import DatasetTrainer
        feats = [c for c in train.columns if c != target]
        num = [c for c in feats if pd.api.types.is_numeric_dtype(train[c])]
        cat = [c for c in feats if c not in num]
        suspects = {c: s for c, s in DatasetTrainer.__new__(DatasetTrainer)._leakage_scores(
            train[feats], train[target], num, cat, clf, target).items() if c not in keep}
        if suspects and len(feats) > len(suspects):
            for c, s in suspects.items():
                checks.append({"kind": "leak", "columns": [c], "score": round(s, 4), "message":
                               f"Столбец «{c}» почти полностью предсказывает цель (оценка {s:.3f}) - похоже на утечку данных. "
                               f"Он исключён из обучающего файла. Если он законный, скажите «оставь столбец {c}»."})
            train, holdout = train.drop(columns=list(suspects)), holdout.drop(columns=list(suspects))
            dropped += list(suspects)
    return DataPlan(train=train, holdout=holdout, target=target, task=task, n_rows=len(df), checks=checks, dropped=dropped)


def evaluate_holdout(model, plan: DataPlan, target: str, features: List[str]) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """Score `model` on the hidden test set. -> (metrics, error_for_the_LLM, contract).
    contract is "raw_dataframe" or "numeric_vector"."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                                 mean_squared_error, r2_score, roc_auc_score)
    from app.core.engine.trainer import DatasetTrainer

    missing = [c for c in features if c not in plan.holdout.columns]
    if missing or target not in plan.holdout.columns:
        removed = f" The platform removed these columns from train.csv (ID / probable leakage): {', '.join(plan.dropped)}." if plan.dropped else ""
        return None, (f"METRICS_JSON 'features'/'target' refer to columns that are not in train.csv: "
                      f"{missing or [target]}. Use only the columns of train.csv.{removed}"), None
    X, y = plan.holdout[features], plan.holdout[target]
    task = plan.task or infer_task(plan.train[target])
    clf = "classification" in task

    pred, contract, first_error = None, None, None
    try:
        pred, contract = np.asarray(model.predict(X)), "raw_dataframe"
    except Exception as e1:
        first_error = e1
        try:
            pred, contract = np.asarray(model.predict(X.to_numpy(dtype=float))), "numeric_vector"
        except Exception:
            return None, (f"The saved model cannot predict a DataFrame with the original columns {features} "
                          f"(error: {first_error}). Save ONE sklearn Pipeline - ColumnTransformer for imputation / "
                          f"encoding inside, then the estimator - so predict() accepts the raw columns of train.csv "
                          f"(strings, dates, missing values) and returns the original target labels."), None
    if len(pred) != len(y):
        return None, f"predict() returned {len(pred)} values for {len(y)} rows.", contract
    if clf:
        known = set(map(str, plan.train[target].unique()))
        if not set(map(str, pred)) <= known:
            return None, ("predict() must return the ORIGINAL target labels "
                          f"({sorted(known)[:6]}...), not encoded values like {sorted(set(map(str, pred)))[:4]}. "
                          "Decode inside the pipeline or train directly on the original labels."), contract

    metrics: Dict[str, Any] = {"n_test": int(len(y)), "task": task, "verified": True}
    if clf:
        yt, yp = y.astype(str).to_numpy(), pred.astype(str)
        acc = float(accuracy_score(yt, yp))
        base = float((y.astype(str) == plan.train[target].astype(str).mode().iloc[0]).mean())
        metrics.update({"accuracy_test": round(acc, 4), "f1_test": round(float(f1_score(yt, yp, average="weighted")), 4),
                        "balanced_accuracy_test": round(float(balanced_accuracy_score(yt, yp)), 4),
                        "baseline_accuracy_test": round(base, 4),
                        "accuracy_test_ci": DatasetTrainer._wilson(int(round(acc * len(y))), len(y))})
        if task == "binary_classification" and hasattr(model, "predict_proba") and contract == "raw_dataframe":
            try:
                classes = [str(c) for c in model.classes_]
                if len(classes) == 2 and y.nunique() == 2:
                    metrics["roc_auc_test"] = round(float(roc_auc_score(
                        (y.astype(str) == classes[1]).to_numpy(), model.predict_proba(X)[:, 1])), 4)
            except Exception:
                pass
    else:
        yt, yp = y.astype(float).to_numpy(), pred.astype(float)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        base = float(np.sqrt(np.mean((yt - float(plan.train[target].astype(float).mean())) ** 2)))
        metrics.update({"r2_test": round(float(r2_score(yt, yp)), 4), "rmse_test": round(rmse, 2),
                        "mae_test": round(float(mean_absolute_error(yt, yp)), 2), "baseline_rmse_test": round(base, 2),
                        "r2_test_ci": DatasetTrainer._bootstrap_r2(yt, yp)})
    return metrics, None, contract


def assess(metrics: Dict, plan: DataPlan, self_reported: Dict) -> Dict:
    """Trust verdict for a generated model: same spirit as DatasetTrainer._assess_trust."""
    concerns: List[Tuple[int, str]] = []
    clf = "classification" in metrics["task"]
    n_test = metrics["n_test"]
    ci = metrics.get("accuracy_test_ci") or metrics.get("r2_test_ci")
    if n_test < 100:
        concerns.append((1, f"Скрытая тестовая выборка мала ({n_test} строк) - реальная метрика может заметно отличаться."))
    elif ci and ci[1] - ci[0] > (0.15 if clf else 0.3):
        concerns.append((1, f"Доверительный интервал метрики широк ({ci[0]}-{ci[1]})."))
    if plan.n_rows < 200:
        concerns.append((1, f"Мало данных ({plan.n_rows} строк)."))
    if clf and metrics["accuracy_test"] <= metrics["baseline_accuracy_test"] + 0.01:
        concerns.append((2, "Модель не лучше простого наивного прогноза (всегда самый частый класс)."))
    elif clf and ci and ci[0] <= metrics["baseline_accuracy_test"]:
        concerns.append((1, f"Интервал точности {ci[0]}-{ci[1]} включает наивный прогноз ({metrics['baseline_accuracy_test']})."))
    if not clf and metrics["rmse_test"] >= metrics["baseline_rmse_test"] * 0.99:
        concerns.append((2, "Модель не лучше простого наивного прогноза (среднее значение)."))
    claimed = self_reported.get("accuracy" if clf else "r2")
    verified = metrics.get("accuracy_test" if clf else "r2_test")
    if isinstance(claimed, (int, float)) and verified is not None and claimed - verified > 0.05:
        gap = claimed - verified
        concerns.append((2 if gap > 0.15 else 1, f"Скрипт заявил {'accuracy' if clf else 'R²'} {claimed:.3f}, а платформа на скрытом "
                                                 f"тесте получила {verified:.3f}: заявленная метрика была завышена."))
    worst = max([c[0] for c in concerns], default=0)
    return {"level": {0: "high", 1: "medium", 2: "low"}[worst], "reasons": [t for _, t in sorted(concerns, key=lambda c: -c[0])]}
