"""Bring a model trained on the user's OWN computer back to the platform, with a score the platform measured itself.

Flow (all three steps need a login; none of it needs the server-training plan, because the platform spends almost no CPU):
  1. PREPARE   The user picks an uploaded dataset and a target. The platform cleans it exactly like the verified-model path
               (duplicates, ID columns, probable leakage), cuts a hidden test set, keeps the test set on the server and gives
               the user ONLY train.csv.
  2. TRAIN     On the user's machine: `python3 train_local.py train.csv --target <target>` (app/core/local_training.py).
  3. REGISTER  The user uploads model.pkl. The platform loads it with the restricted unpickler (never a plain pickle.load),
               scores it on the hidden test set, adds a trust verdict and registers it - the same numbers and the same
               verdict a model written by the AI code path gets (engine/verification.py).

What this does NOT prove, and the registry says so: the user owns the full dataset, so nothing can stop a seller from training
on the whole file instead of train.csv. The test-set score is real for the rows the platform withheld; "was it trained on
train.csv only" is not something the platform can check. Registered models carry built_via="local_training" and
trained_on="user_machine" so a marketplace page can show that they were not trained by the platform.
"""
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

STORE = Path("datasets") / "local_training"
TTL_SECONDS = 7 * 24 * 3600
MAX_MODEL_BYTES = 25 * 1024 * 1024
_ID_RE = re.compile(r"^[0-9a-f]{16}$")
NOTE = ("Scores were measured by the platform on rows withheld from train.csv. The platform cannot check that the model "
        "was trained on train.csv only: this model was trained on the owner's own computer, not by the platform.")


class LocalModelError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _user_dir(user_id: str) -> Path:
    return STORE / (re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))[:64] or "anonymous")


def _cleanup(now: Optional[float] = None) -> None:
    now = now or time.time()
    if not STORE.exists():
        return
    for d in STORE.glob("*/*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > TTL_SECONDS:
                from app.db.database import GenesisDB
                GenesisDB().delete_artifacts_under(d)
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _match_column(name: str, columns) -> Optional[str]:
    cols = [str(c) for c in columns]
    if name in cols:
        return name
    low = {c.lower().strip(): c for c in cols}
    return low.get(str(name).lower().strip())


def prepare(user_id: str, upload_id: str, target: str) -> Dict[str, Any]:
    from app.core import chat_uploads
    from app.core.engine.verification import load_table, prepare_data
    from app.core.local_training import local_command
    try:
        items = chat_uploads.resolve(user_id, [upload_id])      # looks ONLY in this user's own upload folder
    except chat_uploads.UploadError as e:
        raise LocalModelError(str(e), getattr(e, "status_code", 400))
    if items[0]["kind"] != "dataset":
        raise LocalModelError("That upload is not a table (CSV, XLSX, JSON or Parquet).")
    try:
        df = load_table(str(items[0]["path"]))
    except Exception as e:
        raise LocalModelError(f"Could not read the dataset: {e}")
    col = _match_column(target or "", df.columns)
    if not col:
        raise LocalModelError(f"There is no column '{target}'. Columns: {', '.join(map(str, df.columns[:40]))}")
    plan = prepare_data(df, col)
    if not plan.verifiable:
        raise LocalModelError("Too few rows with a value in the target column to hold back a test set (need about 12).")

    _cleanup()
    pid = uuid.uuid4().hex[:16]
    d = _user_dir(user_id) / pid
    d.mkdir(parents=True, exist_ok=True)
    plan.train.to_csv(d / "train.csv", index=False, encoding="utf-8")
    plan.holdout.to_csv(d / "holdout.csv", index=False, encoding="utf-8")        # never leaves the server
    meta = {"target": col, "task": plan.task, "n_rows": plan.n_rows, "checks": plan.checks, "dropped": plan.dropped,
            "created": time.time()}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
    from app.db.database import GenesisDB
    db = GenesisDB()
    for file in d.iterdir():
        db.persist_artifact(file, user_id)
    return {"prepare_id": pid, "target": col, "task": plan.task, "rows_train": int(len(plan.train)),
            "rows_hidden": int(len(plan.holdout)), "checks": plan.checks, "dropped": plan.dropped,
            "download_url": f"/api/local-training/prepared/{pid}/train.csv",
            "command": local_command("train.csv", col)}


def _dir_for(user_id: str, prepare_id: str) -> Path:
    if not isinstance(prepare_id, str) or not _ID_RE.match(prepare_id):
        raise LocalModelError("Unknown prepared dataset.", 404)
    from app.db.database import GenesisDB
    GenesisDB().restore_owned_artifacts(user_id, "datasets/local_training/")
    d = _user_dir(user_id) / prepare_id                  # the sender's own folder: another user's id is simply not found
    if not (d / "meta.json").exists():
        raise LocalModelError("Unknown or expired prepared dataset - prepare it again.", 404)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if time.time() - float(meta.get("created", 0)) > TTL_SECONDS:
        from app.db.database import GenesisDB
        GenesisDB().delete_artifacts_under(d)
        shutil.rmtree(d, ignore_errors=True)
        raise LocalModelError("This prepared dataset has expired - prepare it again.", 404)
    return d


def train_csv_path(user_id: str, prepare_id: str) -> Path:
    return _dir_for(user_id, prepare_id) / "train.csv"


def _inner(model):
    m = model
    for _ in range(5):
        if hasattr(m, "feature_names_in_") or not hasattr(m, "estimator"):
            break
        m = m.estimator
    return m


def _type_name(model) -> str:
    m = _inner(model)
    steps = getattr(m, "steps", None)
    if steps:
        m = steps[-1][1]
    return type(m).__name__


def register(user_id: str, prepare_id: str, model_bytes: bytes, db, registry=None) -> Dict[str, Any]:
    from app.core import plans
    from app.core.engine.predict_utils import RAW_FORMAT
    from app.core.engine.registry import ModelRegistry
    from app.core.engine.verification import DataPlan, assess, evaluate_holdout
    from app.core.safe_pickle import UnsafePickleError, safe_loads

    limit = plans.gate_message(db, user_id, "model")           # saved-model slots count like any other model
    if limit:
        raise LocalModelError(limit, 403)
    if not model_bytes:
        raise LocalModelError("The model file is empty.")
    if len(model_bytes) > MAX_MODEL_BYTES:
        raise LocalModelError(f"The model file is too large (limit {MAX_MODEL_BYTES // (1024 * 1024)} MB).", 413)
    d = _dir_for(user_id, prepare_id)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    import tempfile
    from app.core.model_isolation import evaluate_model_file
    target = meta["target"]
    plan = DataPlan(train=pd.read_csv(d / "train.csv"), holdout=pd.read_csv(d / "holdout.csv"), target=target,
                    task=meta["task"], n_rows=int(meta["n_rows"]), checks=meta["checks"], dropped=meta["dropped"])
    with tempfile.TemporaryDirectory(prefix="genesis-local-model-") as directory:
        model_path = Path(directory) / "model.pkl"
        model_path.write_bytes(model_bytes)
        try:
            verified = evaluate_model_file(str(model_path),plan)
        except UnsafePickleError as error:
            raise LocalModelError(f"This file was refused and nothing in it was run: {error}") from error
        except Exception as error:
            raise LocalModelError(str(error)) from error
    metrics,contract,features = verified["metrics"],verified["contract"],verified["features"]
    metrics["trust"] = assess(metrics, plan, {})
    metrics["contract"] = contract
    if plan.checks:
        metrics["data_checks"] = plan.checks

    num = [c for c in features if c in plan.train.columns and pd.api.types.is_numeric_dtype(plan.train[c])]
    schema = {"numeric": num, "categorical": [c for c in features if c not in num], "datetime": [], "text": []}
    metadata = {"author": user_id, "built_via": "local_training", "trained_on": "user_machine", "verified": True,
                "trust": metrics["trust"], "data_checks": plan.checks, "local_note": NOTE}
    if contract == "raw_dataframe":
        metadata.update({"input_format": RAW_FORMAT, "input_schema": schema})
    reg = registry or ModelRegistry()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl") as stored:
        stored.write(model_bytes)
        stored.flush()
        model_id = reg.register_file(
        source_path=stored.name,
        metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        task=metrics["task"], model_type=verified["model_type"], features=features, target=target, metadata=metadata)
    return {"model_id": model_id, "task": metrics["task"], "model_type": verified["model_type"], "metrics": metrics,
            "trust": metrics["trust"], "note": NOTE}
