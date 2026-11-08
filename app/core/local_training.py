"""Train on the user's OWN computer instead of the platform's server.

Why: server training costs the platform CPU and memory, so it is a paid feature (plans.py, "server_training").
Training on your own machine costs the platform nothing, so it is free for everybody.

What the user gets: a small zip with the SAME trainer the server uses (leak-free pipeline, tuning, imbalance
handling, dates and text, data audit) plus one command-line script. The platform never touches the user's
machine: it only hands out code, and the user decides whether to run it. The code is a fixed, public part
of this repository - NOT code written by an LLM per request - so there is nothing generated to trust.
The data and the trained model stay on the user's computer; nothing is uploaded and nothing is registered.
"""
import io
import os
import re
import shlex
import threading
import zipfile
from typing import Dict, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BUNDLE_URL = "/api/local-training/bundle"
BUNDLE_NAME = "genesis-local-trainer.zip"
_TOP = "genesis-local-trainer"

# (path in the repo, path inside the zip)
_FILES = [
    ("scripts/train_local.py", "train_local.py"),
    ("app/core/engine/trainer.py", "app/core/engine/trainer.py"),
    ("app/core/engine/transformers.py", "app/core/engine/transformers.py"),
    ("app/core/engine/predict_utils.py", "app/core/engine/predict_utils.py"),
]
_EMPTY_INITS = ["app/__init__.py", "app/core/__init__.py", "app/core/engine/__init__.py"]

REQUIREMENTS = """pandas>=2.1.3
numpy>=1.26.2
scikit-learn>=1.3.2
xgboost>=2.0.2
openpyxl>=3.1.0
"""

README = """Genesis AI - local trainer
==========================

Trains a model on YOUR computer. Your data and the model never leave it.

1. Install Python 3.10 or newer, then in this folder:
       python3 -m venv .venv
       source .venv/bin/activate          (Windows: .venv\\\\Scripts\\\\activate)
       pip install -r requirements-local.txt

2. Train (put your file next to this folder or give its full path):
       python3 train_local.py sales.csv --target churn

   Useful options:
       --budget fast|balanced|thorough      how long to search for better settings
       --algorithm auto|xgboost|random_forest|hist_gradient_boosting|logistic|linear
       --exclude col1 col2                  columns the model must not use
       --out my_model                       output folder (default: genesis_model)

   You get my_model/model.pkl and my_model/metrics.json. The scores are measured on rows the model never saw.
   Read the warnings in metrics.json: the trainer drops ID columns and columns that give away the answer.

3. Predict on new rows (same columns as the training file, without the target):
       python3 train_local.py new_rows.csv --predict --model genesis_model/model.pkl

Keep this folder as it is: model.pkl needs the files in app/ to be loaded.
Only open model.pkl files you trained yourself (a pickle file can run code when it is loaded).
"""

_cache: Dict[str, object] = {"sig": None, "data": None}
_lock = threading.Lock()


def _signature() -> Tuple:
    sig = []
    for src, _ in _FILES:
        p = os.path.join(ROOT, src)
        sig.append((src, os.path.getmtime(p) if os.path.exists(p) else None))
    return tuple(sig)


def build_bundle() -> bytes:
    """The zip as bytes. Built from the repo files, cached until one of them changes."""
    with _lock:
        sig = _signature()
        if _cache["sig"] == sig and _cache["data"]:
            return _cache["data"]  # type: ignore[return-value]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for src, dst in _FILES:
                path = os.path.join(ROOT, src)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"local trainer source missing: {src}")
                z.write(path, f"{_TOP}/{dst}")
            for dst in _EMPTY_INITS:
                z.writestr(f"{_TOP}/{dst}", "")
            z.writestr(f"{_TOP}/requirements-local.txt", REQUIREMENTS)
            z.writestr(f"{_TOP}/README.txt", README)
        _cache["sig"], _cache["data"] = sig, buf.getvalue()
        return buf.getvalue()


def _safe_name(value: Optional[str], fallback: str) -> str:
    """A file or column name that is safe to show inside a shell command (it is quoted, and stripped of control chars)."""
    v = re.sub(r"[\x00-\x1f\x7f]", "", str(value or "")).strip()
    return v if v else fallback


def local_command(dataset_file: Optional[str] = None, target: Optional[str] = None, budget: Optional[str] = None) -> str:
    """The one-line command the user runs in the unpacked folder. Names are quoted for the shell."""
    name = _safe_name(os.path.basename(dataset_file or ""), "your_data.csv")
    parts = ["python3", "train_local.py", shlex.quote(name)]
    if target:
        parts += ["--target", shlex.quote(_safe_name(target, "target"))]
    else:
        parts += ["--target", "<column to predict>"]
    if budget in ("fast", "balanced", "thorough"):
        parts += ["--budget", budget]
    return " ".join(parts)


def local_instructions(dataset_file: Optional[str] = None, target: Optional[str] = None,
                       budget: Optional[str] = None) -> str:
    """Chat text: how to train on your own computer. Used by the agent and by the plan-limit message."""
    return (
        f"Train it on your own computer (free, your data never leaves it):\n"
        f"1. Download the trainer: {BUNDLE_URL} and unzip it.\n"
        f"2. In that folder: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-local.txt\n"
        f"3. Put your file there and run:\n"
        f"   {local_command(dataset_file, target, budget)}\n"
        f"You get model.pkl and metrics.json (scores on rows the model never saw). "
        f"I only give you the command; I never run anything on your computer."
    )
