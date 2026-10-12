"""Check that every model file already on disk still loads under the restricted unpickler.

Run this ONCE after upgrading (needs scikit-learn / xgboost installed, same as the app):

    python scripts/check_model_pickles.py

For each .pkl under ./registry/models and ./models it prints OK or BLOCKED with the reason.
A BLOCKED model would fail to predict after the upgrade: either retrain it, or - only if the
named global is a harmless data/reconstruction helper - add it to app/core/safe_pickle.py.
It also lists files that look malicious (os/subprocess/eval) so you can delete them.
Exit code 1 if anything was blocked.
"""
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from app.core.safe_pickle import safe_load, UnsafePickleError  # noqa: E402


def main() -> int:
    files = sorted(glob.glob(os.path.join(ROOT, "registry", "models", "*.pkl"))
                   + glob.glob(os.path.join(ROOT, "models", "*.pkl")))
    if not files:
        print("No model files found.")
        return 0
    blocked = 0
    for path in files:
        rel = os.path.relpath(path, ROOT)
        try:
            with open(path, "rb") as f:
                safe_load(f)
            print(f"OK       {rel}")
        except UnsafePickleError as e:
            blocked += 1
            print(f"BLOCKED  {rel}\n         {e}")
        except Exception as e:   # corrupt file, missing library, ...
            blocked += 1
            print(f"ERROR    {rel}\n         {type(e).__name__}: {e}")
    print(f"\n{len(files) - blocked} ok, {blocked} need attention.")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
