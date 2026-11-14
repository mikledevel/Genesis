"""What the platform itself got out of a model build, kept on the server so that a later "publish" can't be fed made-up values.

POST /api/model-builder/generate runs the build and stores {saved_model_path, metrics, verification} here, per user, and returns a
`build_id`. POST /api/model-builder/publish then takes the model file, the scores and the verdict from THIS record - never from the
request body: before, the client sent `saved_model_path` (a server file path) and `metrics` (the numbers shown on the listing) back,
so anyone could register another user's model file or advertise any accuracy.
"""
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

STORE = Path("datasets") / "builds"
_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def _dir(user_id: str) -> Path:
    return STORE / (re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))[:64] or "anonymous")


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except Exception:
        return False


def save(user_id: str, result: Dict[str, Any]) -> Optional[str]:
    """Keep a successful build; returns its build_id (None if there is nothing worth publishing)."""
    from app.db.database import GenesisDB
    path = result.get("saved_model_path")
    if not result.get("success") or not path or not os.path.exists(path):
        return None
    bid = uuid.uuid4().hex[:16]
    d = _dir(user_id)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"build_id": bid, "created": time.time(), "saved_model_path": os.path.relpath(path),
           "metrics": result.get("metrics") or {}, "verified": bool(result.get("verified")),
           "verification": result.get("verification"), "model_id": None}
    (d / f"{bid}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
    db = GenesisDB()
    db.persist_artifact(path, user_id)
    db.persist_artifact(d / f"{bid}.json", user_id)
    return bid


def find(user_id: str, build_id: Optional[str] = None, saved_model_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """This user's own build, by id or (for older clients) by the model path that generate returned. Never another user's."""
    from app.db.database import GenesisDB
    GenesisDB().restore_owned_artifacts(user_id, "datasets/builds/")
    d = _dir(user_id)
    if build_id:
        if not isinstance(build_id, str) or not _ID_RE.match(build_id):
            return None
        p = d / f"{build_id}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    if saved_model_path and d.exists():
        for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if _same_file(rec.get("saved_model_path", ""), str(saved_model_path)):
                return rec
    return None


def mark_registered(user_id: str, build_id: str, model_id: str) -> None:
    p = _dir(user_id) / f"{build_id}.json"
    if p.exists():
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["model_id"] = model_id
        p.write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
        from app.db.database import GenesisDB
        GenesisDB().persist_artifact(p, user_id)

