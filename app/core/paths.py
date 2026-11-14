"""Which files a request may point the platform at (datasets to profile / train on, models, generated files).

A client-supplied path is only used after is_allowed_path() says yes. The check compares real paths (symlinks and ../ resolved)
and requires the allowed folder itself or a path BELOW it: a plain startswith("/app/datasets") would also accept the sibling
folder "/app/datasets_backup"."""
import os
from typing import Iterable, Optional

ALLOWED_DIRS = ["./datasets", "./models", "./static", "./generated"]


def is_allowed_path(path: str, dirs: Optional[Iterable[str]] = None) -> bool:
    if not isinstance(path,(str,os.PathLike)) or not path or "\x00" in str(path):
        return False
    rp = os.path.realpath(path)
    for d in (ALLOWED_DIRS if dirs is None else dirs):
        rd = os.path.realpath(d)
        if rp == rd or rp.startswith(rd.rstrip(os.sep) + os.sep):
            return True
    return False


def is_owned_path(path: str, user_id: Optional[str], dirs: Optional[Iterable[str]] = None) -> bool:
    """Общие демоданные доступны всем, пользовательские каталоги — только владельцу."""
    if not is_allowed_path(path, dirs):
        return False
    from pathlib import Path
    from app.core.chat_uploads import user_dir
    resolved = Path(path).resolve()
    for root in ("datasets/uploads", "datasets/local_training", "datasets/builds"):
        base = Path(root).resolve()
        if resolved == base:
            return False
        if base in resolved.parents:
            owner_root = base / user_dir(user_id or "anonymous").name
            return bool(user_id and user_id != "anonymous" and owner_root in resolved.parents)
    models = Path("models").resolve()
    if resolved == models or models in resolved.parents:
        owner_root = models / user_dir(user_id or "anonymous").name
        return bool(user_id and user_id != "anonymous" and owner_root in resolved.parents)
    if Path("generated").resolve() in resolved.parents:
        return bool(user_id and resolved.name.startswith(f"user_{user_id}_"))
    return True
