"""Which files a request may point the platform at (datasets to profile / train on, models, generated files).

A client-supplied path is only used after is_allowed_path() says yes. The check compares real paths (symlinks and ../ resolved)
and requires the allowed folder itself or a path BELOW it: a plain startswith("/app/datasets") would also accept the sibling
folder "/app/datasets_backup"."""
import os
from typing import Iterable, Optional

ALLOWED_DIRS = ["./datasets", "./models", "./static", "./generated"]


def is_allowed_path(path: str, dirs: Optional[Iterable[str]] = None) -> bool:
    if not path or "\x00" in str(path):
        return False
    rp = os.path.realpath(path)
    for d in (ALLOWED_DIRS if dirs is None else dirs):
        rd = os.path.realpath(d)
        if rp == rd or rp.startswith(rd.rstrip(os.sep) + os.sep):
            return True
    return False
