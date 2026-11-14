"""Один жизненный цикл резерва и результата для оплачиваемого вызова."""
import hashlib
import json
from contextlib import contextmanager
from fastapi import HTTPException
from app.db.reliability import UsageRejected


def operation_key(user_id, scope, key=None):
    if key is None:
        return None
    if not 1 <= len(key) <= 128:
        raise HTTPException(status_code=422, detail="Idempotency-Key must contain 1 to 128 characters")
    return hashlib.sha256(f"{user_id}:{scope}:{key}".encode()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


@contextmanager
def reserved_usage(db, user_id, amount, **kwargs):
    try:
        operation = db.reserve_usage(user_id, amount, **kwargs)
    except UsageRejected as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    try:
        yield operation
    except BaseException:
        if not operation["replayed"]:
            db.finish_usage(operation["id"], refund=True)
        raise
