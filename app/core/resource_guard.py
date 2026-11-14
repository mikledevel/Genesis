"""Сериализация создания ресурсов аккаунта перед проверкой и записью лимита."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
from pathlib import Path
import threading
from fastapi import HTTPException

_local = threading.local()


@contextmanager
def account_creation(db,user_id):
    identity = (db.database_url or str(Path(db.db_path).resolve())) + ":" + user_id
    held = getattr(_local,"held",set())
    if identity in held:
        yield
        return
    digest = hashlib.sha256(identity.encode()).hexdigest()
    handle,conn = None,None
    try:
        if db.is_postgres:
            import psycopg
            conn = psycopg.connect(db.database_url,autocommit=True,connect_timeout=5)
            key = int(digest[:15],16)
            if not conn.execute("SELECT pg_try_advisory_lock(%s)",(key,)).fetchone()[0]:
                raise HTTPException(status_code=409,detail="Another resource creation is running for this account")
        else:
            import fcntl
            folder = Path(db.db_path).resolve().parent / ".genesis-locks"
            folder.mkdir(exist_ok=True)
            handle = (folder/digest).open("a+b")
            try:
                fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise HTTPException(status_code=409,detail="Another resource creation is running for this account") from error
        _local.held = held | {identity}
        yield
    finally:
        _local.held = held
        if conn:
            conn.close()
        if handle:
            handle.close()


def resource_guard(kind,user_arg="user_id"):
    def decorate(method):
        @wraps(method)
        def run(*args,**kwargs):
            bound = inspect.signature(method).bind(*args,**kwargs)
            instance = bound.arguments.get("self")
            user = bound.arguments.get(user_arg)
            if user_arg == "config":
                user = user.author
            elif user_arg == "metadata":
                user = (user or {}).get("author")
            elif user_arg == "current_user_id":
                user = getattr(instance,"current_user_id",None)
            from app.db.database import GenesisDB
            db = getattr(instance,"db",None) or GenesisDB()
            if not isinstance(getattr(db,"database_url",None),str):
                return method(*args,**kwargs)
            if not user or user == "anonymous" or db.get_session_state(user) is None:
                return method(*args,**kwargs)
            with account_creation(db,user):
                from app.core.plans import gate_message
                reason = gate_message(db,user,kind) if kind else None
                if reason:
                    raise HTTPException(status_code=403,detail=reason)
                return method(*args,**kwargs)
        return run
    return decorate
