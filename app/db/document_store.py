"""Совместимое хранение небольших конфигураций в транзакционной БД."""
import copy
import json
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


class DatabaseDocument:
    def __init__(self, namespace, initial, db=None):
        from app.db.database import GenesisDB
        self.db = db or GenesisDB()
        self.namespace = namespace
        self.initial = initial
        self.local = threading.local()
        self.lock = threading.RLock()
        with self.db._connect() as conn:
            exists = conn.execute("SELECT 1 FROM app_documents WHERE namespace=?", (namespace,)).fetchone()
        if not exists:
            with self.db.document_transaction(namespace,initial):
                pass

    def read(self):
        current = getattr(self.local, "current", None)
        return current if current is not None else self.db.read_document(self.namespace, copy.deepcopy(self.initial))

    @contextmanager
    def transaction(self):
        with self.lock:
            if getattr(self.local, "current", None) is not None:
                yield self.local.current
                return
            with self.db.document_transaction(self.namespace, self.initial) as data:
                self.local.current = data
                try:
                    yield data
                finally:
                    self.local.current = None

    def replace(self, data):
        replacement = copy.deepcopy(data)
        with self.transaction() as current:
            current.clear()
            current.update(replacement)


def legacy_document(path, fallback):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else fallback


def document_mutation(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._document.transaction():
            if hasattr(self, "_refresh_document"):
                self._refresh_document()
            return method(self, *args, **kwargs)
    return run
