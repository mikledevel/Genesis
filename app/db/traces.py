"""Ограниченное хранение технических traces без содержимого сообщений."""
import json
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_traces (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at INTEGER NOT NULL,
    operation TEXT NOT NULL, status TEXT NOT NULL, duration_ms REAL NOT NULL,
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_owner_time ON execution_traces(user_id,created_at);
"""


class TraceMixin:
    def save_trace(self, user_id, trace):
        with self.write_transaction() as conn:
            if self.is_postgres:
                conn.execute("SELECT id FROM users WHERE id=? FOR UPDATE", (user_id,))
            conn.execute("INSERT INTO execution_traces VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                         "status=excluded.status,duration_ms=excluded.duration_ms,trace_json=excluded.trace_json",
                         (trace["id"], user_id, trace["created_at"], trace["operation"], trace["status"],
                          trace["duration_ms"], json.dumps(trace, allow_nan=False)))
            # Лимит на аккаунт сдерживает рост даже при интенсивном использовании.
            conn.execute("DELETE FROM execution_traces WHERE user_id=? AND id NOT IN "
                         "(SELECT id FROM execution_traces WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT 200)",
                         (user_id, user_id))

    def get_trace(self, user_id, trace_id):
        with self._connect() as conn:
            row = conn.execute("SELECT trace_json FROM execution_traces WHERE id=? AND user_id=?",
                               (trace_id, user_id)).fetchone()
        return json.loads(row[0]) if row else None

    def list_traces(self, user_id, limit=30, offset=0):
        with self._connect() as conn:
            rows = conn.execute("SELECT trace_json FROM execution_traces WHERE user_id=? "
                                "ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                                (user_id, min(100, max(1, limit)), max(0, offset))).fetchall()
        return [{k: v for k, v in json.loads(row[0]).items() if k != "spans"} for row in rows]

    def prune_traces(self, days=7):
        with self.write_transaction() as conn:
            conn.execute("DELETE FROM execution_traces WHERE created_at<?", (int(time.time()) - days * 86400,))
