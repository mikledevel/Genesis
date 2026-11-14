"""Транзакционные операции, счётчики и постоянные документы приложения."""
import json
import time
import uuid
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS app_documents (namespace TEXT PRIMARY KEY, content TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS monthly_counters (user_id TEXT NOT NULL, month TEXT NOT NULL, feature TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id, month, feature));
CREATE TABLE IF NOT EXISTS token_reservations (id TEXT PRIMARY KEY,user_id TEXT NOT NULL,month TEXT NOT NULL,amount INTEGER NOT NULL,created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS usage_operations (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, amount_cents INTEGER NOT NULL, author_id TEXT, author_cents INTEGER NOT NULL DEFAULT 0, api_key TEXT, related_id TEXT, status TEXT NOT NULL, result_json TEXT, request_hash TEXT NOT NULL DEFAULT '', created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS owned_datasets (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL, created_at INTEGER NOT NULL, UNIQUE(user_id,path));
CREATE TABLE IF NOT EXISTS background_jobs (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, error TEXT, progress_json TEXT, lease_until INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS incoming_events (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL, lease_until INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS outgoing_events (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_attempt INTEGER NOT NULL, lease_until INTEGER NOT NULL DEFAULT 0, error TEXT);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id,id);
CREATE INDEX IF NOT EXISTS idx_groq_usage_owner_time ON groq_usage_log(user_id,created_at);
CREATE INDEX IF NOT EXISTS idx_feature_usage_owner_time ON feature_usage(user_id,feature,created_at);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_agent ON kb_chunks(agent_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON background_jobs(status,lease_until);
CREATE INDEX IF NOT EXISTS idx_outbox_due ON outgoing_events(status,next_attempt);
"""


class UsageRejected(ValueError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class ReliabilityMixin:
    def reserve_tokens(self,user_id,amount,limit):
        from datetime import datetime, timezone
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        rid = uuid.uuid4().hex
        with self.write_transaction() as conn:
            conn.execute("INSERT INTO monthly_counters(user_id,month,feature,used) SELECT ?,?,'tokens',COALESCE(SUM(total_tokens),0) "
                         "FROM groq_usage_log WHERE user_id=? AND success=1 AND created_at>=? ON CONFLICT DO NOTHING",
                         (user_id,month,user_id,month+"-01"))
            if not conn.execute("UPDATE monthly_counters SET used=used+? WHERE user_id=? AND month=? AND feature='tokens' AND used+?<=?",
                                (amount,user_id,month,amount,limit)).rowcount:
                raise UsageRejected(429,"The monthly token budget is exhausted or already reserved")
            conn.execute("INSERT INTO token_reservations(id,user_id,month,amount,created_at) VALUES (?,?,?,?,?)",(rid,user_id,month,amount,int(time.time())))
        return rid

    def settle_tokens(self,reservation_id,actual):
        with self.write_transaction() as conn:
            lock = " FOR UPDATE" if self.is_postgres else ""
            row = conn.execute("SELECT user_id,month,amount FROM token_reservations WHERE id=?"+lock,(reservation_id,)).fetchone()
            if row:
                conn.execute("UPDATE monthly_counters SET used=used-?+? WHERE user_id=? AND month=? AND feature='tokens'",(row[2],actual,row[0],row[1]))
                conn.execute("DELETE FROM token_reservations WHERE id=?",(reservation_id,))

    @contextmanager
    def write_transaction(self):
        with self._connect() as conn:
            conn.execute("BEGIN" if self.is_postgres else "BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    @contextmanager
    def document_transaction(self, namespace, initial):
        with self.write_transaction() as conn:
            conn.execute("INSERT INTO app_documents(namespace,content) VALUES (?,?) ON CONFLICT DO NOTHING",
                         (namespace, json.dumps(initial, ensure_ascii=False)))
            lock = " FOR UPDATE" if self.is_postgres else ""
            row = conn.execute("SELECT content FROM app_documents WHERE namespace=?" + lock, (namespace,)).fetchone()
            document = json.loads(row[0])
            yield document
            conn.execute("UPDATE app_documents SET content=? WHERE namespace=?", (json.dumps(document, ensure_ascii=False), namespace))

    def read_document(self, namespace, initial):
        with self._connect() as conn:
            row = conn.execute("SELECT content FROM app_documents WHERE namespace=?", (namespace,)).fetchone()
        return json.loads(row[0]) if row else initial

    def claim_monthly_use(self, user_id, feature, limit):
        from datetime import datetime, timezone
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        with self.write_transaction() as conn:
            conn.execute("INSERT INTO monthly_counters(user_id,month,feature,used) "
                         "SELECT ?,?,?,COUNT(*) FROM feature_usage WHERE user_id=? AND feature=? AND created_at>=? "
                         "ON CONFLICT DO NOTHING", (user_id, month, feature, user_id, feature, month + "-01"))
            changed = conn.execute("UPDATE monthly_counters SET used=used+1 WHERE user_id=? AND month=? AND feature=? AND used<?",
                                   (user_id, month, feature, limit)).rowcount
            if changed:
                conn.execute("INSERT INTO feature_usage(user_id,feature) VALUES (?,?)", (user_id, feature))
            return bool(changed)

    def register_dataset(self, user_id, path, size_bytes):
        import os
        path = os.path.relpath(os.path.realpath(path))
        dataset_id = uuid.uuid4().hex
        with self.write_transaction() as conn:
            conn.execute("INSERT INTO owned_datasets(id,user_id,path,size_bytes,created_at) VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING",
                         (dataset_id, user_id, path, size_bytes, int(time.time())))
            return conn.execute("SELECT id FROM owned_datasets WHERE user_id=? AND path=?", (user_id, path)).fetchone()[0]

    def dataset_path(self, user_id, dataset_id):
        with self._connect() as conn:
            row = conn.execute("SELECT path FROM owned_datasets WHERE user_id=? AND id=?", (user_id, dataset_id)).fetchone()
            return row[0] if row else None

    def reserve_usage(self, user_id, amount_cents, *, author_id=None, api_key=None, operation_id=None, request_hash="", related_id=None):
        if isinstance(amount_cents, bool) or not isinstance(amount_cents, int) or amount_cents < 0:
            raise ValueError("Invalid charge")
        from app.core.secrets_store import key_digest
        api_key = key_digest(api_key) if api_key else None
        operation_id = operation_id or uuid.uuid4().hex
        now = int(time.time())
        with self.write_transaction() as conn:
            created = conn.execute("INSERT INTO usage_operations(id,user_id,amount_cents,author_id,api_key,status,created_at,updated_at,request_hash,related_id) "
                                   "VALUES (?,?,?,?,?,'initial',?,?,?,?) ON CONFLICT DO NOTHING",
                                   (operation_id, user_id, amount_cents, author_id, api_key, now, now,request_hash,related_id)).rowcount
            row = conn.execute("SELECT user_id,status,result_json,request_hash FROM usage_operations WHERE id=?", (operation_id,)).fetchone()
            if not created:
                if row[0] != user_id or row[1] != "succeeded" or row[3] != request_hash:
                    raise UsageRejected(409, "This operation is already running or was previously rejected")
                return {"id": operation_id, "replayed": True, "result": json.loads(row[2]) if row[2] else None}
            if api_key:
                if not conn.execute("UPDATE api_keys SET calls_used=calls_used+1 WHERE api_key=? AND user_id=? AND active=1 AND calls_used<calls_limit",
                                    (api_key, user_id)).rowcount:
                    raise UsageRejected(429, "API call limit exceeded")
            if not conn.execute("UPDATE users SET balance_cents=balance_cents-? WHERE id=? AND disabled=0 AND balance_cents>=?",
                                (amount_cents, user_id, amount_cents)).rowcount:
                raise UsageRejected(402, "Insufficient balance or inactive account")
            if amount_cents:
                conn.execute("INSERT INTO transactions(user_id,type,amount_cents,description,related_id) VALUES (?,'usage_charge',?,'Reserved usage',?)",
                             (user_id, -amount_cents, operation_id))
            conn.execute("UPDATE usage_operations SET status='reserved' WHERE id=?", (operation_id,))
        return {"id": operation_id, "replayed": False}

    def finish_usage(self, operation_id, *, result=None, refund=False, actual_cents=None, public_usage=None):
        with self.write_transaction() as conn:
            lock = " FOR UPDATE" if self.is_postgres else ""
            row = conn.execute("SELECT user_id,amount_cents,author_id,api_key,status FROM usage_operations WHERE id=?" + lock, (operation_id,)).fetchone()
            if not row or row[4] != "reserved":
                return False
            user_id, reserved, author_id, api_key, _ = row
            if self.is_postgres:
                if api_key:
                    conn.execute("SELECT id FROM api_keys WHERE api_key=? FOR UPDATE",(api_key,))
                for account in sorted({u for u in (user_id,author_id) if u}):
                    conn.execute("SELECT id FROM users WHERE id=? FOR UPDATE",(account,))
            actual = 0 if refund else (reserved if actual_cents is None else actual_cents)
            if not isinstance(actual, int) or not 0 <= actual <= reserved:
                raise ValueError("Actual charge must fit the reservation")
            if actual < reserved:
                conn.execute("UPDATE users SET balance_cents=balance_cents+? WHERE id=?", (reserved-actual, user_id))
                conn.execute("INSERT INTO transactions(user_id,type,amount_cents,description,related_id) VALUES (?,'usage_refund',?,'Unused reservation',?)",
                             (user_id, reserved-actual, operation_id))
            share = round(actual * 0.75) if author_id and author_id != user_id else 0
            if share:
                if not conn.execute("UPDATE users SET balance_cents=balance_cents+? WHERE id=?", (share, author_id)).rowcount:
                    raise ValueError("Author account no longer exists")
                conn.execute("INSERT INTO transactions(user_id,type,amount_cents,description,related_id) VALUES (?,'usage_earning',?,'Model usage',?)",
                             (author_id, share, operation_id))
            if refund and api_key:
                conn.execute("UPDATE api_keys SET calls_used=calls_used-1 WHERE api_key=? AND calls_used>0", (api_key,))
            if public_usage and not refund:
                from app.core.secrets_store import key_digest
                key,model,prompt,completion = public_usage
                conn.execute("INSERT INTO public_api_usage(api_key,user_id,model,prompt_tokens,completion_tokens,charged_cents) VALUES (?,?,?,?,?,?)",
                             (key_digest(key),user_id,model,prompt,completion,actual))
            conn.execute("UPDATE usage_operations SET status=?,amount_cents=?,author_cents=?,result_json=?,updated_at=? WHERE id=?",
                         ("refunded" if refund else "succeeded", actual, share, json.dumps(result, ensure_ascii=False, default=str), int(time.time()), operation_id))
            return True

    def recover_usage(self, max_age=3600):
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM usage_operations WHERE status='reserved' AND created_at<?", (int(time.time())-max_age,)).fetchall()]
            tokens = conn.execute("SELECT id,amount FROM token_reservations WHERE created_at<?",(int(time.time())-max_age,)).fetchall()
        for operation_id in ids:
            self.finish_usage(operation_id, refund=True)
        for reservation,amount in tokens:
            self.settle_tokens(reservation,amount)
        return len(ids)
