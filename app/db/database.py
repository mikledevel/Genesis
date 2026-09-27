"""Database layer - SQLite storage with API keys support."""

import sqlite3, uuid, os, json
from typing import Dict, List, Any, Optional


class GenesisDB:
    def __init__(self, db_path: str = "genesis.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Every new sqlite3 connection starts with foreign_keys OFF by default -
        PRAGMA foreign_keys=ON must be re-issued on EACH connection, not just once
        at init time, or ON DELETE CASCADE silently never fires.

        Also sets:
          - WAL journal mode: readers no longer block writers and vice versa (default
            rollback-journal mode takes a lock that serializes ALL access, readers included,
            for the duration of any write - WAL is the standard fix for "database is locked"
            under concurrent request handling). This is a durable, on-disk setting (persists
            after being set once) but is cheap and safe to reissue every connection.
          - busy_timeout: when two writes genuinely do collide, retry for up to 5s instead of
            raising `sqlite3.OperationalError: database is locked` immediately. Turns a rare
            hard failure under load into a small added latency instead.
        This is a mitigation, not a replacement for a real multi-writer database - SQLite
        still only allows one writer at a time. If concurrent write volume grows past what
        WAL + busy_timeout comfortably absorbs, that's the signal to migrate to Postgres
        (settings.database_url already exists for this - see CHANGES.md for the migration
        plan), not to tune these settings further.
        """
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL, name TEXT DEFAULT '',
                    balance_cents INTEGER DEFAULT 0,
                    selected_model_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS bot_skills (
                    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, owner_user_id TEXT NOT NULL,
                    name TEXT NOT NULL, description TEXT DEFAULT '',
                    code TEXT NOT NULL,
                    allowed_domains_json TEXT NOT NULL DEFAULT '[]',
                    input_schema_json TEXT NOT NULL DEFAULT '{}',
                    action_type TEXT NOT NULL DEFAULT 'read_only',
                    test_passed INTEGER DEFAULT 0,
                    test_log TEXT DEFAULT '',
                    active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS pending_actions (
                    id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT 'chat',
                    status TEXT NOT NULL DEFAULT 'pending',
                    result_json TEXT, error TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    decided_at TEXT
                );
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id TEXT PRIMARY KEY, skill_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    interval_minutes INTEGER NOT NULL,
                    notify_on_change INTEGER DEFAULT 1,
                    next_run_at TEXT NOT NULL,
                    last_run_at TEXT,
                    last_result_hash TEXT,
                    last_result_json TEXT,
                    last_error TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS scheduled_job_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    success INTEGER NOT NULL, result_json TEXT, error TEXT,
                    changed_from_previous INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
                    type TEXT NOT NULL, amount_cents INTEGER NOT NULL,
                    description TEXT DEFAULT '', related_id TEXT,
                    stripe_session_id TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, title TEXT DEFAULT 'Untitled',
                    user_id TEXT,
                    created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, data TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                    description TEXT DEFAULT '', task TEXT DEFAULT 'other',
                    icon TEXT DEFAULT '🤖', model_count INTEGER DEFAULT 0,
                    user_id TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT NOT NULL,
                    predicate TEXT NOT NULL, object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.7, created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT NOT NULL,
                    api_key TEXT UNIQUE NOT NULL, calls_limit INTEGER DEFAULT 100,
                    calls_used INTEGER DEFAULT 0, created_at TEXT DEFAULT (datetime('now')),
                    active INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, api_key TEXT NOT NULL,
                    model_id TEXT NOT NULL, timestamp TEXT DEFAULT (datetime('now')),
                    status TEXT DEFAULT 'success'
                );
                CREATE TABLE IF NOT EXISTS groq_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                    call_site TEXT NOT NULL, model TEXT NOT NULL,
                    prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
                    success INTEGER NOT NULL, error TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    name TEXT NOT NULL, description TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]', current_version INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, prompt_id TEXT NOT NULL,
                    version INTEGER NOT NULL, content TEXT NOT NULL,
                    change_note TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    name TEXT NOT NULL, status TEXT DEFAULT 'running',
                    created_at TEXT DEFAULT (datetime('now')), ended_at TEXT,
                    winner_variant_id TEXT
                );
                CREATE TABLE IF NOT EXISTS ab_variants (
                    id TEXT PRIMARY KEY, test_id TEXT NOT NULL, label TEXT NOT NULL,
                    prompt_id TEXT, prompt_version INTEGER,
                    system_prompt TEXT NOT NULL, weight REAL DEFAULT 1.0,
                    FOREIGN KEY (test_id) REFERENCES ab_tests(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ab_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, test_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(test_id, conversation_id)
                );
                CREATE TABLE IF NOT EXISTS kb_documents (
                    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    filename TEXT NOT NULL, content TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS kb_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES kb_documents(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS handoff_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL, user_message TEXT, reason TEXT,
                    delivered INTEGER NOT NULL, delivery_error TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS telegram_links (
                    id TEXT PRIMARY KEY, agent_id TEXT NOT NULL UNIQUE, user_id TEXT NOT NULL,
                    bot_token TEXT NOT NULL, bot_id INTEGER, bot_username TEXT,
                    webhook_secret TEXT NOT NULL UNIQUE, status TEXT DEFAULT 'connecting',
                    last_error TEXT, created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS telegram_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, link_id TEXT NOT NULL,
                    event_type TEXT NOT NULL, detail TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                PRAGMA journal_mode=WAL;
            """)
            conn.commit()
            # Idempotent migration: databases created before user accounts existed won't have
            # these columns yet. ALTER TABLE ADD COLUMN is safe to retry - ignore if already there.
            for table, column, coldef in [("chats", "user_id", "TEXT"), ("projects", "user_id", "TEXT"),
                                          ("messages", "data", "TEXT"), ("users", "balance_cents", "INTEGER DEFAULT 0"),
                                          ("users", "selected_model_json", "TEXT"),
                                          ("messages", "rating", "INTEGER"),
                                          ("chats", "agent_id", "TEXT"),
                                          ("users", "monthly_groq_token_limit", "INTEGER"),
                                          # api_keys previously had no link to a real platform account (user_name was
                                          # an arbitrary, unauthenticated client-supplied string) - see create_api_key.
                                          ("api_keys", "user_id", "TEXT"),
                                          # Observability for LLM provider calls (see app.core.llm_errors /
                                          # GenesisAgent._log_llm_call) - previously only success/error/tokens were
                                          # recorded, with no latency or error classification, making outages
                                          # indistinguishable from slow-but-working calls in the logs.
                                          ("groq_usage_log", "latency_ms", "INTEGER"),
                                          ("groq_usage_log", "error_type", "TEXT"),
                                          ("groq_usage_log", "provider", "TEXT"),
                                          ("users", "email_verified", "INTEGER DEFAULT 0"),
                                          ("bot_skills", "action_type", "TEXT NOT NULL DEFAULT 'read_only'")]:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass  # column already exists

    # --- Users ---
    def create_user(self, email: str, password_hash: str, name: str = "") -> Dict:
        user_id = f"user_{uuid.uuid4().hex[:16]}"
        with self._connect() as conn:
            conn.execute("INSERT INTO users (id, email, password_hash, name) VALUES (?, ?, ?, ?)",
                         (user_id, email.lower().strip(), password_hash, name))
            conn.commit()
        return {"id": user_id, "email": email.lower().strip(), "name": name}

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT id, email, password_hash, name FROM users WHERE email = ?",
                                (email.lower().strip(),)).fetchone()
            return {"id": row[0], "email": row[1], "password_hash": row[2], "name": row[3]} if row else None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT id, email, name, balance_cents FROM users WHERE id = ?", (user_id,)).fetchone()
            return {"id": row[0], "email": row[1], "name": row[2], "balance_cents": row[3]} if row else None

    def update_password(self, user_id: str, new_password_hash: str):
        with self._connect() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_password_hash, user_id))
            conn.commit()

    def is_email_verified(self, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT email_verified FROM users WHERE id = ?", (user_id,)).fetchone()
            return bool(row and row[0])

    def mark_email_verified(self, user_id: str):
        with self._connect() as conn:
            conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))
            conn.commit()

    # --- Auth tokens (email verification + password reset) ---
    # A single table with a "purpose" column rather than two tables - same lifecycle
    # (create, check validity+expiry, consume once) for both, just different actions taken
    # by the route handler on successful consumption.
    def create_auth_token(self, user_id: str, purpose: str, ttl_seconds: int) -> str:
        import secrets as _secrets
        from datetime import datetime, timedelta
        token = _secrets.token_urlsafe(32)
        expires_at = (datetime.utcnow() + timedelta(seconds=ttl_seconds)).isoformat()
        with self._connect() as conn:
            conn.execute("INSERT INTO auth_tokens (token, user_id, purpose, expires_at) VALUES (?, ?, ?, ?)",
                         (token, user_id, purpose, expires_at))
            conn.commit()
        return token

    def consume_auth_token(self, token: str, purpose: str) -> Optional[str]:
        """Validates the token is unused, unexpired, and for the expected purpose, marks it
        used, and returns the user_id it belongs to (or None if invalid/expired/already
        used/wrong purpose). Marking used_at happens in the SAME connection/commit as the
        validity check to avoid a token being usable twice via a race between two concurrent
        requests - not perfectly atomic under SQLite's threading model, but the UPDATE...
        WHERE used_at IS NULL below means only one of two racing requests can ever succeed."""
        from datetime import datetime
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, purpose, expires_at, used_at FROM auth_tokens WHERE token = ?", (token,)).fetchone()
            if not row:
                return None
            user_id, row_purpose, expires_at, used_at = row
            if row_purpose != purpose or used_at is not None:
                return None
            if datetime.fromisoformat(expires_at) < datetime.utcnow():
                return None
            cur = conn.execute(
                "UPDATE auth_tokens SET used_at = datetime('now') WHERE token = ? AND used_at IS NULL", (token,))
            conn.commit()
            return user_id if cur.rowcount > 0 else None

    # --- Bot skills (AI-generated, sandbox-tested code capabilities for a bot) ---
    # Ownership is enforced the same way as agents/models elsewhere in this file: every
    # mutating method takes owner_user_id and only acts on rows that actually belong to that
    # user - see the agent-store IDOR fix and the model-registry ownership fix for why this
    # matters (a skill's code and allowed_domains are exactly the kind of thing another user
    # should not be able to read, edit, or delete).
    def create_skill(self, agent_id: str, owner_user_id: str, name: str, description: str,
                      code: str, allowed_domains: list, input_schema: dict,
                      action_type: str = "read_only") -> Dict:
        if action_type not in ("read_only", "outbound_action"):
            raise ValueError("action_type must be 'read_only' or 'outbound_action'")
        import uuid as _uuid
        skill_id = f"skill_{_uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bot_skills (id, agent_id, owner_user_id, name, description, code, "
                "allowed_domains_json, input_schema_json, action_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (skill_id, agent_id, owner_user_id, name, description, code,
                 json.dumps(allowed_domains), json.dumps(input_schema), action_type))
            conn.commit()
        return self.get_skill(skill_id)

    def get_skill(self, skill_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, agent_id, owner_user_id, name, description, code, allowed_domains_json, "
                "input_schema_json, test_passed, test_log, active, created_at, action_type FROM bot_skills WHERE id = ?",
                (skill_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "agent_id": row[1], "owner_user_id": row[2], "name": row[3],
                "description": row[4], "code": row[5], "allowed_domains": json.loads(row[6]),
                "input_schema": json.loads(row[7]), "test_passed": bool(row[8]),
                "test_log": row[9], "active": bool(row[10]), "created_at": row[11],
                "action_type": row[12],
            }

    def list_skills_for_agent(self, agent_id: str) -> List[Dict]:
        """Deliberately does NOT filter by owner - this is called from within a chat session
        to find which skills a bot can use, and the caller (agent.py) already establishes the
        bot's own identity/ownership context separately. Do not expose this list (with code)
        directly to arbitrary users via an API route without an ownership check there - see
        the /api/skills route in main.py, which does check ownership before returning."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, allowed_domains_json, input_schema_json, action_type "
                "FROM bot_skills WHERE agent_id = ? AND active = 1 AND test_passed = 1", (agent_id,)).fetchall()
            return [{"id": r[0], "name": r[1], "description": r[2],
                     "allowed_domains": json.loads(r[3]), "input_schema": json.loads(r[4]),
                     "action_type": r[5]} for r in rows]

    def list_skills_for_owner(self, owner_user_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, agent_id, name, description, test_passed, active, created_at, action_type "
                "FROM bot_skills WHERE owner_user_id = ? ORDER BY created_at DESC", (owner_user_id,)).fetchall()
            return [{"id": r[0], "agent_id": r[1], "name": r[2], "description": r[3],
                     "test_passed": bool(r[4]), "active": bool(r[5]), "created_at": r[6],
                     "action_type": r[7]} for r in rows]

    def update_skill_test_result(self, skill_id: str, passed: bool, log: str, code: Optional[str] = None):
        with self._connect() as conn:
            if code is not None:
                conn.execute(
                    "UPDATE bot_skills SET test_passed = ?, test_log = ?, code = ?, updated_at = datetime('now') WHERE id = ?",
                    (int(passed), log, code, skill_id))
            else:
                conn.execute(
                    "UPDATE bot_skills SET test_passed = ?, test_log = ?, updated_at = datetime('now') WHERE id = ?",
                    (int(passed), log, skill_id))
            conn.commit()

    def delete_skill(self, skill_id: str, owner_user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM bot_skills WHERE id = ? AND owner_user_id = ?", (skill_id, owner_user_id))
            conn.commit()
            return cur.rowcount > 0

    # --- Pending actions ("Level 4": a skill that would do something to a third party -
    # send a message, post something, etc. - never runs on its own, whether triggered from
    # chat or from a scheduled job. It always creates one of these instead, and the actual
    # skill code only executes once the owner explicitly approves it here. See
    # app/core/agent.py's _do_run_skill and app/core/scheduler.py's _run_one_job, both of
    # which check skill["action_type"] == "outbound_action" and route here instead of
    # executing directly - this table is the ONLY path an outbound_action skill can run
    # through, by construction, not by convention.) ---
    def create_pending_action(self, skill_id: str, agent_id: str, owner_user_id: str,
                               params: dict, source: str) -> Dict:
        import uuid as _uuid
        action_id = f"pact_{_uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO pending_actions (id, skill_id, agent_id, owner_user_id, params_json, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (action_id, skill_id, agent_id, owner_user_id, json.dumps(params), source))
            conn.commit()
        return self.get_pending_action(action_id)

    def get_pending_action(self, action_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, skill_id, agent_id, owner_user_id, params_json, source, status, "
                "result_json, error, created_at, decided_at FROM pending_actions WHERE id = ?",
                (action_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "skill_id": row[1], "agent_id": row[2], "owner_user_id": row[3],
                "params": json.loads(row[4]), "source": row[5], "status": row[6],
                "result": json.loads(row[7]) if row[7] else None, "error": row[8],
                "created_at": row[9], "decided_at": row[10],
            }

    def list_pending_actions_for_owner(self, owner_user_id: str, status: Optional[str] = None) -> List[Dict]:
        query = "SELECT id FROM pending_actions WHERE owner_user_id = ?"
        params = [owner_user_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute(query, params).fetchall()]
        return [self.get_pending_action(aid) for aid in ids]

    def decide_pending_action(self, action_id: str, owner_user_id: str, approve: bool) -> Optional[Dict]:
        """Moves a pending action from 'pending' to 'approved' or 'rejected' - ownership
        checked, and only succeeds if it's still actually pending (can't re-decide an action
        that was already approved/rejected, whether by a race or a duplicate click). Returns
        the updated record, or None if it wasn't found/owned/still-pending. Approving does
        NOT execute the skill itself - see main.py's route, which calls this THEN runs the
        skill and calls record_pending_action_result with the outcome."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE pending_actions SET status = ?, decided_at = datetime('now') "
                "WHERE id = ? AND owner_user_id = ? AND status = 'pending'",
                ("approved" if approve else "rejected", action_id, owner_user_id))
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_pending_action(action_id)

    def record_pending_action_result(self, action_id: str, success: bool, result: Optional[dict], error: Optional[str]):
        with self._connect() as conn:
            conn.execute(
                "UPDATE pending_actions SET status = ?, result_json = ?, error = ? WHERE id = ?",
                ("executed" if success else "failed",
                 json.dumps(result) if result is not None else None, error, action_id))
            conn.commit()

    # --- Scheduled jobs (background/autonomous skill runs - "Level 3": a bot that checks
    # something on its own, without a user message triggering each run) ---
    MIN_SCHEDULE_INTERVAL_MINUTES = 15  # hard floor, regardless of what's requested - see
    # create_scheduled_job. Prevents one misconfigured or malicious job from hammering an
    # external site (and, if it were LLM-backed, burning the owner's Groq budget) every few
    # seconds forever.
    MAX_ACTIVE_JOBS_PER_OWNER = 10
    MAX_CONSECUTIVE_FAILURES = 5  # auto-pause after this many failed runs in a row, rather
    # than retrying forever against something that's clearly broken (wrong domain, dead
    # site, a bug introduced by editing the skill) and quietly wasting compute/API calls.

    def create_scheduled_job(self, skill_id: str, agent_id: str, owner_user_id: str,
                              params: dict, interval_minutes: int, notify_on_change: bool = True) -> Dict:
        if interval_minutes < self.MIN_SCHEDULE_INTERVAL_MINUTES:
            raise ValueError(f"interval_minutes must be at least {self.MIN_SCHEDULE_INTERVAL_MINUTES}")
        active_count = len(self.list_scheduled_jobs_for_owner(owner_user_id, active_only=True))
        if active_count >= self.MAX_ACTIVE_JOBS_PER_OWNER:
            raise ValueError(f"You already have {active_count} active scheduled jobs - the limit is "
                              f"{self.MAX_ACTIVE_JOBS_PER_OWNER}. Pause or delete one before adding another.")
        import uuid as _uuid
        job_id = f"job_{_uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO scheduled_jobs (id, skill_id, agent_id, owner_user_id, params_json, "
                "interval_minutes, notify_on_change, next_run_at) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, datetime('now', ?))",
                (job_id, skill_id, agent_id, owner_user_id, json.dumps(params), interval_minutes,
                 int(notify_on_change), f"+{interval_minutes} minutes"))
            conn.commit()
        return self.get_scheduled_job(job_id)

    def get_scheduled_job(self, job_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, skill_id, agent_id, owner_user_id, params_json, interval_minutes, "
                "notify_on_change, next_run_at, last_run_at, last_result_hash, last_result_json, "
                "last_error, consecutive_failures, active, created_at FROM scheduled_jobs WHERE id = ?",
                (job_id,)).fetchone()
            if not row:
                return None
            return {
                "id": row[0], "skill_id": row[1], "agent_id": row[2], "owner_user_id": row[3],
                "params": json.loads(row[4]), "interval_minutes": row[5], "notify_on_change": bool(row[6]),
                "next_run_at": row[7], "last_run_at": row[8], "last_result_hash": row[9],
                "last_result": json.loads(row[10]) if row[10] else None, "last_error": row[11],
                "consecutive_failures": row[12], "active": bool(row[13]), "created_at": row[14],
            }

    def list_scheduled_jobs_for_owner(self, owner_user_id: str, active_only: bool = False) -> List[Dict]:
        query = "SELECT id FROM scheduled_jobs WHERE owner_user_id = ?"
        params = [owner_user_id]
        if active_only:
            query += " AND active = 1"
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute(query, params).fetchall()]
        return [self.get_scheduled_job(jid) for jid in ids]

    def claim_due_jobs(self, limit: int = 10) -> List[Dict]:
        """Atomically claims up to `limit` jobs that are due to run, safe even if multiple
        server processes call this concurrently (see docstring in app/core/scheduler.py for
        the full reasoning). A "claim" is implemented as a compare-and-swap on next_run_at:
        we read it, then try to update it ONLY if it still holds the exact value we just
        read - if another process already claimed the job in between, our UPDATE affects 0
        rows and we simply skip it, rather than two processes both running the same job.

        Claimed jobs get next_run_at bumped 10 minutes into the future as a "running" marker
        - complete_scheduled_job_run() sets the real next_run_at once the run actually
        finishes. If a process crashes mid-run without calling that, the job self-heals: it
        simply becomes claimable again after that 10-minute marker passes, rather than being
        stuck forever."""
        with self._connect() as conn:
            candidates = conn.execute(
                "SELECT id, next_run_at FROM scheduled_jobs WHERE active = 1 AND next_run_at <= datetime('now') "
                "ORDER BY next_run_at LIMIT ?", (limit,)).fetchall()
            claimed_ids = []
            for job_id, next_run_at in candidates:
                cur = conn.execute(
                    "UPDATE scheduled_jobs SET next_run_at = datetime('now', '+10 minutes') "
                    "WHERE id = ? AND next_run_at = ?", (job_id, next_run_at))
                if cur.rowcount > 0:
                    claimed_ids.append(job_id)
            conn.commit()
        return [self.get_scheduled_job(jid) for jid in claimed_ids]

    def complete_scheduled_job_run(self, job_id: str, success: bool, result: Optional[dict],
                                    error: Optional[str]) -> Dict:
        """Records the outcome of a claimed run, schedules the next one (real interval this
        time, not the 10-minute claim marker), and auto-pauses after too many consecutive
        failures. Returns {"changed": bool} so the caller (app/core/scheduler.py) knows
        whether to send a change notification."""
        import hashlib
        job = self.get_scheduled_job(job_id)
        result_json = json.dumps(result) if result is not None else None
        result_hash = hashlib.sha256(result_json.encode()).hexdigest() if result_json else None
        changed = success and job["last_result_hash"] is not None and result_hash != job["last_result_hash"]
        new_failures = 0 if success else job["consecutive_failures"] + 1
        should_deactivate = new_failures >= self.MAX_CONSECUTIVE_FAILURES

        with self._connect() as conn:
            conn.execute(
                "UPDATE scheduled_jobs SET last_run_at = datetime('now'), "
                "next_run_at = datetime('now', ?), last_result_hash = ?, last_result_json = ?, "
                "last_error = ?, consecutive_failures = ?, active = ? WHERE id = ?",
                (f"+{job['interval_minutes']} minutes", result_hash, result_json, error,
                 new_failures, 0 if should_deactivate else 1, job_id))
            conn.execute(
                "INSERT INTO scheduled_job_runs (job_id, success, result_json, error, changed_from_previous) "
                "VALUES (?, ?, ?, ?, ?)", (job_id, int(success), result_json, error, int(changed)))
            conn.commit()
        return {"changed": changed, "auto_paused": should_deactivate}

    def set_scheduled_job_active(self, job_id: str, owner_user_id: str, active: bool) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE scheduled_jobs SET active = ? WHERE id = ? AND owner_user_id = ?",
                               (int(active), job_id, owner_user_id))
            conn.commit()
            return cur.rowcount > 0

    def delete_scheduled_job(self, job_id: str, owner_user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM scheduled_jobs WHERE id = ? AND owner_user_id = ?", (job_id, owner_user_id))
            conn.commit()
            return cur.rowcount > 0

    def get_balance(self, user_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT balance_cents FROM users WHERE id = ?", (user_id,)).fetchone()
            return row[0] if row else 0

    def set_selected_model(self, user_id: str, data: dict):
        with self._connect() as conn:
            conn.execute("UPDATE users SET selected_model_json = ? WHERE id = ?", (json.dumps(data), user_id))
            conn.commit()

    def get_selected_model(self, user_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT selected_model_json FROM users WHERE id = ?", (user_id,)).fetchone()
            if row and row[0]:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return {}
            return {}

    def stripe_session_already_processed(self, stripe_session_id: str) -> bool:
        """Idempotency guard - Stripe can (and does) redeliver the same webhook event more
        than once. Without this check a retried delivery would credit the balance twice."""
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM transactions WHERE stripe_session_id = ?", (stripe_session_id,)).fetchone()
            return row is not None

    def add_transaction(self, user_id: str, type_: str, amount_cents: int, description: str = "",
                         related_id: str = None, stripe_session_id: str = None) -> Dict:
        """Atomically record a ledger entry AND update the user's balance. amount_cents is
        signed: positive credits the user, negative debits them."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "INSERT INTO transactions (user_id, type, amount_cents, description, related_id, stripe_session_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, type_, amount_cents, description, related_id, stripe_session_id))
                conn.execute("UPDATE users SET balance_cents = balance_cents + ? WHERE id = ?", (amount_cents, user_id))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"user_id": user_id, "type": type_, "amount_cents": amount_cents}

    def get_transactions(self, user_id: str, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT type, amount_cents, description, related_id, created_at FROM transactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit)).fetchall()
            return [{"type": r[0], "amount_cents": r[1], "description": r[2], "related_id": r[3], "created_at": r[4]} for r in rows]

    def charge_for_usage(self, buyer_user_id: str, amount_cents: int, description: str,
                          author_user_id: str = None, platform_fee_pct: float = 25.0, related_id: str = None) -> Dict:
        """Deduct amount_cents from the buyer's balance for using a paid bot/model, and credit
        the author their share (minus the platform fee) if there is one (author_user_id may be
        None for platform-owned demo content). Returns {"ok": False, "reason": "insufficient_balance"}
        if the buyer doesn't have enough balance - the caller must not fulfil the request in that case."""
        balance = self.get_balance(buyer_user_id)
        if balance < amount_cents:
            return {"ok": False, "reason": "insufficient_balance", "balance_cents": balance, "needed_cents": amount_cents}
        self.add_transaction(buyer_user_id, "usage_charge", -amount_cents, description, related_id=related_id)
        if author_user_id and author_user_id != buyer_user_id:
            author_share = round(amount_cents * (1 - platform_fee_pct / 100))
            if author_share > 0:
                self.add_transaction(author_user_id, "usage_earning", author_share,
                                     f"Earning: {description}", related_id=related_id)
        return {"ok": True, "charged_cents": amount_cents}

    # --- Chats ---
    def create_chat(self, chat_id: str, title: str = "Untitled", user_id: str = None) -> Dict:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO chats (id, title, user_id) VALUES (?, ?, ?)", (chat_id, title, user_id))
            conn.commit()
        return {"id": chat_id, "title": title}

    def get_chat_owner(self, chat_id: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            return row[0] if row else None

    def tag_chat_with_agent(self, chat_id: str, agent_id: str):
        """Records which published bot a conversation is actually with, the first time it's
        used - this is the missing link that makes per-bot analytics possible at all
        (conversation_id alone doesn't otherwise carry which bot was selected, since bot
        selection lives in users.selected_model_json, a per-USER field, not per-conversation).
        Only ever sets this once per chat (COALESCE keeps the existing value) - a
        conversation shouldn't retroactively change which bot it's attributed to."""
        with self._connect() as conn:
            conn.execute("UPDATE chats SET agent_id = COALESCE(agent_id, ?) WHERE id = ?", (agent_id, chat_id))
            conn.commit()

    def list_chats(self, user_id: str = None) -> List[Dict]:
        with self._connect() as conn:
            if user_id is not None:
                rows = conn.execute("SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM chats ORDER BY updated_at DESC").fetchall()
            return [{"id": r[0], "title": r[1], "user_id": r[2], "created_at": r[3], "updated_at": r[4]} for r in rows]

    def rename_chat(self, chat_id: str, title: str):
        with self._connect() as conn:
            conn.execute("UPDATE chats SET title = ?, updated_at = datetime('now') WHERE id = ?", (title, chat_id))
            conn.commit()

    def delete_chat(self, chat_id: str):
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))  # belt-and-braces even with FK cascade now enabled
            conn.commit()
            return cur.rowcount

    # --- Messages ---
    def add_message(self, chat_id: str, role: str, content: str, data: dict = None) -> int:
        data_json = json.dumps(data, ensure_ascii=False) if data else None
        with self._connect() as conn:
            cur = conn.execute("INSERT INTO messages (chat_id, role, content, data) VALUES (?, ?, ?, ?)", (chat_id, role, content, data_json))
            conn.execute("UPDATE chats SET updated_at = datetime('now') WHERE id = ?", (chat_id,))
            conn.commit()
            return cur.lastrowid

    def get_messages(self, chat_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, role, content, data, created_at, rating FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,)).fetchall()
            result = []
            for r in rows:
                msg = {"id": r[0], "role": r[1], "content": r[2], "timestamp": r[4], "rating": r[5]}
                if r[3]:
                    try:
                        msg["data"] = json.loads(r[3])
                    except (json.JSONDecodeError, TypeError):
                        pass
                result.append(msg)
            return result

    def rate_message(self, message_id: int, chat_id: str, rating: Optional[int]) -> bool:
        """Sets (or clears, if rating is None) a thumbs up(1)/down(-1) rating on a message.
        chat_id is required and checked so a caller can't rate a message in a conversation
        they don't own by guessing a message_id."""
        with self._connect() as conn:
            cur = conn.execute("UPDATE messages SET rating = ? WHERE id = ? AND chat_id = ?",
                               (rating, message_id, chat_id))
            conn.commit()
            return cur.rowcount > 0

    def get_message_count(self, chat_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)).fetchone()
            return row[0] if row else 0

    # --- Projects ---
    def get_activity_last_7_days(self, user_id: str = None) -> List[Dict]:
        """Count of messages sent per day for each of the last 7 days (oldest first),
        used to draw the real Dashboard activity chart instead of mock data."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if user_id is not None:
                rows = conn.execute("""
                    SELECT date(m.created_at) AS day, COUNT(*) AS count
                    FROM messages m JOIN chats c ON m.chat_id = c.id
                    WHERE date(m.created_at) >= date('now', '-6 days') AND c.user_id = ?
                    GROUP BY day
                """, (user_id,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT date(created_at) AS day, COUNT(*) AS count
                    FROM messages
                    WHERE date(created_at) >= date('now', '-6 days')
                    GROUP BY day
                """).fetchall()
        counts = {r["day"]: r["count"] for r in rows}
        from datetime import date, timedelta
        today = date.today()
        return [
            {"day": (today - timedelta(days=offset)).isoformat(),
             "count": counts.get((today - timedelta(days=offset)).isoformat(), 0)}
            for offset in range(6, -1, -1)
        ]

    def get_projects(self, user_id: str = None) -> List[Dict]:
        with self._connect() as conn:
            if user_id is not None:
                rows = conn.execute("SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
            return [{"name": r[1], "desc": r[2], "task": r[3], "icon": r[4], "modelCount": r[5], "user_id": r[6], "created": r[7]} for r in rows]

    def add_project(self, name: str, desc: str = "", task: str = "other", icon: str = "🤖", user_id: str = None) -> Dict:
        with self._connect() as conn:
            conn.execute("INSERT INTO projects (name, description, task, icon, user_id) VALUES (?, ?, ?, ?, ?)", (name, desc, task, icon, user_id))
            conn.commit()
        return {"name": name, "desc": desc, "task": task, "icon": icon, "modelCount": 0}

    def delete_project(self, name: str, user_id: str = None):
        with self._connect() as conn:
            if user_id is not None:
                conn.execute("DELETE FROM projects WHERE name = ? AND user_id = ?", (name, user_id))
            else:
                conn.execute("DELETE FROM projects WHERE name = ?", (name,))
            conn.commit()

    # --- Memory ---
    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 0.7):
        with self._connect() as conn:
            conn.execute("INSERT INTO memory_facts (subject, predicate, object, confidence) VALUES (?, ?, ?, ?)", (subject, predicate, str(obj), confidence))
            conn.commit()

    def get_facts(self, subject: str = "user") -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT predicate, object, confidence FROM memory_facts WHERE subject = ? ORDER BY confidence DESC LIMIT 20", (subject,)).fetchall()
            return [{"predicate": r[0], "object": r[1], "confidence": r[2]} for r in rows]

    # --- API Keys ---
    # NOTE: api_keys.user_id is the real, authenticated platform account (from the JWT via
    # get_current_user_id) that owns this key. user_name remains as a free-text display label
    # only (e.g. "laptop", "prod server") - it is NEVER used for authorization. Previously
    # these routes accepted an arbitrary client-supplied user_name with no login required at
    # all, which meant anyone could list, create, or delete ANY api key for ANY account.
    def create_api_key(self, owner_user_id: str, user_name: str = "", calls_limit: int = 100) -> Dict:
        api_key = f"gen-{uuid.uuid4().hex[:24]}"
        with self._connect() as conn:
            conn.execute("INSERT INTO api_keys (user_id, user_name, api_key, calls_limit) VALUES (?, ?, ?, ?)",
                         (owner_user_id, user_name, api_key, calls_limit))
            conn.commit()
        return {"user_id": owner_user_id, "user_name": user_name, "api_key": api_key, "calls_limit": calls_limit, "calls_used": 0}

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT id, user_name, api_key, calls_limit, calls_used, active, user_id "
                                "FROM api_keys WHERE api_key = ? AND active = 1", (api_key,)).fetchone()
            if row:
                return {"id": row[0], "user_name": row[1], "api_key": row[2], "calls_limit": row[3],
                        "calls_used": row[4], "user_id": row[6]}
            return None

    def use_api_key(self, api_key: str, model_id: str) -> bool:
        """Increments the call counter and logs usage."""
        with self._connect() as conn:
            row = conn.execute("SELECT calls_used, calls_limit FROM api_keys WHERE api_key = ? AND active = 1", (api_key,)).fetchone()
            if not row or row[0] >= row[1]:
                return False
            conn.execute("UPDATE api_keys SET calls_used = calls_used + 1 WHERE api_key = ?", (api_key,))
            conn.execute("INSERT INTO usage_log (api_key, model_id) VALUES (?, ?)", (api_key, model_id))
            conn.commit()
            return True

    def list_api_keys(self, owner_user_id: str) -> List[Dict]:
        """Only ever returns keys owned by owner_user_id - callers must always pass the
        authenticated caller's own id (see /api/keys in main.py). There is deliberately no
        "list everyone's keys" code path left reachable from the API."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id, user_name, api_key, calls_limit, calls_used, active, user_id "
                                 "FROM api_keys WHERE user_id = ?", (owner_user_id,)).fetchall()
            return [{"id": r[0], "user_name": r[1], "api_key": r[2], "calls_limit": r[3], "calls_used": r[4], "active": r[5]} for r in rows]

    def get_api_key_owner(self, api_key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM api_keys WHERE api_key = ?", (api_key,)).fetchone()
            return row[0] if row else None

    def delete_api_key(self, api_key: str, owner_user_id: str) -> bool:
        """Only deletes the key if it belongs to owner_user_id. Returns False (nothing
        deleted) for a key owned by someone else or that doesn't exist, distinguishable from
        a real delete by the route handler."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM api_keys WHERE api_key = ? AND user_id = ?", (api_key, owner_user_id))
            conn.commit()
            return cur.rowcount > 0

    def get_usage_stats(self, api_key: str = None) -> List[Dict]:
        with self._connect() as conn:
            if api_key:
                rows = conn.execute("SELECT * FROM usage_log WHERE api_key = ? ORDER BY timestamp DESC LIMIT 50", (api_key,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM usage_log ORDER BY timestamp DESC LIMIT 50").fetchall()
            return [{"id": r[0], "api_key": r[1], "model_id": r[2], "timestamp": r[3], "status": r[4]} for r in rows]

    # --- Internal Groq usage tracking ---
    # This is DISTINCT from usage_log/api_keys above, which meters the external developer
    # /api/models/predict API. There was previously NO tracking anywhere of the platform's
    # own Groq API calls (chat routing, bot builder, codegen, pipeline suggestions) - every
    # one of those draws from the platform's own GROQ_API_KEY with zero visibility into who
    # is generating how much cost. This does not enforce any limit by itself (no call is
    # blocked based on it) - it exists so that decision can be made deliberately later with
    # real data, rather than the platform having no idea how much internal Groq usage exists.
    def log_groq_usage(self, user_id: Optional[str], call_site: str, model: str,
                        prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None,
                        total_tokens: Optional[int] = None, success: bool = True,
                        error: Optional[str] = None, latency_ms: Optional[int] = None,
                        error_type: Optional[str] = None, provider: str = "groq"):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO groq_usage_log (user_id, call_site, model, prompt_tokens, completion_tokens, "
                "total_tokens, success, error, latency_ms, error_type, provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, call_site, model, prompt_tokens, completion_tokens, total_tokens, int(success), error,
                 latency_ms, error_type, provider))
            conn.commit()

    def get_groq_usage_summary(self, user_id: Optional[str] = None, limit: int = 200) -> Dict:
        """Aggregate token/call counts, optionally scoped to one user. Also returns the most
        recent raw log rows (capped at `limit`) for a detail view."""
        with self._connect() as conn:
            where = "WHERE user_id = ?" if user_id else ""
            params = (user_id,) if user_id else ()
            totals = conn.execute(
                f"SELECT COUNT(*), SUM(CASE WHEN success=1 THEN 1 ELSE 0 END), "
                f"SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), SUM(prompt_tokens), "
                f"SUM(completion_tokens), SUM(total_tokens), AVG(latency_ms) FROM groq_usage_log {where}", params).fetchone()
            by_site = conn.execute(
                f"SELECT call_site, COUNT(*), SUM(total_tokens) FROM groq_usage_log {where} "
                f"GROUP BY call_site ORDER BY COUNT(*) DESC", params).fetchall()
            by_error_type = conn.execute(
                f"SELECT error_type, COUNT(*) FROM groq_usage_log {(where + ' AND ' if where else 'WHERE ') + 'success=0'} "
                f"GROUP BY error_type ORDER BY COUNT(*) DESC", params).fetchall()
            rows = conn.execute(
                f"SELECT call_site, model, prompt_tokens, completion_tokens, total_tokens, success, error, created_at, "
                f"latency_ms, error_type, provider "
                f"FROM groq_usage_log {where} ORDER BY id DESC LIMIT ?", params + (limit,)).fetchall()
            return {
                "total_calls": totals[0] or 0, "successful_calls": totals[1] or 0,
                "failed_calls": totals[2] or 0, "total_prompt_tokens": totals[3] or 0,
                "total_completion_tokens": totals[4] or 0, "total_tokens": totals[5] or 0,
                "avg_latency_ms": round(totals[6], 1) if totals[6] else None,
                "by_call_site": [{"call_site": r[0], "calls": r[1], "total_tokens": r[2] or 0} for r in by_site],
                "by_error_type": [{"error_type": r[0] or "unknown", "count": r[1]} for r in by_error_type],
                "recent": [{"call_site": r[0], "model": r[1], "prompt_tokens": r[2], "completion_tokens": r[3],
                           "total_tokens": r[4], "success": bool(r[5]), "error": r[6], "created_at": r[7],
                           "latency_ms": r[8], "error_type": r[9], "provider": r[10]} for r in rows],
            }

    def get_groq_usage_this_month(self, user_id: str) -> int:
        """Sum of total_tokens logged for this user since the start of the current calendar
        month (UTC) - the actual number app.core.quota checks against a limit. Only counts
        successful calls - a failed call attempt didn't cost anything worth capping on."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(total_tokens) FROM groq_usage_log WHERE user_id = ? AND success = 1 "
                "AND created_at >= datetime('now', 'start of month')", (user_id,)).fetchone()
            return row[0] or 0

    def get_user_groq_limit(self, user_id: str) -> Optional[int]:
        """A per-user override of the platform-default monthly token limit, if one has been
        set for this account. None means 'use the platform default' (see
        config.settings.default_monthly_groq_token_limit) - there's no public/self-service
        way for a user to raise their own limit, since the whole point is protecting the
        platform's own shared GROQ_API_KEY from any one account's unbounded cost."""
        with self._connect() as conn:
            row = conn.execute("SELECT monthly_groq_token_limit FROM users WHERE id = ?", (user_id,)).fetchone()
            return row[0] if row and row[0] is not None else None

    def set_user_groq_limit(self, user_id: str, limit: Optional[int]):
        with self._connect() as conn:
            conn.execute("UPDATE users SET monthly_groq_token_limit = ? WHERE id = ?", (limit, user_id))
            conn.commit()

    # --- Prompt Library (versioned) ---
    def create_prompt(self, user_id: str, name: str, content: str, description: str = "",
                       tags: Optional[List[str]] = None) -> Dict:
        prompt_id = f"prompt_{uuid.uuid4().hex[:16]}"
        tags_json = json.dumps(tags or [])
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "INSERT INTO prompts (id, user_id, name, description, tags, current_version) VALUES (?, ?, ?, ?, ?, 1)",
                    (prompt_id, user_id, name, description, tags_json))
                conn.execute(
                    "INSERT INTO prompt_versions (prompt_id, version, content, change_note) VALUES (?, 1, ?, ?)",
                    (prompt_id, content, "Initial version"))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"id": prompt_id, "name": name, "description": description, "tags": tags or [],
                "current_version": 1, "content": content}

    def list_prompts(self, user_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT p.id, p.name, p.description, p.tags, p.current_version, p.created_at, p.updated_at, pv.content
                FROM prompts p
                JOIN prompt_versions pv ON pv.prompt_id = p.id AND pv.version = p.current_version
                WHERE p.user_id = ? ORDER BY p.updated_at DESC
            """, (user_id,)).fetchall()
            return [{"id": r[0], "name": r[1], "description": r[2], "tags": json.loads(r[3] or "[]"),
                     "current_version": r[4], "created_at": r[5], "updated_at": r[6], "content": r[7]} for r in rows]

    def get_prompt(self, prompt_id: str, user_id: str) -> Optional[Dict]:
        """Returns the prompt with its full version history, or None if it doesn't exist or
        isn't owned by user_id."""
        with self._connect() as conn:
            row = conn.execute("SELECT id, user_id, name, description, tags, current_version, created_at, updated_at "
                               "FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
            if not row or row[1] != user_id:
                return None
            versions = conn.execute(
                "SELECT version, content, change_note, created_at FROM prompt_versions "
                "WHERE prompt_id = ? ORDER BY version DESC", (prompt_id,)).fetchall()
            return {
                "id": row[0], "name": row[2], "description": row[3], "tags": json.loads(row[4] or "[]"),
                "current_version": row[5], "created_at": row[6], "updated_at": row[7],
                "versions": [{"version": v[0], "content": v[1], "change_note": v[2], "created_at": v[3]} for v in versions],
            }

    def add_prompt_version(self, prompt_id: str, user_id: str, content: str, change_note: str = "") -> Optional[Dict]:
        """Editing a prompt creates a NEW version rather than overwriting - the point of a
        versioned library is that nothing is ever silently lost. Returns None if the prompt
        doesn't exist or isn't owned by user_id."""
        with self._connect() as conn:
            row = conn.execute("SELECT current_version, user_id FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
            if not row or row[1] != user_id:
                return None
            new_version = row[0] + 1
            conn.execute("BEGIN")
            try:
                conn.execute("INSERT INTO prompt_versions (prompt_id, version, content, change_note) VALUES (?, ?, ?, ?)",
                            (prompt_id, new_version, content, change_note))
                conn.execute("UPDATE prompts SET current_version = ?, updated_at = datetime('now') WHERE id = ?",
                            (new_version, prompt_id))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"id": prompt_id, "current_version": new_version, "content": content}

    def revert_prompt(self, prompt_id: str, user_id: str, to_version: int) -> Optional[Dict]:
        """Reverting doesn't delete history either - it copies the target version's content
        forward as a brand new version, so the version list stays a complete, honest timeline."""
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
            if not row or row[0] != user_id:
                return None
            target = conn.execute("SELECT content FROM prompt_versions WHERE prompt_id = ? AND version = ?",
                                  (prompt_id, to_version)).fetchone()
            if not target:
                return None
        return self.add_prompt_version(prompt_id, user_id, target[0], f"Reverted to version {to_version}")

    def delete_prompt(self, prompt_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
            if not row or row[0] != user_id:
                return False
            conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
            conn.execute("DELETE FROM prompt_versions WHERE prompt_id = ?", (prompt_id,))  # belt-and-braces w/ FK cascade
            conn.commit()
            return True

    # --- A/B Testing ---
    def create_ab_test(self, user_id: str, agent_id: str, name: str,
                        variants: List[Dict]) -> Dict:
        """variants: list of {label, system_prompt, weight, prompt_id?, prompt_version?}.
        Only one 'running' test per agent_id is allowed - check before calling this."""
        test_id = f"abtest_{uuid.uuid4().hex[:16]}"
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute("INSERT INTO ab_tests (id, user_id, agent_id, name, status) VALUES (?, ?, ?, ?, 'running')",
                            (test_id, user_id, agent_id, name))
                for v in variants:
                    variant_id = f"variant_{uuid.uuid4().hex[:12]}"
                    conn.execute(
                        "INSERT INTO ab_variants (id, test_id, label, prompt_id, prompt_version, system_prompt, weight) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (variant_id, test_id, v["label"], v.get("prompt_id"), v.get("prompt_version"),
                         v["system_prompt"], v.get("weight", 1.0)))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"id": test_id, "agent_id": agent_id, "name": name, "status": "running"}

    def get_running_ab_test_for_agent(self, agent_id: str) -> Optional[Dict]:
        """Used both to enforce 'one running test per bot' on creation, and to find the
        active test when a message comes in for that bot."""
        with self._connect() as conn:
            row = conn.execute("SELECT id, user_id, name FROM ab_tests WHERE agent_id = ? AND status = 'running'",
                               (agent_id,)).fetchone()
            if not row:
                return None
            variants = conn.execute(
                "SELECT id, label, system_prompt, weight FROM ab_variants WHERE test_id = ?", (row[0],)).fetchall()
            return {"id": row[0], "user_id": row[1], "name": row[2],
                    "variants": [{"id": v[0], "label": v[1], "system_prompt": v[2], "weight": v[3]} for v in variants]}

    def list_ab_tests(self, user_id: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, agent_id, name, status, created_at, ended_at, winner_variant_id FROM ab_tests "
                "WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
            return [{"id": r[0], "agent_id": r[1], "name": r[2], "status": r[3],
                     "created_at": r[4], "ended_at": r[5], "winner_variant_id": r[6]} for r in rows]

    def get_or_assign_variant(self, test_id: str, variants: List[Dict], conversation_id: str) -> Dict:
        """Sticky assignment: a conversation is assigned one variant the first time it's seen
        and keeps that same variant for every subsequent message, so a single conversation's
        results aren't split across arms. Weighted-random on first assignment."""
        with self._connect() as conn:
            row = conn.execute("SELECT variant_id FROM ab_assignments WHERE test_id = ? AND conversation_id = ?",
                               (test_id, conversation_id)).fetchone()
            if row:
                match = next((v for v in variants if v["id"] == row[0]), None)
                if match:
                    return match
            # No existing assignment (or its variant vanished) - weighted-random pick.
            import random
            weights = [max(0.0001, v.get("weight", 1.0)) for v in variants]
            chosen = random.choices(variants, weights=weights, k=1)[0]
            try:
                conn.execute("INSERT INTO ab_assignments (test_id, variant_id, conversation_id) VALUES (?, ?, ?)",
                            (test_id, chosen["id"], conversation_id))
                conn.commit()
            except sqlite3.IntegrityError:
                # Race: another call assigned this conversation between our SELECT and INSERT -
                # re-read the now-existing assignment rather than double-count it.
                conn.rollback()
                row = conn.execute("SELECT variant_id FROM ab_assignments WHERE test_id = ? AND conversation_id = ?",
                                   (test_id, conversation_id)).fetchone()
                if row:
                    match = next((v for v in variants if v["id"] == row[0]), None)
                    if match:
                        return match
            return chosen

    def get_ab_test_stats(self, test_id: str) -> Dict:
        """Per-variant: conversations assigned, assistant messages sent, thumbs up/down counts.
        Joins through ab_assignments -> chats/messages, since conversation_id == chats.id."""
        with self._connect() as conn:
            variants = conn.execute("SELECT id, label, weight FROM ab_variants WHERE test_id = ?", (test_id,)).fetchall()
            stats = {}
            for v in variants:
                variant_id = v[0]
                convs = conn.execute("SELECT conversation_id FROM ab_assignments WHERE test_id = ? AND variant_id = ?",
                                     (test_id, variant_id)).fetchall()
                conv_ids = [c[0] for c in convs]
                msg_count = up = down = 0
                if conv_ids:
                    placeholders = ",".join("?" * len(conv_ids))
                    row = conn.execute(
                        f"SELECT COUNT(*), SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END), "
                        f"SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) FROM messages "
                        f"WHERE chat_id IN ({placeholders}) AND role = 'assistant'", conv_ids).fetchone()
                    msg_count, up, down = row[0] or 0, row[1] or 0, row[2] or 0
                stats[variant_id] = {"label": v[1], "weight": v[2], "conversations": len(conv_ids),
                                     "assistant_messages": msg_count, "thumbs_up": up, "thumbs_down": down}
            return stats

    def stop_ab_test(self, test_id: str, user_id: str, winner_variant_id: Optional[str] = None) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM ab_tests WHERE id = ?", (test_id,)).fetchone()
            if not row or row[0] != user_id:
                return False
            conn.execute("UPDATE ab_tests SET status = 'completed', ended_at = datetime('now'), "
                        "winner_variant_id = ? WHERE id = ?", (winner_variant_id, test_id))
            conn.commit()
            return True

    def delete_ab_test(self, test_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM ab_tests WHERE id = ?", (test_id,)).fetchone()
            if not row or row[0] != user_id:
                return False
            conn.execute("DELETE FROM ab_tests WHERE id = ?", (test_id,))
            conn.execute("DELETE FROM ab_variants WHERE test_id = ?", (test_id,))
            conn.execute("DELETE FROM ab_assignments WHERE test_id = ?", (test_id,))
            conn.commit()
            return True

    # --- Knowledge Base (per-bot documents, retrieved via app.core.knowledge_base) ---
    def add_kb_document(self, agent_id: str, user_id: str, filename: str, content: str, chunks: List[str]) -> str:
        doc_id = f"kbdoc_{uuid.uuid4().hex[:16]}"
        with self._connect() as conn:
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "INSERT INTO kb_documents (id, agent_id, user_id, filename, content) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, agent_id, user_id, filename, content))
                for i, chunk in enumerate(chunks):
                    conn.execute(
                        "INSERT INTO kb_chunks (document_id, agent_id, chunk_index, content) VALUES (?, ?, ?, ?)",
                        (doc_id, agent_id, i, chunk))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return doc_id

    def list_kb_documents(self, agent_id: str, user_id: str) -> List[Dict]:
        """Ownership-checked - this is the MANAGEMENT view (a bot owner seeing what they've
        uploaded), unlike get_kb_chunks_for_agent below which is deliberately NOT
        ownership-checked because it's used at chat time by whoever is talking to the bot."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, created_at, LENGTH(content) FROM kb_documents "
                "WHERE agent_id = ? AND user_id = ? ORDER BY created_at DESC", (agent_id, user_id)).fetchall()
            return [{"id": r[0], "filename": r[1], "created_at": r[2], "content_length": r[3]} for r in rows]

    def delete_kb_document(self, document_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT user_id FROM kb_documents WHERE id = ?", (document_id,)).fetchone()
            if not row or row[0] != user_id:
                return False
            conn.execute("DELETE FROM kb_documents WHERE id = ?", (document_id,))
            conn.execute("DELETE FROM kb_chunks WHERE document_id = ?", (document_id,))
            conn.commit()
            return True

    def get_kb_chunks_for_agent(self, agent_id: str) -> List[Dict]:
        """Deliberately NOT ownership-checked - called at CHAT time by whoever is talking to
        a published bot, which is normally a different person than the bot's owner (that's
        the whole point of publishing it). Document management (add/list/delete above) IS
        ownership-checked; reading chunks to actually answer a question is not, since the
        knowledge base's purpose is to be used by the bot's real users."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT kb_chunks.id, kb_chunks.content, kb_documents.filename FROM kb_chunks "
                "JOIN kb_documents ON kb_documents.id = kb_chunks.document_id "
                "WHERE kb_chunks.agent_id = ?", (agent_id,)).fetchall()
            return [{"id": r[0], "content": r[1], "filename": r[2]} for r in rows]

    # --- Human Handoff ---
    def log_handoff_event(self, agent_id: str, conversation_id: str, user_message: str,
                           reason: str, delivered: bool, delivery_error: Optional[str] = None):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO handoff_events (agent_id, conversation_id, user_message, reason, "
                "delivered, delivery_error) VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, conversation_id, user_message, reason, int(delivered), delivery_error))
            conn.commit()

    def get_handoff_events(self, agent_id: str, limit: int = 50) -> List[Dict]:
        """Not ownership-checked at this layer - the API endpoint calling this is (see
        main.py), same split as get_kb_chunks_for_agent vs list_kb_documents above."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conversation_id, user_message, reason, delivered, delivery_error, created_at "
                "FROM handoff_events WHERE agent_id = ? ORDER BY id DESC LIMIT ?", (agent_id, limit)).fetchall()
            return [{"conversation_id": r[0], "user_message": r[1], "reason": r[2],
                    "delivered": bool(r[3]), "delivery_error": r[4], "created_at": r[5]} for r in rows]

    # --- Telegram integration ---
    def upsert_telegram_link(self, agent_id: str, user_id: str, bot_token: str, bot_id: int,
                              bot_username: str, webhook_secret: str, status: str) -> str:
        """One Telegram bot per GenesisAI bot (agent_id is UNIQUE) - reconnecting the same
        bot updates the existing link (new token/secret/status) rather than creating a
        duplicate row."""
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM telegram_links WHERE agent_id = ?", (agent_id,)).fetchone()
            if existing:
                link_id = existing[0]
                conn.execute(
                    "UPDATE telegram_links SET bot_token=?, bot_id=?, bot_username=?, webhook_secret=?, "
                    "status=?, last_error=NULL, updated_at=datetime('now') WHERE id=?",
                    (bot_token, bot_id, bot_username, webhook_secret, status, link_id))
            else:
                link_id = f"tglink_{uuid.uuid4().hex[:16]}"
                conn.execute(
                    "INSERT INTO telegram_links (id, agent_id, user_id, bot_token, bot_id, bot_username, "
                    "webhook_secret, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (link_id, agent_id, user_id, bot_token, bot_id, bot_username, webhook_secret, status))
            conn.commit()
        return link_id

    def get_telegram_link(self, agent_id: str, user_id: str) -> Optional[Dict]:
        """Ownership-checked management view - the bot's token is masked, never returned in
        full once stored (same principle as never echoing a password hash back)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, bot_id, bot_username, status, last_error, created_at, updated_at, bot_token "
                "FROM telegram_links WHERE agent_id = ? AND user_id = ?", (agent_id, user_id)).fetchone()
            if not row:
                return None
            token = row[7] or ""
            masked = f"...{token[-6:]}" if len(token) > 6 else "..."
            return {"id": row[0], "bot_id": row[1], "bot_username": row[2], "status": row[3],
                    "last_error": row[4], "created_at": row[5], "updated_at": row[6], "token_masked": masked}

    def get_telegram_link_by_secret(self, webhook_secret: str) -> Optional[Dict]:
        """NOT ownership-checked - this is how the webhook receiver (main.py) identifies
        which bot/owner an incoming Telegram update belongs to. The secret itself IS the
        credential here (Telegram sends it back in both the URL path and a header on every
        call - see main.py's webhook route), not a user session."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, agent_id, user_id, bot_token, bot_id, bot_username, status "
                "FROM telegram_links WHERE webhook_secret = ?", (webhook_secret,)).fetchone()
            if not row:
                return None
            return {"id": row[0], "agent_id": row[1], "user_id": row[2], "bot_token": row[3],
                    "bot_id": row[4], "bot_username": row[5], "status": row[6]}

    def update_telegram_link_status(self, link_id: str, status: str, last_error: Optional[str] = None):
        with self._connect() as conn:
            conn.execute("UPDATE telegram_links SET status = ?, last_error = ?, updated_at = datetime('now') "
                        "WHERE id = ?", (status, last_error, link_id))
            conn.commit()

    def delete_telegram_link(self, agent_id: str, user_id: str) -> Optional[Dict]:
        """Returns the deleted link's bot_token so the caller can call deleteWebhook on
        Telegram's side before the record disappears, or None if not found/not owned."""
        with self._connect() as conn:
            row = conn.execute("SELECT id, bot_token FROM telegram_links WHERE agent_id = ? AND user_id = ?",
                               (agent_id, user_id)).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM telegram_links WHERE id = ?", (row[0],))
            conn.execute("DELETE FROM telegram_events WHERE link_id = ?", (row[0],))
            conn.commit()
            return {"id": row[0], "bot_token": row[1]}

    def log_telegram_event(self, link_id: str, event_type: str, detail: str = ""):
        with self._connect() as conn:
            conn.execute("INSERT INTO telegram_events (link_id, event_type, detail) VALUES (?, ?, ?)",
                        (link_id, event_type, detail))
            conn.commit()

    def get_telegram_events(self, link_id: str, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_type, detail, created_at FROM telegram_events WHERE link_id = ? "
                "ORDER BY id DESC LIMIT ?", (link_id, limit)).fetchall()
            return [{"event_type": r[0], "detail": r[1], "created_at": r[2]} for r in rows]


    def get_bot_analytics(self, agent_id: str) -> Dict:
        """Aggregates what's already being collected (conversations, ratings) into something
        a bot owner can actually act on - previously all of this existed only as raw rows
        nobody could see. Not ownership-checked here (main.py checks bot ownership before
        calling this, same pattern as list_kb_documents' caller) - kept as a pure aggregation
        query so it's easy to test in isolation.

        Known limitation, noted rather than silently omitted: this does NOT include a
        cost/Groq-usage breakdown per bot - groq_usage_log is keyed by user_id and call_site,
        not agent_id, since a single user's chat_routing calls aren't currently tagged with
        which bot was selected at the time. Attributing Groq cost per bot would need that
        added at the call site in agent.py - a reasonable follow-up, not done here to keep
        this change's scope to what's already cleanly attributable."""
        with self._connect() as conn:
            conv_row = conn.execute(
                "SELECT COUNT(*), MIN(created_at), MAX(updated_at) FROM chats WHERE agent_id = ?",
                (agent_id,)).fetchone()
            total_conversations, first_used_at, last_used_at = conv_row

            msg_row = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END), SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) "
                "FROM messages WHERE chat_id IN (SELECT id FROM chats WHERE agent_id = ?)",
                (agent_id,)).fetchone()
            total_messages, assistant_messages, thumbs_up, thumbs_down = msg_row

            # A simple "what are people actually asking" signal: the first user message of
            # each conversation with this bot, most recent first - not a fancy topic model,
            # just the raw questions a bot owner would want to skim.
            first_messages = conn.execute(
                "SELECT m.content, m.created_at FROM messages m "
                "JOIN chats c ON c.id = m.chat_id "
                "WHERE c.agent_id = ? AND m.role = 'user' AND m.id = ("
                "  SELECT MIN(id) FROM messages WHERE chat_id = m.chat_id AND role = 'user'"
                ") ORDER BY m.created_at DESC LIMIT 20", (agent_id,)).fetchall()

            return {
                "total_conversations": total_conversations or 0,
                "total_messages": total_messages or 0,
                "assistant_messages": assistant_messages or 0,
                "thumbs_up": thumbs_up or 0,
                "thumbs_down": thumbs_down or 0,
                "first_used_at": first_used_at,
                "last_used_at": last_used_at,
                "recent_opening_questions": [{"content": r[0], "created_at": r[1]} for r in first_messages],
            }


