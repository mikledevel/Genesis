# SQLite → Postgres migration plan

`GenesisDB` (`app/db/database.py`) is 68 methods of raw `sqlite3` calls, and
`settings.database_url` already exists in `app/config.py` but is currently unused - it's a
placeholder for this migration, not a working option today. I'm writing this as a plan rather
than shipping a rewrite in this pass because I don't have a Postgres instance or network access
in this environment to run it against - a blind rewrite of the entire data layer that I can only
syntax-check, not execute, is more likely to ship a subtle bug than to help. This is the concrete
plan for when that access exists.

## Why this matters, and when it actually matters
SQLite (even with the WAL + busy_timeout mitigation applied in this pass) only ever allows one
writer at a time. That's fine for the current single-process deployment. It becomes the ceiling
the moment either of these is true:
- The app runs as more than one process/worker (multiple uvicorn workers, multiple containers).
- Write volume (chats, training runs, transactions) gets high enough that writers start queuing.

Don't migrate before you need to - Postgres adds real operational surface (a server to run,
back up, and monitor) that SQLite gives you for free. Migrate when either bullet above becomes
true, not preemptively.

## Recommended approach: SQLAlchemy Core, not raw SQL, not full ORM
- **Not raw SQL per-backend**: maintaining two parallel sets of hand-written SQL (SQLite
  dialect + Postgres dialect) for 68 methods is a maintenance trap - the two will drift.
- **Not a full ORM rewrite** (SQLAlchemy ORM with declarative models): the current code is
  already organized as a clean method-per-operation repository (`create_user`,
  `add_transaction`, etc.) - that shape is worth keeping. A full ORM rewrite would touch every
  call site in `app/main.py` and `app/core/*` for no behavioral gain.
- **SQLAlchemy Core** (`sqlalchemy.Table` + `select()/insert()/update()`) is the right level:
  it keeps the existing method-per-operation shape, but the same code runs against SQLite,
  Postgres, or anything else SQLAlchemy supports, and query-building is dialect-agnostic.

## Concrete steps
1. **Add `sqlalchemy>=2.0` to `requirements.txt`.**
2. **Define the schema once**, as `sqlalchemy.Table` objects in a new `app/db/schema.py`,
   translating the `CREATE TABLE IF NOT EXISTS` blocks currently inline in
   `GenesisDB._init_db()`. Two things need dialect-aware handling:
   - `INTEGER PRIMARY KEY AUTOINCREMENT` (SQLite) → `Integer, primary_key=True` (SQLAlchemy
     picks the right autoincrement syntax per dialect automatically).
   - `datetime('now')` as a column default → `server_default=func.now()`, which SQLAlchemy
     translates correctly per backend.
3. **Replace `GenesisDB.__init__`** to build an `Engine` from `settings.database_url`, falling
   back to `sqlite:///genesis.db` when unset (preserves today's zero-config behavior).
4. **Port methods in dependency order, in small PRs, each with a test that runs against BOTH
   backends** (a local SQLite file and a real Postgres - `pytest` + `testcontainers-python` for
   a throwaway Postgres in CI is the standard way to do this without a permanent test server).
   Suggested batch order, smallest blast radius first:
   - `users` + `auth` methods (create_user, get_user_by_email, get_user_by_id, balance) - no
     foreign-key dependents yet, easiest to verify in isolation.
   - `transactions` / `payments` - touches money, so this batch gets the most scrutiny and the
     Stripe webhook idempotency check (`stripe_session_already_processed`) re-verified
     explicitly, since that's the one method where a race condition would double-credit a user.
   - `chats` / `messages` - highest write volume, so this is where WAL was buying the most
     headroom in the interim; also the batch to load-test before/after.
   - Everything else (`projects`, `memory_facts`, `api_keys`, `usage_log`,
     `groq_usage_log`, `prompts`, `ab_tests`).
5. **Cut over with the WAL mitigation still in place as the rollback path** - keep
   `database_url` empty (SQLite) in production until each batch is verified, then flip it once
   the full surface is ported. Don't do a big-bang cutover.
6. **After cutover**: remove the now-dead SQLite-specific PRAGMA calls in `_connect()`
   (`journal_mode`, `busy_timeout`) - those are meaningless once Postgres is the backend - but
   only after `database_url` is unconditionally required, not optional, or local dev without
   Postgres configured breaks silently.

## What NOT to do
Don't reach for an ORM-mapped-object rewrite, and don't try to keep raw `sqlite3.connect()`
calls "for now" alongside a partial Postgres path with an if/else per method - that's the
worst of both worlds (two schemas to keep in sync, and no dialect abstraction either).
