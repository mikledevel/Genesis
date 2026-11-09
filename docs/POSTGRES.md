# Persistent database on Render Free

GenesisDB uses PostgreSQL when `DATABASE_URL` is set; explicit SQLite paths in local
scripts and tests continue to use SQLite. The server never silently falls back to
SQLite when PostgreSQL is unavailable. Application startup verifies the database.

The GenesisAI Supabase project is `staqccchqykvcbpdoxpr`, in organization MMM.
Provisioning was quoted at $0/month. Its application tables are in private schema
`genesis`, owned by dedicated login `genesis_app`. RLS is enabled on every table;
`PUBLIC`, `anon`, and `authenticated` have no schema access. The backend owner can
access its own tables. Browser requests continue through GenesisAI's authenticated
API; do not expose the database URL or application password to the frontend.

Use the project's **Connect → Session pooler** address (port 5432) when IPv6 direct
connections are unavailable. For the dedicated login, the pooler username is
`genesis_app.staqccchqykvcbpdoxpr`. Require TLS in the connection string. The backend
uses at most four database connections per process and disables prepared statements.

## Before switching an existing deployment

Export the current SQLite database first. A new Render deployment cannot recover
files from the previous instance's ephemeral filesystem. Do not treat an empty
Supabase database as an automatic migration of existing accounts. The new database
must be populated from an export, or the owner must explicitly accept a fresh start.

Set `DATABASE_URL` in Render and deploy the PostgreSQL-capable code together.
Keep the Stripe webhook disabled until the database, API key, signature verification,
and subscription fulfillment have been verified. On Render, Checkout refuses to
start if persistent database storage is not configured.

Database storage preserves accounts, billing, chats, knowledge-base text and usage.
Model binaries, uploaded datasets, and other files still need external object storage;
PostgreSQL does not make Render's local filesystem persistent.

## Local verification

Run a disposable PostgreSQL server on loopback, create a private `genesis` schema,
then run:

```sh
GENESIS_TEST_POSTGRES_URL=postgresql://127.0.0.1:55439/postgres \
  ./venv/bin/python -m pytest tests/test_postgres_database.py -q
```

These tests truncate the disposable schema; remote hosts are refused. Tests cover
reconnection, token consumption, balance atomicity, webhook deduplication, checkout
locking, messages, dates, versioned prompts, job claims, and RLS configuration.
