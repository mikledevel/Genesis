# SQLite → PostgreSQL

`DATABASE_URL` selects the implemented PostgreSQL backend; an empty value selects local SQLite. The adapter in `app/db/postgres.py` preserves the repository contracts with bound parameters and private schema `genesis`. No ORM migration is required.

## Moving an existing installation

1. Stop writes and create a consistent SQLite backup with `scripts/backup_db.py`. Separately preserve local datasets, models, registry artifacts and generated files.
2. Provision an empty PostgreSQL database with a dedicated backend role and private `genesis` schema. Use TLS for remote connections. See [POSTGRES.md](POSTGRES.md).
3. Run `scripts/migrate.py` with the destination DATABASE_URL. This creates/version-checks the destination schema and imports locally available configuration JSON. It does **not** copy existing accounts and transactions out of SQLite.
4. Transfer SQLite rows with a reviewed data migration, preserving IDs, relationships, balances, ownership and timestamps. This repository does not provide an automatic cross-database data-transfer tool. Keep both snapshots until reconciliation is complete.
5. Validate row counts, total balances, financial operations, disabled accounts, chat ownership, API keys, encrypted integration tokens and model files against the source. Transfer pre-existing file artifacts explicitly.
6. Test the destination with application writes, backup and restore into another disposable database. Set DATABASE_AUTO_MIGRATE=false for the serving process, then switch traffic.

Do not treat an empty destination as a successful migration of existing customer data. Rollback requires keeping the original snapshot and accounting for writes made after cutover.

## Verification and operations

Run `GENESIS_TEST_POSTGRES_URL=postgresql://127.0.0.1:55439/postgres python scripts/check_backend.py tests/test_postgres_database.py -q` against a disposable local database only. The suite truncates its private schema and includes a dump/restore test when PostgreSQL client tools are installed.

The current schema version is 3. See [BACKEND_HARDENING.md](BACKEND_HARDENING.md) for release migration, artifact persistence, encryption-key stability and backup commands.
