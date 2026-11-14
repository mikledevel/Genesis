"""Build the PostgreSQL schema from the repository's current SQLite schema.

This is an operator utility, not a runtime migration or data export. It uses an
empty temporary SQLite database and never reads a user's genesis.db.
"""
import tempfile
from pathlib import Path

from app.db.database import GenesisDB
from app.db.postgres import postgres_sql


def schema_sql():
    with tempfile.TemporaryDirectory() as directory:
        db = GenesisDB(str(Path(directory) / 'schema.sqlite'))
        with db._connect() as conn:
            rows = conn.execute("SELECT name, type, sql FROM sqlite_master WHERE sql IS NOT NULL "
                                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall()
    statements = ['SET search_path TO genesis, pg_catalog;']
    for name, kind, sql in rows:
        statements.append(postgres_sql(sql).replace('%%', '%') + ';')
        if kind == 'table':
            statements.append(f'ALTER TABLE "{name}" ENABLE ROW LEVEL SECURITY;')
    return '\n'.join(statements)


if __name__ == '__main__':
    print(schema_sql())
