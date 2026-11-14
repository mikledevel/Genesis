"""Import an exported Genesis SQLite database into an empty PostgreSQL database.

Usage: DATABASE_URL=... python -m scripts.import_sqlite_postgres /path/to/export.sqlite
The source is opened read-only. Existing PostgreSQL data is never overwritten.
"""
import os
import sqlite3
from pathlib import Path

import psycopg
from psycopg import sql

from app.db.database import GenesisDB


def import_database(source_path, database_url):
    source_path = Path(source_path).resolve(strict=True)
    GenesisDB(database_url=database_url)
    counts = {}
    source = sqlite3.connect(source_path.as_uri() + '?mode=ro', uri=True)
    try:
        source.execute('BEGIN')
        source_tables = [r[0] for r in source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid")]
        with psycopg.connect(database_url, prepare_threshold=None) as target:
            target.execute('SET search_path TO genesis, pg_catalog')
            tables = [r[0] for r in target.execute("SELECT tablename FROM pg_tables WHERE schemaname='genesis'")]
            target.execute(sql.SQL('LOCK TABLE {} IN ACCESS EXCLUSIVE MODE').format(
                sql.SQL(',').join(sql.Identifier('genesis', t) for t in tables)))
            for table in tables:
                if table == "schema_migrations":
                    continue
                if target.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier(table))).fetchone():
                    raise ValueError('Target database is not empty; import was refused')
            if set(source_tables) - set(tables):
                raise ValueError('The export contains unknown tables; import was refused')
            for table in source_tables:
                if table == "schema_migrations":
                    continue
                rows = source.execute('SELECT * FROM "' + table.replace('"', '""') + '"')
                columns = [c[0] for c in rows.description]
                statement = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                    sql.Identifier(table), sql.SQL(',').join(map(sql.Identifier, columns)),
                    sql.SQL(',').join(sql.Placeholder() for _ in columns))
                count = 0
                while batch := rows.fetchmany(500):
                    with target.cursor() as cursor:
                        cursor.executemany(statement, batch)
                    count += len(batch)
                counts[table] = count
            for table, column in target.execute(
                "SELECT table_name,column_name FROM information_schema.columns "
                "WHERE table_schema='genesis' AND is_identity='YES'").fetchall():
                maximum = target.execute(sql.SQL('SELECT MAX({}) FROM {}').format(
                    sql.Identifier(column), sql.Identifier(table))).fetchone()[0]
                sequence = target.execute('SELECT pg_get_serial_sequence(%s,%s)', ('genesis.' + table, column)).fetchone()[0]
                target.execute('SELECT setval(%s,%s,%s)', (sequence, maximum or 1, maximum is not None))
        return counts
    finally:
        source.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Exported SQLite database; opened read-only')
    args = parser.parse_args()
    url = os.environ.get('DATABASE_URL')
    if not url:
        parser.error('Set DATABASE_URL in the process environment')
    counts = import_database(args.source, url)
    print(f'Imported {sum(counts.values())} rows across {len(counts)} tables')
