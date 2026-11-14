"""Восстановить staging snapshot только в пустую одноразовую PostgreSQL на loopback."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def restore(path, url):
    if urlsplit(url).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('Restore is restricted to a disposable loopback database')
    with gzip.open(path, 'rb') as file:
        content = file.read(32 * 1024 * 1024 + 1)
    if len(content) > 32 * 1024 * 1024:
        raise ValueError('Snapshot is too large')
    snapshot = json.loads(content)
    if snapshot.get('format') != 'genesis-staging-snapshot-v1' or snapshot.get('schema_version') != 4:
        raise ValueError('Unsupported snapshot format')
    import psycopg
    from psycopg import sql
    with psycopg.connect(url) as conn:
        conn.execute('CREATE SCHEMA IF NOT EXISTS genesis')
    from app.db.database import GenesisDB
    db = GenesisDB(database_url=url)
    with db._connect() as adapter:
        conn = adapter.raw
        existing = {r[0] for r in conn.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='genesis'")}
        if existing != set(snapshot['tables']):
            raise ValueError('Snapshot tables differ from the current schema')
        for name in sorted(existing - {'schema_migrations'}):
            if conn.execute(sql.SQL('SELECT 1 FROM {} LIMIT 1').format(sql.Identifier('genesis', name))).fetchone():
                raise ValueError('Destination database is not empty')
        # Требует локального оператора PostgreSQL; endpoint не выполняет восстановление.
        conn.execute('SET LOCAL session_replication_role = replica')
        for name, table in snapshot['tables'].items():
            if name == 'schema_migrations':
                continue
            statement = sql.SQL('INSERT INTO {} ({}) VALUES ({})').format(
                sql.Identifier('genesis', name), sql.SQL(',').join(map(sql.Identifier, table['columns'])),
                sql.SQL(',').join(sql.Placeholder() for _ in table['columns']))
            for row in table['rows']:
                values = [base64.b64decode(v['$bytes'], validate=True) if isinstance(v, dict) and set(v) == {'$bytes'} else v for v in row]
                conn.execute(statement, values)
        for name, column, sequence in conn.execute("SELECT table_name,column_name,pg_get_serial_sequence(quote_ident(table_schema)||'.'||quote_ident(table_name),column_name) FROM information_schema.columns WHERE table_schema='genesis' AND is_identity='YES'"):
            maximum = conn.execute(sql.SQL('SELECT MAX({}) FROM {}').format(sql.Identifier(column), sql.Identifier('genesis', name))).fetchone()[0]
            conn.execute('SELECT setval(%s::regclass,%s,%s)', (sequence, maximum or 1, maximum is not None))
    print('Staging snapshot restored into the disposable database')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--database-url', required=True)
    args = parser.parse_args()
    restore(args.snapshot, args.database_url)
