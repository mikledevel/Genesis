"""Ограниченный логический snapshot PostgreSQL для проверки маленького staging."""
import base64
import json
import time
from psycopg import sql

MAX_BYTES = 32 * 1024 * 1024
MAX_ROWS = 5000


def _encode(value):
    if isinstance(value, (bytes, memoryview)):
        return {'$bytes': base64.b64encode(bytes(value)).decode('ascii')}
    raise TypeError('Unsupported database value')


def snapshot_bytes(db):
    with db._connect() as adapter:
        conn = adapter.raw
        # Все таблицы читаются из одного согласованного снимка, без блокировки записей.
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        size = conn.execute('SELECT COALESCE(SUM(size_bytes),0) FROM genesis.artifact_files').fetchone()[0]
        if size > MAX_BYTES // 2:
            raise ValueError('Staging snapshot exceeds the 16 MB artifact limit')
        names = [row[0] for row in conn.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='genesis' ORDER BY tablename")]
        tables = {}
        total = 0
        for name in names:
            cursor = conn.execute(sql.SQL('SELECT * FROM {} LIMIT %s').format(sql.Identifier('genesis', name)), (MAX_ROWS + 1,))
            rows = cursor.fetchall()
            if len(rows) > MAX_ROWS:
                raise ValueError('Staging snapshot exceeds the row limit')
            data = {'columns': [c.name for c in cursor.description], 'rows': rows}
            encoded = json.dumps(data, default=_encode, allow_nan=False, separators=(',', ':'))
            total += len(encoded.encode())
            if total > MAX_BYTES:
                raise ValueError('Staging snapshot exceeds the size limit')
            tables[name] = data
        result = {'format': 'genesis-staging-snapshot-v1', 'schema_version': 4,
                  'created_at': int(time.time()), 'tables': tables}
        payload = json.dumps(result, default=_encode, allow_nan=False, separators=(',', ':')).encode()
        if len(payload) > MAX_BYTES:
            raise ValueError('Staging snapshot exceeds the size limit')
        return payload
