#!/usr/bin/env python3
"""
Backs up genesis.db using Python's built-in sqlite3.backup() API, which is safe to run
against a live database (unlike a plain file copy, which can copy a half-written page if the
server is mid-write - especially relevant here since the DB runs in WAL mode). This uses only
the Python standard library, deliberately avoiding a dependency on the separate `sqlite3` CLI
binary, which is NOT guaranteed to be installed on every server/container (it's a different
thing from the `sqlite3` Python module, which ships with Python itself).

Usage:
    python3 scripts/backup_db.py
    DB_PATH=/path/to/genesis.db BACKUP_DIR=/path/to/backups python3 scripts/backup_db.py

To run automatically every day at 3am, add this to `crontab -e`:
    0 3 * * * cd /path/to/GenesisAI-main && /path/to/venv/bin/python3 scripts/backup_db.py >> /var/log/genesis-backup.log 2>&1

With DATABASE_URL, uses pg_dump custom format for the private genesis schema.
SQLite file artifacts require a separate directory backup; see docs/BACKEND_HARDENING.md.
"""
import gzip
import os
import shutil
import sqlite3
import sys
import subprocess
from urllib.parse import urlsplit, parse_qs
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = os.environ.get("DB_PATH", "genesis.db")
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "backups"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))


def backup_postgres(url):
    parsed = urlsplit(url)
    env = dict(os.environ,PGHOST=parsed.hostname or "",PGPORT=str(parsed.port or 5432),
               PGDATABASE=parsed.path.lstrip("/"),PGUSER=parsed.username or "",PGPASSWORD=parsed.password or "")
    from urllib.parse import unquote
    for name in ("PGUSER","PGPASSWORD","PGDATABASE"):
        env[name] = unquote(env[name])
    env["PGSSLMODE"] = parse_qs(parsed.query).get("sslmode",["require"])[0]
    BACKUP_DIR.mkdir(parents=True,exist_ok=True)
    destination = BACKUP_DIR / ("genesis_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".dump")
    try:
        subprocess.run(["pg_dump","--format=custom","--schema=genesis","--file",str(destination)],env=env,check=True,capture_output=True)
        os.chmod(destination,0o600)
    except Exception:
        destination.unlink(missing_ok=True)
        raise RuntimeError("PostgreSQL backup failed; verify pg_dump availability and connection settings") from None
    print(f"Backup created: {destination.name}")
    prune_old_backups()
    return destination


def backup():
    if os.environ.get("DATABASE_URL"):
        return backup_postgres(os.environ["DATABASE_URL"])
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found - nothing to back up.", file=sys.stderr)
        sys.exit(1)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_path = BACKUP_DIR / f"genesis_{timestamp}.db"

    # sqlite3's backup() API performs a live, page-consistent copy - safe to run while the
    # server is actively serving requests against DB_PATH, including under WAL mode.
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(str(dest_path))
    with dst:
        src.backup(dst)
    src.close()
    dst.close()

    gz_path = f"{dest_path}.gz"
    with open(dest_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(dest_path)
    os.chmod(gz_path, 0o600)

    print(f"Backed up {DB_PATH} -> {gz_path}")
    prune_old_backups()


def prune_old_backups():
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    for f in [f for f in BACKUP_DIR.glob("genesis_*") if f.name.endswith((".db.gz",".dump"))]:
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
            print(f"Pruned old backup: {f}")

    print(f"Current backups in {BACKUP_DIR}:")
    for f in sorted([f for f in BACKUP_DIR.glob("genesis_*") if f.name.endswith((".db.gz",".dump"))]):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    backup()
