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

NOTE: this script is for the current SQLite setup. If/when this migrates to Postgres (see
CHANGES.md), replace this with `pg_dump` on the same cron schedule instead - this script will
stop being correct at that point since there will be no genesis.db file to copy.
"""
import gzip
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = os.environ.get("DB_PATH", "genesis.db")
BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "backups"))
RETENTION_DAYS = int(os.environ.get("RETENTION_DAYS", "14"))


def backup():
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

    print(f"Backed up {DB_PATH} -> {gz_path}")
    prune_old_backups()


def prune_old_backups():
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    for f in BACKUP_DIR.glob("genesis_*.db.gz"):
        if datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
            print(f"Pruned old backup: {f}")

    print(f"Current backups in {BACKUP_DIR}:")
    for f in sorted(BACKUP_DIR.glob("genesis_*.db.gz")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    backup()
