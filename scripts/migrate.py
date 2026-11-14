"""Применить схему и импортировать старые конфигурации; данные не удаляются."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Проверить версию без изменения схемы")
    args = parser.parse_args()
    from app.config import settings
    if args.check:
        if settings.database_url:
            from app.db.postgres import pool_for
            with pool_for(settings.database_url).connection() as conn:
                row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        else:
            import sqlite3
            with sqlite3.connect("file:genesis.db?mode=ro",uri=True) as conn:
                row = conn.execute("PRAGMA user_version").fetchone()
        if not row or row[0] != 4:
            raise SystemExit("Schema migration is required")
    else:
        settings.database_auto_migrate = True
        from app.db.database import GenesisDB
        db = GenesisDB()
        from app.core.agent_store import AgentStore
        from app.core.custom_blocks import CustomBlockStore
        from app.core.marketplace import Marketplace
        from app.core.engine.registry import ModelRegistry
        AgentStore(db=db)
        CustomBlockStore()
        Marketplace()
        ModelRegistry()
        from app.core.secrets_store import encrypt_secret, key_digest
        with db.write_transaction() as conn:
            for link_id,token in conn.execute("SELECT id,bot_token FROM telegram_links").fetchall():
                if not token.startswith("enc:v1:"):
                    conn.execute("UPDATE telegram_links SET bot_token=? WHERE id=?",(encrypt_secret(token),link_id))
            for key_id,key in conn.execute("SELECT id,api_key FROM api_keys").fetchall():
                if not key.startswith("sha256:"):
                    digest = key_digest(key)
                    conn.execute("UPDATE api_keys SET api_key=? WHERE id=?",(digest,key_id))
                    conn.execute("UPDATE usage_log SET api_key=? WHERE api_key=?",(digest,key))
                    conn.execute("UPDATE public_api_usage SET api_key=? WHERE api_key=?",(digest,key))
                    conn.execute("UPDATE transactions SET related_id=? WHERE related_id=?",(digest,key))
    print("Schema version 4 is ready")


if __name__ == "__main__":
    main()
