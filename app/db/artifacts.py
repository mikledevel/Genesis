"""Постоянные артефакты небольшого развёртывания; локальный диск — проверяемый кэш."""
import hashlib
import os
from pathlib import Path
import tempfile
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifact_files (path TEXT PRIMARY KEY, user_id TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, content BYTEA NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_artifacts_owner ON artifact_files(user_id);
"""


def artifact_key(path):
    root = Path.cwd().resolve()
    resolved = Path(path).resolve()
    if root not in resolved.parents or resolved.relative_to(root).parts[0] not in {"datasets","registry","models","generated"}:
        raise ValueError("Artifact must be inside an application data directory")
    return resolved.relative_to(root).as_posix()


class ArtifactMixin:
    def persist_artifact(self,path,user_id="system"):
        if not self.is_postgres:
            return
        file = Path(path)
        size = file.stat().st_size
        if size > 64*1024*1024:
            raise ValueError("Artifact exceeds the persistent storage limit of 64 MB")
        data = file.read_bytes()
        key = artifact_key(file)
        with self.write_transaction() as conn:
            conn.execute("SELECT id FROM users WHERE id=? FOR UPDATE",(user_id,))
            used = conn.execute("SELECT COALESCE(SUM(size_bytes),0) FROM artifact_files WHERE user_id=? AND path<>?",(user_id,key)).fetchone()[0]
            if used + size > 500*1024*1024:
                raise ValueError("Persistent artifact storage is full for this account")
            conn.execute("INSERT INTO artifact_files(path,user_id,sha256,size_bytes,content,created_at) VALUES (?,?,?,?,?,?) "
                         "ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256,size_bytes=excluded.size_bytes,content=excluded.content",
                         (key,user_id,hashlib.sha256(data).hexdigest(),size,data,int(time.time())))

    def restore_artifact(self,path):
        if not self.is_postgres:
            return Path(path).exists()
        key = artifact_key(path)
        with self._connect() as conn:
            row = conn.execute("SELECT sha256,content FROM artifact_files WHERE path=?",(key,)).fetchone()
        if not row:
            return Path(path).exists()
        destination = Path(key)
        data = bytes(row[1])
        if hashlib.sha256(data).hexdigest() != row[0]:
            raise ValueError("Artifact checksum mismatch")
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == row[0]:
            return True
        destination.parent.mkdir(parents=True,exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=destination.parent,prefix=".restore-")
        try:
            with os.fdopen(fd,"wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary,destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return True

    def restore_owned_artifacts(self,user_id,prefix="datasets/"):
        if not self.is_postgres:
            return
        with self._connect() as conn:
            paths = [r[0] for r in conn.execute("SELECT path FROM artifact_files WHERE user_id=? AND path LIKE ?",(user_id,prefix+"%")).fetchall()]
        for path in paths:
            self.restore_artifact(path)

    def delete_artifact(self,path):
        if self.is_postgres:
            with self.write_transaction() as conn:
                conn.execute("DELETE FROM artifact_files WHERE path=?",(artifact_key(path),))
        Path(path).unlink(missing_ok=True)

    def delete_artifacts_under(self,path):
        if self.is_postgres:
            prefix = artifact_key(path).rstrip("/") + "/"
            with self.write_transaction() as conn:
                conn.execute("DELETE FROM artifact_files WHERE substr(path,1,?)=?",(len(prefix),prefix))
