"""
Tests for the GenesisDB knowledge base methods (add/list/delete documents, get chunks for
a bot). Uses a throwaway SQLite file per test, never touches the real genesis.db.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.db.database import GenesisDB
from app.core.knowledge_base import chunk_text


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    d = GenesisDB(db_path=path)
    yield d
    os.remove(path)
    for ext in ("-wal", "-shm"):
        try:
            os.remove(path + ext)
        except FileNotFoundError:
            pass


class TestKnowledgeBaseStorage:
    def test_add_and_list_document(self, db):
        chunks = chunk_text("Our return policy allows returns within 30 days.")
        doc_id = db.add_kb_document("agent_1", "user_1", "policy.txt", "full content here", chunks)
        docs = db.list_kb_documents("agent_1", "user_1")
        assert len(docs) == 1
        assert docs[0]["id"] == doc_id
        assert docs[0]["filename"] == "policy.txt"

    def test_chunks_are_retrievable_for_chat_time_use(self, db):
        chunks = chunk_text("Chunk one content.\n\nChunk two content.")
        db.add_kb_document("agent_1", "user_1", "doc.txt", "original", chunks)
        retrieved = db.get_kb_chunks_for_agent("agent_1")
        assert len(retrieved) == len(chunks)
        assert all("filename" in c and c["filename"] == "doc.txt" for c in retrieved)

    def test_get_kb_chunks_not_ownership_checked_by_design(self, db):
        """Deliberately different from list_kb_documents - chat-time retrieval must work
        for whoever is talking to a published bot, not just its owner."""
        db.add_kb_document("agent_1", "owner_user", "doc.txt", "content", ["a chunk"])
        chunks = db.get_kb_chunks_for_agent("agent_1")  # no user_id param at all
        assert len(chunks) == 1

    def test_list_is_ownership_checked(self, db):
        db.add_kb_document("agent_1", "owner_user", "doc.txt", "content", ["a chunk"])
        assert db.list_kb_documents("agent_1", "someone_else") == []
        assert len(db.list_kb_documents("agent_1", "owner_user")) == 1

    def test_delete_ownership_checked(self, db):
        doc_id = db.add_kb_document("agent_1", "owner_user", "doc.txt", "content", ["a chunk"])
        assert db.delete_kb_document(doc_id, "someone_else") is False
        assert db.delete_kb_document(doc_id, "owner_user") is True
        assert db.list_kb_documents("agent_1", "owner_user") == []

    def test_delete_removes_chunks_too(self, db):
        doc_id = db.add_kb_document("agent_1", "owner_user", "doc.txt", "content", ["chunk a", "chunk b"])
        assert len(db.get_kb_chunks_for_agent("agent_1")) == 2
        db.delete_kb_document(doc_id, "owner_user")
        assert db.get_kb_chunks_for_agent("agent_1") == []

    def test_multiple_documents_for_same_bot_all_contribute_chunks(self, db):
        db.add_kb_document("agent_1", "owner_user", "doc1.txt", "content1", ["chunk from doc1"])
        db.add_kb_document("agent_1", "owner_user", "doc2.txt", "content2", ["chunk from doc2"])
        chunks = db.get_kb_chunks_for_agent("agent_1")
        assert len(chunks) == 2
        filenames = {c["filename"] for c in chunks}
        assert filenames == {"doc1.txt", "doc2.txt"}

    def test_different_bots_dont_share_knowledge(self, db):
        db.add_kb_document("agent_1", "owner_user", "doc.txt", "content", ["agent1's chunk"])
        db.add_kb_document("agent_2", "owner_user", "doc.txt", "content", ["agent2's chunk"])
        chunks1 = db.get_kb_chunks_for_agent("agent_1")
        chunks2 = db.get_kb_chunks_for_agent("agent_2")
        assert len(chunks1) == 1 and chunks1[0]["content"] == "agent1's chunk"
        assert len(chunks2) == 1 and chunks2[0]["content"] == "agent2's chunk"
