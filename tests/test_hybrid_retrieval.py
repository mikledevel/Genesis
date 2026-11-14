import pytest
from app.core.semantic_retrieval import hybrid_retrieve
from app.core.knowledge_base import retrieve_relevant_chunks

CHUNKS = [{"filename": "a", "content": "A policy"}, {"filename": "b", "content": "B policy"}]


def test_semantic_paraphrase_and_abstention():
    assert hybrid_retrieve("synonym", CHUNKS, [], scorer=lambda *_: [0.8, 0.2]) == [CHUNKS[0]]
    assert hybrid_retrieve("unknown", CHUNKS, [], scorer=lambda *_: [0.4, 0.3]) == []
    assert hybrid_retrieve("ambiguous", CHUNKS, [], scorer=lambda *_: [0.8, 0.79]) == []


def test_lexical_result_is_preserved_and_results_are_bounded():
    result = hybrid_retrieve("A", CHUNKS, [CHUNKS[0]], top_k=1, scorer=lambda *_: [0.1, 0.9])
    assert result == [CHUNKS[0]]


def test_failed_embeddings_fall_back_without_mutating_chunks(monkeypatch):
    chunks = [{"filename": "returns", "content": "Returns accepted within fourteen days."}]
    baseline = retrieve_relevant_chunks("Returns", chunks, mode="lexical")
    def fail(*_):
        raise RuntimeError("offline")
    monkeypatch.setattr("app.core.semantic_retrieval.dense_scores", fail)
    assert retrieve_relevant_chunks("Returns", chunks, mode="hybrid") == baseline
    assert list(chunks[0]) == ["filename", "content"]


def test_malformed_embeddings_are_rejected():
    with pytest.raises(ValueError):
        hybrid_retrieve("x", CHUNKS, [], scorer=lambda *_: [float("nan"), 0])
