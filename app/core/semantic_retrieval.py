"""Опциональный локальный dense-поиск: документы не отправляются провайдеру."""
from collections import OrderedDict
import hashlib
import threading
import numpy as np

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_lock = threading.RLock()
_model = None
_indexes = OrderedDict()


def embedding_model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=MODEL, threads=1)
        return _model


def _normalize(matrix):
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Invalid embedding matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Empty embedding")
    return array / norms


def dense_scores(query, chunks):
    if len(chunks) > 5000:
        raise ValueError("Local semantic index is limited to 5000 passages")
    texts = [str(c["content"]) for c in chunks]
    key = hashlib.sha256((MODEL + "\0" + "\0".join(texts)).encode()).hexdigest()
    with _lock:
        matrix = _indexes.pop(key, None)
        model = embedding_model()
        if matrix is None:
            matrix = _normalize(list(model.embed(texts, batch_size=32)))
        _indexes[key] = matrix
        while len(_indexes) > 4:
            _indexes.popitem(last=False)
        vector = _normalize(list(model.query_embed(query)))[0]
    if matrix.shape[1] != len(vector):
        raise ValueError("Embedding dimensions changed")
    return matrix @ vector


def hybrid_retrieve(query, chunks, lexical, top_k=3, threshold=0.6, margin=0.05, scorer=None):
    if not chunks or not query.strip() or top_k <= 0:
        return []
    scores = np.asarray((scorer or dense_scores)(query, chunks))
    if scores.shape != (len(chunks),) or not np.isfinite(scores).all():
        raise ValueError("Invalid semantic retrieval scores")
    dense = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    best = float(scores[dense[0]])
    gap = best - float(scores[dense[1]]) if len(dense) > 1 else best
    # При отсутствии лексических совпадений требуются и абсолютный порог, и отрыв.
    if not lexical and (best < threshold or gap < margin):
        return []
    candidates = {i for i in dense[:top_k] if scores[i] >= threshold and scores[i] >= best - 0.12}
    from app.core.knowledge_base import source_id
    lex_ranks = {source_id(c): i for i, c in enumerate(lexical)}
    candidates.update(i for i, c in enumerate(chunks) if source_id(c) in lex_ranks)
    dense_ranks = {i: rank for rank, i in enumerate(dense)}
    def rank(i):
        lexical_rank = lex_ranks.get(source_id(chunks[i]))
        return (0.6 / (10 + lexical_rank) if lexical_rank is not None else 0) + 0.4 / (10 + dense_ranks[i])
    return [chunks[i] for i in sorted(candidates, key=rank, reverse=True)[:top_k]]
