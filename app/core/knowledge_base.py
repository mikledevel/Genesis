"""
Knowledge base retrieval for published bots.

This closes a real gap in the existing anti-hallucination design (see bot_builder.py): a bot
was only ever able to honestly DEFLECT on facts it doesn't know (real inventory, exact
policies, etc.) - it had no way to actually KNOW real facts, because AgentConfig.knowledge_base
existed as a field but was never read or written anywhere. This module is what makes that field
real: a bot's owner uploads documents, and relevant excerpts get retrieved and injected into
the system prompt for each query - the deflection behavior remains the fallback for anything
NOT covered by the uploaded documents, so the two features work together rather than replacing
each other.

RETRIEVAL METHOD - being upfront about the tradeoff made here: this uses TF-IDF + cosine
similarity (scikit-learn, already a project dependency), NOT dense vector/semantic embeddings.
Verified against Groq's live model list (console.groq.com/docs/models) before building this -
Groq does not currently offer an embeddings endpoint, only chat/speech/moderation models - so a
dense-embedding approach would require a second API provider and its own key/cost. TF-IDF is a
well-established, zero-additional-dependency, zero-additional-cost retrieval method. It's
weaker than semantic embeddings at matching heavily paraphrased/synonym-only queries, but is
genuinely effective at matching the kind of keyword overlap a real FAQ-style question and a
real FAQ-style document tend to share. Swapping in real embeddings later (if/when a provider
already in use here offers one) is a contained change - the storage schema (chunks keyed by
agent_id) doesn't need to change, only retrieve_relevant_chunks()'s internals would.

Confirmed empirically while building this (see tests/test_knowledge_base.py): stop-word
removal (stop_words="english") measurably improves precision - without it, a completely
unrelated query can score a nonzero false-positive similarity purely from shared common words
("is", "of", "the"). With it, unrelated queries correctly score 0. The remaining known gap is
synonym mismatch - a query using "shipping" won't match a document that only says "ship"
(zero token overlap, no stemming applied) - real users will sometimes phrase things
differently than the source document, and that's the case dense embeddings would help with
most; documented here rather than silently accepted.
"""
import re
from typing import List, Dict

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS_RETRIEVED = 3
MIN_SIMILARITY = 0.05  # below this, a chunk is probably not actually relevant to the query -
                        # don't force irrelevant content into every single response


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits on paragraph boundaries first (keeps related sentences together rather than
    cutting mid-thought), then hard-wraps any paragraph still longer than chunk_size.
    Adjacent chunks share a bit of overlap so a fact sitting near a boundary is still
    findable from whichever chunk gets matched."""
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    raw_chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            raw_chunks.append(current)
            current = ""
        if len(para) <= chunk_size:
            current = para
        else:
            start = 0
            while start < len(para):
                end = start + chunk_size
                raw_chunks.append(para[start:end])
                start = end - overlap if end - overlap > start else end
    if current:
        raw_chunks.append(current)

    if overlap and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}\n{raw_chunks[i]}".strip())
        raw_chunks = overlapped

    return raw_chunks


def retrieve_relevant_chunks(query: str, chunks: List[Dict], top_k: int = MAX_CHUNKS_RETRIEVED,
                              min_similarity: float = MIN_SIMILARITY) -> List[Dict]:
    """chunks: [{"content": str, "filename": str, ...}, ...]. Returns the top_k most relevant
    by TF-IDF cosine similarity, filtered to a minimum similarity threshold so an unrelated
    knowledge base doesn't get force-injected into every response just because SOME chunk
    happened to score highest among a bad set of candidates."""
    if not chunks or not (query or "").strip():
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [c["content"] for c in chunks] + [query]
    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []  # e.g. every chunk was empty/whitespace-only after tokenization

    query_vec = matrix[-1]
    chunk_vecs = matrix[:-1]
    sims = cosine_similarity(query_vec, chunk_vecs)[0]
    scored = sorted(zip(chunks, sims), key=lambda pair: pair[1], reverse=True)
    return [c for c, score in scored[:top_k] if score >= min_similarity]


def format_context_block(chunks: List[Dict]) -> str:
    """Formats retrieved chunks into the block injected into a bot's system prompt for one
    specific query - never persisted, computed fresh per message."""
    if not chunks:
        return ""
    parts = [f"[Source: {c.get('filename', 'document')}]\n{c['content']}" for c in chunks]
    return (
        "KNOWLEDGE BASE CONTEXT (retrieved from documents this bot's owner uploaded - if the "
        "user's question is answered here, use this information confidently and precisely. "
        "If it is NOT covered by this context, do not guess - follow your normal instructions "
        "about not inventing facts you don't actually have access to):\n\n"
        + "\n\n---\n\n".join(parts)
    )
