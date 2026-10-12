"""
Tests for the knowledge_base.py chunking + TF-IDF retrieval logic - pure functions, no
server or Groq dependency needed, so these are fully testable in isolation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.knowledge_base import chunk_text, retrieve_relevant_chunks, format_context_block


class TestChunking:
    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text("This is a short FAQ answer.")
        assert len(chunks) == 1
        assert "short FAQ answer" in chunks[0]

    def test_paragraphs_are_kept_together_when_they_fit(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) == 1
        assert "Para one" in chunks[0] and "Para three" in chunks[0]

    def test_long_text_splits_into_multiple_chunks(self):
        text = "\n\n".join([f"Paragraph number {i} with some real content in it." for i in range(30)])
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        # nothing should have been silently dropped - every paragraph's distinctive number
        # should appear somewhere across the chunks
        joined = " ".join(chunks)
        for i in range(30):
            assert f"number {i} " in joined

    def test_single_overlong_paragraph_is_hard_wrapped(self):
        text = "word " * 500  # one giant "paragraph", no blank lines
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(c) <= 250 for c in chunks)  # allows a little slack for the overlap prefix

    def test_adjacent_chunks_overlap(self):
        text = "\n\n".join([f"Sentence {i}." * 10 for i in range(10)])
        chunks = chunk_text(text, chunk_size=150, overlap=30)
        if len(chunks) > 1:
            # the tail of chunk N should reappear at the start of chunk N+1
            tail = chunks[0][-30:]
            assert tail in chunks[1]


class TestRetrieval:
    def _chunks(self):
        return [
            {"id": 1, "filename": "faq.txt", "content": "Our return policy allows returns within 30 days of purchase with a receipt."},
            {"id": 2, "filename": "faq.txt", "content": "We ship worldwide via DHL and FedEx, typically arriving in 5-7 business days."},
            {"id": 3, "filename": "faq.txt", "content": "The store is open Monday through Saturday, 9am to 6pm, closed on Sundays."},
        ]

    def test_relevant_chunk_is_retrieved_for_matching_query(self):
        results = retrieve_relevant_chunks("What is your return policy?", self._chunks())
        assert len(results) >= 1
        assert results[0]["id"] == 1

    def test_different_query_retrieves_different_chunk(self):
        results = retrieve_relevant_chunks("How long does it take to ship my order?", self._chunks())
        assert len(results) >= 1
        assert results[0]["id"] == 2

    def test_unrelated_query_returns_nothing_above_threshold(self):
        """A query about something the knowledge base genuinely doesn't cover shouldn't force
        an irrelevant chunk into the context just because it scored 'highest of three bad
        options' - this is what keeps the anti-hallucination deflection behavior intact for
        anything not actually covered."""
        results = retrieve_relevant_chunks("What is the airspeed velocity of an unladen swallow?", self._chunks())
        assert results == []

    def test_empty_query_returns_nothing(self):
        assert retrieve_relevant_chunks("", self._chunks()) == []
        assert retrieve_relevant_chunks("   ", self._chunks()) == []

    def test_empty_chunk_list_returns_nothing(self):
        assert retrieve_relevant_chunks("return policy", []) == []

    def test_top_k_is_respected(self):
        many_chunks = [{"id": i, "filename": "f.txt", "content": f"Return policy detail number {i} about returns and refunds."}
                       for i in range(10)]
        results = retrieve_relevant_chunks("return policy", many_chunks, top_k=2)
        assert len(results) <= 2

    def test_whitespace_only_chunks_dont_crash_retrieval(self):
        chunks = [{"id": 1, "filename": "f.txt", "content": "   "}, {"id": 2, "filename": "f.txt", "content": ""}]
        results = retrieve_relevant_chunks("anything", chunks)
        assert results == []


class TestContextFormatting:
    def test_empty_chunks_produce_empty_block(self):
        assert format_context_block([]) == ""

    def test_formatted_block_includes_source_and_content(self):
        block = format_context_block([{"filename": "policy.txt", "content": "30 day returns."}])
        assert "policy.txt" in block
        assert "30 day returns." in block
        assert "KNOWLEDGE BASE CONTEXT" in block

    def test_formatted_block_instructs_fallback_for_uncovered_questions(self):
        """The whole point of pairing this with the existing anti-hallucination design -
        confirm the instruction to NOT guess on uncovered questions is actually present."""
        block = format_context_block([{"filename": "f.txt", "content": "some fact"}])
        assert "not covered" in block.lower() or "do not guess" in block.lower()
