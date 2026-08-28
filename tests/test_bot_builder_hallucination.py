"""
Tests for the BotBuilder anti-hallucination fix: test_draft() previously only checked
that the trap question's response was non-empty, which let confidently-invented answers
through undetected. These tests mock the Groq calls (no live GROQ_API_KEY in this
environment - see note in the final summary about what could/couldn't be tested live)
to verify the judging and wiring logic itself is correct.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unittest.mock import patch, Mock
from app.core.bot_builder import BotBuilder


SPEC = {
    "name": "ShoeStep Support",
    "description": "Shoe store support bot",
    "system_prompt": "You help customers with shoe questions. Never invent stock/order info.",
    "greeting": "Hi! How can I help?",
    "test_questions": [
        "What are your store hours?",
        "Is size 10 of the Air Runner in stock right now?",
        "What's your return policy?",
    ],
    "trap_question_index": 1,
}


class TestHallucinationDetection:
    def test_trap_question_with_invented_fact_marked_not_ok(self):
        bb = BotBuilder()
        bb.groq_key = "fake-key-for-test"  # bypass the "no key -> skip" early return
        bb._db = Mock()  # avoid touching the real genesis.db for usage logging in these tests

        def fake_chat(system, user, history=None, model=None, max_tokens=600, call_site=None):
            if "size 10" in user:
                return "Yes! We have 3 pairs of size 10 Air Runner in stock right now."
            return "Some normal response."

        def fake_json(system, user, model=None, max_tokens=1500, call_site=None):
            # only the hallucination judge calls _groq_json inside test_draft
            return {"hallucinated": True, "reason": "invented a specific stock count"}

        with patch.object(bb, "_groq_chat", side_effect=fake_chat), \
             patch.object(bb, "_groq_json", side_effect=fake_json):
            results = bb.test_draft(SPEC)

        trap_result = results[2]  # index 0 = greeting, 1 = hours, 2 = trap (size 10)
        assert trap_result["question"] == SPEC["test_questions"][1]
        assert trap_result["ok"] is False
        assert "hallucinated" in trap_result["error"]

    def test_trap_question_with_honest_deflection_marked_ok(self):
        bb = BotBuilder()
        bb.groq_key = "fake-key-for-test"
        bb._db = Mock()  # avoid touching the real genesis.db for usage logging in these tests

        def fake_chat(system, user, history=None, model=None, max_tokens=600, call_site=None):
            if "size 10" in user:
                return "I can't check live stock myself - please contact our support team or check the website."
            return "Some normal response."

        def fake_json(system, user, model=None, max_tokens=1500, call_site=None):
            return {"hallucinated": False, "reason": "correctly deflected to support"}

        with patch.object(bb, "_groq_chat", side_effect=fake_chat), \
             patch.object(bb, "_groq_json", side_effect=fake_json):
            results = bb.test_draft(SPEC)

        trap_result = results[2]
        assert trap_result["ok"] is True
        assert trap_result["error"] is None

    def test_non_trap_questions_unaffected_by_judge(self):
        bb = BotBuilder()
        bb.groq_key = "fake-key-for-test"
        bb._db = Mock()  # avoid touching the real genesis.db for usage logging in these tests

        def fake_chat(system, user, history=None, model=None, max_tokens=600, call_site=None):
            return "A perfectly normal non-empty response."

        judge_calls = []

        def fake_json(system, user, model=None, max_tokens=1500, call_site=None):
            judge_calls.append(user)
            return {"hallucinated": True, "reason": "should never be called for non-trap questions"}

        with patch.object(bb, "_groq_chat", side_effect=fake_chat), \
             patch.object(bb, "_groq_json", side_effect=fake_json):
            results = bb.test_draft(SPEC)

        # The judge should only ever be invoked once - for the trap question - not for
        # "store hours" or "return policy".
        assert len(judge_calls) == 1
        assert results[1]["ok"] is True  # store hours - non-trap, non-empty -> ok regardless of judge
        assert results[3]["ok"] is True  # return policy - non-trap, non-empty -> ok regardless of judge

    def test_missing_trap_index_skips_hallucination_check_gracefully(self):
        """Backward compatibility: an older/malformed spec without a valid trap_question_index
        must not crash - it just falls back to the old empty-only check for every question."""
        bb = BotBuilder()
        bb.groq_key = "fake-key-for-test"
        bb._db = Mock()  # avoid touching the real genesis.db for usage logging in these tests
        spec_no_trap = {**SPEC, "trap_question_index": None}

        def fake_chat(system, user, history=None, model=None, max_tokens=600, call_site=None):
            return "Yes, definitely in stock!"  # would be a hallucination if judged

        judge_calls = []

        def fake_json(system, user, model=None, max_tokens=1500, call_site=None):
            judge_calls.append(user)
            return {"hallucinated": True, "reason": "n/a"}

        with patch.object(bb, "_groq_chat", side_effect=fake_chat), \
             patch.object(bb, "_groq_json", side_effect=fake_json):
            results = bb.test_draft(spec_no_trap)

        assert len(judge_calls) == 0  # judge never invoked - no valid trap index
        assert all(r["ok"] for r in results)  # every response was non-empty -> all pass

    def test_judge_call_failure_fails_open(self):
        """If the judge call itself fails (e.g. transient API error), don't block a
        potentially-legitimate bot on an unverifiable check - but the failure is
        distinguishable via _check_hallucination's 'verified' flag for callers that want it."""
        bb = BotBuilder()
        bb.groq_key = "fake-key-for-test"
        bb._db = Mock()  # avoid touching the real genesis.db for usage logging in these tests

        def fake_chat(system, user, history=None, model=None, max_tokens=600, call_site=None):
            return "Some response to the trap question."

        with patch.object(bb, "_groq_chat", side_effect=fake_chat), \
             patch.object(bb, "_groq_json", return_value=None):  # simulates judge call failure
            results = bb.test_draft(SPEC)
            verdict = bb._check_hallucination("q", "r")

        assert results[2]["ok"] is True  # fails open - not marked as a failure
        assert verdict["verified"] is False

    def test_fix_loop_triggers_on_hallucination_failure(self):
        """End-to-end: build() should invoke fix_spec when the trap question hallucinates,
        exactly as it already does for hard/empty failures."""
        bb = BotBuilder()
        bb.groq_key = "fake-key-for-test"
        bb._db = Mock()  # avoid touching the real genesis.db for usage logging in these tests

        call_count = {"n": 0}

        def fake_generate_spec(description, model=None):
            return dict(SPEC)

        def fake_chat(system, user, history=None, model=None, max_tokens=600, call_site=None):
            if "size 10" in user:
                call_count["n"] += 1
                # First pass hallucinates, second (post-fix) pass deflects correctly
                if call_count["n"] == 1:
                    return "Yes, 3 pairs in stock!"
                return "I can't check live stock - please contact support."
            return "Fine."

        def fake_json(system, user, model=None, max_tokens=1500, call_site=None):
            payload = user
            if "the_question" in payload:  # hallucination judge
                return {"hallucinated": "3 pairs" in payload, "reason": "checked"}
            return dict(SPEC)  # fix_spec call - just return the same spec unchanged

        with patch.object(bb, "generate_spec", side_effect=fake_generate_spec), \
             patch.object(bb, "_groq_chat", side_effect=fake_chat), \
             patch.object(bb, "_groq_json", side_effect=fake_json):
            result = bb.build("a shoe store support bot")

        assert result["iterations"] >= 1  # the fix loop actually ran
        assert result["success"] is True  # and the retest passed after the fix
