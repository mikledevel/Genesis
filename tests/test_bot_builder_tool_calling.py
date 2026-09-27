"""
Tests for the real bug reported from production: the bot-builder's test loop and live
preview chat both talked to the LLM with no tool-calling access to any skill at all, so a
question the bot's own skill was built to answer either got a lucky guess (if the model's
training data happened to know, e.g. a famous GitHub repo) or an honest "I can't check that"
deflection (for anything the model didn't already know) - a real skill was never actually
called from either code path. These tests exercise the fixed _groq_chat's tool-calling
round-trip end to end, executing a REAL skill (real subprocess sandbox, real network call
through SafeFetcher) in response to a mocked Groq tool_calls response - not a mock of the
skill execution itself, only of the Groq completion API boundary.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from app.core.bot_builder import BotBuilder

REAL_SKILL_CODE = '''
def run(params):
    r = fetch("https://github.com/" + params["repo"])
    return {"status": r["status"], "exists": r["status"] == 200}
TEST_PARAMS = {"repo": "anthropics"}
'''

TESTED_SKILL = {
    "name": "check_github_repo", "description": "Checks if a GitHub repo exists",
    "code": REAL_SKILL_CODE, "allowed_domains": ["github.com"],
    "action_type": "read_only", "test_passed": True, "log": [],
}


def _tool_call_message(tool_name: str, arguments: dict):
    """Builds a fake Groq completion response shaped like a real tool-call response - only
    the attributes bot_builder.py's _groq_chat actually reads via duck typing."""
    tool_call = SimpleNamespace(
        id="call_abc123",
        function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def _plain_text_message(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class TestGroqChatToolCallingRoundTrip:
    def test_no_skills_sends_no_tools_param(self):
        """Regression guard: when there are no tested skills, the request must not include a
        tools param at all - this is also what keeps ordinary conversation (no skills bot)
        behaving exactly as before this change."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key"
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _plain_text_message("Hi there!")
            result = bb._groq_chat("system", "hello", skills=[])
        assert result == "Hi there!"
        call_kwargs = MockGroq.return_value.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_untested_skill_not_offered_as_a_tool(self):
        """A skill that failed its own sandbox test must never be offered to the model here
        either - same rule as the live published-bot path (GenesisDB.list_skills_for_agent)."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key"
        untested = {**TESTED_SKILL, "test_passed": False}
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = _plain_text_message("I can't check that.")
            bb._groq_chat("system", "does anthropics/claude-code exist?", skills=[untested])
        call_kwargs = MockGroq.return_value.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_tool_call_actually_executes_the_real_skill_with_real_network_call(self):
        """The core fix, verified end to end: when the model's FIRST response is a tool call,
        the actual skill code runs for real (real sandbox, real SafeFetcher network call to
        github.com - not mocked), and the SECOND completion call receives the genuine result
        as a tool message, not a guess."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key"

        first_response = _tool_call_message("check_github_repo", {"repo": "anthropics/claude-code"})
        second_response = _plain_text_message("**Result:** The repository anthropics/claude-code exists on GitHub.")

        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = [first_response, second_response]
            result = bb._groq_chat("You check GitHub repos.", "Does anthropics/claude-code exist?",
                                    skills=[TESTED_SKILL])

        assert result == "**Result:** The repository anthropics/claude-code exists on GitHub."
        # confirm the SECOND call actually received the real tool result, not a placeholder
        second_call_messages = MockGroq.return_value.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        tool_result = json.loads(tool_messages[0]["content"])
        assert tool_result == {"status": 200, "exists": True}  # the REAL result from github.com

    def test_tool_call_for_a_nonexistent_repo_reports_that_too(self):
        """Same round-trip, but the skill's own real result is 'doesn't exist' - proving this
        isn't just echoing a hardcoded success."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key"
        first_response = _tool_call_message("check_github_repo", {"repo": "this-repo-does-not-exist-xyz-123"})
        second_response = _plain_text_message("That repository doesn't exist.")

        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = [first_response, second_response]
            bb._groq_chat("You check GitHub repos.", "Does this-repo-does-not-exist-xyz-123 exist?",
                          skills=[TESTED_SKILL])

        second_call_messages = MockGroq.return_value.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_result = json.loads([m for m in second_call_messages if m.get("role") == "tool"][0]["content"])
        assert tool_result["exists"] is False

    def test_followup_call_never_includes_tools(self):
        """The follow-up call must not re-offer tools - this is what prevents the model from
        chaining further tool calls indefinitely and closes off another tool_use_failed loop
        on the synthesis step itself."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key"
        first_response = _tool_call_message("check_github_repo", {"repo": "anthropics"})
        second_response = _plain_text_message("It exists.")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = [first_response, second_response]
            bb._groq_chat("system", "does it exist?", skills=[TESTED_SKILL])
        second_call_kwargs = MockGroq.return_value.chat.completions.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs

    def test_unknown_tool_name_handled_gracefully(self):
        """If the model somehow calls a tool name that doesn't match any tested skill
        (shouldn't happen given the tools list we send, but defensive), the round-trip must
        not crash - it reports the mismatch as a tool result and still gets a final answer."""
        bb = BotBuilder(user_id="test_user")
        bb.groq_key = "fake-key"
        first_response = _tool_call_message("some_other_tool", {})
        second_response = _plain_text_message("I couldn't check that.")
        with patch("groq.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.side_effect = [first_response, second_response]
            result = bb._groq_chat("system", "question", skills=[TESTED_SKILL])
        assert result == "I couldn't check that."


class TestChatWithDraftUsesSpecSkills:
    def test_chat_with_draft_pulls_skills_from_spec_automatically(self):
        """This is the exact fix for the reported bug: chat_with_draft (the live preview) now
        reads skills_needed off the spec itself and passes them through - the caller (the
        /api/bot-builder/chat route) doesn't need to do anything extra."""
        bb = BotBuilder(user_id="test_user")
        spec = {"system_prompt": "You check GitHub repos.", "skills_needed": [TESTED_SKILL]}
        with patch.object(bb, "_groq_chat", return_value="checked it") as mock_chat:
            bb.chat_with_draft(spec, "does anthropics/claude-code exist?")
        mock_chat.assert_called_once()
        assert mock_chat.call_args.kwargs["skills"] == [TESTED_SKILL]

    def test_chat_with_draft_with_no_skills_passes_empty_list(self):
        bb = BotBuilder(user_id="test_user")
        spec = {"system_prompt": "You help with FAQs.", "skills_needed": []}
        with patch.object(bb, "_groq_chat", return_value="answer") as mock_chat:
            bb.chat_with_draft(spec, "hello")
        assert mock_chat.call_args.kwargs["skills"] == []


class TestBuildOrdersSkillsBeforeTesting:
    def test_skills_are_generated_before_test_draft_is_called(self):
        """The actual ordering fix: build() must generate+test skills BEFORE calling
        test_draft, so test_draft can receive real, already-tested skill code - verified by
        checking that test_draft is called with a non-empty skills list whenever
        skills_needed was requested."""
        import tempfile, shutil
        d = tempfile.mkdtemp()
        prev = os.getcwd()
        os.chdir(d)
        try:
            bb = BotBuilder(user_id="test_user")
            bb.groq_key = "fake-key"
            spec = {
                "name": "Bot", "description": "x", "system_prompt": "x", "greeting": "hi",
                "test_questions": ["q1", "q2", "q3"], "trap_question_index": 1,
                "skills_needed": [{"name": "check_repo", "description": "x",
                                   "allowed_domains": ["github.com"], "action_type": "read_only"}],
            }
            fake_skill_result = {"code": REAL_SKILL_CODE, "success": True, "log": []}
            passing_results = [{"question": "q", "response": "a", "ok": True, "error": None}] * 4

            with patch.object(bb, "generate_spec", return_value=spec), \
                 patch("app.core.skills.skill_generator.SkillGenerator.build", return_value=fake_skill_result), \
                 patch.object(bb, "test_draft", return_value=passing_results) as mock_test_draft:
                bb.build("a bot that checks github")

            # test_draft must have received the ALREADY-GENERATED skill (test_passed=True,
            # real code present) - not an empty list, and not the raw un-generated request.
            call_kwargs = mock_test_draft.call_args
            passed_skills = call_kwargs.kwargs.get("skills") or call_kwargs.args[2] if len(call_kwargs.args) > 2 else call_kwargs.kwargs.get("skills")
            assert passed_skills[0]["test_passed"] is True
            assert passed_skills[0]["code"] == REAL_SKILL_CODE
        finally:
            os.chdir(prev)
            shutil.rmtree(d, ignore_errors=True)
