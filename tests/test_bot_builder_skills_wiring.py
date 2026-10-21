"""
Tests for BotBuilder's skills_needed wiring - the connection between "user describes a bot
in chat" and "the bot actually gets real, sandbox-tested capabilities", not just a system
prompt. Mocks Groq calls (no live GROQ_API_KEY in this environment) and SkillGenerator.build
(covered by its own tests in test_skill_generator.py) to isolate and verify the WIRING logic:
does build() call the skill generator for each requested skill, does the write-method
defense-in-depth override apply here too, does publish() persist only tested skills and tie
them to the real bot id, does an untested/failed skill get left out.
"""
import os, sys, tempfile, shutil, json as json_module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from unittest.mock import patch
from app.core.bot_builder import BotBuilder
from app.db.database import GenesisDB

BASE_SPEC = {
    "name": "GitHubHelper", "description": "Helps with GitHub",
    "system_prompt": "You help with GitHub questions.",
    "greeting": "Hi! How can I help with GitHub?",
    "test_questions": ["What is Git?", "Does the repo octocat/Hello-World exist?", "How do I clone a repo?"],
    "trap_question_index": 1,
}


def _spec_with_skills(skills_needed):
    return {**BASE_SPEC, "skills_needed": skills_needed}


def _passing_test_results():
    return [{"question": "(greeting)", "response": "hi", "ok": True, "error": None},
            {"question": "q1", "response": "a1", "ok": True, "error": None},
            {"question": "q2", "response": "a2", "ok": True, "error": None},
            {"question": "q3", "response": "a3", "ok": True, "error": None}]


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp()
    prev = os.getcwd()
    os.chdir(d)
    yield d
    os.chdir(prev)
    shutil.rmtree(d, ignore_errors=True)


class TestBuildRequestsSkillGeneration:
    def test_no_skills_needed_means_no_skill_generation_calls(self, workdir):
        bb = BotBuilder(user_id="user_A")
        bb.groq_key = "fake-key"
        with patch.object(bb, "generate_spec", return_value=_spec_with_skills([])), \
             patch.object(bb, "test_draft", return_value=_passing_test_results()), \
             patch("app.core.skills.skill_generator.SkillGenerator.build") as mock_skill_build:
            result = bb.build("a simple FAQ bot")
        mock_skill_build.assert_not_called()
        assert result["skills"] == []

    def test_requested_skill_is_generated_and_attached(self, workdir):
        bb = BotBuilder(user_id="user_A")
        bb.groq_key = "fake-key"
        requested = [{"name": "check_repo", "description": "Check if a GitHub repo exists",
                      "allowed_domains": ["github.com"], "action_type": "read_only"}]
        fake_build_result = {"code": "def run(params):\n    return fetch(params['url'])\nTEST_PARAMS={}",
                              "success": True, "log": [{"iteration": 0, "success": True}]}
        with patch.object(bb, "generate_spec", return_value=_spec_with_skills(requested)), \
             patch.object(bb, "test_draft", return_value=_passing_test_results()), \
             patch("app.core.skills.skill_generator.SkillGenerator.build", return_value=fake_build_result) as mock_build:
            result = bb.build("a bot that checks GitHub repos")
        mock_build.assert_called_once()
        assert len(result["skills"]) == 1
        assert result["skills"][0]["test_passed"] is True
        assert result["skills"][0]["name"] == "check_repo"
        # the final spec's skills_needed is replaced with outcomes, not left as the request
        assert result["spec"]["skills_needed"][0]["code"] == fake_build_result["code"]

    def test_failed_skill_generation_is_reported_not_hidden(self, workdir):
        bb = BotBuilder(user_id="user_A")
        bb.groq_key = "fake-key"
        requested = [{"name": "flaky_skill", "description": "x", "allowed_domains": ["x.com"], "action_type": "read_only"}]
        fake_build_result = {"code": "def run(params):\n    return 1/0\nTEST_PARAMS={}",
                              "success": False, "log": [{"iteration": 0, "success": False}]}
        with patch.object(bb, "generate_spec", return_value=_spec_with_skills(requested)), \
             patch.object(bb, "test_draft", return_value=_passing_test_results()), \
             patch("app.core.skills.skill_generator.SkillGenerator.build", return_value=fake_build_result):
            result = bb.build("a bot with a broken skill")
        assert result["skills"][0]["test_passed"] is False

    def test_skill_with_no_allowed_domains_is_skipped_without_calling_groq(self, workdir):
        bb = BotBuilder(user_id="user_A")
        bb.groq_key = "fake-key"
        requested = [{"name": "no_domains", "description": "x", "allowed_domains": [], "action_type": "read_only"}]
        with patch.object(bb, "generate_spec", return_value=_spec_with_skills(requested)), \
             patch.object(bb, "test_draft", return_value=_passing_test_results()), \
             patch("app.core.skills.skill_generator.SkillGenerator.build") as mock_build:
            result = bb.build("bad request")
        mock_build.assert_not_called()
        assert result["skills"][0]["test_passed"] is False

    def test_at_most_five_skills_generated_per_bot(self, workdir):
        bb = BotBuilder(user_id="user_A")
        bb.groq_key = "fake-key"
        requested = [{"name": f"skill_{i}", "description": "x", "allowed_domains": ["x.com"], "action_type": "read_only"}
                     for i in range(8)]
        fake_build_result = {"code": "def run(params):\n    return {}\nTEST_PARAMS={}", "success": True, "log": []}
        with patch.object(bb, "generate_spec", return_value=_spec_with_skills(requested)), \
             patch.object(bb, "test_draft", return_value=_passing_test_results()), \
             patch("app.core.skills.skill_generator.SkillGenerator.build", return_value=fake_build_result) as mock_build:
            result = bb.build("a bot that wants way too many skills")
        assert mock_build.call_count == 5
        assert len(result["skills"]) == 5

    def test_declared_read_only_but_code_writes_is_overridden_to_outbound_action(self, workdir):
        """Same defense-in-depth as /api/skills/create - even a skill the architect labeled
        read_only gets forced to outbound_action if its generated code actually does a
        write HTTP call."""
        bb = BotBuilder(user_id="user_A")
        bb.groq_key = "fake-key"
        requested = [{"name": "sneaky", "description": "x", "allowed_domains": ["x.com"], "action_type": "read_only"}]
        fake_build_result = {"code": 'def run(params):\n    return fetch("https://x.com", method="POST")\nTEST_PARAMS={}',
                              "success": True, "log": []}
        with patch.object(bb, "generate_spec", return_value=_spec_with_skills(requested)), \
             patch.object(bb, "test_draft", return_value=_passing_test_results()), \
             patch("app.core.skills.skill_generator.SkillGenerator.build", return_value=fake_build_result):
            result = bb.build("a sneaky bot")
        assert result["skills"][0]["action_type"] == "outbound_action"


class TestPublishPersistsOnlyPassedSkills:
    @pytest.fixture
    def db_path(self, workdir):
        return os.path.join(workdir, "test.db")

    def test_passed_skill_is_saved_and_linked_to_the_new_bot(self, workdir, db_path):
        db = GenesisDB(db_path=db_path)
        bb = BotBuilder(user_id="user_A")
        bb._db = db
        spec = {**BASE_SPEC, "skills_needed": [
            {"name": "check_repo", "description": "checks a repo", "allowed_domains": ["github.com"],
             "action_type": "read_only", "code": "def run(params):\n    return {}\nTEST_PARAMS={}",
             "test_passed": True, "log": []},
        ]}
        result = bb.publish(spec, author="user_A")
        assert len(result["saved_skill_ids"]) == 1
        saved = db.get_skill(result["saved_skill_ids"][0])
        assert saved["agent_id"] == result["id"]  # linked to the REAL, newly-created bot
        assert saved["owner_user_id"] == "user_A"
        assert saved["test_passed"] is True

    def test_failed_skill_is_not_saved(self, workdir, db_path):
        db = GenesisDB(db_path=db_path)
        bb = BotBuilder(user_id="user_A")
        bb._db = db
        spec = {**BASE_SPEC, "skills_needed": [
            {"name": "broken", "description": "x", "allowed_domains": ["x.com"], "action_type": "read_only",
             "code": "def run(params):\n    return 1/0\nTEST_PARAMS={}", "test_passed": False, "log": []},
        ]}
        result = bb.publish(spec, author="user_A")
        assert result["saved_skill_ids"] == []
        assert db.list_skills_for_owner("user_A") == []

    def test_bot_with_no_skills_needed_publishes_normally(self, workdir, db_path):
        db = GenesisDB(db_path=db_path)
        bb = BotBuilder(user_id="user_A")
        bb._db = db
        result = bb.publish({**BASE_SPEC, "skills_needed": []}, author="user_A")
        assert result["saved_skill_ids"] == []
        assert result["name"] == "GitHubHelper"

    def test_publish_does_not_call_groq_again_for_already_tested_skill_code(self, workdir, db_path):
        """The whole point of generating+testing during build() is that publish() is free -
        it must not re-invoke SkillGenerator (which would mean another real Groq call)."""
        db = GenesisDB(db_path=db_path)
        bb = BotBuilder(user_id="user_A")
        bb._db = db
        spec = {**BASE_SPEC, "skills_needed": [
            {"name": "check_repo", "description": "x", "allowed_domains": ["github.com"], "action_type": "read_only",
             "code": "def run(params):\n    return {}\nTEST_PARAMS={}", "test_passed": True, "log": []},
        ]}
        with patch("app.core.skills.skill_generator.SkillGenerator.build") as mock_build:
            bb.publish(spec, author="user_A")
        mock_build.assert_not_called()
