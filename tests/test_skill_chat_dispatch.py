"""
Tests for GenesisAgent._do_run_skill and the skills block in _get_effective_system_prompt -
the wiring that lets a published bot actually USE a sandbox-tested skill during a real
conversation, not just have one sitting unused in the database.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.core.agent import GenesisAgent
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry
from app.db.database import GenesisDB
from app.core.agent_store import AgentStore, AgentConfig

WORKING_SKILL_CODE = '''
def run(params):
    r = fetch("https://github.com/" + params["repo"])
    return {"status": r["status"], "exists": r["status"] == 200}
TEST_PARAMS = {"repo": "anthropics"}
'''


@pytest.fixture
def env(monkeypatch):
    d = tempfile.mkdtemp()
    prev_cwd = os.getcwd()
    os.chdir(d)
    os.makedirs("datasets", exist_ok=True)
    monkeypatch.setenv("DEBUG", "true")  # allow the subprocess sandbox fallback for these tests
    db = GenesisDB(db_path=os.path.join(d, "test.db"))
    agent = GenesisAgent(profiler=DatasetProfiler(), trainer=DatasetTrainer(),
                          registry=ModelRegistry(storage_dir=os.path.join(d, "registry")), db=db)
    agent.current_user_id = "owner_user"

    cfg = AgentConfig(name="Test Bot", system_prompt="You help with things.")
    cfg.author = "owner_user"
    AgentStore().create_agent(cfg)
    agent.agent_override = {"agent_id": cfg.id, "system_prompt": cfg.system_prompt}

    yield {"agent": agent, "db": db, "bot_id": cfg.id}
    os.chdir(prev_cwd)
    shutil.rmtree(d, ignore_errors=True)


class TestSkillVisibilityInSystemPrompt:
    def test_no_skills_no_skills_block(self, env):
        prompt = env["agent"]._get_effective_system_prompt("hello")
        assert "SKILLS:" not in prompt

    def test_untested_skill_not_listed(self, env):
        """A skill that hasn't passed its sandbox test must never be offered to the LLM -
        list_skills_for_agent filters on test_passed=1, this checks that filter is actually
        wired through to what the LLM sees."""
        env["db"].create_skill(agent_id=env["bot_id"], owner_user_id="owner_user",
                                name="broken_skill", description="x", code="def run(params): return {}",
                                allowed_domains=["example.com"], input_schema={})
        # deliberately never call update_skill_test_result - stays test_passed=False
        prompt = env["agent"]._get_effective_system_prompt("hello")
        assert "SKILLS:" not in prompt
        assert "broken_skill" not in prompt

    def test_tested_skill_is_listed_with_id_and_description(self, env):
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="check_github_repo",
            description="Checks if a GitHub repo exists", code=WORKING_SKILL_CODE,
            allowed_domains=["github.com"], input_schema={"repo": "string"})
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")

        prompt = env["agent"]._get_effective_system_prompt("check a repo")
        assert "SKILLS:" in prompt
        assert "check_github_repo" in prompt
        assert skill["id"] in prompt
        assert "Checks if a GitHub repo exists" in prompt

    def test_inactive_skill_not_listed(self, env):
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="deactivated",
            description="x", code=WORKING_SKILL_CODE, allowed_domains=["github.com"], input_schema={})
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")
        with env["db"]._connect() as conn:
            conn.execute("UPDATE bot_skills SET active = 0 WHERE id = ?", (skill["id"],))
            conn.commit()
        prompt = env["agent"]._get_effective_system_prompt("hello")
        assert "deactivated" not in prompt


class TestRunSkillDispatch:
    def test_no_agent_id_rejected(self, env):
        resp = env["agent"]._do_run_skill(None, {"skill_id": "anything", "params": {}})
        assert not resp.success
        assert "specific bot" in resp.message

    def test_missing_skill_id_rejected(self, env):
        resp = env["agent"]._do_run_skill(env["bot_id"], {"params": {}})
        assert not resp.success

    def test_nonexistent_skill_rejected(self, env):
        resp = env["agent"]._do_run_skill(env["bot_id"], {"skill_id": "skill_doesnotexist", "params": {}})
        assert not resp.success
        assert "isn't available" in resp.message

    def test_skill_belonging_to_a_different_bot_is_not_reachable(self, env):
        """A skill is scoped to the specific bot it was created for - even a skill the SAME
        owner created for a different bot must not be callable from this one."""
        other_cfg = AgentConfig(name="Other Bot", system_prompt="x")
        other_cfg.author = "owner_user"
        AgentStore().create_agent(other_cfg)
        skill = env["db"].create_skill(
            agent_id=other_cfg.id, owner_user_id="owner_user", name="other_bots_skill",
            description="x", code=WORKING_SKILL_CODE, allowed_domains=["github.com"], input_schema={})
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")

        resp = env["agent"]._do_run_skill(env["bot_id"], {"skill_id": skill["id"], "params": {}})
        assert not resp.success
        assert "isn't available" in resp.message

    def test_untested_skill_cannot_be_run_even_if_id_is_known(self, env):
        """Defense in depth: even if something (a bug, a stale ID cached client-side) tries
        to invoke a skill that never passed its test, the dispatcher itself must refuse -
        not just rely on it being absent from the system prompt."""
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="untested",
            description="x", code=WORKING_SKILL_CODE, allowed_domains=["github.com"], input_schema={})
        resp = env["agent"]._do_run_skill(env["bot_id"], {"skill_id": skill["id"], "params": {}})
        assert not resp.success

    def test_working_skill_executes_and_returns_real_result(self, env):
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="check_github_repo",
            description="x", code=WORKING_SKILL_CODE, allowed_domains=["github.com"], input_schema={})
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")

        resp = env["agent"]._do_run_skill(
            env["bot_id"], {"skill_id": skill["id"], "params": {"repo": "anthropics/claude-code"}})
        assert resp.success, resp.message
        assert resp.data["result"]["exists"] is True

    def test_skill_restricted_to_domain_cannot_reach_another(self, env):
        """End-to-end proof that the allowlist set at skill-creation time is what's actually
        enforced at call time, not something the LLM's chosen params can override."""
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="restricted",
            description="x", code=WORKING_SKILL_CODE, allowed_domains=["pypi.org"], input_schema={})
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")

        resp = env["agent"]._do_run_skill(
            env["bot_id"], {"skill_id": skill["id"], "params": {"repo": "anthropics/claude-code"}})
        assert not resp.success
        assert "only allowed to contact" in resp.data["error"]

    def test_run_skill_tool_name_survives_bot_scoping_filter(self, env):
        """BOT_SCOPED_ALLOWED_TOOLS is what stops a published bot from reaching
        platform-builder tools like train_model - confirm run_skill was actually added there
        (a regression here would silently make every skill unreachable in real chat, while
        direct _do_run_skill tests above would still pass)."""
        assert "run_skill" in GenesisAgent.BOT_SCOPED_ALLOWED_TOOLS
