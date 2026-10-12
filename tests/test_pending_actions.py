"""
Tests for "Level 4": skills that take an action outside the platform (send a message, post
something) never execute directly - they always create a pending_action first, and only run
once a human explicitly approves it. Covers three layers:
  - GenesisDB's pending_actions methods directly
  - GenesisAgent._do_run_skill routing an outbound_action skill to a pending action instead
    of running it (chat trigger)
  - the scheduler's _run_one_job doing the same for a scheduled trigger
  - the defense-in-depth write-method detector that forces action_type regardless of what
    was declared
"""
import sys, os, tempfile, shutil, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.db.database import GenesisDB
from app.core.skills.skill_generator import _detect_write_http_method

READ_ONLY_CODE = '''
def run(params):
    r = fetch("https://github.com/" + params["repo"])
    return {"status": r["status"]}
TEST_PARAMS = {"repo": "anthropics"}
'''

OUTBOUND_CODE = '''
def run(params):
    r = fetch("https://example.com/contact", method="POST", body=params["message"])
    return {"status": r["status"]}
TEST_PARAMS = {"message": "hi"}
'''


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


class TestWriteMethodDetection:
    def test_get_only_code_not_flagged(self):
        assert _detect_write_http_method(READ_ONLY_CODE) is False

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "post", "Put"])
    def test_write_methods_flagged_case_insensitively(self, method):
        code = f'def run(params):\n    return fetch("https://x.com", method="{method}")\nTEST_PARAMS = {{}}'
        assert _detect_write_http_method(code) is True

    def test_syntax_error_defaults_to_flagged_conservatively(self):
        assert _detect_write_http_method("def run(params:\n    return {}") is True


class TestPendingActionsDBLayer:
    def test_create_and_get(self, db):
        action = db.create_pending_action("skill_x", "agent_x", "user_A", {"message": "hi"}, source="chat")
        assert action["status"] == "pending"
        assert db.get_pending_action(action["id"])["params"] == {"message": "hi"}

    def test_approve_moves_to_approved_and_only_once(self, db):
        action = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        decided = db.decide_pending_action(action["id"], "user_A", approve=True)
        assert decided["status"] == "approved"
        # a second decision attempt on an already-decided action must fail
        assert db.decide_pending_action(action["id"], "user_A", approve=False) is None

    def test_reject_moves_to_rejected(self, db):
        action = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        decided = db.decide_pending_action(action["id"], "user_A", approve=False)
        assert decided["status"] == "rejected"

    def test_other_user_cannot_decide_someone_elses_action(self, db):
        action = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        assert db.decide_pending_action(action["id"], "user_B", approve=True) is None
        assert db.get_pending_action(action["id"])["status"] == "pending"

    def test_record_result_success(self, db):
        action = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        db.decide_pending_action(action["id"], "user_A", approve=True)
        db.record_pending_action_result(action["id"], success=True, result={"ok": 1}, error=None)
        updated = db.get_pending_action(action["id"])
        assert updated["status"] == "executed"
        assert updated["result"] == {"ok": 1}

    def test_record_result_failure(self, db):
        action = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        db.decide_pending_action(action["id"], "user_A", approve=True)
        db.record_pending_action_result(action["id"], success=False, result=None, error="network error")
        updated = db.get_pending_action(action["id"])
        assert updated["status"] == "failed"
        assert updated["error"] == "network error"

    def test_list_filters_by_status(self, db):
        a1 = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        a2 = db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        db.decide_pending_action(a1["id"], "user_A", approve=True)
        pending = db.list_pending_actions_for_owner("user_A", status="pending")
        approved = db.list_pending_actions_for_owner("user_A", status="approved")
        assert [a["id"] for a in pending] == [a2["id"]]
        assert [a["id"] for a in approved] == [a1["id"]]

    def test_list_scoped_to_owner(self, db):
        db.create_pending_action("skill_x", "agent_x", "user_A", {}, source="chat")
        db.create_pending_action("skill_y", "agent_y", "user_B", {}, source="chat")
        assert len(db.list_pending_actions_for_owner("user_A")) == 1


class TestSkillActionTypeStorage:
    def test_invalid_action_type_rejected(self, db):
        with pytest.raises(ValueError):
            db.create_skill("agent_x", "user_A", "s", "d", READ_ONLY_CODE, ["x.com"], {}, action_type="delete_everything")

    def test_default_action_type_is_read_only(self, db):
        skill = db.create_skill("agent_x", "user_A", "s", "d", READ_ONLY_CODE, ["x.com"], {})
        assert skill["action_type"] == "read_only"

    def test_outbound_action_type_persisted(self, db):
        skill = db.create_skill("agent_x", "user_A", "s", "d", OUTBOUND_CODE, ["x.com"], {}, action_type="outbound_action")
        assert skill["action_type"] == "outbound_action"

    def test_list_skills_for_agent_includes_action_type(self, db):
        skill = db.create_skill("agent_x", "user_A", "s", "d", OUTBOUND_CODE, ["x.com"], {}, action_type="outbound_action")
        db.update_skill_test_result(skill["id"], passed=True, log="ok")
        listed = db.list_skills_for_agent("agent_x")
        assert listed[0]["action_type"] == "outbound_action"


class TestAgentDispatchRoutesOutboundActionsToApproval:
    @pytest.fixture
    def env(self, monkeypatch):
        from app.core.agent import GenesisAgent
        from app.core.engine.profiler import DatasetProfiler
        from app.core.engine.trainer import DatasetTrainer
        from app.core.engine.registry import ModelRegistry
        from app.core.agent_store import AgentStore, AgentConfig

        d = tempfile.mkdtemp()
        prev_cwd = os.getcwd()
        os.chdir(d)
        os.makedirs("datasets", exist_ok=True)
        monkeypatch.setenv("DEBUG", "true")
        db = GenesisDB(db_path=os.path.join(d, "test.db"))
        agent = GenesisAgent(profiler=DatasetProfiler(), trainer=DatasetTrainer(),
                              registry=ModelRegistry(storage_dir=os.path.join(d, "registry")), db=db)
        agent.current_user_id = "owner_user"
        cfg = AgentConfig(name="Bot", system_prompt="x")
        cfg.author = "owner_user"
        AgentStore().create_agent(cfg)
        agent.agent_override = {"agent_id": cfg.id, "system_prompt": cfg.system_prompt}
        yield {"agent": agent, "db": db, "bot_id": cfg.id}
        os.chdir(prev_cwd)
        shutil.rmtree(d, ignore_errors=True)

    def test_outbound_action_skill_creates_pending_action_not_executed(self, env):
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="contact_seller",
            description="x", code=OUTBOUND_CODE, allowed_domains=["example.com"],
            input_schema={}, action_type="outbound_action")
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")

        resp = env["agent"]._do_run_skill(env["bot_id"], {"skill_id": skill["id"], "params": {"message": "hi there"}})
        assert resp.success
        assert resp.data["requires_approval"] is True
        pending = env["db"].list_pending_actions_for_owner("owner_user")
        assert len(pending) == 1
        assert pending[0]["params"] == {"message": "hi there"}
        assert pending[0]["status"] == "pending"

    def test_read_only_skill_still_executes_directly(self, env):
        """Regression guard: adding the outbound_action gate must not accidentally route
        read_only skills through the approval flow too."""
        skill = env["db"].create_skill(
            agent_id=env["bot_id"], owner_user_id="owner_user", name="check_repo",
            description="x", code=READ_ONLY_CODE, allowed_domains=["github.com"],
            input_schema={}, action_type="read_only")
        env["db"].update_skill_test_result(skill["id"], passed=True, log="ok")

        resp = env["agent"]._do_run_skill(env["bot_id"], {"skill_id": skill["id"], "params": {"repo": "anthropics"}})
        assert resp.success
        assert "requires_approval" not in (resp.data or {})
        assert env["db"].list_pending_actions_for_owner("owner_user") == []


class TestSchedulerRoutesOutboundActionsToApproval:
    @pytest.fixture
    def env(self, monkeypatch):
        d = tempfile.mkdtemp()
        prev_cwd = os.getcwd()
        os.chdir(d)
        monkeypatch.setenv("DEBUG", "true")
        db = GenesisDB(db_path=os.path.join(d, "test.db"))
        yield {"db": db}
        os.chdir(prev_cwd)
        shutil.rmtree(d, ignore_errors=True)

    def test_scheduled_outbound_action_creates_pending_action_not_executed(self, env):
        from app.core.scheduler import _run_one_job
        db = env["db"]
        skill = db.create_skill(agent_id="agent_x", owner_user_id="owner_user", name="contact_seller",
                                 description="x", code=OUTBOUND_CODE, allowed_domains=["example.com"],
                                 input_schema={}, action_type="outbound_action")
        db.update_skill_test_result(skill["id"], passed=True, log="ok")
        job = db.create_scheduled_job(skill["id"], "agent_x", "owner_user", {"message": "hi"}, interval_minutes=15)

        asyncio.run(_run_one_job(db, job))

        pending = db.list_pending_actions_for_owner("owner_user")
        assert len(pending) == 1
        assert pending[0]["source"] == "scheduled_job"
        # the job itself should be marked as having completed a (successful) run - creating
        # the pending action IS what this run did - so it doesn't get reclaimed every tick
        updated_job = db.get_scheduled_job(job["id"])
        assert updated_job["last_run_at"] is not None
