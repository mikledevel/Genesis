"""
Tests for GenesisDB's scheduled_jobs methods - the "Level 3" autonomous scheduling layer.
These focus on the parts that are easy to get subtly wrong and hard to notice in manual
testing: the atomic double-claim protection, and the failure/change-detection bookkeeping
that decides when to notify an owner or auto-pause a broken job.
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.db.database import GenesisDB


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


def _make_overdue(db, job_id):
    with db._connect() as conn:
        conn.execute("UPDATE scheduled_jobs SET next_run_at = datetime('now', '-1 minute') WHERE id = ?", (job_id,))
        conn.commit()


class TestCreateScheduledJob:
    def test_interval_below_minimum_rejected(self, db):
        with pytest.raises(ValueError, match="at least"):
            db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=14)

    def test_minimum_interval_exactly_accepted(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        assert job["interval_minutes"] == 15

    def test_max_active_jobs_per_owner_enforced(self, db):
        for i in range(db.MAX_ACTIVE_JOBS_PER_OWNER):
            db.create_scheduled_job(f"skill_{i}", "agent_x", "user_x", {}, interval_minutes=15)
        with pytest.raises(ValueError, match="already have"):
            db.create_scheduled_job("skill_over_limit", "agent_x", "user_x", {}, interval_minutes=15)

    def test_paused_jobs_dont_count_toward_the_limit(self, db):
        jobs = [db.create_scheduled_job(f"skill_{i}", "agent_x", "user_x", {}, interval_minutes=15)
                for i in range(db.MAX_ACTIVE_JOBS_PER_OWNER)]
        db.set_scheduled_job_active(jobs[0]["id"], "user_x", active=False)
        # one paused now, so there's room for exactly one more active job
        db.create_scheduled_job("skill_new", "agent_x", "user_x", {}, interval_minutes=15)

    def test_different_owners_dont_share_the_limit(self, db):
        for i in range(db.MAX_ACTIVE_JOBS_PER_OWNER):
            db.create_scheduled_job(f"skill_{i}", "agent_x", "user_A", {}, interval_minutes=15)
        # user_B has their own quota, unaffected by user_A's jobs
        job = db.create_scheduled_job("skill_y", "agent_y", "user_B", {}, interval_minutes=15)
        assert job is not None


class TestAtomicClaiming:
    def test_due_job_is_claimed(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        _make_overdue(db, job["id"])
        claimed = db.claim_due_jobs()
        assert len(claimed) == 1
        assert claimed[0]["id"] == job["id"]

    def test_not_yet_due_job_is_not_claimed(self, db):
        db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        # freshly created - next_run_at is 15 minutes in the future, not due yet
        assert db.claim_due_jobs() == []

    def test_inactive_job_is_never_claimed_even_if_overdue(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        _make_overdue(db, job["id"])
        db.set_scheduled_job_active(job["id"], "user_x", active=False)
        assert db.claim_due_jobs() == []

    def test_double_claim_is_prevented(self, db):
        """The core correctness property this whole mechanism exists for: two concurrent
        callers (simulating two server processes/workers) must never both get the same job."""
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        _make_overdue(db, job["id"])
        first = db.claim_due_jobs()
        second = db.claim_due_jobs()
        assert len(first) == 1
        assert len(second) == 0

    def test_claim_respects_limit(self, db):
        for i in range(5):
            job = db.create_scheduled_job(f"skill_{i}", "agent_x", "user_x", {}, interval_minutes=15)
            _make_overdue(db, job["id"])
        claimed = db.claim_due_jobs(limit=3)
        assert len(claimed) == 3


class TestCompleteScheduledJobRun:
    def test_first_run_is_never_marked_changed(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        result = db.complete_scheduled_job_run(job["id"], success=True, result={"a": 1}, error=None)
        assert result["changed"] is False

    def test_identical_result_is_not_marked_changed(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        db.complete_scheduled_job_run(job["id"], success=True, result={"price": 300}, error=None)
        result = db.complete_scheduled_job_run(job["id"], success=True, result={"price": 300}, error=None)
        assert result["changed"] is False

    def test_different_result_is_marked_changed(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        db.complete_scheduled_job_run(job["id"], success=True, result={"price": 300}, error=None)
        result = db.complete_scheduled_job_run(job["id"], success=True, result={"price": 250}, error=None)
        assert result["changed"] is True

    def test_next_run_at_advances_by_interval_after_completion(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        _make_overdue(db, job["id"])
        db.claim_due_jobs()  # bumps next_run_at to the +10min claim marker
        db.complete_scheduled_job_run(job["id"], success=True, result={"a": 1}, error=None)
        updated = db.get_scheduled_job(job["id"])
        # should no longer be immediately due (claim marker replaced with the real interval)
        assert db.claim_due_jobs() == []
        assert updated["last_run_at"] is not None

    def test_consecutive_failures_increment_and_reset(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        db.complete_scheduled_job_run(job["id"], success=False, result=None, error="boom")
        db.complete_scheduled_job_run(job["id"], success=False, result=None, error="boom")
        assert db.get_scheduled_job(job["id"])["consecutive_failures"] == 2
        db.complete_scheduled_job_run(job["id"], success=True, result={"ok": True}, error=None)
        assert db.get_scheduled_job(job["id"])["consecutive_failures"] == 0

    def test_auto_pauses_after_max_consecutive_failures(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        result = None
        for _ in range(db.MAX_CONSECUTIVE_FAILURES):
            result = db.complete_scheduled_job_run(job["id"], success=False, result=None, error="boom")
        assert result["auto_paused"] is True
        assert db.get_scheduled_job(job["id"])["active"] is False

    def test_not_yet_at_failure_threshold_stays_active(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_x", {}, interval_minutes=15)
        result = None
        for _ in range(db.MAX_CONSECUTIVE_FAILURES - 1):
            result = db.complete_scheduled_job_run(job["id"], success=False, result=None, error="boom")
        assert result["auto_paused"] is False
        assert db.get_scheduled_job(job["id"])["active"] is True


class TestOwnershipAndLifecycle:
    def test_other_user_cannot_pause_a_job_they_dont_own(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_A", {}, interval_minutes=15)
        assert db.set_scheduled_job_active(job["id"], "user_B", active=False) is False
        assert db.get_scheduled_job(job["id"])["active"] is True

    def test_owner_can_pause_and_resume(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_A", {}, interval_minutes=15)
        assert db.set_scheduled_job_active(job["id"], "user_A", active=False) is True
        assert db.get_scheduled_job(job["id"])["active"] is False
        assert db.set_scheduled_job_active(job["id"], "user_A", active=True) is True
        assert db.get_scheduled_job(job["id"])["active"] is True

    def test_other_user_cannot_delete_a_job_they_dont_own(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_A", {}, interval_minutes=15)
        assert db.delete_scheduled_job(job["id"], "user_B") is False
        assert db.get_scheduled_job(job["id"]) is not None

    def test_owner_can_delete_their_own_job(self, db):
        job = db.create_scheduled_job("skill_x", "agent_x", "user_A", {}, interval_minutes=15)
        assert db.delete_scheduled_job(job["id"], "user_A") is True
        assert db.get_scheduled_job(job["id"]) is None

    def test_list_only_shows_owners_own_jobs(self, db):
        db.create_scheduled_job("skill_a", "agent_x", "user_A", {}, interval_minutes=15)
        db.create_scheduled_job("skill_b", "agent_x", "user_B", {}, interval_minutes=15)
        jobs_a = db.list_scheduled_jobs_for_owner("user_A")
        assert len(jobs_a) == 1
        assert jobs_a[0]["owner_user_id"] == "user_A"
