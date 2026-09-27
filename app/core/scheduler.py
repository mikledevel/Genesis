"""
Background scheduler for "Level 3" bot capabilities - skills that run on their own on a
timer, without a user message triggering each run (e.g. "check this site every 30 minutes
and tell me if the price drops").

Design notes:

- No new infrastructure (no Celery, no Redis, no separate worker process) - this runs as a
  background asyncio task inside the same FastAPI process, started at app startup. That's a
  deliberate choice for where this project is right now: a single-process deployment doesn't
  need a distributed task queue, and adding one would be real operational complexity (a
  broker to run, a worker process to deploy and monitor) for no benefit at this scale. If/when
  this app runs with multiple worker processes (e.g. `uvicorn --workers 4`) or multiple
  server instances, this file's polling loop would run once per process/instance - that's
  SAFE (not a correctness bug) because GenesisDB.claim_due_jobs() uses an atomic
  compare-and-swap so only one process can ever actually claim and run a given due job - but
  it means N processes each spend a small amount of CPU polling on the same schedule. Revisit
  with a real task queue only if that polling overhead ever actually matters.

- No Groq/LLM calls happen here. A skill's code was already generated and sandbox-tested at
  CREATION time (see app/core/skills/skill_generator.py) - running it again later is just
  executing that already-written, already-vetted function. This is why autonomous scheduling
  doesn't reopen the "will this silently burn through the owner's Groq budget" concern that
  came up when we first discussed autonomy - there is no LLM call in this loop at all.

- Every claimed job still goes through the exact same SafeFetcher domain-allowlist and
  private-IP blocking as a chat-triggered skill call (see run_stored_skill) - scheduling adds
  a trigger mechanism, it does not add any new capability or relax any existing guardrail.
"""
import asyncio
import time

POLL_INTERVAL_SECONDS = 30
CLAIM_BATCH_SIZE = 10

_scheduler_task = None


async def _run_one_job(db, job: dict):
    from app.core.skills.skill_generator import SkillGenerator
    from app.core.email_sender import send_email

    skill = db.get_skill(job["skill_id"])
    if not skill or not skill["active"] or not skill["test_passed"]:
        db.complete_scheduled_job_run(job["id"], success=False, result=None,
                                       error="The skill this job depends on is no longer available or active.")
        return

    if skill["action_type"] == "outbound_action":
        # Same rule as chat (see GenesisAgent._do_run_skill's docstring): an outbound_action
        # skill never runs on its own, whether the trigger is a chat message or a timer. A
        # schedule can create AS MANY pending actions as it likes, but each one still needs
        # a human to click approve - autonomy here means "checks on its own", never "acts on
        # its own". We still record this as a completed "run" (creating the pending action
        # IS what this run does) so next_run_at advances normally and the job doesn't get
        # reclaimed and re-queue duplicate pending actions every 30 seconds.
        action = db.create_pending_action(
            skill_id=skill["id"], agent_id=job["agent_id"], owner_user_id=job["owner_user_id"],
            params=job["params"], source="scheduled_job")
        db.complete_scheduled_job_run(job["id"], success=True,
                                       result={"pending_action_id": action["id"]}, error=None)
        owner = db.get_user_by_id(job["owner_user_id"])
        if owner:
            send_email(owner["email"], f"\"{skill['name']}\" wants your approval",
                       f"Your scheduled job for \"{skill['name']}\" ran and wants to take an "
                       f"action outside the platform with these details: {job['params']}\n\n"
                       f"Nothing has been sent or done yet - review and approve or reject it "
                       f"from your dashboard.")
        return

    # Real sandboxed execution, in a thread so it never blocks the event loop the rest of the
    # app (including live chat requests) is running on - SkillGenerator.run_stored_skill is
    # synchronous (subprocess.run / blocking Docker SDK calls under the hood).
    outcome = await asyncio.to_thread(
        SkillGenerator().run_stored_skill, skill["code"], job["params"], skill["allowed_domains"])

    result = db.complete_scheduled_job_run(
        job["id"], success=outcome["success"], result=outcome.get("result"), error=outcome.get("error"))

    if result["auto_paused"]:
        owner = db.get_user_by_id(job["owner_user_id"])
        if owner:
            send_email(owner["email"], f"A scheduled skill was paused after repeated failures",
                       f"Your scheduled job for skill \"{skill['name']}\" has failed "
                       f"{db.MAX_CONSECUTIVE_FAILURES} times in a row and has been automatically "
                       f"paused so it doesn't keep spending resources on something that isn't "
                       f"working. Last error: {outcome.get('error')}\n\n"
                       f"Check the skill and re-enable the job from your dashboard when it's fixed.")
        return

    if outcome["success"] and result["changed"] and job["notify_on_change"]:
        owner = db.get_user_by_id(job["owner_user_id"])
        if owner:
            send_email(owner["email"], f"Update from your \"{skill['name']}\" job",
                       f"Your scheduled skill \"{skill['name']}\" found a change:\n\n"
                       f"{outcome['result']}\n\n"
                       f"(This job checks every {job['interval_minutes']} minutes and only "
                       f"emails you when the result actually changes.)")


async def _poll_loop():
    from app.db.database import GenesisDB
    db = GenesisDB()
    while True:
        try:
            jobs = db.claim_due_jobs(limit=CLAIM_BATCH_SIZE)
            for job in jobs:
                try:
                    await _run_one_job(db, job)
                except Exception as e:
                    # A bug in the runner itself (not the skill's own failure, which
                    # complete_scheduled_job_run already handles) must not kill the whole
                    # polling loop - one broken job should never stop every other user's
                    # scheduled jobs from running.
                    print(f"[scheduler] unexpected error running job {job['id']}: {e}")
        except Exception as e:
            print(f"[scheduler] error in poll loop: {e}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start_scheduler():
    """Call once at app startup (see main.py's lifespan). Idempotent - calling it twice
    (e.g. under uvicorn --reload, which can re-run startup code) won't start a second
    duplicate loop in the same process."""
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_poll_loop())
    return _scheduler_task


def stop_scheduler():
    global _scheduler_task
    if _scheduler_task is not None:
        _scheduler_task.cancel()
        _scheduler_task = None
