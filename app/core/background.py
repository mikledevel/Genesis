"""Очередь в БД: процессы обучения, входящие события и доставка outbox."""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path

log = logging.getLogger(__name__)
_tasks = []
_stopping = threading.Event()


def _active(db,user_id):
    state = db.get_session_state(user_id)
    return bool(state and not state[1])


def _telegram_link(db,link_id):
    with db._connect() as conn:
        row = conn.execute("SELECT webhook_secret FROM telegram_links WHERE id=?",(link_id,)).fetchone()
    return db.get_telegram_link_by_secret(row[0]) if row else None


def execute_job(db,job):
    if not _active(db,job["user_id"]):
        db.finish_job(job["id"],error="Account is inactive")
        return
    env = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen([sys.executable,"-m","app.job_runner"],stdin=subprocess.PIPE,
                                   stdout=output,stderr=output,text=True,env=env,start_new_session=True)
        try:
            process.stdin.write(json.dumps(job))
            process.stdin.close()
            started = time.monotonic()
            while process.poll() is None:
                current = db.get_job(job["user_id"],job["id"])
                if _stopping.is_set() or time.monotonic()-started > 600 or not current or current["status"] == "cancelled":
                    raise TimeoutError("Job cancelled or exceeded the execution timeout")
                time.sleep(0.5)
            output.seek(0,2)
            output.seek(max(0,output.tell()-1_000_000))
            result = None
            for line in output:
                if line.startswith("GENESIS_JOB_RESULT:"):
                    result = json.loads(line.partition(":")[2])
            if process.returncode != 0 or result is None:
                raise RuntimeError("The worker could not finish this job")
            db.finish_job(job["id"], result=result,
                          error="The operation did not succeed" if result.get("success") is False else None,
                          attempt=job.get("attempt"))
        except BaseException as error:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            if _stopping.is_set():
                db.release_job(job["id"], job.get("attempt", 1))
            else:
                db.finish_job(job["id"],error="Job failed or was cancelled; consult server logs", attempt=job.get("attempt"))
            log.warning("job_failed id=%s type=%s",job["id"],type(error).__name__)


def process_incoming(db,event):
    if not _active(db,event["user_id"]):
        with db.write_transaction() as conn:
            conn.execute("UPDATE incoming_events SET status='failed' WHERE id=?",(event["id"],))
        return
    data = event["payload"]
    link = _telegram_link(db,data["link_id"])
    if not link or link["user_id"] != event["user_id"]:
        raise ValueError("Telegram connection is no longer available")
    from app.core.agent_store import AgentStore
    from app.core.agent import GenesisAgent
    bot = AgentStore().get_agent(link["agent_id"])
    if not bot:
        raise ValueError("Bot no longer exists")
    db.log_telegram_event(link["id"],"message_in",data["text"][:200])
    agent = GenesisAgent(agent_override={"model":bot.get("model","openai/gpt-oss-120b"),
        "system_prompt":bot.get("system_prompt",""),"agent_id":link["agent_id"],
        "external_channel":"telegram","allow_management":False})
    response = agent.process(data["text"],conversation_id=f"telegram_{link['agent_id']}_{data['chat_id']}",user_id=event["user_id"])
    if not response.success:
        raise RuntimeError("Bot could not generate a response")
    db.complete_incoming(event,{"link_id":link["id"],"chat_id":data["chat_id"],"text":response.message})


def deliver_outgoing(db,event):
    if not _active(db,event["user_id"]):
        raise ValueError("Account is inactive")
    data = event["payload"]
    if event["kind"] == "telegram":
        from app.core.telegram_integration import send_message
        link = _telegram_link(db,data["link_id"])
        if not link or link["user_id"] != event["user_id"]:
            raise ValueError("Connection not found")
        try:
            send_message(link["bot_token"],data["chat_id"],data["text"])
            db.log_telegram_event(link["id"],"message_out",data["text"][:200])
            db.update_telegram_link_status(link["id"],"connected")
        except Exception as error:
            db.log_telegram_event(link["id"],"error",type(error).__name__)
            db.update_telegram_link_status(link["id"],"error","Telegram delivery failed")
            raise
    elif event["kind"] == "handoff":
        from app.core.handoff import send_handoff_webhook
        result = send_handoff_webhook(data["url"],data["body"])
        if data.get("event_id"):
            with db.write_transaction() as conn:
                conn.execute("UPDATE handoff_events SET delivered=?,delivery_error=? WHERE id=?",
                             (1 if result["delivered"] else 0,None if result["delivered"] else "Webhook delivery failed",data["event_id"]))
        if not result["delivered"]:
            raise RuntimeError("Webhook delivery failed")
    else:
        raise ValueError("Unsupported outgoing event")


async def _loop(kind):
    from app.db.database import GenesisDB
    db = GenesisDB()
    while True:
        try:
            if kind == "job":
                job = await asyncio.to_thread(db.claim_job)
                if job:
                    await asyncio.to_thread(execute_job,db,job)
            elif kind == "incoming":
                event = await asyncio.to_thread(db.claim_incoming)
                if event:
                    try:
                        await asyncio.to_thread(process_incoming,db,event)
                    except Exception:
                        with db.write_transaction() as conn:
                            conn.execute("UPDATE incoming_events SET status='failed' WHERE id=?",(event["id"],))
                        log.warning("incoming_event_failed id=%s",event["id"])
            elif kind == "maintenance":
                await asyncio.to_thread(db.recover_usage)
                await asyncio.to_thread(db.prune_traces)
                await asyncio.sleep(59)
            else:
                event = await asyncio.to_thread(db.claim_outgoing)
                if event:
                    try:
                        await asyncio.to_thread(deliver_outgoing,db,event)
                        await asyncio.to_thread(db.finish_outgoing,event)
                    except Exception as error:
                        await asyncio.to_thread(db.finish_outgoing,event,type(error).__name__)
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning("background_loop_failed kind=%s type=%s",kind,type(error).__name__)
            await asyncio.sleep(5)


def start_workers():
    _stopping.clear()
    if not _tasks:
        _tasks.extend(asyncio.create_task(_loop(kind)) for kind in ("job","job","incoming","outgoing","maintenance"))


async def stop_workers():
    _stopping.set()
    tasks,_tasks[:] = list(_tasks),[]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks,return_exceptions=True)
