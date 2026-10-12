"""chat_agent routes (split out of app/main.py)."""
import asyncio
import contextvars
import json
import queue
import threading
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.deps import (
    UploadFile,
    File,
    Any,
    Depends,
    Dict,
    HTTPException,
    MessageRequest,
    Optional,
    get_agent,
    get_current_user_id,
    os,
    settings
)

router = APIRouter()

# ── Agent ──
@router.get("/api/agent/conversations")
async def agent_conversations(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"conversations": GenesisDB().list_chats(user_id=user_id)}

@router.get("/api/agent/chat/{conversation_id}")
async def agent_chat(conversation_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    owner = db.get_chat_owner(conversation_id)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": db.get_messages(conversation_id)}

def _agent_message_sync(req: MessageRequest, user_id: str) -> dict:
    """Shared by the plain JSON endpoint and the SSE streaming endpoint below: ownership check,
    balance pre-check, the agent call, billing AFTER success, bot tagging. Raises HTTPException."""
    from app.db.database import GenesisDB
    from app.core.llm_errors import LLMError
    cid = req.conversation_id or "default"
    db = GenesisDB()
    owner = db.get_chat_owner(cid)
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="This conversation belongs to another account")

    # If the user has a paid bot selected (per-user, stored in DB), a message to it costs
    # money. We only CHECK the balance here (no mutation) so an obviously-unaffordable
    # request fails fast without spending an LLM call. The actual charge happens AFTER
    # agent.process() succeeds, below - previously the charge happened here, before the LLM
    # was ever called, so a Groq outage or any other generation failure still cost the buyer
    # money for a message that was never actually generated.
    selected = db.get_selected_model(user_id)
    price = selected.get("price_per_call", 0) or 0
    bot_author = selected.get("author")
    price_cents = round(price * 100)
    billable = price > 0 and bot_author and bot_author != user_id
    if billable and db.get_balance(user_id) < price_cents:
        raise HTTPException(status_code=402, detail=(
            f"Insufficient balance - this bot costs ${price:.3f}/message, "
            f"you have ${db.get_balance(user_id)/100:.2f}. Add funds to continue."))

    agent = get_agent(cid, user_id)
    # If the currently-selected bot has a running A/B test, use this conversation's assigned
    # variant's system_prompt instead of the bot's normal stored prompt - sticky per
    # conversation_id (see get_or_assign_variant) so a single chat doesn't flip between arms.
    # Reuses the same agent_override mechanism built for external channels (Telegram): it
    # bypasses users.selected_model_json entirely for this one field rather than mutating it,
    # so it can't race with the user's own unrelated model selection.
    agent.agent_override = None
    agent_id = selected.get("agent_id")
    if agent_id:
        ab_test = db.get_running_ab_test_for_agent(agent_id)
        if ab_test and ab_test["variants"]:
            variant = db.get_or_assign_variant(ab_test["id"], ab_test["variants"], cid)
            agent.agent_override = {**selected, "system_prompt": variant["system_prompt"]}

    process_kwargs = {}
    agent_message = req.message
    if req.attachments:
        from app.core import chat_uploads
        try:
            items = chat_uploads.resolve(user_id, req.attachments)
            extra, shown = chat_uploads.build_context(items)
        except chat_uploads.UploadError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e))
        agent_message = req.message + extra           # what the model/tools see
        process_kwargs["display_message"] = req.message + shown   # what the history shows
    try:
        response = agent.process(agent_message, cid, req.projects, user_id=user_id, **process_kwargs)
    except LLMError as e:
        # Generation genuinely failed - surface a real error instead of a fake 200, and
        # (critically) never reach the charge_for_usage call below, so nobody is billed for
        # a message the model never produced. See app.core.llm_errors for the status-code
        # mapping (config -> 503, rate limit -> 429, timeout -> 504, generic -> 502).
        # The exact exception (incl. its class name) is already logged server-side above, in
        # _log_groq_usage/print - no need to leak it into the user-facing chat bubble, which
        # previously showed raw text like "... (LLMTimeoutError)" verbatim in the UI.
        from app.core.llm_errors import LLMConfigError, LLMRateLimitError, LLMTimeoutError
        if isinstance(e, LLMTimeoutError):
            friendly = "That took longer than expected to answer. Please try again."
        elif isinstance(e, LLMRateLimitError):
            friendly = "The AI provider is busy right now. Please wait a moment and try again."
        elif isinstance(e, LLMConfigError):
            friendly = "The AI provider isn't configured correctly on this server. Please let the site operator know."
        else:
            friendly = "The AI provider failed to generate a response. Please try again in a moment."
        raise HTTPException(status_code=e.http_status, detail=friendly)

    if billable:
        charge = db.charge_for_usage(
            buyer_user_id=user_id, amount_cents=price_cents,
            description=f"Message to {selected.get('agent_name', 'bot')}",
            author_user_id=bot_author, related_id=agent_id,
        )
        if not charge["ok"]:
            # Balance changed between our pre-check and now (e.g. a concurrent spend) - the
            # response was already generated, so we still return it rather than discard real
            # output, but we do NOT pretend the charge succeeded.
            print(f"[billing] charge_for_usage failed post-generation for user={user_id} agent={agent_id}: "
                  f"balance={charge.get('balance_cents')}")

    if agent_id:
        # Must run AFTER agent.process() - that's what actually creates the chats row
        # (via create_chat inside process()) for a brand-new conversation_id. Tagging before
        # that would silently update zero rows on the very first message of a chat.
        db.tag_chat_with_agent(cid, agent_id)  # links this conversation to the bot for
                                                # analytics - see GenesisDB.get_bot_analytics
    return {"response": response.message, "conversation_id": cid, "data": response.data, "message_id": response.message_id}


@router.get("/api/llm/status")
async def llm_status(user_id: str = Depends(get_current_user_id)):
    """Which LLM providers are configured and which are cooling down after a failure (no secrets)."""
    from app.config import settings as _settings
    from app.core import groq_client
    return groq_client.status(_settings.groq_api_key)


@router.post("/api/agent/upload")
async def agent_upload(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    """The chat paperclip: store a dataset (CSV/TSV/XLSX/XLS/JSON/Parquet) or a document
    (TXT/MD/PDF) for this user and return an id to send with the next message."""
    from app.core import chat_uploads
    limit = max(chat_uploads.MAX_DATA_BYTES, chat_uploads.MAX_DOC_BYTES)
    data = await file.read(limit + 1)             # one byte past the cap -> rejected without reading it all
    try:
        return chat_uploads.save_upload(user_id, file.filename or "file", data)
    except chat_uploads.UploadError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/api/agent/message")
async def agent_message(req: MessageRequest, user_id: str = Depends(get_current_user_id)):
    return _agent_message_sync(req, user_id)


@router.post("/api/agent/message/stream")
async def agent_message_stream(req: MessageRequest, user_id: str = Depends(get_current_user_id)):
    """Same as /api/agent/message, but streamed as Server-Sent Events so the UI can show live
    progress ("Cross-validating... done") while a long action runs. Events, one JSON per
    `data:` line: plan / step / fail (see app/core/progress.py), then exactly one terminal event:
    {"type":"final", response, conversation_id, data, message_id} or
    {"type":"error", status, detail}. Auth, billing and error mapping are identical to the
    JSON endpoint because both call _agent_message_sync."""
    from app.core import progress
    q: "queue.Queue" = queue.Queue()
    ctx = contextvars.copy_context()      # the callback lives in a ContextVar: this request only

    def run():
        progress.set_callback(q.put)
        try:
            q.put({"type": "final", **_agent_message_sync(req, user_id)})
        except HTTPException as e:
            q.put({"type": "error", "status": e.status_code, "detail": e.detail})
        except Exception as e:
            print(f"[agent stream] unhandled error: {type(e).__name__}: {e}")
            q.put({"type": "error", "status": 500, "detail": "Something went wrong while processing your message."})
        finally:
            q.put(None)

    threading.Thread(target=ctx.run, args=(run,), daemon=True).start()

    async def events():
        while True:
            try:
                item = await asyncio.to_thread(q.get, True, 15)
            except queue.Empty:
                yield ": keep-alive\n\n"          # comment frame: stops proxies closing an idle stream
                continue
            if item is None:
                return
            yield "data: " + json.dumps(item, ensure_ascii=False, default=str) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@router.post("/api/agent/chat/{conversation_id}/messages/{message_id}/rate")
async def rate_message(conversation_id: str, message_id: int, data: dict, user_id: str = Depends(get_current_user_id)):
    """Thumbs up(1)/down(-1)/clear(null) on an assistant message - the signal A/B test stats
    are built from (see GenesisDB.get_ab_test_stats)."""
    from app.db.database import GenesisDB
    db = GenesisDB()
    if db.get_chat_owner(conversation_id) != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")
    rating = data.get("rating")
    if rating not in (1, -1, None):
        raise HTTPException(status_code=400, detail="rating must be 1, -1, or null")
    ok = db.rate_message(message_id, conversation_id, rating)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found in this conversation")
    return {"status": "ok", "rating": rating}

@router.post("/api/agent/chat/{conversation_id}/rename")
async def rename_chat(conversation_id: str, data: Dict[str, Any], user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    if db.get_chat_owner(conversation_id) != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")
    db.rename_chat(conversation_id, data.get('title', conversation_id))
    return {"status": "renamed"}

@router.delete("/api/agent/chat/{conversation_id}")
async def delete_chat(conversation_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    if db.get_chat_owner(conversation_id) != user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")
    db.delete_chat(conversation_id)
    return {"status": "deleted"}

# ── Memory ──
@router.get("/api/memory/facts")
async def memory_facts(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"facts": GenesisDB().get_facts(user_id)}

@router.get("/api/dashboard/activity")
async def dashboard_activity(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"activity": GenesisDB().get_activity_last_7_days(user_id=user_id)}

@router.get("/api/memory/persona")
async def memory_persona():
    from app.core.memory.graph import KnowledgeGraph
    from app.core.memory.persona import PersonaGenerator
    return {"persona": PersonaGenerator(KnowledgeGraph()).generate()}

@router.get("/api/memory/graph")
async def memory_graph():
    from app.core.memory.graph import KnowledgeGraph
    return KnowledgeGraph().to_dict()

# ── Code Generation ──
@router.post("/api/codegen/generate")
async def generate_code(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.codegen.generator import CodeGenerator
    from app.db.database import GenesisDB
    gen = CodeGenerator(GenesisDB())
    description = data.get("description", "")
    code = gen.generate_from_description(description)
    path = gen.save_code(code, owner_user_id=user_id)
    return {"code_path": path, "code": code}

@router.get("/api/codegen/list")
async def list_generated(user_id: str = Depends(get_current_user_id)):
    import glob
    # Only ever return the caller's own generated files - see save_code's owner-tagged
    # filenames. Previously this globbed every file in the shared "generated/" directory
    # and returned all of it, unauthenticated, regardless of who generated what.
    prefix = f"user_{user_id}_"
    files = [f for f in glob.glob("generated/*.py") if os.path.basename(f).startswith(prefix)]
    return {"files": [{"name": os.path.basename(f), "path": f} for f in files]}

# ── Health ──

@router.post("/api/agent/select-model")
async def select_model(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    GenesisDB().set_selected_model(user_id, data)
    return {"status": "ok", "model": data.get("model", "")}

# ── Agent Browser ("computer use" style agency) ──
# See app/core/agent_browser.py's module docstring for the full architecture and security
# model before touching anything here - in particular why this container gets real network
# access (unlike the skills sandbox) and what that does and doesn't protect against.
@router.post("/api/agent-browser/sessions")
async def agent_browser_create_session(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.agent_browser import AgentBrowserSession, AgentBrowserError
    if not settings.agent_browser_enabled:
        raise HTTPException(status_code=503, detail="The agent browser isn't enabled on this server.")
    db = GenesisDB()
    balance = db.get_balance(user_id)
    if balance < settings.agent_browser_min_balance_cents:
        raise HTTPException(status_code=402, detail=(
            f"A balance of at least {settings.agent_browser_min_balance_cents} cents is required to "
            f"start an agent browser session (you have {balance}). Top up at POST /api/payments/checkout."))
    try:
        session = AgentBrowserSession.create(user_id)
    except AgentBrowserError as e:
        raise HTTPException(status_code=503, detail=str(e))
    db.create_agent_browser_session_record(session.session_id, user_id)
    return {"session_id": session.session_id,
            "max_session_minutes": settings.agent_browser_max_session_minutes,
            "max_actions": settings.agent_browser_max_actions_per_session,
            "price_cents_per_minute": settings.agent_browser_price_cents_per_minute}

@router.post("/api/agent-browser/sessions/{session_id}/actions")
async def agent_browser_perform_action(session_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.agent_browser import AgentBrowserSession, AgentBrowserError
    db = GenesisDB()
    record = db.get_agent_browser_session(session_id)
    if not record or record["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="No such session")
    if record["status"] != "active":
        raise HTTPException(status_code=409, detail=f"This session is {record['status']}, not active")

    action_type = data.get("type", "")
    try:
        session = AgentBrowserSession.resume(session_id)
        result = session.perform_action(action_type, data.get("params") or {})
    except AgentBrowserError as e:
        # A session-level failure (expired, over its action cap, container already gone) -
        # the caller should call POST .../close next, which independently handles a
        # missing/already-exited container (estimates duration from the DB record's
        # started_at and closes out billing from that) - no need to duplicate that logic here.
        raise HTTPException(status_code=409, detail=str(e))
    db.record_agent_browser_action(session_id)
    return result

@router.post("/api/agent-browser/sessions/{session_id}/close")
async def agent_browser_close_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.agent_browser import AgentBrowserSession, AgentBrowserError, compute_browser_cost_cents
    db = GenesisDB()
    record = db.get_agent_browser_session(session_id)
    if not record or record["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="No such session")
    if record["status"] != "active":
        return {"status": record["status"], "charged_cents": record["charged_cents"]}

    try:
        session = AgentBrowserSession.resume(session_id)
        close_result = session.close()
        duration_seconds = close_result["duration_seconds"]
    except AgentBrowserError:
        # Container already gone (e.g. it hit its own internal cap and exited) - still need
        # to close out the billing record using the best duration estimate we have.
        import datetime as _dt
        started = _dt.datetime.fromisoformat(record["started_at"].replace(" ", "T"))
        duration_seconds = (_dt.datetime.utcnow() - started).total_seconds()

    cost_cents = compute_browser_cost_cents(duration_seconds)
    charge = db.charge_for_usage(user_id, cost_cents, f"Agent browser session ({duration_seconds:.0f}s)",
                                 related_id=session_id)
    # The container is ALREADY stopped/removed at this point regardless of whether the charge
    # succeeds - a user can't rack up unbounded debt by draining their balance mid-session,
    # because the hard wall-clock/action caps already bound worst-case duration (see
    # settings.agent_browser_max_session_minutes), and the session is always torn down before
    # billing is even attempted, never after a failed charge.
    final_charged = cost_cents if charge["ok"] else 0
    db.close_agent_browser_session_record(session_id, duration_seconds, final_charged,
                                          status="closed" if charge["ok"] else "closed_unpaid")
    return {"status": "closed", "duration_seconds": duration_seconds, "charged_cents": final_charged,
            "balance_cents_remaining": db.get_balance(user_id)}

@router.get("/api/agent-browser/sessions")
async def agent_browser_list_sessions(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"sessions": GenesisDB().list_agent_browser_sessions(user_id)}

@router.get("/api/agent-browser/sessions/{session_id}")
async def agent_browser_get_session(session_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    record = GenesisDB().get_agent_browser_session(session_id)
    if not record or record["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="No such session")
    return record


# ---- Pending actions ("Level 4": outbound_action skills always stop here first) ----
@router.get("/api/pending-actions")
async def list_pending_actions(status: Optional[str] = None, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"actions": GenesisDB().list_pending_actions_for_owner(user_id, status=status)}

@router.post("/api/pending-actions/{action_id}/approve")
async def approve_pending_action(action_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core.skills.skill_generator import SkillGenerator
    db = GenesisDB()
    action = db.decide_pending_action(action_id, user_id, approve=True)
    if not action:
        raise HTTPException(status_code=404, detail=(
            "Pending action not found, doesn't belong to you, or was already decided"))

    skill = db.get_skill(action["skill_id"])
    if not skill or not skill["active"]:
        db.record_pending_action_result(action_id, success=False, result=None,
                                         error="The skill this action depends on is no longer available.")
        raise HTTPException(status_code=409, detail="The underlying skill is no longer available")

    outcome = SkillGenerator().run_stored_skill(skill["code"], action["params"], skill["allowed_domains"])
    db.record_pending_action_result(action_id, success=outcome["success"],
                                     result=outcome.get("result"), error=outcome.get("error"))
    return {"action": db.get_pending_action(action_id)}

@router.post("/api/pending-actions/{action_id}/reject")
async def reject_pending_action(action_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    action = GenesisDB().decide_pending_action(action_id, user_id, approve=False)
    if not action:
        raise HTTPException(status_code=404, detail=(
            "Pending action not found, doesn't belong to you, or was already decided"))
    return {"action": action}
