"""bots routes (split out of app/main.py)."""
from fastapi import APIRouter
from app.deps import (
    Depends,
    File,
    Form,
    HTTPException,
    List,
    Optional,
    Request,
    SUPPORTED_KB_EXTENSIONS,
    UploadFile,
    _enforce_groq_quota,
    _require_bot_owner,
    _secrets,
    get_current_user_id,
    get_current_user_id_optional,
    json,
    re,
    settings
)

router = APIRouter()

@router.get("/api/my/models")
async def my_models(user_id: str = Depends(get_current_user_id)):
    """The caller's OWN registered ML models (for the bot admin panel's "Linked ML model"
    picker). Unlike /api/registry/models this is authenticated and per-user, and returns only
    what the picker needs - never file paths."""
    from app.core.engine.registry import ModelRegistry
    return {"models": [
        {"model_id": m["model_id"], "task": m.get("task"), "model_type": m.get("model_type"),
         "target": m.get("target", ""), "features": m.get("features", []),
         "metrics": {k: v for k, v in (m.get("metrics") or {}).items() if isinstance(v, (int, float))},
         "created_at": m.get("created_at")}
        for m in ModelRegistry().list_models_by_author(user_id)]}

# ── Agent Store ──
@router.get("/api/agents")
async def list_agents(author: str = None):
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().list_agents(author=author)}

@router.get("/api/agents/featured")
async def featured_agents():
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().get_featured()}

@router.get("/api/agents/search")
async def search_agents(q: str):
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().search_agents(q)}

@router.get("/api/agents/mine")
async def my_agents(user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    return {"agents": AgentStore().list_agents(author=user_id, published_only=False)}

@router.get("/api/agents/tools")
async def available_tools():
    from app.core.agent_store import AgentStore
    return {"tools": AgentStore.AVAILABLE_TOOLS}

@router.get("/api/agents/models")
async def available_models():
    """Flat list of every catalog model (chat + audio), richest single source of truth.
    Frontend model-selector cards and the Create Agent / Bot Settings dropdowns should
    all read from this instead of hardcoding model data."""
    from app.core.models_catalog import MODEL_CATALOG, to_dict
    return {"models": [to_dict(spec) for spec in MODEL_CATALOG.values()]}

@router.get("/api/agents/models/by-category")
async def available_models_by_category():
    """Category metadata (id/label/icon) plus the full flat model list. The frontend
    filters client-side by category membership, since a model can belong to several
    categories at once and grouping would duplicate cards."""
    from app.core.models_catalog import MODEL_CATALOG, CATEGORIES, to_dict
    return {
        "categories": CATEGORIES,
        "models": [to_dict(spec) for spec in MODEL_CATALOG.values()],
    }

@router.post("/api/agents/models/auto")
async def auto_select_model(data: dict):
    """Auto mode: given a task hint and hard constraints, return the best-fit chat model."""
    from app.core.models_catalog import pick_auto_model, MODEL_CATALOG, to_dict
    model_id = pick_auto_model(
        task_hint=data.get("task_hint", ""),
        needs_vision=bool(data.get("needs_vision", False)),
        needs_json_schema=bool(data.get("needs_json_schema", False)),
        prioritize_speed=bool(data.get("prioritize_speed", False)),
        prioritize_cost=bool(data.get("prioritize_cost", False)),
    )
    return {"model_id": model_id, "model": to_dict(MODEL_CATALOG[model_id])}

# NOTE: the routes above (tools/models/models/by-category/models/auto) MUST stay registered
# before /api/agents/{agent_id} below - FastAPI matches path routes in registration order,
# and {agent_id} is a single-segment wildcard that would otherwise swallow /api/agents/models
# (treating "models" as an agent_id) before the more specific route below ever gets a chance.
# This was a pre-existing bug that silently broke /api/agents/models entirely; fixed here.
@router.get("/api/agents/{agent_id}")
async def get_agent_by_id(agent_id: str, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    from app.core.agent_store import AgentStore
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Not found")
    # Published agents are meant to be publicly browsable (that's the marketplace). An
    # unpublished/private agent - including its system_prompt - must only be visible to its
    # own author. Previously this endpoint had no auth dependency at all and returned any
    # agent, published or not, to anyone who knew or guessed its id. Returning 404 (not 403)
    # for a private agent owned by someone else avoids confirming that a given id exists.
    if not agent.get("published") and agent.get("author") != user_id:
        raise HTTPException(status_code=404, detail="Not found")
    return agent

@router.post("/api/agents/{agent_id}/knowledge")
async def upload_kb_document(agent_id: str, file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.core.knowledge_base import chunk_text
    from app.db.database import GenesisDB

    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")

    filename = file.filename or "document.txt"
    if not filename.lower().endswith(SUPPORTED_KB_EXTENSIONS):
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_KB_EXTENSIONS)} "
                   f"(PDF/DOCX not yet supported - convert to plain text first)")

    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text")
    if not content.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    chunks = chunk_text(content)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract any content to index")

    db = GenesisDB()
    doc_id = db.add_kb_document(agent_id, user_id, filename, content, chunks)
    return {"id": doc_id, "filename": filename, "chunk_count": len(chunks)}

@router.get("/api/agents/{agent_id}/knowledge")
async def list_kb_documents(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    return {"documents": GenesisDB().list_kb_documents(agent_id, user_id)}

@router.delete("/api/agents/{agent_id}/knowledge/{document_id}")
async def delete_kb_document(agent_id: str, document_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_kb_document(document_id, user_id):
        raise HTTPException(status_code=404, detail="Document not found or not yours")
    return {"status": "deleted"}

@router.get("/api/agents/{agent_id}/analytics")
async def bot_analytics(agent_id: str, user_id: str = Depends(get_current_user_id)):
    """Aggregates what's already being collected (conversations, ratings) - see
    GenesisDB.get_bot_analytics for what is/isn't included and why."""
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    return GenesisDB().get_bot_analytics(agent_id)

@router.get("/api/agents/{agent_id}/handoffs")
async def bot_handoffs(agent_id: str, user_id: str = Depends(get_current_user_id)):
    """Recent human-handoff events for this bot - see app.core.handoff and
    GenesisAgent._do_request_handoff for how these get created."""
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    return {"handoffs": GenesisDB().get_handoff_events(agent_id)}

@router.get("/api/agents/{agent_id}/conversations")
async def bot_conversations(agent_id: str, q: Optional[str] = None, date_from: Optional[str] = None,
                            date_to: Optional[str] = None, limit: int = 25, offset: int = 0,
                            user_id: str = Depends(get_current_user_id)):
    """Admin panel conversation log: searchable/filterable list of conversations with this bot."""
    _require_bot_owner(agent_id, user_id)
    from app.db.database import GenesisDB
    for d in (date_from, date_to):
        if d and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD")
    return GenesisDB().list_agent_conversations(agent_id, search=q, date_from=date_from,
                                                date_to=date_to, limit=limit, offset=offset)

@router.get("/api/agents/{agent_id}/conversations/{conversation_id}")
async def bot_conversation_messages(agent_id: str, conversation_id: str,
                                    user_id: str = Depends(get_current_user_id)):
    _require_bot_owner(agent_id, user_id)
    from app.db.database import GenesisDB
    messages = GenesisDB().get_agent_conversation(agent_id, conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found for this bot")
    return {"messages": messages}

@router.post("/api/agents/{agent_id}/telegram/connect")
async def telegram_connect(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    from app.core import telegram_integration as tg

    bot = AgentStore().get_agent(agent_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if bot.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")

    token = (data.get("bot_token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="bot_token is required")

    try:
        identity = tg.get_me(token)
    except tg.TelegramAPIError as e:
        raise HTTPException(status_code=400, detail=f"Invalid bot token: {e.description}")

    db = GenesisDB()
    webhook_secret = _secrets.token_urlsafe(24)
    status, warning = "connected_no_webhook", None
    if settings.public_base_url:
        webhook_url = f"{settings.public_base_url.rstrip('/')}/api/telegram/webhook/{webhook_secret}"
        try:
            tg.set_webhook(token, webhook_url, webhook_secret)
            status = "connected"
        except tg.TelegramAPIError as e:
            warning = f"Token is valid but webhook setup failed: {e.description}"
    else:
        warning = ("No public server URL is configured (PUBLIC_BASE_URL), so Telegram can't "
                   "deliver live messages yet - use the built-in test chat below, or deploy "
                   "publicly and reconnect once you have a real URL.")

    imported = {}
    try:
        imported = tg.import_bot_settings(token, identity["id"])
    except tg.TelegramAPIError:
        pass  # import is best-effort - a fresh bot with nothing configured yet is normal, not an error

    link_id = db.upsert_telegram_link(agent_id, user_id, token, identity["id"],
                                      identity.get("username", ""), webhook_secret, status)
    db.log_telegram_event(link_id, "connected", f"@{identity.get('username', '')}")
    return {"status": status, "bot_username": identity.get("username"), "bot_id": identity["id"],
            "imported": imported, "warning": warning}

@router.get("/api/agents/{agent_id}/telegram")
async def telegram_status(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    link = GenesisDB().get_telegram_link(agent_id, user_id)
    if not link:
        return {"connected": False}
    return {"connected": True, **link}

@router.delete("/api/agents/{agent_id}/telegram")
async def telegram_disconnect(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    from app.core import telegram_integration as tg
    deleted = GenesisDB().delete_telegram_link(agent_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No Telegram connection found for this bot")
    try:
        tg.delete_webhook(deleted["bot_token"])
    except tg.TelegramAPIError:
        pass  # best-effort - the link is already gone from our side either way
    return {"status": "disconnected"}

@router.get("/api/agents/{agent_id}/telegram/events")
async def telegram_events(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    agent = AgentStore().get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Bot not found")
    if agent.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    db = GenesisDB()
    link = db.get_telegram_link(agent_id, user_id)
    if not link:
        return {"events": []}
    return {"events": db.get_telegram_events(link["id"])}

@router.post("/api/agents/{agent_id}/telegram/test")
async def telegram_test_chat(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Runs a message through the EXACT SAME dispatch pipeline the real Telegram webhook
    uses, without touching Telegram at all - lets a bot owner verify their bot's behavior
    (including RAG/predict/handoff) before ever connecting a real token, and remains fully
    testable in environments (like this one) that can't reach api.telegram.org."""
    from app.core.agent_store import AgentStore
    from app.core.agent import GenesisAgent
    bot = AgentStore().get_agent(agent_id)
    if not bot:
        raise HTTPException(status_code=404, detail="Bot not found")
    if bot.get("author") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this bot")
    agent = GenesisAgent()
    agent.agent_override = {"model": bot.get("model", "openai/gpt-oss-120b"),
                            "system_prompt": bot.get("system_prompt", ""), "agent_id": agent_id}
    cid = f"telegram_test_{agent_id}_{user_id}"
    resp = agent.process(data.get("message", ""), conversation_id=cid, user_id=user_id)
    return {"response": resp.message, "data": resp.data}

@router.post("/api/telegram/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    """The real webhook Telegram calls on every incoming message. No auth dependency - the
    secret in the URL path IS the credential, additionally confirmed via the
    X-Telegram-Bot-Api-Secret-Token header Telegram sends on every call (defense in depth -
    URL paths can end up in logs/proxies more easily than a header)."""
    from app.db.database import GenesisDB
    db = GenesisDB()
    link = db.get_telegram_link_by_secret(secret)
    if not link:
        raise HTTPException(status_code=404, detail="Unknown webhook")
    header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    if header_secret != secret:
        raise HTTPException(status_code=403, detail="Secret mismatch")

    update = await request.json()
    message = update.get("message") or {}
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")
    if not chat_id or not text:
        return {"ok": True}  # nothing to process (e.g. a sticker/photo) - ack so Telegram doesn't retry

    from app.core.agent_store import AgentStore
    from app.core.agent import GenesisAgent
    from app.core import telegram_integration as tg

    bot = AgentStore().get_agent(link["agent_id"])
    if not bot:
        return {"ok": True}

    db.log_telegram_event(link["id"], "message_in", text[:200])
    agent = GenesisAgent()
    agent.agent_override = {"model": bot.get("model", "openai/gpt-oss-120b"),
                            "system_prompt": bot.get("system_prompt", ""), "agent_id": link["agent_id"]}
    conversation_id = f"telegram_{link['agent_id']}_{chat_id}"
    # Attributed to the BOT OWNER's account, not the (unauthenticated, no GenesisAI account)
    # Telegram user - the owner's own Groq quota funds their bot's real traffic, same as any
    # other cost their published bot generates.
    resp = agent.process(text, conversation_id=conversation_id, user_id=link["user_id"])

    try:
        tg.send_message(link["bot_token"], chat_id, resp.message)
        db.log_telegram_event(link["id"], "message_out", resp.message[:200])
        db.update_telegram_link_status(link["id"], "connected")
    except tg.TelegramAPIError as e:
        db.log_telegram_event(link["id"], "error", str(e))
        db.update_telegram_link_status(link["id"], "error", str(e))
    return {"ok": True}

@router.post("/api/agents/create")
async def create_agent(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore, AgentConfig
    config = AgentConfig(
        name=data.get("name", "New Agent"),
        system_prompt=data.get("system_prompt", "You are a helpful assistant.")
    )
    config.description = data.get("description", "")
    config.model = data.get("model", "openai/gpt-oss-120b")
    config.temperature = data.get("temperature", 0.5)
    config.max_tokens = data.get("max_tokens", 1000)
    config.tools = data.get("tools", [])
    config.author = user_id
    store = AgentStore()
    store.create_agent(config)
    return config.to_dict()

@router.patch("/api/agents/{agent_id}")
async def update_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    if "linked_model_id" in data and data["linked_model_id"]:
        from app.core.engine.registry import ModelRegistry
        try:
            record = ModelRegistry().get_record(data["linked_model_id"])
        except KeyError:
            raise HTTPException(status_code=400, detail="linked_model_id does not exist in the registry")
        if record.get("metadata", {}).get("author", user_id) != user_id:
            # Older registry records (before ownership tracking existed on models) have no
            # owner_user_id at all - default to allowing rather than retroactively locking
            # a solo dev out of their own pre-existing models.
            raise HTTPException(status_code=403, detail="You can only link a bot to a model you own")
    if "handoff_webhook_url" in data and data["handoff_webhook_url"]:
        from app.core.handoff import is_plausible_webhook_url
        if not is_plausible_webhook_url(data["handoff_webhook_url"]):
            raise HTTPException(status_code=400, detail="handoff_webhook_url must be a valid http(s) URL")
    updated = AgentStore().update_agent(agent_id, data, owner_user_id=user_id)
    if not updated:
        raise HTTPException(status_code=403, detail="Not your bot, or it doesn't exist")
    return updated

@router.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    if not AgentStore().delete_agent(agent_id, owner_user_id=user_id):
        raise HTTPException(status_code=403, detail="Not your bot, or it doesn't exist")
    return {"status": "deleted"}

@router.post("/api/agents/{agent_id}/publish")
async def publish_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    if not AgentStore().publish_agent(agent_id, data.get("price", 0), owner_user_id=user_id):
        raise HTTPException(status_code=403, detail="Not your bot, or it doesn't exist")
    return {"status": "published"}

@router.post("/api/agents/{agent_id}/clone")
async def clone_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    config = AgentStore().clone_agent(agent_id, user_id)
    if not config:
        raise HTTPException(status_code=404, detail="Not found")
    return config.to_dict()

@router.post("/api/agents/{agent_id}/review")
async def review_agent(agent_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.agent_store import AgentStore
    AgentStore().add_review(agent_id, user_id, data.get("rating", 5), data.get("comment", ""))
    return {"status": "reviewed"}

@router.post("/api/bot-builder/generate")
async def bot_builder_generate(data: dict, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    _enforce_groq_quota(user_id)
    from app.core.bot_builder import BotBuilder
    result = BotBuilder(user_id=user_id).build(
        description=data.get("description", ""),
        model=data.get("model"),
        max_iterations=data.get("max_iterations"),
    )
    return result

@router.post("/api/bot-builder/chat")
async def bot_builder_chat(data: dict, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    _enforce_groq_quota(user_id)
    from app.core.bot_builder import BotBuilder
    reply = BotBuilder(user_id=user_id).chat_with_draft(
        spec=data.get("spec", {}),
        message=data.get("message", ""),
        history=data.get("history", []),
        model=data.get("model"),
    )
    return {"response": reply}

@router.post("/api/bot-builder/chat-with-attachments")
async def bot_builder_chat_with_attachments(
    spec: str = Form(...), message: str = Form(""), history: str = Form("[]"),
    model: Optional[str] = Form(None), files: List[UploadFile] = File(default=[]),
    user_id: Optional[str] = Depends(get_current_user_id_optional),
):
    """Preview chat with image/PDF attachments (multipart). See app.core.attachments for the
    limits and for how images vs PDFs reach the model. If the draft bot's model can't see
    images (e.g. openai/gpt-oss-120b) this turn is routed to a vision-capable model and the
    response says so (`routed_to_vision_model` / `model_used`)."""
    import json as _json
    from app.core.attachments import (AttachmentError, MAX_FILE_BYTES, process_attachments,
                                      resolve_model, build_user_content)
    from app.core.bot_builder import BotBuilder
    try:
        spec_obj, history_obj = _json.loads(spec), _json.loads(history or "[]")
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="spec and history must be valid JSON")
    if not isinstance(spec_obj, dict) or not isinstance(history_obj, list):
        raise HTTPException(status_code=400, detail="spec must be an object and history a list")
    if not message.strip() and not files:
        raise HTTPException(status_code=400, detail="Write a message or attach a file")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Too many files: up to 5 per message.")
    raw_files = []
    for f in files:
        # read at most one byte past the cap so an oversized upload is rejected without
        # pulling the whole thing into memory
        raw_files.append((f.filename or "file", await f.read(MAX_FILE_BYTES + 1)))
    try:
        processed = process_attachments(raw_files)
    except AttachmentError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    _enforce_groq_quota(user_id)
    base_model = model or BotBuilder.DEFAULT_MODEL
    model_used, routed = resolve_model(base_model, has_images=bool(processed["images"]))
    reply = BotBuilder(user_id=user_id).chat_with_draft(
        spec=spec_obj, message=build_user_content(message, processed),
        history=history_obj, model=model_used)
    return {"response": reply, "model_used": model_used, "routed_to_vision_model": routed,
            "notes": processed["notes"],
            "attachments": [{"name": i["name"], "kind": "image"} for i in processed["images"]] +
                           [{"name": d["name"], "kind": "document"} for d in processed["documents"]]}

@router.post("/api/bot-builder/revise")
async def bot_builder_revise(data: dict, user_id: Optional[str] = Depends(get_current_user_id_optional)):
    _enforce_groq_quota(user_id)
    from app.core.bot_builder import BotBuilder
    bb = BotBuilder(user_id=user_id)
    spec = bb.revise(spec=data.get("spec", {}), feedback=data.get("feedback", ""), model=data.get("model"))
    test_results = bb.test_draft(spec, model=data.get("model"), skills=spec.get("skills_needed"))
    return {"spec": spec, "test_results": test_results, "success": all(r["ok"] for r in test_results)}

@router.post("/api/bot-builder/publish")
async def bot_builder_publish(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.bot_builder import BotBuilder
    config = BotBuilder(user_id=user_id).publish(
        spec=data.get("spec", {}),
        model=data.get("model"),
        author=user_id,
        price=data.get("price", 0.0),
        publish=bool(data.get("publish", True)),
    )
    return {"status": "published" if data.get("publish", True) else "draft_saved", "agent": config}


# ---- Bot Skills (AI-generated, sandbox-tested network capabilities for a bot) ----
@router.post("/api/skills/create")
async def create_skill(data: dict, user_id: str = Depends(get_current_user_id)):
    """Generates, sandbox-tests, and (if the test passes) saves a new skill for a bot the
    caller owns. This can take several seconds and several real Groq calls (the
    generate -> test -> fix loop) - same cost shape as /api/bot-builder/generate, so it's
    gated by the same quota check."""
    _enforce_groq_quota(user_id)
    from app.core.agent_store import AgentStore
    from app.db.database import GenesisDB
    from app.core.skills.skill_generator import SkillGenerator

    agent_id = data.get("agent_id")
    description = (data.get("description") or "").strip()
    allowed_domains = data.get("allowed_domains") or []
    if not agent_id or not description:
        raise HTTPException(status_code=400, detail="agent_id and description are required")
    if not allowed_domains or not isinstance(allowed_domains, list):
        raise HTTPException(status_code=400, detail=(
            "allowed_domains must be a non-empty list of domains this skill is allowed to "
            "contact, e.g. [\"ebay.com\"] - a skill with no declared domains can't reach the "
            "network at all, and one is required so the sandbox can enforce it."))

    agent = AgentStore().get_agent(agent_id)
    if not agent or agent.get("author") != user_id:
        raise HTTPException(status_code=404, detail="Bot not found, or it doesn't belong to you")

    declared_action_type = data.get("action_type", "read_only")
    if declared_action_type not in ("read_only", "outbound_action"):
        raise HTTPException(status_code=400, detail="action_type must be 'read_only' or 'outbound_action'")

    result = SkillGenerator(user_id=user_id).build(description, allowed_domains)
    from app.core.skills.skill_generator import _detect_write_http_method
    # Defense in depth: a skill that writes (POST/PUT/DELETE/PATCH) is ALWAYS treated as an
    # outbound_action, regardless of what the creator declared - see _detect_write_http_method's
    # docstring. This can only make the gate stricter than requested, never looser.
    action_type = "outbound_action" if (result["code"] and _detect_write_http_method(result["code"])) else declared_action_type

    db = GenesisDB()
    skill = db.create_skill(
        agent_id=agent_id, owner_user_id=user_id,
        name=data.get("name") or description[:60],
        description=description, code=result["code"] or "",
        allowed_domains=allowed_domains, input_schema=data.get("input_schema") or {},
        action_type=action_type)
    db.update_skill_test_result(skill["id"], passed=result["success"],
                                 log=json.dumps(result["log"])[:4000])
    skill = db.get_skill(skill["id"])
    del skill["code"]  # never return the generated source to the client - see get_skill's docstring note
    return {"skill": skill, "test_success": result["success"], "log": result["log"]}

@router.get("/api/skills")
async def list_my_skills(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"skills": GenesisDB().list_skills_for_owner(user_id)}

@router.delete("/api/skills/{skill_id}")
async def delete_skill(skill_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_skill(skill_id, user_id):
        raise HTTPException(status_code=404, detail="Skill not found, or it doesn't belong to you")
    return {"status": "deleted"}


# ---- Scheduled jobs ("Level 3": autonomous, timer-driven skill runs) ----
@router.post("/api/scheduled-jobs/create")
async def create_scheduled_job(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    skill_id = data.get("skill_id")
    if not skill_id:
        raise HTTPException(status_code=400, detail="skill_id is required")
    skill = db.get_skill(skill_id)
    if not skill or skill["owner_user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Skill not found, or it doesn't belong to you")
    if not skill["active"] or not skill["test_passed"]:
        raise HTTPException(status_code=400, detail="Only an active, sandbox-tested skill can be scheduled")

    try:
        job = db.create_scheduled_job(
            skill_id=skill_id, agent_id=skill["agent_id"], owner_user_id=user_id,
            params=data.get("params") or {},
            interval_minutes=int(data.get("interval_minutes", 0)),
            notify_on_change=bool(data.get("notify_on_change", True)))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job": job}

@router.get("/api/scheduled-jobs")
async def list_scheduled_jobs(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"jobs": GenesisDB().list_scheduled_jobs_for_owner(user_id)}

@router.post("/api/scheduled-jobs/{job_id}/pause")
async def pause_scheduled_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().set_scheduled_job_active(job_id, user_id, active=False):
        raise HTTPException(status_code=404, detail="Job not found, or it doesn't belong to you")
    return {"status": "paused"}

@router.post("/api/scheduled-jobs/{job_id}/resume")
async def resume_scheduled_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().set_scheduled_job_active(job_id, user_id, active=True):
        raise HTTPException(status_code=404, detail="Job not found, or it doesn't belong to you")
    return {"status": "resumed"}

@router.delete("/api/scheduled-jobs/{job_id}")
async def delete_scheduled_job(job_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_scheduled_job(job_id, user_id):
        raise HTTPException(status_code=404, detail="Job not found, or it doesn't belong to you")
    return {"status": "deleted"}
