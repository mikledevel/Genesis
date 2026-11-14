"""Выбор бота с серверными правами, ценой и конфигурацией."""
from fastapi import HTTPException


def resolve_selection(db,user_id,data):
    if not isinstance(data,dict):
        raise HTTPException(status_code=422,detail="Invalid model selection")
    agent_id = data.get("agent_id")
    if agent_id:
        if not isinstance(agent_id,str):
            raise HTTPException(status_code=422,detail="agent_id must be text")
        from app.core.agent_store import AgentStore
        bot = AgentStore(db=db).get_agent(agent_id)
        if not bot or (not bot.get("published") and bot.get("author") != user_id):
            raise HTTPException(status_code=404,detail="Bot not found")
        owner = db.get_session_state(bot.get("author"))
        if not owner or owner[1]:
            raise HTTPException(status_code=404,detail="Bot is unavailable")
        return {"agent_id":bot["id"],"agent_name":bot["name"],"model":bot["model"],
                "system_prompt":bot.get("system_prompt",""),"author":bot["author"],
                "price_per_call":bot.get("price_per_call",0)}
    from app.core.models_catalog import MODEL_CATALOG
    from app.core.agent import GenesisAgent
    model = data.get("model") or "openai/gpt-oss-120b"
    if not isinstance(model,str) or (model not in MODEL_CATALOG and model not in GenesisAgent.MODEL_MAP):
        raise HTTPException(status_code=400,detail="Unsupported model")
    prompt = data.get("system_prompt","")
    if not isinstance(prompt,str) or len(prompt)>100000:
        raise HTTPException(status_code=422,detail="Invalid system prompt")
    return {"model":model,"system_prompt":prompt,"agent_name":str(data.get("agent_name",model))[:200]}
