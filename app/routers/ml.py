"""ml routes (split out of app/main.py)."""
from fastapi import APIRouter
from app.deps import (
    Any,
    Depends,
    Dict,
    HTTPException,
    Header,
    Optional,
    _enforce_groq_quota,
    get_current_user_id,
    os,
    settings
)

router = APIRouter()

# ── Registry ──
@router.get("/api/registry/models")
async def registry_models(task: Optional[str] = None):
    from app.core.engine.registry import ModelRegistry
    return {"models": ModelRegistry().list_models(task=task)}

@router.get("/api/registry/models/{model_id}")
async def registry_model(model_id: str):
    from app.core.engine.registry import ModelRegistry
    try:
        return ModelRegistry().get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")

@router.get("/api/registry/models/{model_id}/history")
async def registry_model_history(model_id: str):
    """Every version in this model's lineage (newest first) - see ModelRegistry.register's
    family_id/version and retrain_model, which is what actually grows this list."""
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    try:
        rec = reg.get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    family_id = rec.get("family_id", rec.get("model_id"))
    return {"family_id": family_id, "versions": reg.get_family_history(family_id)}

@router.post("/api/registry/models/{model_id}/retrain")
async def registry_retrain_model(model_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Retrains an existing registered model on a new/updated dataset, producing a new
    VERSION in the same lineage (see GenesisAgent._do_retrain / ModelRegistry.register)."""
    from app.core.agent import GenesisAgent
    agent = GenesisAgent()
    agent.current_user_id = user_id
    dataset_path = data.get("dataset_path")
    resp = agent._do_retrain(model_id, dataset_path, data)
    if not resp.success:
        raise HTTPException(status_code=400, detail=resp.message)
    return {"message": resp.message, "data": resp.data}

@router.get("/api/registry/best")
async def registry_best(task: str = "binary_classification", metric: str = "accuracy"):
    from app.core.engine.registry import ModelRegistry
    return {"best": ModelRegistry().find_best(task, metric)}

@router.get("/api/registry/stats")
async def registry_stats():
    from app.core.engine.registry import ModelRegistry
    return ModelRegistry().get_stats()

# ── Datasets ──
@router.get("/api/datasets/search")
async def dataset_search(q: str):
    from app.core.engine.dataset_finder import DatasetFinder
    return {"query": q, "results": [{"title": d.title, "url": d.url, "source": d.source} for d in DatasetFinder().search(q)]}

# ── Projects ──
@router.get("/api/projects")
async def get_projects(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"projects": GenesisDB().get_projects(user_id=user_id)}

@router.post("/api/projects")
async def save_projects(data: Dict[str, Any], user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    for p in data.get("projects", []):
        db.add_project(p.get("name",""), p.get("desc",""), p.get("task","other"), p.get("icon","🤖"), user_id=user_id)
    return {"status": "saved"}

# ── Model Predict ──
@router.post("/api/models/{model_id}/predict")
async def model_predict(
    model_id: str,
    data: dict = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = None,  # deprecated: query-param fallback, logs/URLs can leak this - prefer X-API-Key header
):
    from app.db.database import GenesisDB
    db = GenesisDB()
    key = x_api_key or api_key
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key - send it as the X-API-Key header")
    key_info = db.validate_api_key(key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    try:
        record = reg.get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")

    # Authorization: an API key lets its owner call their OWN registered models freely.
    # A model with no "author" in metadata is a legacy/platform model with no owner to
    # protect, so it stays open (unchanged prior behavior). A model owned by someone ELSE
    # is only reachable this way if it has an active, priced marketplace listing - in which
    # case it's billed exactly like /api/marketplace/{listing_id}/use, so an API key can no
    # longer be used to get someone else's paid model for free by calling this route instead
    # of the metered marketplace endpoint. If it has no listing, it's private - denied.
    record_author = (record.get("metadata") or {}).get("author")
    caller_id = key_info.get("user_id")
    listing = None
    price_cents = 0
    if record_author and record_author != caller_id:
        from app.core.marketplace import Marketplace
        mkt = Marketplace()
        listing = next((l for l in mkt.listings.values()
                         if l.get("model_id") == model_id and l.get("active")), None)
        if not listing:
            raise HTTPException(status_code=403, detail="This model is private and you don't have access to it")
        price_cents = round(listing.get("price_per_call", 0) * 100)
        if price_cents > 0:
            if not caller_id:
                raise HTTPException(status_code=401, detail="This is a paid model - use an API key tied to a logged-in account")
            charge = db.charge_for_usage(
                buyer_user_id=caller_id, amount_cents=price_cents,
                description=f"Predict via API: {listing.get('title', model_id)}",
                author_user_id=record_author, platform_fee_pct=25.0, related_id=model_id)
            if not charge["ok"]:
                raise HTTPException(status_code=402, detail=(
                    f"Insufficient balance - this model costs ${price_cents/100:.4f}/call, "
                    f"you have ${charge['balance_cents']/100:.2f}. Add funds to continue."))

    if not db.use_api_key(key, model_id):
        raise HTTPException(status_code=429, detail="API call limit exceeded")
    try:
        model = reg.get_model(model_id)
        features = data.get("features", []) if data else []
        if features:
            from app.core.engine.predict_utils import build_model_input_from_list
            prediction = model.predict(build_model_input_from_list(record, features))
            if listing is not None:
                from app.core.marketplace import Marketplace
                Marketplace().use_model(listing["listing_id"], caller_id)
            return {"prediction": prediction.tolist(), "model_id": model_id}
        return {"message": "Model loaded", "model_id": model_id, "metrics": record.get("metrics", {})}
    except HTTPException:
        raise
    except Exception as e:
        # Mirrors /api/marketplace/{listing_id}/use: if we already charged for a paid model
        # and the prediction itself then failed, refund both sides rather than keep a charge
        # for a call that produced no result.
        if price_cents > 0 and listing is not None:
            db.add_transaction(caller_id, "usage_refund", price_cents,
                               f"Refund: {listing.get('title', model_id)} (prediction error)", related_id=model_id)
            author_share = round(price_cents * 0.75)
            if author_share > 0:
                db.add_transaction(record_author, "usage_earning_reversal", -author_share,
                                   f"Reversal: {listing.get('title', model_id)} (prediction error)", related_id=model_id)
        raise HTTPException(status_code=500, detail=str(e))

# ── Pipeline Builder ──
@router.get("/api/pipeline/templates")
async def pipeline_templates():
    return {
        "templates": [
            {"name": "Binary Classification", "blocks": ["load_data","profile","clean","train_xgboost","evaluate","save"]},
            {"name": "Regression", "blocks": ["load_data","profile","clean","train_xgboost_reg","evaluate_reg","save"]},
            {"name": "Quick Classify", "blocks": ["load_data","train_xgboost","evaluate"]},
        ]
    }

@router.post("/api/pipeline/build")
async def build_pipeline(data: dict):
    from app.core.codegen.generator import CodeGenerator
    gen = CodeGenerator(None)
    code = gen.generate_from_description(data.get("description", ""))
    path = gen.save_code(code)
    return {"code": code, "code_path": path}

@router.post("/api/pipeline/suggest")
async def suggest_pipeline_blocks(data: dict):
    """Used by the Pipeline Builder's 'build from prompt' box - asks the real LLM to pick
    blocks from the fixed catalog, instead of doing naive keyword matching in JS.

    BUGFIX: _ai_build_pipeline returns a (blocks, custom_defs) TUPLE on success, or None
    on failure (see its docstring) - the same contract GenesisAgent._do_suggest_pipeline
    already unpacks correctly. This route used to do
    `agent._ai_build_pipeline(description) or agent._keyword_build_pipeline(description)`,
    which - whenever the AI call actually succeeded - assigned that raw 2-tuple to `blocks`
    instead of the block-id list inside it. The very next line then called
    `PIPELINE_BLOCKS.get(b, b)` on `b = <that list>`, which raises "unhashable type: 'list'"
    since a list can't be a dict key - an unhandled 500 every time the AI path succeeded.
    It also called _ai_build_pipeline a SECOND time just to compute `used_ai` (a redundant
    real Groq call/cost), and never materialized any AI-requested custom blocks the way
    _do_suggest_pipeline does. Fixed by unpacking the tuple once, like the chat path does.
    """
    from app.core.agent import GenesisAgent
    from app.core.custom_blocks import CustomBlockStore
    description = data.get("description", "")
    if not description:
        return {"blocks": [], "explanation": "Опишите, что должен делать пайплайн."}
    agent = GenesisAgent()
    ai_result = agent._ai_build_pipeline(description)
    used_ai = ai_result is not None
    if ai_result:
        blocks, custom_defs = ai_result
    else:
        blocks, custom_defs = agent._keyword_build_pipeline(description), []

    # Materialize any AI-requested custom blocks into real CustomBlock rows and splice
    # their real ids into the pipeline in place of the "custom:xxx" placeholder - same
    # step _do_suggest_pipeline performs for the chat-driven path.
    store = CustomBlockStore()
    block_labels = dict(agent.PIPELINE_BLOCKS)
    ref_to_real_id = {}
    for cdef in custom_defs:
        ref = cdef.get("ref")
        name = cdef.get("name", ref)
        desc = cdef.get("description", "")
        if not ref or ref in ref_to_real_id:
            continue
        created = store.create(
            name=name,
            code=f"# TODO: implement '{name}'\n# {desc}\ndef run(data):\n    raise NotImplementedError({desc!r})",
            icon="🔧", category="custom",
        )
        ref_to_real_id[ref] = created["id"]
        block_labels[created["id"]] = f"{name} (custom) - {desc}"
    blocks = [ref_to_real_id.get(b, b) for b in blocks]

    explanation = "\n".join(f"{i+1}. {block_labels.get(b, b)}" for i, b in enumerate(blocks))
    return {"blocks": blocks, "explanation": explanation, "ai_generated": used_ai}

@router.post("/api/model-builder/suggest")
async def model_builder_suggest(data: dict, user_id: str = Depends(get_current_user_id)):
    """Given just a dataset, suggests what's actually worth modeling - the step a beginner
    with a real idea but no ML background is usually stuck on before they even get to
    'describe the model you want' (see app/core/dataset_advisor.py). No training happens
    here; this is a cheap, fast preview step the UI can show before committing to a full
    build. Does NOT consume Groq quota via _enforce_groq_quota - unlike /generate, a failed
    or low-value suggestion costs the user nothing to ask for."""
    from app.core.dataset_advisor import suggest_model_ideas
    dataset_path = data.get("dataset_path", "")
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Valid dataset_path is required - upload/select a dataset first")
    groq_key = os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key
    return suggest_model_ideas(dataset_path, groq_key=groq_key)

@router.post("/api/model-builder/generate")
async def model_builder_generate(data: dict, user_id: str = Depends(get_current_user_id)):
    _enforce_groq_quota(user_id)
    from app.core.model_builder import ModelBuilder
    dataset_path = data.get("dataset_path", "")
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Valid dataset_path is required - upload/select a dataset first")

    description, target_column = data.get("description", ""), data.get("target_column")
    auto_suggestion = None
    if not description or not target_column:
        # Fully automated path ("just hand me a CSV") - auto-pick the best suggestion
        # instead of making description/target_column hard requirements. A caller who DID
        # provide both skips this entirely and nothing changes from before.
        from app.core.dataset_advisor import suggest_model_ideas
        groq_key = os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key
        advice = suggest_model_ideas(dataset_path, groq_key=groq_key)
        if not advice.get("suggestions"):
            raise HTTPException(status_code=400, detail=(
                "Couldn't find a sensible modeling target in this dataset automatically - "
                "please provide 'description' and 'target_column' explicitly. " +
                (advice.get("note") or "")))
        auto_suggestion = advice["suggestions"][0]
        target_column = target_column or auto_suggestion["target_column"]
        description = description or auto_suggestion["description"]

    result = ModelBuilder().build(
        description=description,
        dataset_path=dataset_path,
        target_column=target_column,
        user_id=user_id,
    )
    if auto_suggestion:
        result["auto_suggestion_used"] = auto_suggestion
    return result

@router.post("/api/model-builder/publish")
async def model_builder_publish(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.model_builder import ModelBuilder
    saved_model_path = data.get("saved_model_path")
    if not saved_model_path or not os.path.exists(saved_model_path):
        raise HTTPException(status_code=400, detail="No trained model to publish - build one first")
    model_id = ModelBuilder().register(saved_model_path, data.get("metrics", {}), author_user_id=user_id)
    if not model_id:
        raise HTTPException(status_code=500, detail="Could not register the model")
    price = data.get("price_per_call", 0.0)
    if price >= 0:
        from app.core.marketplace import Marketplace
        Marketplace().publish(
            author=user_id, model_id=model_id,
            title=data.get("title", "Untitled Model"), description=data.get("description", ""),
            task=data.get("metrics", {}).get("task", "binary_classification"),
            model_type=data.get("metrics", {}).get("model_type", "unknown"),
            metrics={k: v for k, v in data.get("metrics", {}).items() if isinstance(v, (int, float))},
            price_per_call=price,
        )
    return {"status": "published", "model_id": model_id}


# ---- Custom Blocks ----
@router.get("/api/blocks")
async def list_blocks(category: str = None):
    from app.core.custom_blocks import CustomBlockStore
    return {"blocks": CustomBlockStore().list_blocks(category)}

@router.post("/api/blocks/create")
async def create_block(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.core.custom_blocks import CustomBlockStore
    block = CustomBlockStore().create(
        name=data.get("name","New Block"),
        code=data.get("code","# Python code"),
        icon=data.get("icon","🔧"),
        category=data.get("category","custom"),
        author=user_id,
    )
    return block

@router.delete("/api/blocks/{block_id}")
async def delete_block(block_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.custom_blocks import CustomBlockStore
    if not CustomBlockStore().delete_block(block_id, user_id):
        raise HTTPException(status_code=404, detail="Block not found, or it doesn't belong to you")
    return {"status": "deleted"}

# ── Prompt Library (versioned) & A/B Testing ──

@router.post("/api/prompts")
async def create_prompt(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    name = (data.get("name") or "").strip()
    content = (data.get("content") or "").strip()
    if not name or not content:
        raise HTTPException(status_code=400, detail="name and content are required")
    return GenesisDB().create_prompt(user_id, name, content, data.get("description", ""), data.get("tags"))

@router.get("/api/prompts")
async def list_prompts(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"prompts": GenesisDB().list_prompts(user_id)}

@router.get("/api/prompts/{prompt_id}")
async def get_prompt(prompt_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    prompt = GenesisDB().get_prompt(prompt_id, user_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt

@router.put("/api/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Creates a NEW version rather than overwriting - see add_prompt_version."""
    from app.db.database import GenesisDB
    content = (data.get("content") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required")
    result = GenesisDB().add_prompt_version(prompt_id, user_id, content, data.get("change_note", ""))
    if not result:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return result

@router.post("/api/prompts/{prompt_id}/revert/{version}")
async def revert_prompt(prompt_id: str, version: int, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    result = GenesisDB().revert_prompt(prompt_id, user_id, version)
    if not result:
        raise HTTPException(status_code=404, detail="Prompt or version not found")
    return result

@router.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_prompt(prompt_id, user_id):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "deleted"}

@router.post("/api/ab-tests")
async def create_ab_test(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    agent_id = data.get("agent_id")
    variants = data.get("variants") or []
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    if len(variants) < 2:
        raise HTTPException(status_code=400, detail="At least 2 variants are required for an A/B test")
    for v in variants:
        if not (v.get("label") and (v.get("system_prompt") or "").strip()):
            raise HTTPException(status_code=400, detail="Each variant needs a label and non-empty system_prompt")
    if db.get_running_ab_test_for_agent(agent_id):
        raise HTTPException(status_code=409, detail="This bot already has a running A/B test - stop it before starting a new one")
    return db.create_ab_test(user_id, agent_id, data.get("name", "Untitled test"), variants)

@router.get("/api/ab-tests")
async def list_ab_tests(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"tests": GenesisDB().list_ab_tests(user_id)}

@router.get("/api/ab-tests/{test_id}")
async def get_ab_test(test_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    tests = [t for t in db.list_ab_tests(user_id) if t["id"] == test_id]
    if not tests:
        raise HTTPException(status_code=404, detail="Test not found")
    return {**tests[0], "stats": db.get_ab_test_stats(test_id)}

@router.post("/api/ab-tests/{test_id}/stop")
async def stop_ab_test(test_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
    """Stopping optionally promotes a winning variant's prompt to be the bot's live,
    permanent system_prompt (via AgentStore), so the test's result actually sticks."""
    from app.db.database import GenesisDB
    from app.core.agent_store import AgentStore
    db = GenesisDB()
    winner_variant_id = data.get("winner_variant_id")
    tests = [t for t in db.list_ab_tests(user_id) if t["id"] == test_id]
    if not tests:
        raise HTTPException(status_code=404, detail="Test not found")
    if winner_variant_id:
        stats = db.get_ab_test_stats(test_id)
        if winner_variant_id not in stats:
            raise HTTPException(status_code=400, detail="winner_variant_id is not a variant of this test")
        run = db.get_running_ab_test_for_agent(tests[0]["agent_id"])
        variant_prompt = None
        if run:
            match = next((v for v in run["variants"] if v["id"] == winner_variant_id), None)
            variant_prompt = match["system_prompt"] if match else None
        if variant_prompt:
            # update_agent itself enforces ownership (returns None if not owned by user_id) -
            # no separate author check needed here.
            AgentStore().update_agent(tests[0]["agent_id"], {"system_prompt": variant_prompt}, user_id)
    if not db.stop_ab_test(test_id, user_id, winner_variant_id):
        raise HTTPException(status_code=404, detail="Test not found")
    return {"status": "stopped", "winner_variant_id": winner_variant_id}

@router.delete("/api/ab-tests/{test_id}")
async def delete_ab_test(test_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_ab_test(test_id, user_id):
        raise HTTPException(status_code=404, detail="Test not found")
    return {"status": "deleted"}
