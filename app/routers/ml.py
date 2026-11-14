"""ml routes (split out of app/main.py)."""
from fastapi import APIRouter, File, Form, UploadFile, Query
from fastapi.responses import FileResponse
import os
from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, Header
from app.deps import _enforce_groq_quota
from app.core.auth import get_current_user_id
from app.config import settings

from app.api.contracts import DatasetBuildRequest, RetrainRequest

router = APIRouter()

# ── Registry ──
@router.get("/api/registry/models")
def registry_models(task: Optional[str] = None, user_id: str = Depends(get_current_user_id),
                    limit: int = Query(50,ge=1,le=200), offset: int = Query(0,ge=0)):
    from app.core.engine.registry import ModelRegistry
    models = [m for m in ModelRegistry().list_models_by_author(user_id,limit=100000) if task is None or m.get("task")==task]
    return {"models":models[offset:offset+limit],"total":len(models),"limit":limit,"offset":offset}

@router.get("/api/registry/models/{model_id}")
def registry_model(model_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.engine.registry import ModelRegistry
    try:
        record = ModelRegistry().get_record(model_id)
        if (record.get("metadata") or {}).get("author") != user_id:
            raise HTTPException(status_code=404, detail="Model not found")
        return record
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")

@router.get("/api/registry/models/{model_id}/history")
def registry_model_history(model_id: str, user_id: str = Depends(get_current_user_id)):
    """Every version in this model's lineage (newest first) - see ModelRegistry.register's
    family_id/version and retrain_model, which is what actually grows this list."""
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    try:
        rec = reg.get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    if (rec.get("metadata") or {}).get("author") != user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    family_id = rec.get("family_id", rec.get("model_id"))
    return {"family_id": family_id, "versions": [m for m in reg.get_family_history(family_id) if (m.get("metadata") or {}).get("author") == user_id]}

@router.post("/api/registry/models/{model_id}/retrain")
def registry_retrain_model(model_id: str, data: RetrainRequest, user_id: str = Depends(get_current_user_id)):
    """Retrains an existing registered model on a new/updated dataset, producing a new
    VERSION in the same lineage (see GenesisAgent._do_retrain / ModelRegistry.register)."""
    from app.core.agent import GenesisAgent
    agent = GenesisAgent()
    agent.current_user_id = user_id
    dataset_path = data.get("dataset_path")
    resp = agent._do_retrain(model_id, dataset_path, data.model_dump() if hasattr(data,"model_dump") else data)
    if not resp.success:
        raise HTTPException(status_code=resp.data.get("status",400), detail=resp.message)
    return {"message": resp.message, "data": resp.data}

@router.post("/api/registry/models/{model_id}/improve")
def registry_improve_model(model_id: str, data: RetrainRequest, user_id: str = Depends(get_current_user_id)):
    """Tries to improve a weak model (continue training / bigger or class-balanced refit / other families) and registers a new VERSION only if rows
    no decision used confirm the gain (see app/core/model_improver.py). Body: dataset_path, new_data (bool), budget (fast|balanced|thorough).
    Always 200 with data.improved true/false when the attempt ran; 400 for a missing model/data or a plan without server training."""
    from app.core.agent import GenesisAgent
    agent = GenesisAgent()
    agent.current_user_id = user_id
    resp = agent._do_improve_model(model_id, data.get("dataset_path"), data.model_dump() if hasattr(data,"model_dump") else data)
    if not resp.success:
        raise HTTPException(status_code=resp.data.get("status",400), detail=resp.message)
    return {"message": resp.message, "data": resp.data}

@router.get("/api/registry/best")
def registry_best(task: str = "binary_classification", metric: str = "accuracy", user_id: str = Depends(get_current_user_id)):
    from app.core.engine.registry import ModelRegistry
    models = [m for m in ModelRegistry().list_models_by_author(user_id) if m.get("task") == task and metric in m.get("metrics", {})]
    reverse = metric not in {"rmse", "mae", "mse", "log_loss"}
    models.sort(key=lambda m: m["metrics"][metric], reverse=reverse)
    return {"best": models[0] if models else None}

@router.get("/api/registry/stats")
def registry_stats(user_id: str = Depends(get_current_user_id)):
    from app.core.engine.registry import ModelRegistry
    models = ModelRegistry().list_models_by_author(user_id,limit=100000)
    return {"total_models": len(models), "tasks": {t: sum(m.get("task") == t for m in models) for t in {m.get("task") for m in models}}}

# ── Datasets ──
@router.get("/api/datasets/search")
def dataset_search(q: str):
    from app.core.engine.dataset_finder import DatasetFinder
    return {"query": q, "results": [{"title": d.title, "url": d.url, "source": d.source} for d in DatasetFinder().search(q)]}

# ── Projects ──
@router.get("/api/projects")
def get_projects(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"projects": GenesisDB().get_projects(user_id=user_id)}

@router.post("/api/projects")
def save_projects(data: Dict[str, Any], user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    for p in data.get("projects", []):
        db.add_project(p.get("name",""), p.get("desc",""), p.get("task","other"), p.get("icon","🤖"), user_id=user_id)
    return {"status": "saved"}

# ── Model Predict ──
@router.post("/api/models/{model_id}/predict")
def model_predict(model_id: str, data: dict = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = None,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
    from app.db.database import GenesisDB
    from app.core.engine.registry import ModelRegistry
    from app.core.engine.predict_utils import build_model_input_from_list
    from app.core.marketplace import Marketplace
    from app.core.money import price_cents
    from app.core.usage import reserved_usage, operation_key, fingerprint
    from app.core.inference import predict_model
    db = GenesisDB()
    key = x_api_key or api_key
    info = db.validate_api_key(key) if key else None
    if not info:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    caller = info["user_id"]
    reg = ModelRegistry()
    try:
        record = reg.get_record(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    owner = (record.get("metadata") or {}).get("author")
    listing = None
    if owner != caller:
        listing = next((l for l in Marketplace().listings.values() if l.get("model_id") == model_id and l.get("active")), None)
        if not listing:
            raise HTTPException(status_code=403, detail="This model is private")
    features = (data or {}).get("features")
    if not isinstance(features, list) or not features or len(features) > 10000:
        raise HTTPException(status_code=422, detail="A non-empty features array of at most 10000 values is required")
    try:
        prepared = build_model_input_from_list(record, features)
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=422, detail="Invalid model features") from error
    amount = price_cents(listing["price_per_call"]) if listing else 0
    oid = operation_key(caller, "predict:" + model_id, idempotency_key)
    with reserved_usage(db, caller, amount, api_key=key, author_id=owner, operation_id=oid, request_hash=fingerprint(data), related_id=listing["listing_id"] if listing else model_id) as operation:
        if operation["replayed"]:
            return operation["result"]
        try:
            prediction = predict_model(reg, model_id, prepared)
            response = {"prediction": prediction.tolist(), "model_id": model_id, "charged_cents": amount}
            db.finish_usage(operation["id"], result=response)
            return response
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=502, detail="Prediction failed; the reservation was refunded") from error

# ── Pipeline Builder ──
@router.get("/api/pipeline/templates")
def pipeline_templates():
    return {
        "templates": [
            {"name": "Binary Classification", "blocks": ["load_data","profile","clean","train_xgboost","evaluate","save"]},
            {"name": "Regression", "blocks": ["load_data","profile","clean","train_xgboost_reg","evaluate_reg","save"]},
            {"name": "Quick Classify", "blocks": ["load_data","train_xgboost","evaluate"]},
        ]
    }

@router.post("/api/pipeline/build")
def build_pipeline(data: dict):
    from app.core.codegen.generator import CodeGenerator
    gen = CodeGenerator(None)
    code = gen.generate_from_description(data.get("description", ""))
    path = gen.save_code(code)
    return {"code": code, "code_path": path, "bundle_url": "/api/local-training/bundle"}

@router.post("/api/pipeline/suggest")
def suggest_pipeline_blocks(data: dict, user_id: str = Depends(get_current_user_id)):
    _enforce_groq_quota(user_id)
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
    block_labels = dict(agent.PIPELINE_BLOCKS)
    proposals = [{"name": c.get("name",c.get("ref")),"description": c.get("description", ""),
                  "status": "requires_implementation"} for c in custom_defs]
    blocks = [b for b in blocks if b in agent.PIPELINE_BLOCKS]

    explanation = "\n".join(f"{i+1}. {block_labels.get(b, b)}" for i, b in enumerate(blocks))
    return {"blocks": blocks, "explanation": explanation, "ai_generated": used_ai, "proposals": proposals}

@router.post("/api/model-builder/suggest")
def model_builder_suggest(data: DatasetBuildRequest, user_id: str = Depends(get_current_user_id)):
    """Given just a dataset, suggests what's actually worth modeling - the step a beginner
    with a real idea but no ML background is usually stuck on before they even get to
    'describe the model you want' (see app/core/dataset_advisor.py). No training happens
    here; this is a cheap, fast preview step the UI can show before committing to a full
    build. Does NOT consume Groq quota via _enforce_groq_quota - unlike /generate, a failed
    or low-value suggestion costs the user nothing to ask for."""
    from app.core.dataset_advisor import suggest_model_ideas
    from app.core.paths import is_owned_path
    dataset_path = data.get("dataset_path", "")
    from app.db.database import GenesisDB
    if data.get("dataset_id"):
        dataset_path = GenesisDB().dataset_path(user_id, data.get("dataset_id")) or ""
    if dataset_path and is_owned_path(dataset_path, user_id):
        GenesisDB().restore_artifact(dataset_path)
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Valid dataset_path is required - upload/select a dataset first")
    if not is_owned_path(dataset_path, user_id):
        raise HTTPException(status_code=403, detail="dataset_path must be inside the datasets folder")
    _enforce_groq_quota(user_id)
    groq_key = os.environ.get("GROQ_API_KEY", "") or settings.groq_api_key
    return suggest_model_ideas(dataset_path, groq_key=groq_key)

@router.post("/api/model-builder/generate")
def model_builder_generate(data: DatasetBuildRequest, user_id: str = Depends(get_current_user_id)):
    from app.core import plans
    from app.db.database import GenesisDB
    _limit = plans.gate_message(GenesisDB(), user_id, "server_training")   # server CPU: paid; local training is free
    if _limit:
        raise HTTPException(status_code=403, detail=_limit)
    _enforce_groq_quota(user_id)
    from app.core.model_builder import ModelBuilder
    from app.core.paths import is_owned_path
    dataset_path = data.get("dataset_path", "")
    from app.db.database import GenesisDB
    if data.get("dataset_id"):
        dataset_path = GenesisDB().dataset_path(user_id, data["dataset_id"]) or ""
    if dataset_path and is_owned_path(dataset_path, user_id):
        GenesisDB().restore_artifact(dataset_path)
    if not dataset_path or not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Valid dataset_path is required - upload/select a dataset first")
    if not is_owned_path(dataset_path, user_id):
        raise HTTPException(status_code=403, detail="dataset_path must be inside the datasets folder")

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

    # limited per month and counted when it actually starts: after the plan, the token quota and the path have said yes
    _limit = plans.start_server_job(GenesisDB(), user_id)
    if _limit:
        raise HTTPException(status_code=403, detail=_limit)


    result = ModelBuilder().build(
        description=description,
        dataset_path=dataset_path,
        target_column=target_column,
        user_id=user_id,
    )
    if auto_suggestion:
        result["auto_suggestion_used"] = auto_suggestion
    from app.core import build_records
    build_id = build_records.save(user_id, result)         # the server keeps what the build produced; publish reads it back from here
    if build_id:
        result["build_id"] = build_id
    return result

@router.get("/api/local-training/bundle")
def local_training_bundle():
    """Free on every plan, no login needed: a zip with the platform's own trainer and one command-line script, to
    train on YOUR computer (nothing is uploaded; see app/core/local_training.py). It is fixed open code, not generated."""
    from fastapi.responses import Response
    from app.core.local_training import build_bundle, BUNDLE_NAME
    return Response(content=build_bundle(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{BUNDLE_NAME}"', "Cache-Control": "no-cache"})

@router.get("/api/local-training/command")
def local_training_command(dataset: str = "", target: str = "", budget: str = ""):
    """The ready command for the user's terminal (names are shell-quoted). Pure text, nothing runs anywhere."""
    from app.core.local_training import local_command, local_instructions, BUNDLE_URL
    return {"bundle_url": BUNDLE_URL, "command": local_command(dataset or None, target or None, budget or None),
            "instructions": local_instructions(dataset or None, target or None, budget or None)}

@router.post("/api/local-training/prepare")
def local_training_prepare(data: dict, user_id: str = Depends(get_current_user_id)):
    """Step 1 of bringing a locally trained model back (app/core/local_models.py): the platform cleans an uploaded dataset, keeps a
    hidden test set on the server and hands out only train.csv. `upload_id` is the id from POST /api/agent/upload."""
    from app.core import local_models
    try:
        return local_models.prepare(user_id, str(data.get("upload_id", "")), str(data.get("target", "")))
    except local_models.LocalModelError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

@router.get("/api/local-training/prepared/{prepare_id}/train.csv")
def local_training_train_csv(prepare_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core import local_models
    try:
        path = local_models.train_csv_path(user_id, prepare_id)
    except local_models.LocalModelError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    return FileResponse(str(path), media_type="text/csv", filename="train.csv")

@router.post("/api/local-training/register")
async def local_training_register(prepare_id: str = Form(...), model: UploadFile = File(...),
                                  user_id: str = Depends(get_current_user_id)):
    """Step 3: upload model.pkl. It is read with the restricted unpickler, scored by the platform on the hidden test set and
    registered with a trust verdict. Takes no server-training plan (almost no CPU); saved-model slots still count."""
    from app.core import local_models
    from app.db.database import GenesisDB
    data = await model.read(local_models.MAX_MODEL_BYTES + 1)           # one byte past the cap -> refused without reading it all
    try:
        return local_models.register(user_id, prepare_id, data, GenesisDB())
    except local_models.LocalModelError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))

@router.post("/api/model-builder/publish")
def model_builder_publish(data: dict, user_id: str = Depends(get_current_user_id)):
    """Register (and, unless price_per_call < 0, list) a model this user built. The model file, scores and trust verdict come from the
    server's own record of the build (`build_id` from /generate; older clients may send the `saved_model_path` that /generate returned,
    which is only accepted if it is one of THIS user's builds). `metrics` in the body is ignored. Only verified builds can be listed."""
    from app.core import build_records
    from app.core.model_builder import ModelBuilder
    from app.routers.marketplace import listing_args
    rec = build_records.find(user_id, data.get("build_id"), data.get("saved_model_path"))
    if not rec or not os.path.exists(rec["saved_model_path"]):
        raise HTTPException(status_code=400, detail="No trained model to publish - build one first")
    price = data.get("price_per_call", 0.0)
    try:
        price = float(price)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="price_per_call must be a number")
    v = rec.get("verification")
    if price >= 0:
        if not rec.get("verified") or not v:
            raise HTTPException(status_code=400, detail="Only a build the platform verified on a hidden test set can be listed for sale. "
                                                        "Use price_per_call: -1 to just save it to your models.")
        trust = (v.get("metrics") or {}).get("trust") or {}
        if trust.get("level") == "low":
            why = "; ".join(str(r) for r in (trust.get("reasons") or [])[:3])
            raise HTTPException(status_code=400, detail="The platform's check rates this model as not reliable enough to sell"
                                                        + (f": {why}" if why else "") + ".")
    model_id = rec.get("model_id")
    if not model_id:
        model_id = ModelBuilder().register(rec["saved_model_path"], rec.get("metrics") or {}, author_user_id=user_id, verification=v)
        if not model_id:
            raise HTTPException(status_code=500, detail="Could not register the model")
        build_records.mark_registered(user_id, rec["build_id"], model_id)
    if price >= 0:
        from app.core.engine.registry import ModelRegistry
        from app.core.marketplace import Marketplace
        kw = listing_args(ModelRegistry().get_record(model_id), user_id)
        Marketplace().publish(author=user_id, model_id=model_id, title=data.get("title", "Untitled Model"),
                              description=data.get("description", ""), price_per_call=price, **kw)
    return {"status": "published" if price >= 0 else "saved", "model_id": model_id}


# ---- Custom Blocks ----
@router.get("/api/blocks")
def list_blocks(category: str = None):
    from app.core.custom_blocks import CustomBlockStore
    return {"blocks": CustomBlockStore().list_blocks(category)}

@router.post("/api/blocks/create")
def create_block(data: dict, user_id: str = Depends(get_current_user_id)):
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
def delete_block(block_id: str, user_id: str = Depends(get_current_user_id)):
    from app.core.custom_blocks import CustomBlockStore
    if not CustomBlockStore().delete_block(block_id, user_id):
        raise HTTPException(status_code=404, detail="Block not found, or it doesn't belong to you")
    return {"status": "deleted"}

# ── Prompt Library (versioned) & A/B Testing ──

@router.post("/api/prompts")
def create_prompt(data: dict, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    name = (data.get("name") or "").strip()
    content = (data.get("content") or "").strip()
    if not name or not content:
        raise HTTPException(status_code=400, detail="name and content are required")
    from app.core import plans
    _limit = plans.gate_message(GenesisDB(), user_id, "prompt")
    if _limit:
        raise HTTPException(status_code=403, detail=_limit)
    return GenesisDB().create_prompt(user_id, name, content, data.get("description", ""), data.get("tags"))

@router.get("/api/prompts")
def list_prompts(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"prompts": GenesisDB().list_prompts(user_id)}

@router.get("/api/prompts/{prompt_id}")
def get_prompt(prompt_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    prompt = GenesisDB().get_prompt(prompt_id, user_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt

@router.put("/api/prompts/{prompt_id}")
def update_prompt(prompt_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
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
def revert_prompt(prompt_id: str, version: int, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    result = GenesisDB().revert_prompt(prompt_id, user_id, version)
    if not result:
        raise HTTPException(status_code=404, detail="Prompt or version not found")
    return result

@router.delete("/api/prompts/{prompt_id}")
def delete_prompt(prompt_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_prompt(prompt_id, user_id):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"status": "deleted"}

@router.post("/api/ab-tests")
def create_ab_test(data: dict, user_id: str = Depends(get_current_user_id)):
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
    from app.core import plans
    from app.core.agent_store import AgentStore
    bot = AgentStore().get_agent(agent_id)
    # An A/B test replaces the bot's system prompt for whoever chats with it, so only the bot's own author may start one.
    # 404 (not 403) for someone else's bot: do not confirm that an id exists.
    if not bot or bot.get("author") != user_id:
        raise HTTPException(status_code=404, detail="Bot not found, or it doesn't belong to you")
    _limit = plans.gate_message(db, user_id, "ab_test")
    if _limit:
        raise HTTPException(status_code=403, detail=_limit)
    if db.get_running_ab_test_for_agent(agent_id):
        raise HTTPException(status_code=409, detail="This bot already has a running A/B test - stop it before starting a new one")
    return db.create_ab_test(user_id, agent_id, data.get("name", "Untitled test"), variants)

@router.get("/api/ab-tests")
def list_ab_tests(user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    return {"tests": GenesisDB().list_ab_tests(user_id)}

@router.get("/api/ab-tests/{test_id}")
def get_ab_test(test_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    db = GenesisDB()
    tests = [t for t in db.list_ab_tests(user_id) if t["id"] == test_id]
    if not tests:
        raise HTTPException(status_code=404, detail="Test not found")
    return {**tests[0], "stats": db.get_ab_test_stats(test_id)}

@router.post("/api/ab-tests/{test_id}/stop")
def stop_ab_test(test_id: str, data: dict, user_id: str = Depends(get_current_user_id)):
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
def delete_ab_test(test_id: str, user_id: str = Depends(get_current_user_id)):
    from app.db.database import GenesisDB
    if not GenesisDB().delete_ab_test(test_id, user_id):
        raise HTTPException(status_code=404, detail="Test not found")
    return {"status": "deleted"}
