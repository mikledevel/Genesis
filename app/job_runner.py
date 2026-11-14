"""Дочерний процесс выполняет только фиксированные серверные операции."""
import json
import sys


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (600,600))
    request = json.load(sys.stdin)
    from app.config import settings
    if sys.platform.startswith("linux"):
        memory = settings.training_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
    from app.core.auth import get_current_user_id
    from app.db.database import GenesisDB
    from app.core import progress
    progress.set_callback(lambda event: GenesisDB().set_job_progress(request["id"],event,request.get("attempt")))
    state = GenesisDB().get_session_state(request["user_id"])
    if not state or state[1]:
        raise ValueError("Account is inactive")
    kind, data, user = request["kind"], request["payload"], request["user_id"]
    from app.core.tracing import trace_scope
    from app.core.quota import quota_owner
    quota_owner.set(user)
    with trace_scope("job:" + kind, user) as trace:
        trace["job_id"] = request["id"]
        result = run_operation(kind, data, user)
        if result.get("success") is False:
            trace["status"] = "rejected"
        result["trace_id"] = trace["id"]
    print("GENESIS_JOB_RESULT:" + json.dumps(result,default=str))


def run_operation(kind, data, user):
    from app.routers import ml
    if kind == "model_build":
        from app.api.contracts import DatasetBuildRequest
        result = ml.model_builder_generate(DatasetBuildRequest.model_validate(data), user)
    elif kind == "retrain":
        from app.api.contracts import RetrainRequest
        result = ml.registry_retrain_model(data["model_id"],RetrainRequest.model_validate({k:v for k,v in data.items() if k!="model_id"}),user)
    elif kind == "improve":
        from app.api.contracts import RetrainRequest
        result = ml.registry_improve_model(data["model_id"],RetrainRequest.model_validate({k:v for k,v in data.items() if k!="model_id"}),user)
    elif kind in {"train", "forecast", "code_generate", "model_create"}:
        from app.core.agent import GenesisAgent
        from app.core.paths import is_owned_path
        from app.db.database import GenesisDB
        db = GenesisDB()
        if not is_owned_path(data["path"], user):
            raise ValueError("Dataset is not owned by this account")
        db.restore_artifact(data["path"])
        agent = GenesisAgent(db=db)
        agent.current_user_id = user
        agent.current_dataset = data["path"] if kind in {"code_generate", "model_create"} else None
        tool = {"train": "train_model", "forecast": "forecast_model", "code_generate": "generate_code", "model_create": "create_model"}[kind]
        args = {k: v for k, v in data.items() if kind != "code_generate" or k != "path"}
        response = agent._dispatch_tool(tool, args, data.get("description", ""), "job", None, False, {})
        result = {"success": response.success, "message": response.message, "data": response.data}
    else:
        raise ValueError("Unsupported job kind")
    return result


if __name__ == "__main__":
    main()
