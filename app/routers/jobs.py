from typing import Literal, Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, model_validator
from app.core.auth import get_current_user_id
from app.db.database import GenesisDB

router = APIRouter()


class JobRequest(BaseModel):
    kind: Literal["model_build","retrain","improve","train","forecast","code_generate","model_create"]
    payload: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self):
        from app.api.contracts import DatasetBuildRequest, RetrainRequest
        from app.core.tool_contracts import TrainArgs, ForecastArgs, CodeArgs, CreateModelArgs
        if self.kind == "model_build":
            data = DatasetBuildRequest.model_validate(self.payload).model_dump(exclude_none=True)
        elif self.kind in {"retrain", "improve"}:
            model_id = self.payload.get("model_id")
            if not isinstance(model_id, str) or not 0 < len(model_id) <= 200:
                raise ValueError("model_id is required")
            data = RetrainRequest.model_validate({k: v for k, v in self.payload.items() if k != "model_id"}).model_dump(exclude_none=True)
            data["model_id"] = model_id
        elif self.kind == "code_generate":
            data = CodeArgs.model_validate({k: v for k, v in self.payload.items() if k != "path"}).model_dump(exclude_none=True)
            data["path"] = self.payload.get("path")
        else:
            contract = CreateModelArgs if self.kind == "model_create" else TrainArgs if self.kind == "train" else ForecastArgs
            payload = dict(self.payload)
            dataset_id = payload.pop("dataset_id", None) if self.kind in {"train", "forecast"} else None
            data = contract.model_validate(payload).model_dump(exclude_none=True)
            if dataset_id is not None:
                import re
                if not isinstance(dataset_id, str) or not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
                    raise ValueError("Invalid dataset_id")
                data["dataset_id"] = dataset_id
        path = data.get("dataset_path") if self.kind in {"model_build", "retrain", "improve"} else data.get("path")
        if not data.get("dataset_id") and (not isinstance(path, str) or not 0 < len(path) <= 4096):
            raise ValueError("A dataset path or owned dataset_id is required")
        self.payload = data
        return self


@router.post("/api/jobs", status_code=202)
def create_job(body: JobRequest, user_id: str = Depends(get_current_user_id),
               idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")):
    from app.core.usage import operation_key
    from app.core import plans
    db = GenesisDB()
    reason = plans.gate_message(db,user_id,"server_training")
    if reason:
        raise HTTPException(status_code=403,detail=reason)
    from app.core.paths import is_owned_path
    data = dict(body.payload)
    path_key = "dataset_path" if body.kind in {"model_build", "retrain", "improve"} else "path"
    if data.get("dataset_id"):
        data[path_key] = db.dataset_path(user_id, data.pop("dataset_id"))
    path = data.get(path_key)
    if not path or not is_owned_path(path, user_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    from pathlib import Path
    if not db.restore_artifact(path) or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if body.kind == "forecast":
        from app.core.agent import GenesisAgent, AgentResponse
        agent = GenesisAgent(db=db)
        agent.current_user_id = user_id
        prepared = agent._prepare_forecast("", data)
        if isinstance(prepared, AgentResponse):
            raise HTTPException(status_code=422, detail={"message": prepared.message, **(prepared.data or {})})
        data = {k: v for k, v in prepared.items() if k != "df" and v is not None}
    if body.kind in {"retrain", "improve"}:
        from app.core.engine.registry import ModelRegistry
        try:
            record = ModelRegistry().get_record(data["model_id"])
        except KeyError:
            raise HTTPException(status_code=404, detail="Model not found")
        if (record.get("metadata") or {}).get("author") != user_id:
            raise HTTPException(status_code=404, detail="Model not found")
    try:
        jid = db.enqueue_job(user_id,body.kind,data,operation_key(user_id,"job",idempotency_key))
    except ValueError as error:
        raise HTTPException(status_code=409,detail=str(error)) from error
    return {"job_id":jid,"status_url":f"/api/jobs/{jid}"}


@router.get("/api/jobs")
def list_jobs(user_id: str = Depends(get_current_user_id)):
    return {"jobs": GenesisDB().list_jobs(user_id)}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str,user_id: str = Depends(get_current_user_id)):
    job = GenesisDB().get_job(user_id,job_id)
    if not job:
        raise HTTPException(status_code=404,detail="Job not found")
    return job


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str,user_id: str = Depends(get_current_user_id)):
    if not GenesisDB().cancel_job(user_id,job_id):
        raise HTTPException(status_code=409,detail="Job not found or already finished")
    return {"status":"cancelled"}
