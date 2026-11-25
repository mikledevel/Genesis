from typing import Literal, Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import Response
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


class ForecastPreview(BaseModel):
    dataset_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    date_column: str = Field(min_length=1, max_length=200)
    target_column: str = Field(min_length=1, max_length=200)
    horizon: int = Field(default=14, ge=1, le=365)
    agg: Literal["sum", "mean"] = "sum"

    @model_validator(mode="after")
    def different_columns(self):
        if self.date_column == self.target_column:
            raise ValueError("Choose different date and metric columns")
        return self


def _owned_table(db, user_id, dataset_id):
    from pathlib import Path
    from app.core.paths import is_owned_path
    from app.core.engine.verification import load_table
    path = db.dataset_path(user_id, dataset_id)
    if not path or not is_owned_path(path, user_id) or not db.restore_artifact(path) or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        return load_table(path)
    except (ValueError, OSError):
        raise HTTPException(status_code=422, detail="Could not read the uploaded table") from None


@router.post("/api/forecasts/preview")
def preview_forecast(body: ForecastPreview, user_id: str = Depends(get_current_user_id)):
    from app.core.engine.forecaster import prepare_series
    from app.core import plans
    db = GenesisDB()
    frame = _owned_table(db, user_id, body.dataset_id)
    try:
        series, info = prepare_series(frame, body.date_column, body.target_column, body.agg)
    except ValueError as error:
        if "finite numeric" in str(error):
            raise HTTPException(status_code=422, detail="The metric needs numeric values. For decimal-comma CSV, save as XLSX or export with decimal points.") from None
        raise HTTPException(status_code=422, detail="Choose a readable date column and a numeric metric") from None
    except (KeyError, TypeError):
        raise HTTPException(status_code=422, detail="Choose a readable date column and a numeric metric") from None
    need = max(24, 2 * max(info["season"], 3) + body.horizon)
    horizon_limit = max(1, len(series) // 4)
    warnings = list(info.get("notes", []))
    zero_share = float((series == 0).mean())
    if zero_share >= 0.4:
        warnings.append("Many periods have zero values. Sparse sales require special care; do not use this as an inventory purchase plan.")
    return {"points": len(series), "frequency": info["freq"], "gaps": info.get("gaps", 0),
            "duplicates": info.get("duplicates", 0), "warnings": warnings,
            "eligible": len(series) >= need and body.horizon <= horizon_limit,
            "minimum_points": need, "maximum_horizon": horizon_limit,
            "can_run": plans.gate_message(db, user_id, "server_training") is None,
            "note": "Preview does not start training or charge a fee. Price for a reviewed one-off report is agreed separately."}


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


@router.get("/api/jobs/{job_id}/report")
def job_report(job_id: str, lang: Literal["en", "ru"] = "en", format: Literal["html", "csv"] = "html",
               user_id: str = Depends(get_current_user_id)):
    job = GenesisDB().get_job(user_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["kind"] != "forecast" or job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="A completed forecast is required")
    snapshot = ((job.get("result") or {}).get("data") or {}).get("report_snapshot")
    if not snapshot:
        raise HTTPException(status_code=409, detail="This older forecast has no report snapshot. Create a new forecast to export it.")
    from app.core.forecast_reports import render_snapshot
    try:
        content = render_snapshot(snapshot, lang, format)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(status_code=409, detail="The saved report is incomplete or incompatible") from None
    return Response(content, media_type="text/html" if format == "html" else "text/csv",
                    headers={"Content-Disposition": f'attachment; filename="forecast-{lang}.{format}"',
                             "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                             "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox"})
