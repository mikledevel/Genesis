"""Каталог аргументов агента: одна схема для LLM и серверной проверки."""
import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DatasetArgs(Arguments):
    path: StrictStr | None = Field(default=None, max_length=4096)


class TrainArgs(DatasetArgs):
    target: StrictStr | None = Field(default=None, max_length=200)
    target_column: StrictStr | None = Field(default=None, max_length=200)
    model_type: Literal["auto", "xgboost", "random_forest", "hist_gradient_boosting", "logistic", "linear"] | None = None
    budget: Literal["fast", "balanced", "thorough"] | None = None
    keep_columns: list[StrictStr] | None = Field(default=None, max_length=500)
    exclude_columns: list[StrictStr] | None = Field(default=None, max_length=500)

    @field_validator("model_type", mode="before")
    @classmethod
    def algorithm_alias(cls, value):
        if not isinstance(value, str):
            return value
        return {"logistic_regression": "logistic", "linear_regression": "linear", "xgb": "xgboost",
                "randomforest": "random_forest"}.get(value, value)


class ForecastArgs(TrainArgs):
    date_column: StrictStr | None = Field(default=None, max_length=200)
    horizon: int | None = Field(default=None, ge=1, le=10000, strict=True)
    horizon_unit: Literal["step", "steps", "day", "days", "week", "weeks", "month", "months", "year", "years"] | None = None
    agg: Literal["sum", "mean", "last"] | None = None


class RetrainArgs(TrainArgs):
    model_id: StrictStr = Field(min_length=1, max_length=200)
    new_data: StrictBool | None = Field(default=None, description="Set true when the user explicitly says these are new or unseen rows.")
    only_if_weak: StrictBool | None = None


class TaskArgs(Arguments):
    task: StrictStr | None = Field(default=None, max_length=200)


class SearchArgs(Arguments):
    query: StrictStr | None = Field(default=None, max_length=10000)


class DescriptionArgs(Arguments):
    description: StrictStr | None = Field(default=None, max_length=10000)


class CodeArgs(DescriptionArgs):
    target_column: StrictStr | None = Field(default=None, max_length=200)


class CreateModelArgs(TrainArgs):
    description: StrictStr | None = Field(default=None, max_length=10000)


class InstallArgs(Arguments):
    number: int = Field(default=1, ge=1, le=100, strict=True)


class MarketplaceArgs(Arguments):
    task: StrictStr | None = Field(default=None, max_length=200)


class PublishArgs(Arguments):
    model_id: StrictStr = Field(min_length=1, max_length=200)
    name: StrictStr | None = Field(default=None, max_length=200)
    description: StrictStr | None = Field(default=None, max_length=10000)
    price_per_call: float | None = Field(default=None, ge=0, le=1000000, strict=True)


class BlockArgs(Arguments):
    name: StrictStr = Field(min_length=1, max_length=200)
    description: StrictStr = Field(min_length=1, max_length=10000)
    category: StrictStr | None = Field(default=None, max_length=100)


class PredictArgs(Arguments):
    features: dict[StrictStr, float | StrictStr | StrictBool | None] = Field(default_factory=dict, max_length=500)


class HandoffArgs(Arguments):
    reason: StrictStr | None = Field(default=None, max_length=2000)


class SkillArgs(Arguments):
    skill_id: StrictStr = Field(min_length=1, max_length=200)
    params: dict[str, Any] = Field(default_factory=dict)


class AddSkillArgs(DescriptionArgs):
    bot_name: StrictStr | None = Field(default=None, max_length=200)
    skill_description: StrictStr | None = Field(default=None, max_length=10000)


class ReplyArgs(Arguments):
    message: StrictStr = Field(min_length=1, max_length=100000)
    citations: list[StrictStr] = Field(default_factory=list, max_length=20)


CONTRACTS = {
    "analyze_dataset": DatasetArgs, "train_model": TrainArgs, "train_local": TrainArgs,
    "forecast_model": ForecastArgs, "retrain_model": RetrainArgs, "improve_model": RetrainArgs,
    "find_best_model": TaskArgs, "list_models": Arguments, "search_datasets": SearchArgs,
    "research_project": SearchArgs, "suggest_pipeline": DescriptionArgs, "generate_code": CodeArgs,
    "install_dataset": InstallArgs, "list_marketplace": MarketplaceArgs, "list_projects": Arguments,
    "publish_marketplace": PublishArgs, "create_custom_block": BlockArgs, "create_bot": DescriptionArgs,
    "create_model": CreateModelArgs, "predict_with_model": PredictArgs, "request_handoff": HandoffArgs,
    "run_skill": SkillArgs, "add_skill": AddSkillArgs,
}
ALIASES = {}
for canonical, aliases in {
    "analyze_dataset": "load_data load_dataset analyze", "train_model": "train fit_model",
    "train_local": "local_train train_locally local_training", "forecast_model": "forecast time_series predict_future",
    "retrain_model": "retrain update_model", "improve_model": "finetune_model fine_tune fine_tune_model improve",
    "find_best_model": "best_model find_best", "list_models": "list show_models",
    "search_datasets": "search find_datasets", "research_project": "research investigate",
    "suggest_pipeline": "suggest_structure propose_pipeline", "generate_code": "codegen generate_pipeline",
    "install_dataset": "install download_dataset", "list_marketplace": "marketplace show_marketplace",
    "list_projects": "projects show_projects", "publish_marketplace": "publish_to_marketplace publish_model",
    "create_custom_block": "create_block new_block", "create_bot": "build_bot make_bot bot_builder",
    "create_model": "build_model train_full_model", "predict_with_model": "predict run_model",
    "request_handoff": "handoff_to_human escalate", "run_skill": "call_skill use_skill",
    "add_skill": "add_capability create_skill",
}.items():
    ALIASES.update({alias: canonical for alias in aliases.split()})


def validate_call(tool, args):
    if tool is None:
        Arguments.model_validate(args)
        return None, {}
    if not isinstance(tool, str) or not isinstance(args, dict):
        raise ValueError("Tool name and arguments have invalid types")
    name = ALIASES.get(tool, tool)
    if name not in CONTRACTS:
        raise ValueError("Unknown tool")
    return name, CONTRACTS[name].model_validate(args).model_dump(exclude_none=True, exclude_unset=True)


class Decision(Arguments):
    tool: StrictStr | None = Field(max_length=128)
    args: dict = Field(default_factory=dict)
    message: StrictStr = Field(default="", max_length=100000)
    citations: list[StrictStr] = Field(default_factory=list, max_length=20)


def parse_completion(completion):
    message = completion.choices[0].message
    calls = getattr(message, "tool_calls", None)
    if isinstance(calls, list) and calls:
        if len(calls) != 1:
            raise ValueError("Only one tool call per step is supported")
        call = calls[0].function
        args = json.loads(call.arguments)
        if call.name == "reply":
            reply = ReplyArgs.model_validate(args)
            raw = {"tool": None, "args": {}, **reply.model_dump()}
        else:
            raw = {"tool": call.name, "args": args, "message": message.content or ""}
    else:
        raw = json.loads(message.content)
    decision = Decision.model_validate(raw)
    return decision.model_dump()


def tool_definitions(allowed=None):
    models = {**CONTRACTS, "reply": ReplyArgs}
    return [{"type": "function", "function": {"name": name,
             "description": name.replace("_", " "), "parameters": model.model_json_schema()}}
            for name, model in models.items() if allowed is None or name in allowed or name == "reply"]
