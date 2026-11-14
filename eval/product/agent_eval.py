"""Offline workflow replay и независимый live-eval выбора инструментов."""
import argparse
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@contextmanager
def isolated_workspace(live=False):
    # Сначала читаются только настройки; пользовательские БД и файлы не открываются.
    from app.config import settings
    old = settings.model_dump()
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="genesis-eval-") as work:
        os.chdir(work)
        settings.database_url = ""
        settings.database_auto_migrate = True
        settings.debug = True
        settings.secret_key = "isolated-eval-secret-at-least-32-bytes"
        settings.admin_emails = ""
        settings.allowed_origins = ""
        if not live:
            settings.groq_api_key = "offline-replay"
            settings.openrouter_api_key = settings.gemini_api_key = ""
        try:
            yield settings
        finally:
            os.chdir(cwd)
            for key, value in old.items():
                setattr(settings, key, value)


def summarize(rows):
    durations = sorted(row["latency_ms"] for row in rows)
    return {"cases": len(rows), "passed": sum(row["passed"] for row in rows),
            "success_rate": sum(row["passed"] for row in rows) / len(rows),
            "median_latency_ms": statistics.median(durations),
            "p95_latency_ms": durations[min(len(durations) - 1, int(len(durations) * 0.95))]}


def offline_workflow():
    with isolated_workspace() as settings:
        import numpy as np
        import pandas as pd
        from app.core.agent import GenesisAgent
        from app.core.agent_store import AgentStore, AgentConfig
        from app.core.chat_uploads import save_upload, resolve, build_context
        from app.core.knowledge_base import source_id
        from app.db.database import GenesisDB
        db = GenesisDB()
        user = db.create_user("eval@example.test", "not-a-login-password")["id"]
        db.set_user_plan(user, "pro")
        rng = np.random.RandomState(42)
        age = rng.randint(18, 70, 160)
        frame = pd.DataFrame({"age": age, "income": rng.randint(100, 1000, 160),
                              "churn": ((age > 40) ^ (rng.rand(160) < 0.18)).astype(int)})
        uploaded = save_upload(user, "churn.csv", frame.to_csv(index=False).encode())
        context, _ = build_context(resolve(user, [uploaded["id"]]))
        path = db.dataset_path(user, uploaded["dataset_id"])
        agent = GenesisAgent(db=db)
        agent.groq_key = "offline-replay"
        rows = []

        def step(name, message, decision, check):
            started = time.monotonic()
            completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(decision), tool_calls=None))], usage=None)
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
            with patch("app.core.groq_client.provider_plan", return_value=[("replay", "fixture", client)]):
                response = agent.process(message, "workflow", user_id=user)
            passed = bool(check(response))
            trace = db.get_trace(user, response.data["trace_id"])
            rows.append({"id": name, "passed": passed, "latency_ms": round((time.monotonic() - started) * 1000, 3),
                         "trace_status": trace["status"], "tool": decision.get("tool")})
            return response

        step("upload-analyze", "Analyze the attached dataset" + context,
             {"tool": "analyze_dataset", "args": {"path": path}, "message": "Analyzing"}, lambda r: r.success)
        ambiguous = pd.DataFrame({"target_a": rng.randint(0, 2, 100), "target_b": rng.randint(0, 2, 100)})
        other_upload = save_upload(user, "ambiguous.csv", ambiguous.to_csv(index=False).encode())
        other_path = db.dataset_path(user, other_upload["dataset_id"])
        step("clarify-ambiguous-target", "Train a model on " + other_path,
             {"tool": "train_model", "args": {"path": other_path, "budget": "fast"}},
             lambda r: bool(r.data.get("needs_target")) and not agent.registry.list_models_by_author(user))
        trained = step("train-register", f"Train {path}, target churn", {"tool": "train_model", "args": {
            "path": path, "target_column": "churn", "model_type": "logistic", "budget": "fast"}, "message": "Training"},
            lambda r: r.success and "model_id" in r.data and "metrics" in r.data)
        model_id = trained.data.get("model_id")
        if model_id is None:
            return {"mode": "offline_workflow_replay", "summary": summarize(rows), "cases": rows}
        step("publish", "Publish the trained model", {"tool": "publish_marketplace", "args": {
            "model_id": model_id, "name": "Eval churn", "price_per_call": 0.02}, "message": "Publishing"}, lambda r: r.success)
        bot = AgentConfig("EvalBot", "Answer from documents. Predict churn only from supplied values.")
        bot.author, bot.linked_model_id = user, model_id
        AgentStore(db=db).create_agent(bot)
        agent.agent_override = {"agent_id": bot.id, "model": "openai/gpt-oss-120b", "system_prompt": bot.system_prompt}
        step("ask-missing-inputs", "Predict churn, age 35", {"tool": "predict_with_model", "args": {
            "features": {"age": 35}}, "message": "Predict"}, lambda r: not r.success and "income" in r.data.get("missing_features", []))
        step("predict-version-metrics", "Age 35, income 500", {"tool": "predict_with_model", "args": {
            "features": {"age": 35, "income": 500}}, "message": "Predict"}, lambda r: r.success and r.data.get("model_version") == 1
            and bool(r.data.get("evaluation_metrics")) and bool(r.data.get("limitations")))
        step("reject-nonfinite-input", "Income is unknown", {"tool": "predict_with_model", "args": {
            "features": {"age": "Infinity", "income": 500}}, "message": "Predict"}, lambda r: not r.success)
        step("scoped-bot-blocks-training", "Train another model", {"tool": "train_model", "args": {
            "path": path}, "message": "Training"}, lambda r: "model_id" not in r.data and "Training" not in r.message)
        passage = {"filename": "returns.txt", "content": "Returns are accepted within fourteen days of receiving an unused product."}
        db.add_kb_document(bot.id, user, passage["filename"], passage["content"], [passage["content"]])
        step("answer-with-source", "What is the returns policy?", {"tool": None, "message": "Returns: fourteen days for unused products.",
            "citations": [source_id(passage)]}, lambda r: r.success and r.data.get("grounding") == "cited")
        step("reject-invented-source", "What is the returns policy?", {"tool": None, "message": "Thirty days.",
            "citations": ["kb-invented"]}, lambda r: not r.success and r.data.get("grounding") == "invalid_citation")
        agent.agent_override = None
        before = len(agent.registry.list_models_by_author(user))
        step("reject-invalid-tool-args", "Train", {"tool": "train_model", "args": {"path": path, "budget": "unlimited"}},
             lambda r: not r.success and len(agent.registry.list_models_by_author(user)) == before)
        step("reject-unknown-tool", "Run shell", {"tool": "shell", "args": {"command": "anything"}}, lambda r: not r.success)
        return {"mode": "offline_workflow_replay", "llm_quality_measured": False, "provider_cost_usd": 0,
                "notes": "Only provider responses are replayed; upload, contracts, training, registry, inference, publishing and KB run for real.",
                "summary": summarize(rows), "cases": rows}


def routing_args_match(actual, expected):
    expected = dict(expected)
    if 'horizon' in expected and expected.get('horizon_unit') in {'day', 'week'}:
        units = {'day': 1, 'days': 1, 'week': 7, 'weeks': 7}
        actual_unit = units.get(actual.get('horizon_unit'))
        if actual_unit is None or actual.get('horizon', 0) * actual_unit != expected['horizon'] * units[expected['horizon_unit']]:
            return False
        expected.pop('horizon')
        expected.pop('horizon_unit')
    return all(actual.get(k) == v for k, v in expected.items())


def live_routing(model, interval_seconds=45, protocol="json", case_prefix=None):
    with isolated_workspace(live=True) as settings:
        from app.core.agent import GenesisAgent
        from app.core.tool_contracts import validate_call
        from app.core.tracing import trace_scope
        from app.db.database import GenesisDB
        if not (settings.groq_api_key or settings.openrouter_api_key or settings.gemini_api_key):
            raise RuntimeError("No LLM provider is configured")
        if settings.groq_api_key:
            # Измеряется выбранная модель, без подмены результатом резервной модели.
            settings.llm_provider_order = "groq"
            settings.groq_fallback_models = ""
        settings.groq_connect_timeout_seconds = 5
        settings.groq_read_timeout_seconds = 25
        settings.agent_tool_protocol = protocol
        agent = GenesisAgent(db=GenesisDB(), agent_override={"model": model})
        agent.groq_key = settings.groq_api_key
        cases = json.loads(Path(__file__).with_name("routing_cases.json").read_text())
        cases = [c for c in cases if case_prefix is None or c["id"].startswith(case_prefix)]
        if not cases:
            raise ValueError("No matching routing cases")
        rows = []
        for index, case in enumerate(cases):
            if index:
                time.sleep(max(0, interval_seconds))
            started = time.monotonic()
            with trace_scope("routing_eval") as trace:
                try:
                    decision = agent._call_groq(case["message"])
                    tool, args = validate_call(decision["tool"], decision["args"])
                    # target и target_column — совместимые имена одного параметра.
                    if "target_column" not in args and "target" in args:
                        args["target_column"] = args["target"]
                    passed = tool in case["tools"] and routing_args_match(args, case["args"])
                    row = {"id": case["id"], "passed": passed, "actual_tool": tool, "actual_args": args}
                except Exception as error:
                    row = {"id": case["id"], "passed": False, "error_type": type(error).__name__}
            row.update(latency_ms=round((time.monotonic() - started) * 1000, 3),
                       prompt_tokens=trace["prompt_tokens"], completion_tokens=trace["completion_tokens"],
                       estimated_cost_usd=trace["estimated_cost_usd"], known_estimated_cost_usd=trace["known_estimated_cost_usd"],
                       unpriced_calls=trace["unpriced_calls"], unmetered_attempts=trace["unmetered_attempts"],
                       provider_attempts=[{k: s[k] for k in ("provider", "model", "status") if k in s}
                                          for s in trace["spans"] if s["operation"] == "provider_attempt"])
            rows.append(row)
        from app.core.tool_contracts import tool_definitions
        return {"mode": "live_routing", "model": model, "protocol": protocol, "tools_executed": False, "provider_order": settings.llm_provider_order,
                "dataset_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256((agent.NATIVE_SYSTEM_PROMPT if protocol != "json" else agent.SYSTEM_PROMPT).encode()).hexdigest(),
                "tool_schema_sha256": hashlib.sha256(json.dumps(tool_definitions(), sort_keys=True).encode()).hexdigest(),
                "summary": summarize(rows),
                "estimated_cost_usd": sum(r["estimated_cost_usd"] for r in rows) if all(r["estimated_cost_usd"] is not None for r in rows) else None,
                "known_estimated_cost_usd": sum(r["known_estimated_cost_usd"] for r in rows), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--protocol", choices=("json", "native"), default="json")
    parser.add_argument("--interval-seconds", type=float, default=45, help="Пауза между live-запросами для TPM лимита")
    parser.add_argument("--case-prefix", help="Запустить только выбранную группу live-cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report = offline_workflow() if args.mode == "offline" else live_routing(args.model, args.interval_seconds, args.protocol, args.case_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"]))
    return 0 if report["summary"]["passed"] == report["summary"]["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
