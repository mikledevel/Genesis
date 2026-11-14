"""Единый trace LLM, retrieval и инструментов. Сообщения и аргументы не сохраняются."""
import contextvars
from contextlib import contextmanager
from functools import wraps
import logging
import time
import uuid

current_trace = contextvars.ContextVar("execution_trace", default=None)
current_span = contextvars.ContextVar("execution_span", default=None)
log = logging.getLogger("genesis.trace")


@contextmanager
def trace_scope(operation, user_id=None, db=None):
    existing = current_trace.get()
    if existing is not None:
        with span(operation):
            yield existing
        return
    from app.core.observability import request_id
    trace = {"id": uuid.uuid4().hex, "request_id": request_id.get(), "operation": operation,
             "created_at": int(time.time()), "status": "ok", "spans": [],
             "prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0.0,
             "unpriced_calls": 0, "unmetered_attempts": 0, "cost_complete": True, "dropped_spans": 0}
    token = current_trace.set(trace)
    started = time.monotonic()
    try:
        yield trace
    except BaseException:
        trace["status"] = "error"
        raise
    finally:
        trace["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        trace["known_estimated_cost_usd"] = trace["estimated_cost_usd"]
        if not trace["cost_complete"]:
            trace["estimated_cost_usd"] = None
        current_trace.reset(token)
        if user_id and user_id != "anonymous":
            try:
                if db is None:
                    from app.db.database import GenesisDB
                    db = GenesisDB()
                db.save_trace(user_id, trace)
            except Exception as error:
                log.warning("trace_persistence_failed type=%s", type(error).__name__)
        log.info("trace_finished id=%s operation=%s status=%s duration_ms=%s", trace["id"], operation,
                 trace["status"], trace["duration_ms"])


@contextmanager
def span(operation, **attributes):
    trace = current_trace.get()
    event = {"id": uuid.uuid4().hex, "parent_id": current_span.get(), "operation": operation,
             "status": "ok", "started_at_ms": round(time.time() * 1000, 3), **attributes}
    token = current_span.set(event["id"])
    started = time.monotonic()
    try:
        yield event
    except BaseException as error:
        event.update(status="error", error_type=type(error).__name__)
        raise
    finally:
        event["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        current_span.reset(token)
        if trace is not None:
            if len(trace["spans"]) < 200:
                trace["spans"].append(event)
            else:
                trace["dropped_spans"] += 1


def record_usage(event, completion, provider, model):
    usage = getattr(completion, "usage", None)
    def count(name):
        value = getattr(usage, name, None)
        return value if type(value) is int and value >= 0 else None
    prompt, output = count("prompt_tokens"), count("completion_tokens")
    cost = None
    if provider == "groq" and prompt is not None and output is not None:
        from app.core.models_catalog import MODEL_CATALOG
        spec = MODEL_CATALOG.get(model)
        if spec and spec.input_price_per_m is not None and spec.output_price_per_m is not None:
            cost = (prompt * spec.input_price_per_m + output * spec.output_price_per_m) / 1_000_000
    event.update(prompt_tokens=prompt, completion_tokens=output, estimated_cost_usd=cost,
                 pricing_source="catalog" if cost is not None else "unknown")
    trace = current_trace.get()
    if trace is not None:
        trace["prompt_tokens"] += prompt or 0
        trace["completion_tokens"] += output or 0
        trace["estimated_cost_usd"] += cost or 0
        trace["unpriced_calls"] += int(cost is None)
        if cost is None:
            trace["cost_complete"] = False


def traced_agent(method):
    @wraps(method)
    def run(self, message, conversation_id=None, user_projects=None, user_id=None, **kwargs):
        from app.core.quota import quota_owner
        token = quota_owner.set(user_id if user_id and user_id != "anonymous" else None)
        try:
            with trace_scope("agent", user_id or "anonymous", self.db) as trace:
                self.retrieved_sources = []
                result = method(self, message, conversation_id, user_projects, user_id, **kwargs)
                if not result.success:
                    trace["status"] = "rejected"
                result.data = {**result.data, "trace_id": trace["id"]}
                return result
        finally:
            quota_owner.reset(token)
    return run
