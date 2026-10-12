"""Per-request progress reporting for long agent actions (training, bot building, ...).

A request that wants live progress (the SSE endpoint /api/agent/message/stream) installs a
callback with set_callback(); the agent and the engines then call plan()/step()/fail() while
they work. Without a callback every call is a no-op, so the normal JSON endpoint, tests and
background jobs are unaffected. A ContextVar (not a global) keeps concurrent requests and
threads from ever seeing each other's callback.

Events (all plain dicts, JSON-serialisable):
  {"type": "plan", "title": str, "steps": [{"id": str, "label": str}, ...]}   announce steps
  {"type": "step", "id": str, "status": "active|done|error|skipped", "detail": str|None}
  {"type": "fail", "detail": str}      the running step failed (UI marks the active step as error)
  {"type": "note", "text": str}        a short remark, e.g. "Groq is unavailable - answered by OpenRouter"
"""
from contextvars import ContextVar
from typing import Callable, Dict, List, Optional, Tuple

_cb: ContextVar[Optional[Callable[[Dict], None]]] = ContextVar("genesis_progress_cb", default=None)


def set_callback(cb: Optional[Callable[[Dict], None]]) -> None:
    _cb.set(cb)


def emit(event: Dict) -> None:
    cb = _cb.get()
    if cb is None:
        return
    try:
        cb(event)
    except Exception:       # progress is best-effort; it must never break the real work
        pass


def plan(title: str, steps: List[Tuple[str, str]]) -> None:
    emit({"type": "plan", "title": title, "steps": [{"id": i, "label": l} for i, l in steps]})


def step(step_id: str, status: str = "active", detail: Optional[str] = None) -> None:
    emit({"type": "step", "id": step_id, "status": status, "detail": detail})


def fail(detail: str = "") -> None:
    emit({"type": "fail", "detail": detail})


def note(text: str) -> None:
    emit({"type": "note", "text": text})


def stage_callback() -> Callable[..., None]:
    """Adapter handed to engines that take `on_stage(step_id, status="active", detail=None)`."""
    return lambda step_id, status="active", detail=None: step(step_id, status, detail)
