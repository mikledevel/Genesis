"""Live progress: progress module, engine stage hooks, agent plans and the SSE endpoint."""
import json, os, sys, tempfile, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, pytest
from unittest.mock import patch
import app.main as main_module            # import before any test chdirs (main mounts ./static)
from fastapi.testclient import TestClient
from app.core import progress
from app.core.agent import AgentResponse
from app.core.bot_builder import BotBuilder
from app.core.engine.trainer import DatasetTrainer
from app.deps import get_current_user_id


def _collect():
    events = []
    progress.set_callback(events.append)
    return events


def test_no_callback_is_a_noop():
    progress.set_callback(None)
    progress.plan("x", [("a", "A")]); progress.step("a", "done"); progress.fail("boom")   # must not raise


def test_callback_is_isolated_per_thread():
    mine = _collect(); other = []
    def worker():
        progress.set_callback(other.append); progress.step("t", "done")
    t = threading.Thread(target=worker); t.start(); t.join()
    progress.step("m", "done")
    assert [e["id"] for e in mine] == ["m"] and [e["id"] for e in other] == ["t"]
    progress.set_callback(None)


def test_a_failing_callback_never_breaks_the_work():
    progress.set_callback(lambda e: 1 / 0)
    progress.step("x"); progress.set_callback(None)


def test_trainer_reports_every_stage_in_order():
    rng = np.random.RandomState(0); x = rng.randn(300)
    df = pd.DataFrame({"x": x, "y": (x + rng.randn(300) * .5 > 0).astype(int)})
    seen = []
    DatasetTrainer(tempfile.mkdtemp()).train(df, "y", "binary_classification", "auto",
                                             on_stage=lambda sid, status="active", detail=None: seen.append((sid, status)))
    order = [s for s, st in seen if st == "done"]
    assert order == ["prep", "cv", "fit", "eval", "save"]
    assert ("cv", "active") in seen


def test_trainer_without_on_stage_still_works():
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"x": rng.randn(120), "y": rng.randint(0, 2, 120)})
    assert DatasetTrainer(tempfile.mkdtemp()).train(df, "y", "binary_classification", "logistic").model is not None


def _bb():
    bb = BotBuilder(); bb.groq_key = "k"; return bb


def test_bot_builder_reports_stages_and_fix_loop():
    bb, seen = _bb(), []
    spec = {"name": "B", "system_prompt": "p", "greeting": "g", "test_questions": ["q"], "skills_needed": []}
    results = iter([[{"question": "q", "ok": False}], [{"question": "q", "ok": True}]])
    with patch.object(bb, "generate_spec", return_value=dict(spec)), \
         patch.object(bb, "test_draft", side_effect=lambda *a, **k: next(results)), \
         patch.object(bb, "fix_spec", side_effect=lambda s, *a, **k: s):
        out = bb.build("x", on_stage=lambda sid, status="active", detail=None: seen.append((sid, status)))
    assert out["success"] and out["iterations"] == 1
    assert ("skills", "skipped") in seen and ("fix", "active") in seen and ("fix", "done") in seen
    assert seen[-1] == ("ready", "done")


def test_bot_builder_clean_run_skips_fix():
    bb, seen = _bb(), []
    spec = {"name": "B", "system_prompt": "p", "skills_needed": []}
    with patch.object(bb, "generate_spec", return_value=spec), \
         patch.object(bb, "test_draft", return_value=[{"question": "q", "ok": True}]):
        bb.build("x", on_stage=lambda sid, status="active", detail=None: seen.append((sid, status)))
    assert ("fix", "skipped") in seen


# ---------------------------------------------------------------- SSE endpoint
class _FakeAgent:
    agent_override = None
    def process(self, message, cid, projects=None, user_id=None):
        progress.plan("Training a model", [("load", "Loading"), ("fit", "Fitting")])
        progress.step("load", "active"); progress.step("load", "done", "300 rows")
        progress.step("fit", "done")
        return AgentResponse(message=f"echo: {message}", data={"k": 1}, message_id=7)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main_module.app.dependency_overrides[get_current_user_id] = lambda: "u1"
    with patch("app.routers.chat_agent.get_agent", return_value=_FakeAgent()):
        yield TestClient(main_module.app)
    main_module.app.dependency_overrides.pop(get_current_user_id, None)


def _sse(resp):
    return [json.loads(l[6:]) for l in resp.iter_lines() if l.startswith("data: ")]


def test_stream_emits_progress_then_one_final_event(client):
    with client.stream("POST", "/api/agent/message/stream", json={"message": "hi", "conversation_id": "c1"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        ev = _sse(r)
    assert [e["type"] for e in ev] == ["plan", "step", "step", "step", "final"]
    assert ev[-1]["response"] == "echo: hi" and ev[-1]["message_id"] == 7 and ev[-1]["data"] == {"k": 1}


def test_stream_maps_errors_like_the_json_endpoint(client):
    from app.core.llm_errors import LLMRateLimitError
    class Boom(_FakeAgent):
        def process(self, *a, **k): raise LLMRateLimitError("slow down")
    with patch("app.routers.chat_agent.get_agent", return_value=Boom()):
        with client.stream("POST", "/api/agent/message/stream", json={"message": "hi", "conversation_id": "c2"}) as r:
            ev = _sse(r)
        json_resp = client.post("/api/agent/message", json={"message": "hi", "conversation_id": "c3"})
    assert ev[-1]["type"] == "error" and ev[-1]["status"] == json_resp.status_code == 429
    assert ev[-1]["detail"] == json_resp.json()["detail"]


def test_plain_json_endpoint_unchanged(client):
    r = client.post("/api/agent/message", json={"message": "hello", "conversation_id": "c4"})
    assert r.status_code == 200 and r.json()["response"] == "echo: hello"


def test_stream_requires_auth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main_module.app.dependency_overrides.pop(get_current_user_id, None)
    assert TestClient(main_module.app).post("/api/agent/message/stream", json={"message": "x"}).status_code in (401, 403)
