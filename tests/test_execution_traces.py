from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.tracing import trace_scope, span, record_usage, current_trace
from app.db.database import GenesisDB


def test_trace_usage_parentage_and_owner_isolation(tmp_path):
    db = GenesisDB(str(tmp_path / "test.db"))
    with trace_scope("agent", "alice", db) as trace:
        with span("tool", tool="predict") as parent:
            with span("provider_attempt", provider="groq", model="openai/gpt-oss-120b") as event:
                record_usage(event, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20)),
                             "groq", "openai/gpt-oss-120b")
    saved = db.get_trace("alice", trace["id"])
    assert saved["status"] == "ok"
    assert saved["prompt_tokens"] == 100
    assert saved["estimated_cost_usd"] > 0
    assert saved["spans"][0]["parent_id"] == parent["id"]
    assert db.get_trace("bob", trace["id"]) is None
    assert "spans" not in db.list_traces("alice")[0]
    assert current_trace.get() is None


def test_failover_records_failure_actual_model_and_unknown_price():
    from app.core.groq_client import _call_llm
    first = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(TimeoutError("secret")))))
    second = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=None))))
    with trace_scope("test") as trace, patch("app.core.groq_client.provider_plan", return_value=[("groq", "a", first), ("openrouter", "b", second)]):
        _call_llm("test", model="a", messages=[])
    assert [s["model"] for s in trace["spans"]] == ["a", "b"]
    assert trace["spans"][0]["error_type"] == "TimeoutError"
    assert "secret" not in str(trace)
    assert trace["unpriced_calls"] == 1


def test_error_and_parallel_contexts():
    def run(i):
        with trace_scope("task") as trace:
            with span("work", index=i):
                pass
        return trace
    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(run, [1, 2]))
    assert a["id"] != b["id"]
    assert a["spans"][0]["index"] == 1 and b["spans"][0]["index"] == 2
    with pytest.raises(ValueError), trace_scope("fail") as failed:
        raise ValueError("private text")
    assert failed["status"] == "error"


def test_schema_three_migrates_without_losing_data(tmp_path):
    path = str(tmp_path / "test.db")
    db = GenesisDB(path)
    user = db.create_user("a@example.org", "hash")
    with db._connect() as conn:
        conn.execute("DROP TABLE execution_traces")
        conn.execute("DELETE FROM schema_migrations WHERE version=4")
        conn.execute("PRAGMA user_version=3")
        conn.commit()
    migrated = GenesisDB(path)
    assert migrated.get_user_by_id(user["id"])
    assert migrated.list_traces(user["id"]) == []


def test_trace_api_only_returns_current_users_data(tmp_path, monkeypatch):
    from app.core.auth import create_token
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.chdir(tmp_path)
    db = GenesisDB()
    alice = db.create_user("alice@example.test", "hash")["id"]
    bob = db.create_user("bob@example.test", "hash")["id"]
    with trace_scope("test", alice, db) as trace:
        pass
    client = TestClient(app)
    assert client.get("/api/traces").status_code == 401
    bob_headers = {"Authorization": "Bearer " + create_token(bob)}
    assert client.get("/api/traces/" + trace["id"], headers=bob_headers).status_code == 404
    alice_headers = {"Authorization": "Bearer " + create_token(alice)}
    assert client.get("/api/traces/" + trace["id"], headers=alice_headers).json()["operation"] == "test"
    assert len(client.get("/api/traces", headers=alice_headers).json()["traces"]) == 1


def test_unknown_prices_are_not_reported_as_free():
    with trace_scope("test") as trace:
        with span("provider_attempt") as event:
            record_usage(event, SimpleNamespace(usage=None), "openrouter", "unknown")
    assert trace["estimated_cost_usd"] is None
    assert trace["known_estimated_cost_usd"] == 0
    assert trace["unpriced_calls"] == 1 and not trace["cost_complete"]


def test_inference_trace_has_model_version_without_feature_values():
    import numpy as np
    from app.core.inference import predict_model
    model = SimpleNamespace(predict=lambda _: np.array([1]))
    registry = SimpleNamespace(get_record=lambda _: {"version": 4}, get_model=lambda _: model)
    with trace_scope("test") as trace:
        assert predict_model(registry, "m4", [[12345]]).tolist() == [1]
    event = trace["spans"][0]
    assert event["operation"] == "model_inference" and event["model_version"] == 4
    assert "12345" not in str(trace)


def test_nonfinite_prediction_is_an_error_not_invalid_json():
    import numpy as np
    from app.core.inference import predict_model
    registry = SimpleNamespace(get_record=lambda _: {}, get_model=lambda _: SimpleNamespace(predict=lambda _: np.array([float("inf")])))
    with pytest.raises(ValueError, match="non-finite"), trace_scope("test") as trace:
        predict_model(registry, "m", [[1]])
    assert trace["status"] == "error" and trace["spans"][0]["status"] == "error"
