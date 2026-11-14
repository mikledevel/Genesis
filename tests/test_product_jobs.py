import json
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from app.db.database import GenesisDB
from app.routers.jobs import JobRequest, create_job


@pytest.fixture
def account(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = GenesisDB()
    from tests.pro_user import make_pro_user
    make_pro_user(db, "worker")
    path = tmp_path / "datasets" / "sample.csv"
    path.parent.mkdir()
    rng = np.random.RandomState(21)
    age = rng.randint(18, 70, 100)
    pd.DataFrame({"age": age, "income": rng.randint(100, 1000, 100),
                  "churn": (age > 40) ^ (rng.rand(100) < 0.15)}).to_csv(path, index=False)
    return db, "worker", path


def test_job_payloads_and_owner_checked_before_enqueue(account):
    db, user, path = account
    with pytest.raises(ValueError):
        JobRequest(kind="forecast", payload={"path": str(path), "horizon": -1})
    with pytest.raises(ValueError):
        JobRequest(kind="train", payload={"path": str(path), "unexpected": True})
    foreign = path.parent / "uploads" / "someone_else" / "data.csv"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("x,y\n1,2")
    with pytest.raises(HTTPException) as denied:
        create_job(JobRequest(kind="train", payload={"path": str(foreign)}), user, None)
    assert denied.value.status_code == 404
    assert not db.list_jobs(user)


def test_train_queue_claim_execute_and_predict_with_real_model(account):
    from app.core.background import execute_job
    from app.core.engine.registry import ModelRegistry
    from app.core.engine.predict_utils import build_model_input
    from app.core.inference import predict_model
    db, user, path = account
    response = create_job(JobRequest(kind="train", payload={"path": str(path), "target_column": "churn",
                          "model_type": "logistic", "budget": "fast"}), user, "one")
    job = db.claim_job()
    execute_job(db, job)
    result = db.get_job(user, response["job_id"])
    assert result["status"] == "succeeded", result
    data = result["result"]["data"]
    registry = ModelRegistry()
    record = registry.get_record(data["model_id"])
    assert len(predict_model(registry, data["model_id"], build_model_input(record, {"age": 45, "income": 500}))) == 1
    trace = db.get_trace(user, result["result"]["trace_id"])
    assert trace["job_id"] == result["id"] and trace["status"] == "ok"


def test_stale_worker_cannot_overwrite_recovered_job(account):
    db, user, path = account
    jid = db.enqueue_job(user, "train", {"path": str(path)})
    first = db.claim_job()
    with db.write_transaction() as conn:
        conn.execute("UPDATE background_jobs SET lease_until=0 WHERE id=?", (jid,))
    second = db.claim_job()
    db.finish_job(jid, result={"stale": True}, attempt=first["attempt"])
    db.set_job_progress(jid, {"stale": True}, attempt=first["attempt"])
    assert db.get_job(user, jid)["status"] == "running"
    assert db.get_job(user, jid)["progress"] is None
    db.finish_job(jid, result={"fresh": True}, attempt=second["attempt"])
    assert db.get_job(user, jid)["result"] == {"fresh": True}


def test_cancellation_is_terminal_and_shutdown_requeues(account):
    db, user, path = account
    jid = db.enqueue_job(user, "train", {"path": str(path)})
    first = db.claim_job()
    db.release_job(jid, first["attempt"])
    assert db.get_job(user, jid)["status"] == "pending"
    second = db.claim_job()
    assert second["attempt"] == 2
    assert db.cancel_job(user, jid)
    db.finish_job(jid, result={"late": True}, attempt=second["attempt"])
    assert db.get_job(user, jid)["status"] == "cancelled"


def test_chat_queues_once_and_does_not_continue_before_training_finishes(account, monkeypatch):
    from app.core.agent import GenesisAgent
    from app.config import settings
    db, user, path = account
    agent = GenesisAgent(db=db)
    monkeypatch.setattr(settings, "agent_max_steps", 3)
    decision = {"tool": "train_model", "args": {"path": str(path), "target_column": "churn", "budget": "fast"}}
    with patch.object(agent, "_call_llm", return_value=decision) as llm:
        first = agent.process("train", "c", user_id=user, background=True, request_id="request-one")
        second = agent.process("train", "c", user_id=user, background=True, request_id="request-one")
    assert llm.call_count == 2
    assert first.data["job_id"] == second.data["job_id"]
    assert len(db.list_jobs(user)) == 1
    assert db.get_job(user, first.data["job_id"])["status"] == "pending"
    job = db.claim_job()
    db.finish_job(job["id"], result={"success": True, "message": "Trained", "data": {
        "model_id": "measured-model", "metrics": {"accuracy_test": 0.8}}}, attempt=job["attempt"])
    db.add_message("c", "user", "Which model did you train?")
    assert "measured-model" in str(agent._recent_history("c"))
    assert "job succeeded" in str(agent._recent_history("c"))


@pytest.mark.parametrize('message', ['Forecast revenue for 2 weeks', 'Прогноз revenue на 2 недели'])
def test_forecast_queue_resolves_named_column_and_horizon(account, message):
    from app.core.agent import GenesisAgent
    db, user, path = account
    n = 80
    pd.DataFrame({'date': pd.date_range('2025-01-01', periods=n),
                  'revenue': np.sin(np.arange(n)) + 50, 'cost': np.cos(np.arange(n)) + 20}).to_csv(path, index=False)
    agent = GenesisAgent(db=db)
    agent.current_user_id = user
    response = agent._queue_tool('forecast_model', {'path': str(path)}, message)
    assert response.success, response.message
    job = db.claim_job()
    assert job['id'] == response.data['job_id']
    assert job['payload']['target_column'] == 'revenue'
    assert job['payload']['date_column'] == 'date'
    assert job['payload']['horizon'] == 14


def test_ambiguous_forecast_does_not_enqueue(account):
    from app.core.agent import GenesisAgent
    db, user, path = account
    pd.DataFrame({'date': pd.date_range('2025-01-01', periods=80),
                  'revenue': np.sin(np.arange(80)) + 50, 'cost': np.cos(np.arange(80)) + 20}).to_csv(path, index=False)
    agent = GenesisAgent(db=db)
    agent.current_user_id = user
    response = agent._queue_tool('forecast_model', {'path': str(path)}, 'Forecast this dataset')
    assert response.data['needs_target']
    assert not db.list_jobs(user)


def test_multiple_date_columns_require_clarification(account):
    from app.core.agent import GenesisAgent
    db, user, path = account
    pd.DataFrame({'order_date': pd.date_range('2025-01-01', periods=80),
                  'ship_date': pd.date_range('2025-01-02', periods=80),
                  'revenue': np.sin(np.arange(80)) + 50}).to_csv(path, index=False)
    agent = GenesisAgent(db=db)
    agent.current_user_id = user
    response = agent._queue_tool('forecast_model', {'path': str(path)}, 'Forecast revenue')
    assert response.data['needs_date']
    assert not db.list_jobs(user)


@pytest.mark.parametrize('payload', [{}, {'target_column': 'missing'}, {'target_column': 'age', 'date_column': 'income'}])
def test_direct_forecast_api_rejects_invalid_or_ambiguous_series(account, payload):
    db, user, path = account
    with pytest.raises(HTTPException) as error:
        create_job(JobRequest(kind='forecast', payload={'path': str(path), **payload}), user, None)
    assert error.value.status_code == 422
    assert not db.list_jobs(user)


def test_forecast_quoted_target_and_conflicting_llm_parameter(account):
    from app.core.agent import GenesisAgent
    db, user, path = account
    pd.DataFrame({'date': pd.date_range('2025-01-01', periods=80),
                  'net revenue': np.sin(np.arange(80)) + 50, 'cost': np.cos(np.arange(80)) + 20}).to_csv(path, index=False)
    agent = GenesisAgent(db=db)
    agent.current_user_id = user
    request = 'Forecast column "net revenue" for 2 weeks'
    rejected = agent._queue_tool('forecast_model', {'path': str(path), 'target_column': 'cost'}, request)
    assert not rejected.success and not db.list_jobs(user)
    accepted = agent._queue_tool('forecast_model', {'path': str(path)}, request)
    assert accepted.success, accepted.message
    assert db.claim_job()['payload']['target_column'] == 'net revenue'


def test_training_accepts_owned_upload_id_and_rejects_foreign_id(account):
    db, user, path = account
    dataset_id = db.register_dataset(user, str(path), path.stat().st_size)
    request = JobRequest(kind='train', payload={'dataset_id': dataset_id, 'target_column': 'churn'})
    assert create_job(request, user, None)['job_id']
    from tests.pro_user import make_pro_user
    make_pro_user(db, 'other-user')
    with pytest.raises(HTTPException) as error:
        create_job(request, 'other-user', None)
    assert error.value.status_code == 404


def test_followup_training_uses_server_selected_dataset_not_guessed_path(account):
    from app.core.agent import GenesisAgent
    db, user, path = account
    agent = GenesisAgent(db=db)
    agent.current_user_id = user
    assert agent._do_analyze(str(path)).success
    response = agent._queue_tool('train_model', {'path': 'datasets/hallucinated.csv', 'target_column': 'churn'}, 'Train on this file')
    assert response.success, response.message
    assert db.claim_job()['payload']['path'] == str(path)
    rejected = agent._queue_tool('train_model', {'path': 'datasets/missing.csv'}, 'Train datasets/missing.csv')
    assert not rejected.success


def test_analyzed_dataset_context_survives_new_agent(account):
    from app.core.agent import GenesisAgent
    db, user, path = account
    agent = GenesisAgent(db=db)
    agent.current_user_id = user
    result = agent._do_analyze(str(path))
    db.create_chat('restore-dataset', user_id=user)
    db.add_message('restore-dataset', 'assistant', result.message, result.data)
    db.add_message('restore-dataset', 'user', 'Train this file')
    fresh = GenesisAgent(db=db)
    fresh.current_user_id = user
    fresh._recent_history('restore-dataset')
    assert fresh.current_dataset == str(path)
    assert str(path) in fresh._get_effective_system_prompt('Train this file')
