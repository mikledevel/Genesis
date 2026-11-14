import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from app.core.tool_contracts import validate_call, parse_completion, tool_definitions
from app.core.agent import GenesisAgent
from app.db.database import GenesisDB


@pytest.mark.parametrize("tool,args", [
    ("shell", {"command": "echo bad"}), ("train_model", {"path": ["x.csv"]}),
    ("train_model", {"budget": "unlimited"}), ("forecast_model", {"horizon": -1}),
    ("forecast_model", {"horizon": True}), ("retrain_model", {"model_id": "m", "new_data": "false"}),
    ("train_model", {"path": "x.csv", "execute": "bad"}),
    ("predict_with_model", {"features": {"age": float("nan")}}),
    (None, {"path": "x.csv"}), ("run_skill", {"skill_id": "x", "params": []}),
])
def test_rejects_invalid_actions(tool, args):
    with pytest.raises((ValueError, TypeError)):
        validate_call(tool, args)


def test_aliases_keep_missing_fields_absent():
    assert validate_call("fit_model", {"path": "x.csv"}) == ("train_model", {"path": "x.csv"})
    assert validate_call("predict", {"features": {"age": 32, "plan": "Pro"}})[0] == "predict_with_model"


def test_native_and_legacy_completion():
    call = SimpleNamespace(function=SimpleNamespace(name="analyze_dataset", arguments='{"path":"x.csv"}'))
    message = SimpleNamespace(tool_calls=[call], content=None)
    assert parse_completion(SimpleNamespace(choices=[SimpleNamespace(message=message)]))["args"] == {"path": "x.csv"}
    message.tool_calls = None
    message.content = json.dumps({"tool": None, "message": "hello"})
    assert parse_completion(SimpleNamespace(choices=[SimpleNamespace(message=message)]))["tool"] is None
    message.content = "[]"
    with pytest.raises(ValueError):
        parse_completion(SimpleNamespace(choices=[SimpleNamespace(message=message)]))


def test_native_reply_and_request_never_combine_json_mode_and_tools(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "agent_tool_protocol", "native")
    monkeypatch.chdir(tmp_path)
    call = SimpleNamespace(function=SimpleNamespace(name="reply", arguments='{"message":"Hello","citations":[]}'))
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))])
    agent = GenesisAgent(db=GenesisDB(str(tmp_path / "test.db")))
    agent.groq_key = "fixture"
    with patch("app.core.groq_client.call_llm", return_value=completion) as llm:
        assert agent._call_groq("hello")["message"] == "Hello"
    assert llm.call_args.kwargs["tool_choice"] == "required"
    assert "response_format" not in llm.call_args.kwargs
    assert validate_call("train_model", {"model_type": "logistic_regression"})[1]["model_type"] == "logistic"


def test_json_protocol_uses_the_same_validated_decision_contract(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "agent_tool_protocol", "json")
    message = SimpleNamespace(content='{"tool":"train_model","args":{"path":"x.csv"},"message":"Training"}', tool_calls=None)
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    agent = GenesisAgent(db=GenesisDB())
    agent.groq_key = "fixture"
    with patch("app.core.groq_client.call_llm", return_value=completion) as llm:
        result = agent._call_groq("train")
    assert result["tool"] == "train_model"
    assert llm.call_args.kwargs["response_format"] == {"type": "json_object"}
    assert "tools" not in llm.call_args.kwargs


def test_tool_validation_failure_has_one_bounded_regeneration():
    import groq
    import httpx
    from app.core.groq_client import _create
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com/test"))
    error = groq.BadRequestError("Bad tool", response=response, body={"error": {"code": "tool_use_failed", "failed_generation": "never execute this"}})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    from unittest.mock import Mock
    client.chat.completions.create = Mock(side_effect=[error, SimpleNamespace(usage=None)])
    _create(client, "groq", "m", [{"role": "user", "content": "hello"}], {"tools": tool_definitions()}, None)
    assert client.chat.completions.create.call_count == 2
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert "never execute this" not in str(messages)
    client.chat.completions.create = Mock(side_effect=error)
    with pytest.raises(groq.BadRequestError):
        _create(client, "groq", "m", [], {"tools": tool_definitions()}, None)
    assert client.chat.completions.create.call_count == 2


def test_published_bot_has_no_builder_tool_schema():
    definitions = tool_definitions(GenesisAgent.BOT_SCOPED_ALLOWED_TOOLS)
    names = {d["function"]["name"] for d in definitions}
    assert "predict_with_model" in names
    assert "train_model" not in names
    for definition in definitions:
        assert definition["function"]["parameters"]["additionalProperties"] is False


def test_validation_precedes_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = GenesisAgent(db=GenesisDB(str(tmp_path / "test.db")))
    with patch.object(agent, "_do_train") as train:
        response = agent._dispatch_tool("train_model", {"budget": "infinite"}, "train", "c", None, False, {})
    train.assert_not_called()
    assert not response.success and response.data["code"] == "invalid_tool_arguments"
