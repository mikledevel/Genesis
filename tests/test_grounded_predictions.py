import pytest
from app.core.engine.predict_utils import build_model_input, build_model_input_from_list, prediction_provenance
from app.core.knowledge_base import answer_sources, source_id, format_context_block


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "NaN", "", [], {}, 10**400])
def test_invalid_numbers_never_reach_model(value):
    record = {"features": ["age"]}
    with pytest.raises((ValueError, TypeError)):
        build_model_input(record, {"age": value})
    with pytest.raises((ValueError, TypeError)):
        build_model_input_from_list(record, [value])


def test_raw_schema_dates_and_unknown_features():
    record = {"features": ["date", "text"], "metadata": {"input_format": "raw_dataframe",
              "input_schema": {"datetime": ["date"], "text": ["text"]}}}
    with pytest.raises(ValueError):
        build_model_input(record, {"date": "yesterdayish", "text": "hello"})
    with pytest.raises(ValueError):
        build_model_input(record, {"date": "2026-11-11", "text": 1})
    with pytest.raises(ValueError):
        build_model_input({"features": ["a"]}, {"a": 1, "extra": 2})
    assert build_model_input(record, {"date": "2026-11-11", "text": "hello"}).shape == (1, 2)
    with pytest.raises(ValueError):
        build_model_input({"features": ["category"], "metadata": {"input_format": "raw_dataframe"}}, {"category": 10**400})


def test_provenance_does_not_invent_metrics_or_causes():
    result = prediction_provenance({"version": 3, "metrics": {"accuracy_test": 0.8, "bad": float("nan")}})
    assert result["model_version"] == 3
    assert result["evaluation_metrics"] == {"accuracy_test": 0.8}
    assert any("causal" in text for text in result["limitations"])


def test_only_retrieved_passages_can_be_cited():
    chunks = [{"filename": 'x<img>.txt', "content": "Returns within 14 days."}]
    key = source_id(chunks[0])
    assert key in format_context_block(chunks)
    assert answer_sources(chunks, [key])[0]["cited"]
    assert not answer_sources(chunks, [])[0]["cited"]
    with pytest.raises(ValueError):
        answer_sources(chunks, ["kb-invented"])


def test_generated_model_is_not_loaded_on_host_to_compute_optional_confidence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch, Mock
    import numpy as np
    from app.core.agent import GenesisAgent
    from app.db.database import GenesisDB
    monkeypatch.chdir(tmp_path)
    record = {"features": ["x"], "target": "label", "metadata": {"built_via": "AICodeGenerator"}}
    host_loader = Mock(side_effect=AssertionError("Must use isolated runner"))
    registry = SimpleNamespace(get_record=lambda _: record, get_model=host_loader)
    agent = GenesisAgent(registry=registry, db=GenesisDB())
    with patch("app.core.agent_store.AgentStore.get_agent", return_value={"linked_model_id": "m"}), \
         patch("app.core.inference.predict_model", return_value=np.array([1])):
        response = agent._do_bot_predict("bot", {"features": {"x": 1}})
    assert response.success and response.data["prediction"] == 1
    assert response.data["confidence"] is None
    host_loader.assert_not_called()
