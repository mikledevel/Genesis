"""Tests for app/core/dataset_advisor.py - real pandas/DatasetProfiler (both are actually
installed, no need to stub them), with the LLM call mocked via patch("app.core.groq_client.
call_llm") since that's a real network dependency.
"""
import csv
import json
import os
import sys
import tempfile
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pydantic_settings  # noqa
except ImportError:
    fake_ps = types.ModuleType("pydantic_settings")
    class _BaseSettings:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
    fake_ps.BaseSettings = _BaseSettings
    fake_ps.SettingsConfigDict = lambda **kw: kw
    sys.modules["pydantic_settings"] = fake_ps

# app.core.groq_client (imported lazily inside dataset_advisor.suggest_model_ideas) needs
# httpx and groq, neither installed in this sandbox - stub just enough that importing it
# (so patch("app.core.groq_client.call_llm") can resolve the module) doesn't explode. The
# actual call_llm behavior itself is what every test below mocks directly anyway.
for _mod_name in ("httpx", "groq"):
    try:
        __import__(_mod_name)
    except ImportError:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)
if not hasattr(sys.modules["httpx"], "Timeout"):
    sys.modules["httpx"].Timeout = lambda **kw: kw
if not hasattr(sys.modules["groq"], "Groq"):
    sys.modules["groq"].Groq = object
import app.core.groq_client  # noqa: ensures the module is registered under app.core before patch() needs it

from app.core.dataset_advisor import suggest_model_ideas, _profile_to_prompt
from app.core.engine.profiler import DatasetProfiler


def _write_churn_csv() -> str:
    """A small, realistic dataset: a sensible target (churn) plus an id column that should
    never be suggested as a target even though it's also numeric and unique."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["customer_id", "age", "monthly_spend", "tenure_months", "churn"])
        rows = [
            [1001, 34, 59.99, 12, 0], [1002, 51, 19.99, 3, 1], [1003, 29, 89.50, 24, 0],
            [1004, 42, 15.00, 1, 1], [1005, 38, 59.99, 18, 0], [1006, 60, 25.00, 2, 1],
            [1007, 27, 99.99, 36, 0], [1008, 45, 12.50, 1, 1], [1009, 33, 59.99, 15, 0],
            [1010, 55, 20.00, 4, 1],
        ]
        w.writerows(rows)
    return path


def _mock_completion(content_obj):
    completion = MagicMock()
    completion.choices[0].message.content = json.dumps(content_obj)
    return completion


def test_profile_to_prompt_includes_column_stats():
    path = _write_churn_csv()
    try:
        profiler = DatasetProfiler()
        profiler.run_full_profile(path)
        prompt = _profile_to_prompt(profiler.profile)
        assert "churn" in prompt
        assert "customer_id" in prompt
        assert "10 rows" in prompt
    finally:
        os.unlink(path)


def test_no_groq_key_falls_back_to_heuristic_without_calling_llm():
    path = _write_churn_csv()
    try:
        result = suggest_model_ideas(path, groq_key="")
        assert result["success"] is True
        assert result["ai_generated"] is False
        assert len(result["suggestions"]) == 1
    finally:
        os.unlink(path)


def test_ai_suggestion_is_used_when_llm_succeeds():
    path = _write_churn_csv()
    try:
        fake_response = {
            "suggestions": [
                {"target_column": "churn", "task": "binary_classification",
                 "title": "Predict which customers will cancel",
                 "description": "Flags customers likely to churn so you can reach out first.",
                 "confidence": 0.9},
                {"target_column": "monthly_spend", "task": "regression",
                 "title": "Estimate how much a customer will spend",
                 "description": "Predicts monthly spend from their other attributes.",
                 "confidence": 0.4},
            ]
        }
        with patch("app.core.groq_client.call_llm", return_value=_mock_completion(fake_response)):
            result = suggest_model_ideas(path, groq_key="fake-key")
        assert result["success"] is True
        assert result["ai_generated"] is True
        assert len(result["suggestions"]) == 2
        assert result["suggestions"][0]["target_column"] == "churn"
        assert result["suggestions"][0]["confidence"] == 0.9
    finally:
        os.unlink(path)


def test_hallucinated_target_column_is_dropped():
    """The LLM suggesting a column that doesn't actually exist in the dataset must never be
    trusted silently - this is exactly the kind of thing that would otherwise fail loudly
    and confusingly much later, inside the actual training run."""
    path = _write_churn_csv()
    try:
        fake_response = {"suggestions": [
            {"target_column": "this_column_does_not_exist", "task": "regression",
             "title": "x", "description": "x", "confidence": 0.9},
        ]}
        with patch("app.core.groq_client.call_llm", return_value=_mock_completion(fake_response)):
            result = suggest_model_ideas(path, groq_key="fake-key")
        # All suggestions were invalid -> falls back to the heuristic, and is HONEST about
        # that (ai_generated False), not silently passing through a hallucinated column.
        assert result["ai_generated"] is False
        assert result["suggestions"][0]["target_column"] in ("churn", "customer_id")  # real column
    finally:
        os.unlink(path)


def test_mixed_valid_and_hallucinated_columns_keeps_only_the_valid_one():
    path = _write_churn_csv()
    try:
        fake_response = {"suggestions": [
            {"target_column": "made_up_column", "task": "regression", "title": "x", "description": "x", "confidence": 0.9},
            {"target_column": "churn", "task": "binary_classification", "title": "y", "description": "y", "confidence": 0.8},
        ]}
        with patch("app.core.groq_client.call_llm", return_value=_mock_completion(fake_response)):
            result = suggest_model_ideas(path, groq_key="fake-key")
        assert result["ai_generated"] is True
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["target_column"] == "churn"
    finally:
        os.unlink(path)


def test_llm_failure_falls_back_gracefully_not_a_crash():
    path = _write_churn_csv()
    try:
        with patch("app.core.groq_client.call_llm", side_effect=RuntimeError("provider down")):
            result = suggest_model_ideas(path, groq_key="fake-key")
        assert result["success"] is True
        assert result["ai_generated"] is False
        assert len(result["suggestions"]) >= 1
    finally:
        os.unlink(path)


def test_malformed_json_response_falls_back_gracefully():
    path = _write_churn_csv()
    try:
        bad_completion = MagicMock()
        bad_completion.choices[0].message.content = "not valid json at all"
        with patch("app.core.groq_client.call_llm", return_value=bad_completion):
            result = suggest_model_ideas(path, groq_key="fake-key")
        assert result["success"] is True
        assert result["ai_generated"] is False
    finally:
        os.unlink(path)


def test_nonexistent_dataset_path_fails_cleanly():
    result = suggest_model_ideas("/no/such/file.csv", groq_key="fake-key")
    assert result["success"] is False
    assert "error" in result


if __name__ == "__main__":
    passed, failed = 0, 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                import traceback; traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
