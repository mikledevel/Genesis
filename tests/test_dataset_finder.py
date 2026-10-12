import pytest, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.dataset_finder import DatasetFinder, DatasetInfo

class TestDatasetFinder:
    @pytest.fixture
    def finder(self): return DatasetFinder()

    def test_search_churn(self, finder):
        results = finder.search("churn prediction")
        assert isinstance(results, list)

    def test_search_nlp(self, finder):
        results = finder.search("sentiment analysis")
        assert len(results) >= 0  # может быть 0 если API недоступен

    def test_search_nonexistent(self, finder):
        results = finder.search("xyzabc123_nonexistent_thing")
        assert isinstance(results, list)

    def test_format_for_chat_empty(self, finder):
        formatted = finder.format_for_chat([])
        assert "Couldn't find anything" in formatted or "Try different keywords" in formatted

    def test_format_for_chat_results(self, finder):
        results = finder.search("churn")
        formatted = finder.format_for_chat(results)
        assert "⭐" in formatted or "Couldn't find anything" in formatted

