"""
Tests for ModelRegistry versioning/lineage (family_id, version, parent_model_id) - the
foundation "retraining" is built on. Uses an isolated tempdir storage_dir per test, never
touches the real ./registry.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.core.engine.registry import ModelRegistry


class FakeModel:
    """Minimal picklable stand-in for a real sklearn model - registry.register() only
    needs something pickle can serialize, it doesn't need a real fitted estimator."""
    def predict(self, X):
        return [0] * len(X)


@pytest.fixture
def registry():
    d = tempfile.mkdtemp()
    yield ModelRegistry(storage_dir=d)
    shutil.rmtree(d, ignore_errors=True)


class TestVersioning:
    def test_plain_register_starts_a_new_family_at_version_1(self, registry):
        mid = registry.register(FakeModel(), {"accuracy": 0.8}, "binary_classification",
                                "xgboost", ["age", "salary"], "churn")
        rec = registry.get_record(mid)
        assert rec["version"] == 1
        assert rec["family_id"] == mid  # its own family, since no family_id was passed
        assert rec["parent_model_id"] is None

    def test_registering_with_family_id_increments_version(self, registry):
        mid1 = registry.register(FakeModel(), {"accuracy": 0.8}, "binary_classification",
                                 "xgboost", ["age"], "churn")
        mid2 = registry.register(FakeModel(), {"accuracy": 0.85}, "binary_classification",
                                 "xgboost", ["age", "salary"], "churn",
                                 family_id=mid1, parent_model_id=mid1)
        rec2 = registry.get_record(mid2)
        assert rec2["version"] == 2
        assert rec2["family_id"] == mid1
        assert rec2["parent_model_id"] == mid1
        # the original is untouched
        rec1 = registry.get_record(mid1)
        assert rec1["version"] == 1

    def test_three_generations_increment_correctly(self, registry):
        mid1 = registry.register(FakeModel(), {}, "regression", "linear", ["x"], "y")
        mid2 = registry.register(FakeModel(), {}, "regression", "linear", ["x"], "y",
                                 family_id=mid1, parent_model_id=mid1)
        mid3 = registry.register(FakeModel(), {}, "regression", "linear", ["x"], "y",
                                 family_id=mid1, parent_model_id=mid2)
        assert registry.get_record(mid3)["version"] == 3
        assert registry.get_record(mid3)["parent_model_id"] == mid2

    def test_get_family_history_newest_first_and_complete(self, registry):
        mid1 = registry.register(FakeModel(), {"acc": 0.7}, "binary_classification",
                                 "xgboost", ["a"], "y")
        mid2 = registry.register(FakeModel(), {"acc": 0.8}, "binary_classification",
                                 "xgboost", ["a", "b"], "y", family_id=mid1, parent_model_id=mid1)
        history = registry.get_family_history(mid1)
        assert len(history) == 2
        assert history[0]["model_id"] == mid2  # newest first
        assert history[1]["model_id"] == mid1
        assert [h["version"] for h in history] == [2, 1]

    def test_old_records_without_family_id_treated_as_standalone_family(self, registry):
        """Backward compatibility: registry.json written before versioning existed has no
        family_id/version keys at all - get_family_history must still work on it."""
        mid = registry.register(FakeModel(), {}, "regression", "linear", ["x"], "y")
        # simulate an old-format record by stripping the new fields directly from the index
        del registry._index[mid]["family_id"]
        del registry._index[mid]["version"]
        del registry._index[mid]["parent_model_id"]
        registry._save_index()
        history = registry.get_family_history(mid)
        assert len(history) == 1
        assert history[0]["model_id"] == mid

    def test_unrelated_families_dont_interfere(self, registry):
        family_a = registry.register(FakeModel(), {}, "regression", "linear", ["x"], "y")
        family_b = registry.register(FakeModel(), {}, "binary_classification", "xgboost", ["z"], "w")
        registry.register(FakeModel(), {}, "regression", "linear", ["x"], "y",
                          family_id=family_a, parent_model_id=family_a)
        assert len(registry.get_family_history(family_a)) == 2
        assert len(registry.get_family_history(family_b)) == 1
