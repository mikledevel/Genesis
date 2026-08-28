import pytest, pandas as pd, numpy as np, tempfile, os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.registry import ModelRegistry, ModelRecord
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

class TestModelRegistry:
    @pytest.fixture
    def registry(self):
        d = tempfile.mkdtemp()
        return ModelRegistry(storage_dir=d)

    @pytest.fixture
    def model(self):
        m = LogisticRegression(max_iter=100, random_state=42)
        X = np.array([[1,2],[2,3],[3,4],[4,5]])
        y = np.array([0,1,0,1])
        m.fit(X, y)
        return m

    def test_register_model(self, registry, model):
        mid = registry.register(
            model=model,
            metrics={"accuracy": 0.85, "f1": 0.83},
            task="binary_classification",
            model_type="logistic",
            features=["f1", "f2"],
            target="churn",
        )
        assert mid in registry._index
        loaded = registry.get_model(mid)
        assert loaded is not None

    def test_find_best(self, registry, model):
        registry.register(model, {"accuracy": 0.75}, "binary_classification", "logistic", ["f1"], "t")
        registry.register(model, {"accuracy": 0.95}, "binary_classification", "logistic", ["f1"], "t")
        best = registry.find_best("binary_classification", "accuracy")
        assert best is not None
        assert best["metrics"]["accuracy"] == 0.95

    def test_list_models(self, registry, model):
        registry.register(model, {"accuracy": 0.8}, "binary_classification", "logistic", ["f1"], "t")
        registry.register(model, {"r2": 0.9}, "regression", "xgboost", ["f1"], "t")
        clf_models = registry.list_models(task="binary_classification")
        assert len(clf_models) == 1

    def test_compare_models(self, registry, model):
        id1 = registry.register(model, {"accuracy": 0.8}, "binary_classification", "logistic", ["f1"], "t")
        id2 = registry.register(model, {"accuracy": 0.9}, "binary_classification", "logistic", ["f1"], "t")
        df = registry.compare_models([id1, id2])
        assert len(df) == 2
        assert "accuracy" in df.columns

    def test_delete_model(self, registry, model):
        mid = registry.register(model, {"accuracy": 0.8}, "binary_classification", "logistic", ["f1"], "t")
        registry.delete_model(mid)
        assert mid not in registry._index

    def test_get_stats(self, registry, model):
        registry.register(model, {"accuracy": 0.8}, "binary_classification", "logistic", ["f1"], "t")
        stats = registry.get_stats()
        assert stats["total_models"] == 1
        assert "binary_classification" in stats["by_task"]
