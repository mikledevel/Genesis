"""
Tests for the trainer.py audit fixes: stratified splits, string-target label decoding,
capped one-hot encoding, new model types, and real AutoML via model_type="auto".
"""
import pytest, pandas as pd, numpy as np, tempfile, os, sys, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.trainer import DatasetTrainer, _LabelDecodingModel


class TestAuditFixes:
    @pytest.fixture
    def t(self):
        return DatasetTrainer(model_dir=tempfile.mkdtemp())

    @pytest.fixture
    def imbalanced_df(self):
        # 18 of class 0, 2 of class 1 - would previously risk an unstratified test split
        # containing zero examples of the minority class.
        n0, n1 = 18, 2
        return pd.DataFrame({
            "age": list(range(20, 20 + n0)) + [45, 46],
            "salary": list(range(40, 40 + n0)) + [200, 210],
            "churn": [0] * n0 + [1] * n1,
        })

    @pytest.fixture
    def string_target_df(self):
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45, 28, 33, 38, 43, 48],
            "salary": [50, 60, 75, 90, 110, 55, 70, 85, 100, 120],
            "plan": ["basic", "premium", "basic", "basic", "premium",
                     "basic", "premium", "basic", "basic", "premium"],
        })

    @pytest.fixture
    def high_cardinality_df(self):
        # 50 rows, a "city" column with 40 distinct values - must be capped, not
        # one-hot-exploded into 40 columns.
        n = 50
        return pd.DataFrame({
            "age": np.random.RandomState(0).randint(20, 60, n),
            "city": [f"city_{i % 40}" for i in range(n)],
            "churn": np.random.RandomState(1).randint(0, 2, n),
        })

    def test_stratified_split_handles_imbalanced_classes(self, t, imbalanced_df):
        # Should not raise, and should still produce a minority-aware CV score field.
        r = t.train(imbalanced_df, "churn", "binary_classification", "random_forest", cv_folds=3)
        assert "accuracy_test" in r.metrics

    def test_string_target_decodes_back_to_original_labels(self, t, string_target_df):
        r = t.train(string_target_df, "plan", "binary_classification", "logistic")
        assert r.target_classes is not None
        assert set(r.target_classes) == {"basic", "premium"}
        with open(r.model_path, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, _LabelDecodingModel)
        # predict() on the saved artifact must return the original string labels, not ints
        X, _ = t.prepare_data(string_target_df, "plan")
        preds = model.predict(X.values)
        assert set(preds).issubset({"basic", "premium"})

    def test_numeric_target_unchanged_no_wrapper(self, t):
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 28, 33, 38, 43, 48],
                            "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]})
        r = t.train(df, "churn", "binary_classification", "xgboost")
        assert r.target_classes is None
        with open(r.model_path, "rb") as f:
            model = pickle.load(f)
        assert not isinstance(model, _LabelDecodingModel)  # bare estimator, same as before

    def test_high_cardinality_categorical_is_capped(self, t, high_cardinality_df):
        r = t.train(high_cardinality_df, "churn", "binary_classification", "random_forest")
        # age (1) + at most MAX_ONEHOT_CATEGORIES one-hot dummy columns for city (capped,
        # drop_first=True so one fewer than the cap+1 buckets) - must NOT be ~40 columns.
        assert len(r.feature_columns) <= 1 + DatasetTrainer.MAX_ONEHOT_CATEGORIES

    def test_registered_feature_columns_match_saved_model_input_shape(self, t, string_target_df):
        """This is the exact bug found in the audit: registry.register() was fed the raw
        pre-encoding column list instead of the post-encoding one. Verify they now match."""
        r = t.train(string_target_df, "plan", "binary_classification", "random_forest")
        with open(r.model_path, "rb") as f:
            model = pickle.load(f)
        X, _ = t.prepare_data(string_target_df, "plan")
        assert list(X.columns) == r.feature_columns
        # and the saved model actually accepts a vector of exactly that width
        row = X.iloc[[0]].values
        assert row.shape[1] == len(r.feature_columns)
        model.predict(row)  # must not raise a shape-mismatch error

    def test_hist_gradient_boosting_model_type(self, t):
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 28, 33, 38, 43, 48],
                            "salary": [50, 60, 75, 90, 110, 55, 70, 85, 100, 120]})
        r = t.train(df, "salary", "regression", "hist_gradient_boosting")
        assert "r2_test" in r.metrics
        assert r.model_type == "hist_gradient_boosting"

    def test_auto_mode_picks_and_reports_a_real_winner(self, t):
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 28, 33, 38, 43, 48, 26, 31, 36, 41, 46],
                            "salary": [50, 60, 75, 90, 110, 55, 70, 85, 100, 120, 52, 61, 77, 91, 111],
                            "churn": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]})
        r = t.train(df, "churn", "binary_classification", "auto", cv_folds=3)
        assert r.model_type in DatasetTrainer.CANDIDATE_TYPES_CLASSIFICATION
        assert "automl_candidates" in r.metrics
        assert "automl_chosen" in r.metrics
        assert r.metrics["automl_chosen"] == r.model_type
        # every candidate type should have at least attempted to report a score (or None on failure)
        assert set(r.metrics["automl_candidates"].keys()) == set(DatasetTrainer.CANDIDATE_TYPES_CLASSIFICATION)

    def test_auto_mode_regression(self, t):
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 28, 33, 38, 43, 48, 26, 31],
                            "salary": [50, 60, 75, 90, 110, 55, 70, 85, 100, 120, 52, 61]})
        r = t.train(df, "salary", "regression", "auto", cv_folds=3)
        assert r.model_type in DatasetTrainer.CANDIDATE_TYPES_REGRESSION
        assert "r2_test" in r.metrics
