"""
DatasetTrainer - fits a real sklearn/xgboost model on a profiled dataset.

Audit fixes in this version (previously: unstratified splits could crash/mislead on
imbalanced classes, string targets weren't handled, categorical features were
one-hot-encoded with no cardinality cap, scale-sensitive linear models weren't scaled,
and the registry was fed the WRONG feature-column list - see registry.py callers):

  1. Stratified train/test splits for classification (falls back to a plain split only
     if a class has fewer than 2 members, since sklearn can't stratify that).
  2. Non-numeric classification targets are LabelEncoder'd internally, and the saved
     model artifact is wrapped in _LabelDecodingModel so calling .predict() on it still
     returns the ORIGINAL string labels - every existing `model.predict(...)` call site
     keeps working unchanged whether the target was numeric or a string column.
  3. Categorical feature columns are capped at MAX_ONEHOT_CATEGORIES most-frequent
     values before one-hot encoding (rest bucketed into "__other__") to avoid a single
     high-cardinality column blowing up the feature space.
  4. Linear/logistic models are wrapped in a StandardScaler Pipeline (scale-sensitive;
     tree-based models are not, so they're left unscaled).
  5. Two more model types: "hist_gradient_boosting" (sklearn-native, handles missing
     values natively, no new dependency) alongside the existing xgboost/random_forest.
  6. model_type="auto" now actually IS AutoML: it cross-validates every candidate type
     and fits the winner, instead of the old find_best() which only ever looked up
     already-registered models.
  7. ModelResult now exposes feature_columns - the ACTUAL post-encoding columns the
     saved model expects, in order. Callers (see agent.py _do_train) must register
     THIS list with ModelRegistry, not the raw pre-encoding profiler columns, or the
     predict-time feature vector a user builds won't match what the model expects.
"""
import pandas as pd, numpy as np, pickle
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class _LabelDecodingModel:
    """Wraps a fitted classifier trained on LabelEncoder-encoded integer targets so
    .predict() on the saved/pickled artifact returns the ORIGINAL string labels, not
    opaque integers. Kept as a plain top-level class (not a closure) so it pickles and
    unpickles cleanly via its module path."""
    def __init__(self, estimator, classes: List[str]):
        self.estimator = estimator
        self.classes = list(classes)

    def predict(self, X):
        idx = np.asarray(self.estimator.predict(X)).astype(int)
        return np.array([self.classes[i] if 0 <= i < len(self.classes) else str(i) for i in idx])

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        raise AttributeError("Underlying estimator has no predict_proba")


class ModelResult:
    def __init__(self):
        self.model_type = self.task = self.model_path = self.creation_time = ""
        self.metrics: Dict = {}
        self.feature_importance: Dict = {}
        self.feature_columns: List = []       # actual post-encoding columns the model expects
        self.target_classes: Optional[List] = None  # original class labels, if target was string


class DatasetTrainer:
    MAX_ONEHOT_CATEGORIES = 20  # per-column cap before one-hot; rest bucketed as "__other__"
    CANDIDATE_TYPES_CLASSIFICATION = ["xgboost", "random_forest", "hist_gradient_boosting", "logistic"]
    CANDIDATE_TYPES_REGRESSION = ["xgboost", "random_forest", "hist_gradient_boosting", "linear"]

    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir); self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result = ModelResult()

    def prepare_data(self, df, target_col, max_categories: Optional[int] = None):
        max_categories = self.MAX_ONEHOT_CATEGORIES if max_categories is None else max_categories
        df = df.dropna(subset=[target_col]).copy()
        y = df[target_col]; X = df.drop(columns=[target_col])
        for c in X.select_dtypes(include=[np.number]).columns:
            X[c] = X[c].fillna(X[c].median())
        cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)
        for c in cat_cols:
            X[c] = X[c].fillna("__missing__")
            top = X[c].value_counts().nlargest(max_categories).index
            X[c] = X[c].where(X[c].isin(top), "__other__")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        return X, y

    def _build_estimator(self, task: str, model_type: str):
        """Returns a fittable estimator (or scaler+model Pipeline for scale-sensitive types)."""
        is_clf = "classification" in task
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=42))])
        if model_type == "linear":
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier(n_estimators=200, random_state=42) if is_clf else RandomForestRegressor(n_estimators=200, random_state=42)
        if model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            return HistGradientBoostingClassifier(random_state=42) if is_clf else HistGradientBoostingRegressor(random_state=42)
        # default / "xgboost"
        if is_clf:
            from xgboost import XGBClassifier
            return XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, verbosity=0)

    def _stratify_target(self, is_clf: bool, y):
        """sklearn's train_test_split/StratifiedKFold both require every class to have at
        least 2 members - fall back to an unstratified split for degenerate tiny/rare-class
        data instead of crashing."""
        if not is_clf:
            return None
        vc = y.value_counts()
        return y if (vc >= 2).all() else None

    def _cv_score(self, m, X, y, task: str, cv_folds: int, n_samples: int):
        from sklearn.model_selection import cross_val_score
        min_cv = min(cv_folds, n_samples // 2) if n_samples >= 4 else 2
        scoring = "accuracy" if "classification" in task else "r2"
        try:
            scores = cross_val_score(m, X, y, cv=min_cv, scoring=scoring)
            return round(float(scores.mean()), 4), round(float(scores.std()), 4)
        except Exception:
            return None, None

    def _fit_and_finalize(self, X, y, task: str, model_type: str, test_size: int,
                           cv_folds: int, target_classes: Optional[List], warnings_list: List,
                           n_samples: int) -> ModelResult:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

        is_clf = "classification" in task
        stratify = self._stratify_target(is_clf, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify)

        m = self._build_estimator(task, model_type)
        m.fit(X_train, y_train)
        y_pred_train = m.predict(X_train)
        y_pred_test = m.predict(X_test)

        cv_mean, cv_std = self._cv_score(self._build_estimator(task, model_type), X, y, task, cv_folds, n_samples)

        if is_clf:
            self.result.metrics = {
                "accuracy_train": round(accuracy_score(y_train, y_pred_train), 4),
                "accuracy_test": round(accuracy_score(y_test, y_pred_test), 4),
                "f1_test": round(f1_score(y_test, y_pred_test, average="weighted"), 4),
                "cv_mean": cv_mean, "cv_std": cv_std,
            }
        else:
            self.result.metrics = {
                "r2_train": round(r2_score(y_train, y_pred_train), 4),
                "r2_test": round(r2_score(y_test, y_pred_test), 4),
                "rmse_test": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 2),
                "cv_mean": cv_mean, "cv_std": cv_std,
            }
        if warnings_list:
            self.result.metrics["warnings"] = warnings_list

        # Feature importance - only meaningful for tree models with a direct attribute, or a
        # Pipeline whose final step exposes it via named_steps.
        importer = m.named_steps["model"] if hasattr(m, "named_steps") else m
        if hasattr(importer, "feature_importances_"):
            imp = importer.feature_importances_
            idx = np.argsort(imp)[::-1][:10]
            self.result.feature_importance = {X.columns[i]: round(float(imp[i]), 4) for i in idx}

        final_model = _LabelDecodingModel(m, target_classes) if target_classes else m

        fn = f"{task}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
        self.result.model_path = str(self.model_dir / fn)
        with open(self.result.model_path, "wb") as f:
            pickle.dump(final_model, f)

        self.result.model_type = model_type
        self.result.task = task
        self.result.feature_columns = list(X.columns)
        self.result.target_classes = target_classes
        self.result.creation_time = datetime.now().isoformat()
        return self.result

    def train_auto(self, X, y, task: str, test_size: int, cv_folds: int,
                    target_classes: Optional[List], warnings_list: List, n_samples: int) -> ModelResult:
        """Real AutoML: cross-validates every candidate model type on the training data and
        fits the winner. Runs each candidate through the same CV scoring used to report
        cv_mean, so 'auto' actually compares apples to apples rather than defaulting to
        whichever type happened to be requested."""
        is_clf = "classification" in task
        candidates = self.CANDIDATE_TYPES_CLASSIFICATION if is_clf else self.CANDIDATE_TYPES_REGRESSION
        scores = {}
        best_type, best_score = None, None
        for mt in candidates:
            try:
                m = self._build_estimator(task, mt)
                mean, _ = self._cv_score(m, X, y, task, cv_folds, n_samples)
                scores[mt] = mean
                if mean is not None and (best_score is None or mean > best_score):
                    best_type, best_score = mt, mean
            except Exception as e:
                scores[mt] = None
        if best_type is None:
            raise RuntimeError("AutoML: every candidate model type failed to train on this dataset")
        result = self._fit_and_finalize(X, y, task, best_type, test_size, cv_folds,
                                        target_classes, warnings_list, n_samples)
        result.metrics["automl_candidates"] = scores
        result.metrics["automl_chosen"] = best_type
        return result

    def train(self, df, target_col, task, model_type="xgboost", test_size=0.2, cv_folds=5,
              max_categories: Optional[int] = None) -> ModelResult:
        is_clf = "classification" in task
        X, y = self.prepare_data(df, target_col, max_categories=max_categories)
        n_samples = len(X)

        warnings_list = []
        if n_samples < 100:
            warnings_list.append(f"Малый датасет: {n_samples} строк. Рекомендуется > 1000.")
        if n_samples < 50:
            warnings_list.append(f"Очень малый датасет ({n_samples} строк). Результаты недостоверны.")

        target_classes = None
        if is_clf and not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
            target_classes = [str(c) for c in le.classes_]

        if model_type == "auto":
            return self.train_auto(X, y, task, test_size, cv_folds, target_classes, warnings_list, n_samples)

        return self._fit_and_finalize(X, y, task, model_type, test_size, cv_folds,
                                      target_classes, warnings_list, n_samples)

    def get_summary(self):
        return {
            "model_type": self.result.model_type,
            "task": self.result.task,
            "metrics": self.result.metrics,
            "feature_importance": self.result.feature_importance,
            "feature_columns": self.result.feature_columns,
            "target_classes": self.result.target_classes,
            "model_path": self.result.model_path,
        }
