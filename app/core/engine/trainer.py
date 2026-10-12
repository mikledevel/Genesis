"""
DatasetTrainer - fits a real sklearn/xgboost model on a profiled dataset.

Rewrite (leak-free, pipeline-based). What changed versus the previous version:

  1. NO LEAKAGE. The train/test split happens FIRST, on the raw frame. Imputation, scaling
     and one-hot encoding live inside a sklearn Pipeline that is fitted on the training part
     only. Cross-validation (and the AutoML winner choice) runs on the training part only;
     the held-out test set is scored exactly once, at the end.
  2. ONE ARTIFACT. The saved model is Pipeline(preprocessor + estimator) and takes the
     ORIGINAL columns (a DataFrame). Categorical values are passed as raw strings; unseen
     categories are handled instead of crashing. `feature_columns` is therefore the list of
     raw input columns, and `input_schema` says which are numeric / categorical. Registry
     records written from this trainer carry metadata.input_format == "raw_dataframe"
     (see predict_utils.py); older records keep working through the legacy numeric-vector path.
  3. HONEST METRICS. Besides accuracy/F1/R2/RMSE we now report a naive baseline, ROC-AUC for
     binary tasks and MAE for regression, and warn when the model is not better than the
     baseline, when classes are very imbalanced, or when a column looks like an ID.
     CV uses f1_weighted (classification) / r2 (regression) and says so in `cv_metric`.
  4. Kept from before: stratified splits, string-target decoding via _LabelDecodingModel,
     capped one-hot cardinality (MAX_ONEHOT_CATEGORIES), scaled linear models, xgboost /
     random_forest / hist_gradient_boosting / logistic|linear, model_type="auto" AutoML.
     Classification targets that are not already 0..k-1 integers are now label-encoded too
     (XGBoost crashes on labels like {1, 2}).
  5. `prepare_data()` is kept unchanged for backwards compatibility (legacy callers/tests);
     train() no longer uses it.
"""
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from app.core.engine.predict_utils import RAW_FORMAT, coerce_categoricals

log = logging.getLogger(__name__)
# Scoped, not global: the old module-level filterwarnings("ignore") hid every warning in the process.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")


def _noop_stage(*_a, **_k):
    return None


class _LabelDecodingModel:
    """Wraps a fitted classifier trained on LabelEncoder-encoded integer targets so
    .predict() on the saved/pickled artifact returns the ORIGINAL labels, not opaque
    integers. Plain top-level class (not a closure) so it pickles via its module path and is
    allow-listed in app/core/safe_pickle.py."""
    def __init__(self, estimator, classes: List[Any]):
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
        self.feature_columns: List = []        # RAW input columns the saved model expects
        self.target_classes: Optional[List] = None
        self.input_format: str = RAW_FORMAT
        self.input_schema: Dict[str, List[str]] = {"numeric": [], "categorical": []}
        self.training_info: Dict[str, Any] = {}
        self.model = None                      # the fitted artifact itself (not serialised in summaries)


class DatasetTrainer:
    MAX_ONEHOT_CATEGORIES = 20  # per-column cap (incl. the "infrequent" bucket) before one-hot
    CANDIDATE_TYPES_CLASSIFICATION = ["xgboost", "random_forest", "hist_gradient_boosting", "logistic"]
    CANDIDATE_TYPES_REGRESSION = ["xgboost", "random_forest", "hist_gradient_boosting", "linear"]
    SCALED_TYPES = {"logistic", "linear"}

    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir); self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result = ModelResult()
        self._stage = _noop_stage      # on_stage(step_id, status, detail) - see app/core/progress.py

    # ------------------------------------------------------------------ legacy
    def prepare_data(self, df, target_col, max_categories: Optional[int] = None):
        """LEGACY: encodes the whole frame up front (this is what leaked). Kept only so old
        callers keep working; train() does not use it."""
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

    # ---------------------------------------------------------------- building
    def _preprocessor(self, num_cols: List[str], cat_cols: List[str], scale: bool, max_categories: int):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        parts = []
        if num_cols:
            steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if scale:
                steps.append(("scale", StandardScaler()))
            parts.append(("num", Pipeline(steps), num_cols))
        if cat_cols:
            parts.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         max_categories=max(2, max_categories), sparse_output=False)),
            ]), cat_cols))
        return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)

    def _build_model(self, task: str, model_type: str):
        is_clf = "classification" in task
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)
        if model_type == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1) if is_clf \
                else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            return HistGradientBoostingClassifier(random_state=42) if is_clf else HistGradientBoostingRegressor(random_state=42)
        if is_clf:  # default / "xgboost"
            from xgboost import XGBClassifier
            return XGBClassifier(random_state=42, verbosity=0, n_jobs=-1)
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)

    def _build_pipeline(self, task: str, model_type: str, num_cols, cat_cols, max_categories: int):
        from sklearn.pipeline import Pipeline
        prep = self._preprocessor(num_cols, cat_cols, model_type in self.SCALED_TYPES, max_categories)
        return Pipeline([("prep", prep), ("model", self._build_model(task, model_type))])

    # -------------------------------------------------------------- evaluation
    @staticmethod
    def _stratify_target(is_clf: bool, y):
        """sklearn can't stratify when a class has <2 members - fall back to a plain split."""
        if not is_clf:
            return None
        return y if (pd.Series(y).value_counts() >= 2).all() else None

    @staticmethod
    def _splitter(y, is_clf: bool, cv_folds: int, n_samples: int):
        from sklearn.model_selection import KFold, StratifiedKFold
        k = max(2, min(cv_folds, n_samples // 2)) if n_samples >= 4 else 2
        if is_clf:
            min_cls = int(pd.Series(y).value_counts().min())
            if min_cls >= 2:
                return StratifiedKFold(n_splits=min(k, min_cls), shuffle=True, random_state=42)
        return KFold(n_splits=k, shuffle=True, random_state=42)

    def _cv_score(self, pipeline, X, y, task: str, cv_folds: int):
        """CV on whatever (X, y) it is given - callers pass the TRAINING part only."""
        from sklearn.model_selection import cross_val_score
        is_clf = "classification" in task
        try:
            scores = cross_val_score(pipeline, X, y, cv=self._splitter(y, is_clf, cv_folds, len(X)),
                                     scoring="f1_weighted" if is_clf else "r2")
            return round(float(scores.mean()), 4), round(float(scores.std()), 4)
        except Exception as e:
            log.warning("cross-validation failed: %s", e)
            return None, None

    # ---------------------------------------------------------------- fitting
    def _fit_and_finalize(self, X_train, X_test, y_train, y_test, task: str, model_type: str,
                          num_cols, cat_cols, max_categories: int, cv_folds: int,
                          target_classes: Optional[List], decode_classes: Optional[List],
                          warnings_list: List, cv: Optional[Tuple] = None) -> ModelResult:
        from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                     mean_squared_error, r2_score, roc_auc_score)
        is_clf = "classification" in task

        st = self._stage
        if cv is None:
            st("cv", "active", "cross-validating on the training part")
            cv_mean, cv_std = self._cv_score(
                self._build_pipeline(task, model_type, num_cols, cat_cols, max_categories),
                X_train, y_train, task, cv_folds)
            st("cv", "done", (f"CV {'f1' if is_clf else 'r2'} {cv_mean} ± {cv_std}" if cv_mean is not None
                              else "cross-validation unavailable"))
        else:
            cv_mean, cv_std = cv

        st("fit", "active", model_type)
        m = self._build_pipeline(task, model_type, num_cols, cat_cols, max_categories)
        m.fit(X_train, y_train)
        st("fit", "done", model_type)
        st("eval", "active", "scoring on the held-out test set")
        pred_train, pred_test = m.predict(X_train), m.predict(X_test)

        metrics: Dict[str, Any] = {"cv_mean": cv_mean, "cv_std": cv_std,
                                   "cv_metric": "f1_weighted" if is_clf else "r2"}
        if is_clf:
            acc_test = float(accuracy_score(y_test, pred_test))
            baseline = float((y_test == pd.Series(y_train).mode().iloc[0]).mean())
            metrics.update({
                "accuracy_train": round(float(accuracy_score(y_train, pred_train)), 4),
                "accuracy_test": round(acc_test, 4),
                "f1_test": round(float(f1_score(y_test, pred_test, average="weighted")), 4),
                "baseline_accuracy_test": round(baseline, 4),
            })
            if pd.Series(y_train).nunique() == 2 and hasattr(m, "predict_proba") and pd.Series(y_test).nunique() == 2:
                try:
                    metrics["roc_auc_test"] = round(float(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])), 4)
                except Exception as e:
                    log.warning("roc_auc failed: %s", e)
            if acc_test <= baseline + 0.01:
                warnings_list.append("Модель не лучше наивного прогноза (всегда самый частый класс): "
                                     f"accuracy {acc_test:.2f} vs baseline {baseline:.2f}.")
        else:
            rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
            base_rmse = float(np.sqrt(np.mean((np.asarray(y_test, dtype=float) - float(np.mean(y_train))) ** 2)))
            metrics.update({
                "r2_train": round(float(r2_score(y_train, pred_train)), 4),
                "r2_test": round(float(r2_score(y_test, pred_test)), 4),
                "rmse_test": round(rmse, 2),
                "mae_test": round(float(mean_absolute_error(y_test, pred_test)), 2),
                "baseline_rmse_test": round(base_rmse, 2),
            })
            if rmse >= base_rmse * 0.99:
                warnings_list.append("Модель не лучше наивного прогноза (среднее значение): "
                                     f"RMSE {rmse:.2f} vs baseline {base_rmse:.2f}.")
        if warnings_list:
            metrics["warnings"] = warnings_list
        self.result.metrics = metrics
        st("eval", "done", (f"accuracy {metrics['accuracy_test']} (baseline {metrics['baseline_accuracy_test']})"
                            if is_clf else f"R² {metrics['r2_test']}, RMSE {metrics['rmse_test']}"))

        # Feature importance (tree models): mapped back to the post-encoding feature names.
        self.result.feature_importance = {}
        try:
            est = m.named_steps["model"]
            if hasattr(est, "feature_importances_"):
                names = list(m.named_steps["prep"].get_feature_names_out())
                imp = est.feature_importances_
                top = np.argsort(imp)[::-1][:10]
                self.result.feature_importance = {str(names[i]): round(float(imp[i]), 4) for i in top}
        except Exception as e:
            log.warning("feature importance unavailable: %s", e)

        final_model = _LabelDecodingModel(m, decode_classes) if decode_classes is not None else m

        st("save", "active", "writing the model file")
        fn = f"{task}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
        self.result.model_path = str(self.model_dir / fn)
        with open(self.result.model_path, "wb") as f:
            pickle.dump(final_model, f)
        st("save", "done")

        self.result.model = final_model
        self.result.model_type = model_type
        self.result.task = task
        self.result.feature_columns = list(X_train.columns)
        self.result.input_format = RAW_FORMAT
        self.result.input_schema = {"numeric": list(num_cols), "categorical": list(cat_cols)}
        self.result.target_classes = target_classes
        self.result.creation_time = datetime.now().isoformat()
        return self.result

    def train_auto(self, X_train, X_test, y_train, y_test, task: str, num_cols, cat_cols,
                   max_categories: int, cv_folds: int, target_classes, decode_classes,
                   warnings_list: List) -> ModelResult:
        """AutoML: cross-validate every candidate on the TRAINING part only, fit the winner,
        then score it once on the untouched test part."""
        is_clf = "classification" in task
        candidates = self.CANDIDATE_TYPES_CLASSIFICATION if is_clf else self.CANDIDATE_TYPES_REGRESSION
        scores, cvs = {}, {}
        best_type, best_score = None, None
        self._stage("cv", "active", f"comparing {len(candidates)} model types by cross-validation")
        for mt in candidates:
            try:
                pipe = self._build_pipeline(task, mt, num_cols, cat_cols, max_categories)
                mean, std = self._cv_score(pipe, X_train, y_train, task, cv_folds)
                scores[mt], cvs[mt] = mean, (mean, std)
                if mean is not None and (best_score is None or mean > best_score):
                    best_type, best_score = mt, mean
            except Exception as e:
                log.warning("AutoML candidate %s failed: %s", mt, e)
                scores[mt] = None
        if best_type is None:
            raise RuntimeError("AutoML: every candidate model type failed to train on this dataset")
        self._stage("cv", "done", f"best: {best_type} (CV {best_score})")
        result = self._fit_and_finalize(X_train, X_test, y_train, y_test, task, best_type, num_cols, cat_cols,
                                        max_categories, cv_folds, target_classes, decode_classes,
                                        warnings_list, cv=cvs[best_type])
        result.metrics["automl_candidates"] = scores
        result.metrics["automl_chosen"] = best_type
        return result

    def train(self, df, target_col, task, model_type="xgboost", test_size: float = 0.2, cv_folds: int = 5,
              max_categories: Optional[int] = None, on_stage=None) -> ModelResult:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import sklearn

        is_clf = "classification" in task
        max_categories = self.MAX_ONEHOT_CATEGORIES if max_categories is None else max_categories
        self._stage = on_stage or _noop_stage
        self._stage("prep", "active", "splitting first, then building the preprocessing pipeline")

        df = df.dropna(subset=[target_col]).copy()
        df.columns = df.columns.astype(str)
        target_col = str(target_col)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        n_samples = len(X)

        warnings_list: List[str] = []
        dt_cols = list(X.select_dtypes(include=["datetime", "datetimetz"]).columns)
        if dt_cols:
            X = X.drop(columns=dt_cols)
            warnings_list.append(f"Столбцы с датами исключены из признаков: {', '.join(dt_cols)}.")
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in X.columns if c not in num_cols]
        X = coerce_categoricals(X, cat_cols)
        if X.shape[1] == 0:
            raise ValueError("No usable feature columns left after excluding the target and date columns")

        if n_samples < 100:
            warnings_list.append(f"Малый датасет: {n_samples} строк. Рекомендуется > 1000.")
        if n_samples < 50:
            warnings_list.append(f"Очень малый датасет ({n_samples} строк). Результаты недостоверны.")
        if n_samples >= 50:
            id_like = [c for c in X.columns if X[c].nunique() / n_samples >= 0.95]
            if id_like:
                warnings_list.append(f"Похоже на ID-столбцы (почти все значения уникальны): {', '.join(id_like)} "
                                     "- модель может запомнить их вместо закономерностей.")

        # Classification targets must be 0..k-1 ints for XGBoost; anything else is encoded and
        # decoded back at predict time by _LabelDecodingModel.
        target_classes = decode_classes = None
        if is_clf:
            uniq = sorted(pd.Series(y).unique().tolist(), key=str)
            contiguous = pd.api.types.is_numeric_dtype(y) and set(uniq) == set(range(len(uniq)))
            if not contiguous:
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)
                decode_classes = le.classes_.tolist()
                target_classes = [str(c) for c in decode_classes]
            vc = pd.Series(y).value_counts(normalize=True)
            if len(vc) > 1 and vc.min() < 0.10:
                warnings_list.append(f"Сильный дисбаланс классов: редкий класс {vc.min():.1%}. "
                                     "Смотрите F1/ROC-AUC, а не accuracy.")

        # SPLIT FIRST - everything learned from data happens after this line, on X_train only.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=self._stratify_target(is_clf, y))

        self._stage("prep", "done", f"{len(X_train)} train / {len(X_test)} test rows · "
                                    f"{len(num_cols)} numeric + {len(cat_cols)} categorical features")
        if model_type == "auto":
            result = self.train_auto(X_train, X_test, y_train, y_test, task, num_cols, cat_cols,
                                     max_categories, cv_folds, target_classes, decode_classes, warnings_list)
        else:
            result = self._fit_and_finalize(X_train, X_test, y_train, y_test, task, model_type, num_cols,
                                            cat_cols, max_categories, cv_folds, target_classes,
                                            decode_classes, warnings_list)

        try:
            fingerprint = format(int(pd.util.hash_pandas_object(df, index=False).sum()) & (2**64 - 1), "016x")
        except Exception:
            fingerprint = None
        result.training_info = {"sklearn_version": sklearn.__version__, "n_rows": n_samples,
                                "trained_at": result.creation_time, "data_fingerprint": fingerprint,
                                "random_state": 42}
        return result

    def get_summary(self):
        return {
            "model_type": self.result.model_type,
            "task": self.result.task,
            "metrics": self.result.metrics,
            "feature_importance": self.result.feature_importance,
            "feature_columns": self.result.feature_columns,
            "input_format": self.result.input_format,
            "input_schema": self.result.input_schema,
            "target_classes": self.result.target_classes,
            "model_path": self.result.model_path,
        }
