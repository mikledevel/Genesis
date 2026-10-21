"""
DatasetTrainer - fits a real sklearn/xgboost model on a profiled dataset.

Leak-free, pipeline-based, and (this version) tuned for QUALITY:

  1. NO LEAKAGE. The train/test split happens FIRST, on the raw frame. Imputation, scaling, one-hot,
     date features and TF-IDF all live inside one sklearn Pipeline fitted on the training part only.
     CV, hyperparameter search and the decision threshold use the training part only; the held-out
     test set is scored exactly once, at the end.
  2. ONE ARTIFACT. The saved model takes the ORIGINAL columns (a DataFrame): raw categories, raw date
     strings, raw text. `feature_columns` = raw input columns; `input_schema` says which are
     numeric / categorical / datetime / text. Registry records carry metadata.input_format ==
     "raw_dataframe" (see predict_utils.py); older records keep working through the legacy path.
  3. HYPERPARAMETER TUNING with a time budget (budget="fast"|"balanced"|"thorough"). The default
     configuration is always a candidate, so tuning can only help on CV; it is skipped (and says so)
     for fast mode, for linear regression and for datasets under MIN_ROWS_TO_TUNE rows.
  4. IMBALANCED CLASSES. If the rarest class is < IMBALANCE_THRESHOLD of the training rows the trainer
     switches on class weights (xgboost: scale_pos_weight), selects models by PR-AUC (binary) /
     macro-F1 (multiclass) instead of accuracy, and for binary tasks tunes the DECISION THRESHOLD for
     the rare class on out-of-fold training predictions. Metrics then include precision / recall / F1
     of the rare class, balanced accuracy and PR-AUC.
  5. DATES AND TEXT. Date-like columns become calendar features (year, month, day, weekday, weekend,
     days since 1970, hour) instead of being dropped; free-text columns (multi-word, mostly unique)
     get TF-IDF word/bigram features instead of being one-hot-capped into noise.
  6. Kept: stratified splits, string-target decoding via _LabelDecodingModel, capped one-hot
     cardinality, scaled linear models, xgboost / random_forest / hist_gradient_boosting / logistic|
     linear, model_type="auto" AutoML, honest metrics (baseline, ROC-AUC, MAE), warnings.
  7. `prepare_data()` is kept unchanged for backwards compatibility; train() does not use it.
"""
import logging
import pickle
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from app.core.engine.predict_utils import RAW_FORMAT, coerce_categoricals
from app.core.engine.transformers import DateFeatures, detect_datetime_columns, detect_text_columns, looks_like_id

log = logging.getLogger(__name__)
# Scoped, not global: the old module-level filterwarnings("ignore") hid every warning in the process.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")


def _noop_stage(*_a, **_k):
    return None


def _jsonable(obj):
    """numpy scalars / arrays -> plain Python so metrics can be stored as JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


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


class _ThresholdModel:
    """Binary classifier with a tuned decision threshold for the RARE class (imbalanced data).

    predict() returns the rare class when P(rare) >= threshold. predict_proba() returns scores
    re-scaled so that the threshold maps to 0.5 - argmax(predict_proba) then always agrees with
    predict() and "confidence" means confidence in the decision actually made. (They are decision
    scores, not calibrated probabilities; predict_proba_raw() gives the model's own probabilities.)
    Allow-listed in app/core/safe_pickle.py."""
    def __init__(self, estimator, positive_index: int, threshold: float):
        self.estimator = estimator
        self.positive_index = int(positive_index)
        self.threshold = float(threshold)

    def _p(self, X):
        return np.asarray(self.estimator.predict_proba(X))[:, self.positive_index]

    def predict(self, X):
        pos = self.positive_index
        return np.where(self._p(X) >= self.threshold, pos, 1 - pos)

    def predict_proba(self, X):
        p, t = self._p(X), self.threshold
        adj = np.where(p >= t, 0.5 + 0.5 * (p - t) / max(1 - t, 1e-9), 0.5 * p / max(t, 1e-9))
        out = np.zeros((len(p), 2))
        out[:, self.positive_index] = adj
        out[:, 1 - self.positive_index] = 1 - adj
        return out

    def predict_proba_raw(self, X):
        return self.estimator.predict_proba(X)


class ModelResult:
    def __init__(self):
        self.model_type = self.task = self.model_path = self.creation_time = ""
        self.metrics: Dict = {}
        self.feature_importance: Dict = {}
        self.feature_columns: List = []        # RAW input columns the saved model expects
        self.target_classes: Optional[List] = None
        self.input_format: str = RAW_FORMAT
        self.input_schema: Dict[str, List[str]] = {"numeric": [], "categorical": [], "datetime": [], "text": []}
        self.training_info: Dict[str, Any] = {}
        self.model = None                      # the fitted artifact itself (not serialised in summaries)


@dataclass
class FeatureSpec:
    """Everything the pipeline builder needs to know about the data (decided once, after the split)."""
    num: List[str] = field(default_factory=list)
    cat: List[str] = field(default_factory=list)
    dt: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    max_categories: int = 20
    text_features: int = 200
    min_df: int = 2
    balanced: bool = False                    # class-imbalance handling switched on
    pos_weight: Optional[float] = None        # xgboost binary: n(label 0) / n(label 1)


class DatasetTrainer:
    MAX_ONEHOT_CATEGORIES = 20  # per-column cap (incl. the "infrequent" bucket) before one-hot
    CANDIDATE_TYPES_CLASSIFICATION = ["xgboost", "random_forest", "hist_gradient_boosting", "logistic"]
    CANDIDATE_TYPES_REGRESSION = ["xgboost", "random_forest", "hist_gradient_boosting", "linear"]
    SCALED_TYPES = {"logistic", "linear"}
    IMBALANCE_THRESHOLD = 0.15     # rarest class share (of the training rows) below which we compensate
    MIN_RARE_FOR_THRESHOLD = 15    # fewer rare-class training rows than this: a tuned threshold is noise
    MIN_ROWS_TO_TUNE = 100
    DEFAULT_BUDGET = "balanced"
    TUNE_BUDGETS = {"fast": None,
                    "balanced": {"iters": 10, "seconds": 45},
                    "thorough": {"iters": 30, "seconds": 150}}

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
    def _preprocessor(self, spec: FeatureSpec, scale: bool):
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        parts = []
        if spec.num:
            steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if scale:
                steps.append(("scale", StandardScaler()))
            parts.append(("num", Pipeline(steps), spec.num))
        if spec.cat:
            parts.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         max_categories=max(2, spec.max_categories), sparse_output=False)),
            ]), spec.cat))
        if spec.dt:
            date_steps = [("feat", DateFeatures()), ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if scale:
                date_steps.append(("scale", StandardScaler()))
            parts.append(("date", Pipeline(date_steps), spec.dt))
        for i, col in enumerate(spec.text):   # a plain string (not a list) = 1-D input for the vectorizer
            parts.append((f"text{i}", TfidfVectorizer(max_features=spec.text_features, ngram_range=(1, 2),
                                                      min_df=spec.min_df, sublinear_tf=True), col))
        # sparse_threshold=0: always dense (HistGradientBoosting rejects sparse input)
        return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True, sparse_threshold=0)

    def _build_model(self, task: str, model_type: str, spec: FeatureSpec):
        is_clf = "classification" in task
        cw = "balanced" if (is_clf and spec.balanced) else None
        if model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42, class_weight=cw)
        if model_type == "linear":
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                          class_weight="balanced_subsample" if cw else None) if is_clf \
                else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        if model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            return HistGradientBoostingClassifier(random_state=42, class_weight=cw) if is_clf \
                else HistGradientBoostingRegressor(random_state=42)
        if is_clf:  # default / "xgboost"
            from xgboost import XGBClassifier
            extra = {"scale_pos_weight": spec.pos_weight} if (spec.balanced and spec.pos_weight) else {}
            return XGBClassifier(random_state=42, verbosity=0, n_jobs=-1, **extra)
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)

    def _build_pipeline(self, task: str, model_type: str, spec: FeatureSpec):
        from sklearn.pipeline import Pipeline
        prep = self._preprocessor(spec, model_type in self.SCALED_TYPES)
        return Pipeline([("prep", prep), ("model", self._build_model(task, model_type, spec))])

    @staticmethod
    def _param_space(model_type: str, task: str) -> Optional[Dict]:
        from scipy.stats import loguniform, randint, uniform
        if model_type == "xgboost":
            return {"n_estimators": randint(100, 500), "max_depth": randint(3, 9), "learning_rate": loguniform(0.02, 0.3),
                    "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4),
                    "min_child_weight": randint(1, 8), "reg_lambda": loguniform(0.5, 10)}
        if model_type == "random_forest":
            return {"n_estimators": [200, 300, 500], "max_depth": [None, 8, 12, 20],
                    "min_samples_leaf": [1, 2, 4, 8], "max_features": ["sqrt", 0.5, 1.0]}
        if model_type == "hist_gradient_boosting":
            return {"learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(15, 64),
                    "min_samples_leaf": randint(10, 60), "l2_regularization": loguniform(1e-3, 10),
                    "max_iter": [150, 250, 400]}
        if model_type == "logistic":
            return {"C": loguniform(0.01, 100)}
        return None            # plain linear regression has nothing worth tuning

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

    @staticmethod
    def _scoring(task: str, spec: FeatureSpec, y_train) -> Tuple[Any, str]:
        """(scorer for sklearn, name reported as `cv_metric`). Imbalanced data is judged on the RARE
        class (PR-AUC) / macro-F1, never on accuracy."""
        if "classification" not in task:
            return "r2", "r2"
        if not spec.balanced:
            return "f1_weighted", "f1_weighted"
        if pd.Series(y_train).nunique() == 2:
            from sklearn.metrics import average_precision_score
            pos = int(pd.Series(y_train).value_counts().idxmin())

            def pr_auc(estimator, X, y):
                return float(average_precision_score(np.asarray(y) == pos, estimator.predict_proba(X)[:, pos]))
            return pr_auc, "pr_auc"
        return "f1_macro", "f1_macro"

    def _cv_score(self, pipeline, X, y, task: str, cv_folds: int, scoring=None):
        """CV on whatever (X, y) it is given - callers pass the TRAINING part only."""
        from sklearn.model_selection import cross_val_score
        is_clf = "classification" in task
        try:
            scores = cross_val_score(pipeline, X, y, cv=self._splitter(y, is_clf, cv_folds, len(X)),
                                     scoring=scoring or ("f1_weighted" if is_clf else "r2"))
            return round(float(scores.mean()), 4), round(float(scores.std()), 4)
        except Exception as e:
            log.warning("cross-validation failed: %s", e)
            return None, None

    def _tune(self, task, model_type, X, y, spec, scoring, cv_folds, cfg, baseline: Optional[float]):
        """Random search on the TRAINING part only. The default configuration is the incumbent, so the
        result is never worse than not tuning (on CV). -> (best_params, info)"""
        from sklearn.model_selection import ParameterSampler, cross_val_score
        space = self._param_space(model_type, task)
        if not space:
            return {}, {"skipped": "nothing to tune for this model type"}
        splitter = self._splitter(y, "classification" in task, cv_folds, len(X))
        best_params, best_score, best_std = {}, baseline if baseline is not None else -np.inf, None
        started, tried = time.monotonic(), 0
        for params in ParameterSampler(space, n_iter=cfg["iters"], random_state=42):
            if time.monotonic() - started > cfg["seconds"]:
                break
            pipe = self._build_pipeline(task, model_type, spec).set_params(**{f"model__{k}": v for k, v in params.items()})
            try:
                scores = cross_val_score(pipe, X, y, cv=splitter, scoring=scoring)
            except Exception as e:
                log.warning("tuning candidate failed: %s", e)
                continue
            tried += 1
            if scores.mean() > best_score + 1e-4:
                best_params, best_score, best_std = params, float(scores.mean()), float(scores.std())
        return best_params, {"trials": tried, "seconds": round(time.monotonic() - started, 1),
                             "cv_default": baseline, "cv_tuned": round(best_score, 4) if best_params else baseline,
                             "cv_std": round(best_std, 4) if best_std is not None else None,
                             "best_params": _jsonable(best_params)}

    # ---------------------------------------------------------------- fitting
    def _fit_and_finalize(self, X_train, X_test, y_train, y_test, task: str, model_type: str,
                          spec: FeatureSpec, cv_folds: int, target_classes: Optional[List],
                          decode_classes: Optional[List], warnings_list: List, budget: str,
                          cv: Optional[Tuple] = None) -> ModelResult:
        from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                                     mean_absolute_error, mean_squared_error, precision_recall_curve,
                                     precision_recall_fscore_support, r2_score, roc_auc_score)
        from sklearn.model_selection import cross_val_predict
        is_clf = "classification" in task
        st = self._stage
        scoring, cv_metric = self._scoring(task, spec, y_train)
        binary = is_clf and pd.Series(y_train).nunique() == 2
        pos = int(pd.Series(y_train).value_counts().idxmin()) if binary else None

        # 1) CV of the default configuration (AutoML already did this for the winner and passes it in)
        if cv is None:
            st("cv", "active", "cross-validating on the training part")
            cv_mean, cv_std = self._cv_score(self._build_pipeline(task, model_type, spec), X_train, y_train, task,
                                             cv_folds, scoring)
            st("cv", "done", (f"CV {cv_metric} {cv_mean} ± {cv_std}" if cv_mean is not None else "cross-validation unavailable"))
        else:
            cv_mean, cv_std = cv
        cv_default = cv_mean

        # 2) hyperparameter search (training part only, time-boxed)
        cfg = self.TUNE_BUDGETS.get(budget)
        best_params, tuning = {}, {"budget": budget}
        if cfg is None:
            tuning["skipped"] = "budget is 'fast'"
        elif len(X_train) < self.MIN_ROWS_TO_TUNE:
            tuning["skipped"] = f"only {len(X_train)} training rows - tuning would overfit"
        else:
            st("tune", "active", f"random search, up to {cfg['iters']} configs / {cfg['seconds']} s")
            best_params, info = self._tune(task, model_type, X_train, y_train, spec, scoring, cv_folds, cfg, cv_default)
            tuning.update(info)
            if best_params:
                cv_mean, cv_std = info["cv_tuned"], info["cv_std"]
        if "skipped" in tuning:
            st("tune", "skipped", tuning["skipped"])
        else:
            st("tune", "done", (f"{tuning.get('trials', 0)} configs tried, CV {cv_metric} {cv_default} → {cv_mean}" if best_params
                                else f"{tuning.get('trials', 0)} configs tried - the defaults were already best"))

        # 3) final fit (+ decision threshold for the rare class on OUT-OF-FOLD training predictions)
        st("fit", "active", model_type)
        m = self._build_pipeline(task, model_type, spec).set_params(**{f"model__{k}": v for k, v in best_params.items()})
        threshold = None
        if binary and spec.balanced and int((np.asarray(y_train) == pos).sum()) >= self.MIN_RARE_FOR_THRESHOLD:
            try:
                oof = cross_val_predict(m, X_train, y_train, cv=self._splitter(y_train, True, cv_folds, len(X_train)),
                                        method="predict_proba")[:, pos]
                prec, rec, thr = precision_recall_curve(np.asarray(y_train) == pos, oof)
                f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
                threshold = float(np.clip(thr[int(np.argmax(f1s))], 0.02, 0.98))
            except Exception as e:
                log.warning("threshold tuning failed: %s", e)
        m.fit(X_train, y_train)
        inner = _ThresholdModel(m, pos, threshold) if threshold is not None else m
        st("fit", "done", model_type + (f" · decision threshold {threshold:.2f} for the rare class" if threshold is not None else ""))

        # 4) evaluation: the test set is touched exactly once, with the model as it will be deployed
        st("eval", "active", "scoring on the held-out test set")
        pred_train, pred_test = inner.predict(X_train), inner.predict(X_test)
        metrics: Dict[str, Any] = {"cv_mean": cv_mean, "cv_std": cv_std, "cv_metric": cv_metric}
        if is_clf:
            acc_test = float(accuracy_score(y_test, pred_test))
            baseline = float((y_test == pd.Series(y_train).mode().iloc[0]).mean())
            metrics.update({
                "accuracy_train": round(float(accuracy_score(y_train, pred_train)), 4),
                "accuracy_test": round(acc_test, 4),
                "f1_test": round(float(f1_score(y_test, pred_test, average="weighted")), 4),
                "baseline_accuracy_test": round(baseline, 4),
            })
            try:
                metrics["balanced_accuracy_test"] = round(float(balanced_accuracy_score(y_test, pred_test)), 4)
            except Exception:
                pass
            if binary and pd.Series(y_test).nunique() == 2 and hasattr(m, "predict_proba"):
                proba = m.predict_proba(X_test)
                try:
                    metrics["roc_auc_test"] = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
                    metrics["pr_auc_test"] = round(float(average_precision_score(np.asarray(y_test) == pos, proba[:, pos])), 4)
                except Exception as e:
                    log.warning("auc failed: %s", e)
            if binary:
                p, r, f, _ = precision_recall_fscore_support(y_test, pred_test, labels=[pos], zero_division=0)
                metrics.update({"precision_minority_test": round(float(p[0]), 4), "recall_minority_test": round(float(r[0]), 4),
                                "f1_minority_test": round(float(f[0]), 4)})
            if spec.balanced:
                metrics["f1_macro_test"] = round(float(f1_score(y_test, pred_test, average="macro")), 4)
                if threshold is not None:
                    metrics["threshold"] = round(threshold, 3)
                if binary and metrics.get("f1_minority_test", 1) == 0:
                    warnings_list.append("Модель не находит редкий класс на тесте (recall = 0). "
                                         "Нужно больше примеров редкого класса или другие признаки.")
            elif acc_test <= baseline + 0.01:
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
        tuning_report = {k: v for k, v in tuning.items() if k != "best_params" or v}
        metrics["tuning"] = tuning_report
        metrics["n_test"] = int(len(y_test))
        if is_clf:
            metrics["accuracy_test_ci"] = self._wilson(int(round(metrics["accuracy_test"] * len(y_test))), len(y_test))
        else:
            metrics["r2_test_ci"] = self._bootstrap_r2(y_test, pred_test)
        if warnings_list:
            metrics["warnings"] = warnings_list
        audit = getattr(self, "_audit", {})
        if audit.get("checks"):
            metrics["data_checks"] = audit["checks"]
        metrics["trust"] = self._assess_trust(is_clf, metrics, spec, y_train, y_test, getattr(self, "_n_rows_used", len(y_train) + len(y_test)))
        self.result.metrics = _jsonable(metrics)
        st("eval", "done", (f"accuracy {metrics['accuracy_test']} (baseline {metrics['baseline_accuracy_test']})"
                            + (f", rare-class F1 {metrics['f1_minority_test']}" if spec.balanced and binary else "")
                            if is_clf else f"R² {metrics['r2_test']}, RMSE {metrics['rmse_test']}"))

        # Feature importance (tree models), named in terms the user recognises
        self.result.feature_importance = {}
        try:
            est = m.named_steps["model"]
            if hasattr(est, "feature_importances_"):
                names = [self._pretty_feature(n, spec) for n in m.named_steps["prep"].get_feature_names_out()]
                imp = est.feature_importances_
                top = np.argsort(imp)[::-1][:10]
                self.result.feature_importance = {str(names[i]): round(float(imp[i]), 4) for i in top}
        except Exception as e:
            log.warning("feature importance unavailable: %s", e)

        final_model = _LabelDecodingModel(inner, decode_classes) if decode_classes is not None else inner

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
        self.result.input_schema = {"numeric": list(spec.num), "categorical": list(spec.cat),
                                    "datetime": list(spec.dt), "text": list(spec.text)}
        self.result.target_classes = target_classes
        self.result.creation_time = datetime.now().isoformat()
        return self.result

    # ------------------------------------------------------------ data reliability
    # A depth-4 tree scores a PERFECT threshold rule (target = x > 0) at only ~0.99 AUC on held-out folds,
    # so 0.99 is where "practically perfect" starts. Honest strong features stay far below it (breast-cancer's
    # best single column ~0.97, iris ~0.95 accuracy) - tests/test_reliability.py guards against false positives.
    LEAK_AUC, LEAK_ACC, LEAK_R2, LEAK_PURITY = 0.99, 0.99, 0.995, 0.995

    def _leakage_scores(self, X_train, y_train, num_cols, cat_cols, is_clf: bool, target_name: str) -> Dict[str, float]:
        """How well does ONE column alone (a depth-4 tree, 3-fold CV, training rows only) predict the target?
        A single column that is (almost) perfect is practically never a real signal - it is the answer in
        disguise (a derived column, a post-outcome field, a copy). Categorical columns are judged by purity
        (does each category map to one class?), only when categories are not near-unique (that would be an ID)."""
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        n = len(X_train)
        if n < 60:
            return {}
        idx = np.random.RandomState(0).choice(n, size=min(n, 20000), replace=False)
        yv = pd.Series(np.asarray(y_train))[idx].reset_index(drop=True)
        base = float(yv.value_counts(normalize=True).max()) if is_clf else 0.0
        binary = is_clf and yv.nunique() == 2
        out: Dict[str, float] = {}
        for c in list(num_cols) + list(cat_cols):
            x = X_train[c].reset_index(drop=True).iloc[idx].reset_index(drop=True)
            try:
                if c in cat_cols:
                    if not is_clf or x.nunique() == 0 or len(x) / x.nunique() < 3 or base >= 0.9:
                        continue
                    g = pd.DataFrame({"x": x.fillna("__missing__").astype(str), "y": yv}).groupby("x")["y"]
                    purity = float(g.agg(lambda v: v.value_counts().iloc[0]).sum() / len(x))
                    if purity >= self.LEAK_PURITY:
                        out[c] = purity
                    continue
                v = x.fillna(x.median()).to_numpy(dtype=float).reshape(-1, 1)
                if is_clf:
                    if binary:
                        sc = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), v, yv, cv=3, scoring="roc_auc").mean()
                        hit = sc >= self.LEAK_AUC
                    else:
                        sc = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), v, yv, cv=3, scoring="accuracy").mean()
                        hit = sc >= self.LEAK_ACC and base < 0.8
                else:
                    sc = cross_val_score(DecisionTreeRegressor(max_depth=6, random_state=0), v, yv.astype(float), cv=3, scoring="r2").mean()
                    hit = sc >= self.LEAK_R2
                if hit:
                    out[c] = float(sc)
            except Exception as e:
                log.warning("leakage scan skipped %s: %s", c, e)
        return out

    @staticmethod
    def _wilson(k: int, n: int, z: float = 1.96) -> List[float]:
        """95% Wilson interval for a proportion (accuracy)."""
        if n <= 0:
            return [0.0, 1.0]
        p = k / n; d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return [round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4)]

    @staticmethod
    def _bootstrap_r2(y_true, y_pred, B: int = 300) -> List[float]:
        from sklearn.metrics import r2_score
        yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        rng = np.random.RandomState(0); vals = []
        for _ in range(B):
            i = rng.randint(0, len(yt), len(yt))
            if np.ptp(yt[i]) > 0:
                vals.append(r2_score(yt[i], yp[i]))
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else [0.0, 1.0]

    def _assess_trust(self, is_clf: bool, metrics: Dict, spec: FeatureSpec, y_train, y_test, n_rows: int) -> Dict:
        """Should the user believe these numbers? level = the worst concern found. Every concern is a
        plain sentence, because the point is to tell a person - not to hand them another number."""
        concerns: List[Tuple[int, str]] = []          # (2 = low trust, 1 = medium)
        n_test = len(y_test)
        ci = metrics.get("accuracy_test_ci") or metrics.get("r2_test_ci")
        if n_test < 100:
            concerns.append((1, f"Тестовая выборка мала ({n_test} строк) - реальная метрика может заметно отличаться."))
        elif ci and ci[1] - ci[0] > (0.15 if is_clf else 0.3):
            concerns.append((1, f"Доверительный интервал метрики широк ({ci[0]}-{ci[1]})."))
        base = metrics.get("baseline_accuracy_test")
        if is_clf and not spec.balanced and ci and base is not None and ci[0] <= base and \
                not any("не лучше наивного" in w for w in metrics.get("warnings", [])):
            concerns.append((1, f"Интервал точности {ci[0]}-{ci[1]} включает наивный прогноз ({base}): преимущество над ним статистически неочевидно."))
        if not is_clf and ci and ci[0] <= 0.05:
            concerns.append((1, f"R² на тесте может быть близок к нулю (интервал {ci[0]}-{ci[1]})."))
        if n_rows < 200:
            concerns.append((1, f"Мало данных ({n_rows} строк): модель могла выучить случайные совпадения."))
        if any("не лучше наивного" in w for w in metrics.get("warnings", [])) or \
                (spec.balanced and metrics.get("f1_minority_test") == 0):
            concerns.append((2, "Модель не лучше простого наивного прогноза - на этих данных она ничего не выучила."))
        cv_to_test = {"f1_weighted": "f1_test", "pr_auc": "pr_auc_test", "f1_macro": "f1_macro_test", "r2": "r2_test"}
        cv, te = metrics.get("cv_mean"), metrics.get(cv_to_test.get(metrics.get("cv_metric"), ""))
        if cv is not None and te is not None and cv - te > 0.15:
            concerns.append((1, f"На тесте метрика заметно хуже, чем на кросс-валидации ({te} против {cv}): возможна переобученность или неудачное разбиение."))
        if is_clf and spec.balanced:
            rare = min(int((np.asarray(y_test) == v).sum()) for v in pd.Series(y_train).unique())
            if rare < 20:
                concerns.append((1, f"В тесте всего {rare} примеров редкого класса - recall/precision очень неустойчивы."))
        if getattr(self, "_audit", {}).get("unresolved_leak"):
            cols = ", ".join(self._audit["unresolved_leak"])
            concerns.append((2, f"Единственный признак ({cols}) почти идеально предсказывает цель - проверьте, не утечка ли это."))
        worst = max([c[0] for c in concerns], default=0)
        return {"level": {0: "high", 1: "medium", 2: "low"}[worst], "reasons": [t for _, t in sorted(concerns, key=lambda c: -c[0])]}

    @staticmethod
    def _pretty_feature(name: str, spec: FeatureSpec) -> str:
        """ColumnTransformer's 'cat__plan_pro' / 'text0__refund' -> 'plan_pro' / 'review: refund'."""
        m = re.match(r"^text(\d+)__(.*)$", name)
        if m and int(m.group(1)) < len(spec.text):
            return f"{spec.text[int(m.group(1))]}: {m.group(2)}"
        return re.sub(r"^(num|cat|date)__", "", name)

    def train_auto(self, X_train, X_test, y_train, y_test, task: str, spec: FeatureSpec, cv_folds: int,
                   target_classes, decode_classes, warnings_list: List, budget: str) -> ModelResult:
        """AutoML: cross-validate every candidate on the TRAINING part only, tune and fit the winner,
        then score it once on the untouched test part."""
        is_clf = "classification" in task
        candidates = self.CANDIDATE_TYPES_CLASSIFICATION if is_clf else self.CANDIDATE_TYPES_REGRESSION
        scoring, cv_metric = self._scoring(task, spec, y_train)
        scores, cvs = {}, {}
        best_type, best_score = None, None
        self._stage("cv", "active", f"comparing {len(candidates)} model types by cross-validation ({cv_metric})")
        for mt in candidates:
            try:
                pipe = self._build_pipeline(task, mt, spec)
                mean, std = self._cv_score(pipe, X_train, y_train, task, cv_folds, scoring)
                scores[mt], cvs[mt] = mean, (mean, std)
                if mean is not None and (best_score is None or mean > best_score):
                    best_type, best_score = mt, mean
            except Exception as e:
                log.warning("AutoML candidate %s failed: %s", mt, e)
                scores[mt] = None
        if best_type is None:
            raise RuntimeError("AutoML: every candidate model type failed to train on this dataset")
        self._stage("cv", "done", f"best: {best_type} (CV {cv_metric} {best_score})")
        result = self._fit_and_finalize(X_train, X_test, y_train, y_test, task, best_type, spec, cv_folds,
                                        target_classes, decode_classes, warnings_list, budget, cv=cvs[best_type])
        result.metrics["automl_candidates"] = scores
        result.metrics["automl_chosen"] = best_type
        return result

    def train(self, df, target_col, task, model_type="xgboost", test_size: float = 0.2, cv_folds: int = 5,
              max_categories: Optional[int] = None, on_stage=None, budget: Optional[str] = None,
              keep_columns: Optional[List[str]] = None, exclude_columns: Optional[List[str]] = None) -> ModelResult:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import sklearn

        is_clf = "classification" in task
        max_categories = self.MAX_ONEHOT_CATEGORIES if max_categories is None else max_categories
        budget = budget if budget in self.TUNE_BUDGETS else self.DEFAULT_BUDGET
        self._stage = on_stage or _noop_stage
        self._stage("prep", "active", "splitting first, then building the preprocessing pipeline")

        df = df.dropna(subset=[target_col]).copy()
        df.columns = df.columns.astype(str)
        target_col = str(target_col)
        warnings_list: List[str] = []
        data_checks: List[Dict[str, Any]] = []
        keep = set(keep_columns or [])
        self._audit = {"checks": data_checks, "unresolved_leak": []}

        # Exact duplicate rows would land on both sides of the split and inflate the test score. But only
        # when a repeat is SURPRISING: if the columns can take few distinct combinations (a survey of
        # yes/no answers), identical rows are natural, carry frequency information, and are kept. The
        # rule: the number of possible combinations must exceed the row count by 100x or more.
        if len(df) >= 50:
            combos_log10 = sum(np.log10(max(2, df[c].nunique(dropna=True))) for c in df.columns)
            dups = int(df.duplicated().sum()) if combos_log10 >= np.log10(len(df)) + 2 else 0
            if dups >= max(2, 0.005 * len(df)):
                df = df.drop_duplicates().reset_index(drop=True)
                msg = f"Удалено {dups} дублирующихся строк: иначе одинаковые строки попали бы и в обучение, и в тест, и оценка была бы завышена."
                data_checks.append({"kind": "duplicates", "rows": dups, "message": msg}); warnings_list.append(msg)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        # Columns the user (or the plan) asked to leave out. Unknown names are ignored, the target can't be
        # excluded, and excluding everything is refused.
        excl = [str(c) for c in (exclude_columns or []) if str(c) in X.columns and str(c) != target_col]
        if excl:
            if len(X.columns) - len(excl) < 1:
                raise ValueError("Excluding those columns would leave no features to learn from")
            X = X.drop(columns=excl)
            data_checks.append({"kind": "excluded", "columns": excl,
                                "message": f"Не используются по вашей просьбе: {', '.join(excl)}."})
        n_samples = len(X)
        self._n_rows_used = n_samples

        # Column typing: numeric / date / free text / categorical
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        rest = [c for c in X.columns if c not in num_cols]
        dt_cols = detect_datetime_columns(X, rest)
        rest = [c for c in rest if c not in dt_cols]
        text_cols = detect_text_columns(X, rest)
        cat_cols = [c for c in rest if c not in text_cols]

        # Row identifiers carry no signal - a tree memorises them (and in a file sorted by the label they
        # even "predict" it). Dropped as long as something else is left to learn from.
        ids = [c for c in num_cols + cat_cols if c not in keep and looks_like_id(X[c], c)]
        if ids and len(X.columns) > len(ids):
            X = X.drop(columns=ids)
            num_cols = [c for c in num_cols if c not in ids]
            cat_cols = [c for c in cat_cols if c not in ids]
            msg = f"Исключены ID-столбцы (уникальны почти в каждой строке): {', '.join(ids)}. Модель запомнила бы их вместо закономерностей."
            data_checks.append({"kind": "id", "columns": ids, "message": msg}); warnings_list.append(msg)
        X = coerce_categoricals(X, cat_cols)
        for c in text_cols:
            X[c] = X[c].fillna("").astype(str)
        if X.shape[1] == 0:
            raise ValueError("No usable feature columns left after excluding the target")
        notes = []
        if dt_cols:
            notes.append("Даты → календарные признаки (год, месяц, день недели, выходной, …): " + ", ".join(dt_cols))
        if text_cols:
            notes.append("Текст → TF-IDF слова и биграммы: " + ", ".join(text_cols))

        if n_samples < 100:
            warnings_list.append(f"Малый датасет: {n_samples} строк. Рекомендуется > 1000.")
        if n_samples < 50:
            warnings_list.append(f"Очень малый датасет ({n_samples} строк). Результаты недостоверны.")

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
                                     "Смотрите F1/Recall редкого класса, а не accuracy.")

        # SPLIT FIRST - everything learned from data happens after this line, on X_train only.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=self._stratify_target(is_clf, y))

        # Target leakage scan on the TRAINING rows only; flagged columns are removed from both parts.
        suspects = {c: sc for c, sc in self._leakage_scores(X_train, y_train, num_cols, cat_cols, is_clf, target_col).items()
                    if c not in keep}
        n_features = len(num_cols) + len(cat_cols) + len(dt_cols) + len(text_cols)
        if suspects:
            if n_features > len(suspects):
                for c, sc in suspects.items():
                    msg = (f"Столбец «{c}» почти полностью предсказывает цель (оценка {sc:.3f}) - похоже на утечку данных "
                           "(признак, выведенный из ответа или известный только после него). Он исключён из обучения. "
                           f"Если он законный, скажите «оставь столбец {c}».")
                    data_checks.append({"kind": "leak", "columns": [c], "score": round(sc, 4), "message": msg}); warnings_list.append(msg)
                X_train, X_test = X_train.drop(columns=list(suspects)), X_test.drop(columns=list(suspects))
                num_cols = [c for c in num_cols if c not in suspects]
                cat_cols = [c for c in cat_cols if c not in suspects]
            else:
                self._audit["unresolved_leak"] = list(suspects)      # the only feature(s): keep them, but say so loudly

        shares = pd.Series(y_train).value_counts(normalize=True) if is_clf else None
        balanced = bool(is_clf and len(shares) > 1 and shares.min() < self.IMBALANCE_THRESHOLD)
        pos_weight = None
        if balanced and len(shares) == 2:
            n0, n1 = int((np.asarray(y_train) == 0).sum()), int((np.asarray(y_train) == 1).sum())
            pos_weight = (n0 / n1) if n1 else None
        if balanced:
            notes.append(f"Дисбаланс (редкий класс {shares.min():.1%} в обучении): веса классов, отбор по "
                         + ("PR-AUC" if len(shares) == 2 else "macro-F1") + ", порог решения подбирается на обучающей части.")
        spec = FeatureSpec(num=num_cols, cat=cat_cols, dt=dt_cols, text=text_cols, max_categories=max_categories,
                           text_features=int(max(20, min(200, 4e7 // max(1, n_samples * max(1, len(text_cols)))))),
                           min_df=2 if len(X_train) >= 50 else 1, balanced=balanced, pos_weight=pos_weight)

        self._stage("prep", "done", f"{len(X_train)} train / {len(X_test)} test rows · "
                                    f"{len(num_cols)} numeric, {len(cat_cols)} categorical, {len(dt_cols)} date, {len(text_cols)} text features")
        if model_type == "auto":
            result = self.train_auto(X_train, X_test, y_train, y_test, task, spec, cv_folds,
                                     target_classes, decode_classes, warnings_list, budget)
        else:
            result = self._fit_and_finalize(X_train, X_test, y_train, y_test, task, model_type, spec, cv_folds,
                                            target_classes, decode_classes, warnings_list, budget)
        if notes:
            result.metrics["feature_notes"] = notes

        try:
            fingerprint = format(int(pd.util.hash_pandas_object(df, index=False).sum()) & (2**64 - 1), "016x")
        except Exception:
            fingerprint = None
        result.training_info = {"sklearn_version": sklearn.__version__, "n_rows": n_samples,
                                "trained_at": result.creation_time, "data_fingerprint": fingerprint,
                                "random_state": 42, "budget": budget}
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
