"""Improve a weak, already-fitted model - and say honestly whether it worked.

    from app.core.model_improver import improve_model
    res = improve_model(model, df, target="churn", task="binary_classification")
    res.improved, res.model, res.report

`model` is what the registry holds: a fitted sklearn Pipeline (preprocessing steps ... final estimator) that takes a table with the
ORIGINAL columns. Nothing here depends on the platform's trainer; only on that shape.

What it does (and why it is built this way)
* DIAGNOSE. Score on rows kept out of every decision: balanced accuracy (classification) or R2 (regression), against the trivial
  baseline (always the majority class / always the mean). "Weak" = barely above the baseline, or a large train-vs-validation gap. The
  diagnosis is information: whether a model can be improved is not knowable from its score alone (a 0.70 model may have a 0.90 ceiling),
  so the search always runs; the accept rule below is what stops a good model being replaced by noise. only_if_weak=True skips the
  search for a model that is not weak (use it for batch jobs over many models).
* TRY, on the features the existing preprocessing already builds (dates, text, categories are not re-implemented here):
    - CONTINUE the existing estimator (warm start: more trees / more boosting rounds, trained further on the given rows). This is the
      real "incremental" update; it is only tried where the estimator supports it and the labels match.
    - REFIT the same settings, bigger, regularised, class-balanced variants, and a few other model families.
* DECIDE with three disjoint parts of the data: candidates are ranked on VALIDATION; the winner is refitted on train+validation; the
  untouched TEST part is looked at once, to accept or reject. Accept only if the gain is at least MIN_GAIN and a paired bootstrap says
  the winner is better in at least MIN_CONFIDENCE of resamples. Otherwise the ORIGINAL model is returned unchanged.
* data_is_new=True: the rows are data the model has not seen (the "new data arrived" case) - the model as it is, is the reference.
  data_is_new=False: the rows are (a copy of) the training data - the reference is the same settings refitted on the same split, because
  the original model has seen these rows and its score on them would be flattering.

Limits, stated: preprocessing is reused as fitted (unsupervised steps only; a supervised step such as target encoding or feature
selection refuses, because refitting around it needs the trainer); a random split is used (no time ordering); at least MIN_ROWS rows.
"""
from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import clone, is_classifier
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 60
MIN_GAIN = 0.01          # in metric units (balanced accuracy / R2)
MIN_CONFIDENCE = 0.95    # share of paired bootstrap resamples in which the winner must beat the reference
WEAK_MARGIN = 0.10       # weak if (score - trivial baseline) is below this
OVERFIT_GAP = 0.15       # weak if train score exceeds validation score by more than this
BOOTSTRAPS = 300
IMBALANCE_SHARE = 0.25   # a minority class below this share triggers class-balanced candidates
HGB_MAX_FEATURES = 3000  # HistGradientBoosting needs a dense matrix

_SUPERVISED = {"TargetEncoder", "SelectKBest", "SelectPercentile", "SelectFromModel", "RFE", "RFECV", "SequentialFeatureSelector",
               "SMOTE", "SMOTENC", "ADASYN", "BorderlineSMOTE", "RandomOverSampler", "RandomUnderSampler", "TomekLinks",
               "CatBoostEncoder", "WOEEncoder", "LeaveOneOutEncoder", "JamesSteinEncoder", "LinearDiscriminantAnalysis"}


class Unsupported(Exception):
    """The model or data is outside what this module can improve safely; the message says why."""


@dataclass
class ImproveResult:
    model: Any                      # the improved pipeline, or the ORIGINAL object when nothing helped
    improved: bool
    report: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------- structure
def _class_names(obj) -> List[str]:
    names = [type(obj).__name__]
    for sub in getattr(obj, "steps", []) or []:
        names += _class_names(sub[1])
    for t in getattr(obj, "transformers_", None) or getattr(obj, "transformers", None) or []:
        if len(t) >= 2 and not isinstance(t[1], str):
            names += _class_names(t[1])
    for t in getattr(obj, "transformer_list", None) or []:
        names += _class_names(t[1])
    return names


def _unwrap(model) -> Tuple[Optional[Pipeline], str, Any]:
    if not isinstance(model, Pipeline):
        raise Unsupported("the stored model is not a scikit-learn Pipeline (preprocessing + estimator), so it cannot be improved here")
    *prep, (est_name, est) = model.steps
    bad = sorted(set(_class_names(Pipeline(prep) if prep else model)) & _SUPERVISED) if prep else []
    if bad:
        raise Unsupported(f"preprocessing contains supervised steps ({', '.join(bad)}); reusing it on a new split would leak the target")
    for name, step in prep:
        if step not in (None, "passthrough") and not hasattr(step, "transform"):
            raise Unsupported(f"preprocessing step '{name}' has no transform (a sampler?), so it cannot be reused as it is")
    return (Pipeline(copy.deepcopy(prep)) if prep else None), est_name, est


def _features(model, df, target, features):
    if features:
        return list(features)
    names = getattr(model, "feature_names_in_", None)
    return [str(c) for c in names] if names is not None else [c for c in df.columns if c != target]


# ---------------------------------------------------------------------------------------------------------------- metric
def _kind(task: str) -> str:
    return "classification" if "classification" in str(task) else "regression"


def _score(kind, y, pred) -> float:
    return float(balanced_accuracy_score(y, pred) if kind == "classification" else r2_score(y, pred))


def _baseline(kind, y) -> float:
    return 1.0 / max(len(np.unique(y)), 1) if kind == "classification" else 0.0


def _paired_bootstrap(kind, y, pred_a, pred_b, seed) -> float:
    """Share of resamples of the test rows in which prediction B scores higher than A."""
    rng = np.random.RandomState(seed)
    n, wins = len(y), 0
    for _ in range(BOOTSTRAPS):
        idx = rng.randint(0, n, n)
        if kind == "classification" and len(np.unique(y[idx])) < 2:
            continue
        wins += _score(kind, y[idx], pred_b[idx]) > _score(kind, y[idx], pred_a[idx])
    return wins / BOOTSTRAPS


def _standard_metrics(kind, y_te, pred, est, X_te, y_trva) -> Dict[str, float]:
    """The metric keys the platform's trainer writes (accuracy_test, f1_test, roc_auc_test, r2_test, rmse_test ...), computed for the
    delivered model on the untouched test rows, so that an improved version sorts and displays like any other registered model
    (registry.find_best only considers records that have the metric asked for). These rows are NOT the trainer's test rows: the
    numbers of two versions are comparable only through the 'test_before_improvement' key next to them."""
    m: Dict[str, float] = {"n_test": int(len(y_te))}
    if kind == "classification":
        _, counts = np.unique(y_te, return_counts=True)
        m.update(accuracy_test=round(float(accuracy_score(y_te, pred)), 4), f1_test=round(float(f1_score(y_te, pred, average="weighted")), 4),
                 balanced_accuracy_test=round(float(balanced_accuracy_score(y_te, pred)), 4),
                 f1_macro_test=round(float(f1_score(y_te, pred, average="macro")), 4),
                 baseline_accuracy_test=round(float(counts.max() / len(y_te)), 4))
        classes = getattr(est, "classes_", None)
        if classes is not None and len(classes) == 2 and hasattr(est, "predict_proba"):
            try:
                proba = est.predict_proba(_dense_if_needed(X_te, _needs_dense(est)))[:, 1]
                pos = np.asarray(y_te) == classes[1]
                m.update(roc_auc_test=round(float(roc_auc_score(pos, proba)), 4), pr_auc_test=round(float(average_precision_score(pos, proba)), 4))
            except Exception:
                pass
    else:
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        m.update(r2_test=round(float(r2_score(y_te, pred)), 4), rmse_test=round(rmse, 2), mae_test=round(float(mean_absolute_error(y_te, pred)), 2),
                 baseline_rmse_test=round(float(np.sqrt(np.mean((np.asarray(y_te, dtype=float) - float(np.mean(y_trva))) ** 2))), 2))
    return m


# ---------------------------------------------------------------------------------------------------------------- candidates
def _warm_params(est):
    """How to continue this estimator, or None when it has no warm start. Returns (kind, extra_param_name)."""
    if isinstance(est, (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
                        GradientBoostingClassifier, GradientBoostingRegressor)):
        return "n_estimators"
    if isinstance(est, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        return "max_iter"
    try:
        import xgboost
        if isinstance(est, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
            return "xgb"
    except ImportError:
        pass
    return None


def _continue_fitter(est, extra_rounds: int) -> Optional[Callable]:
    how = _warm_params(est)
    if how is None:
        return None

    def fit(X, y):
        e = copy.deepcopy(est)
        if how == "xgb":
            e.set_params(n_estimators=extra_rounds)
            e.fit(X, y, xgb_model=est.get_booster())
        else:
            e.set_params(warm_start=True, **{how: getattr(e, how) + extra_rounds})
            e.fit(X, y)
        return e
    return fit


def _sizes(budget):
    return {"fast": (150, 1), "balanced": (300, 2), "thorough": (600, 3)}.get(budget, (300, 2))


def _candidates(est, kind, X_sparse, n_features, imbalanced, overfit, budget, seed, n_jobs, new_data, classes_ok) -> List[Tuple[str, Callable]]:
    trees, level = _sizes(budget)
    clf = kind == "classification"
    cw = {"class_weight": "balanced"} if (clf and imbalanced) else {}
    out: List[Tuple[str, Callable]] = []

    refit_same = lambda X, y: clone(est).fit(X, y)           # what a plain retrain would give
    out.append(("refit same settings", refit_same))

    cont = _continue_fitter(est, extra_rounds=max(trees // 2, 50)) if (new_data and classes_ok) else None
    if cont is not None:
        out.append(("continue training (warm start)", cont))
    if not new_data and _warm_params(est) is not None:        # on the training data itself: the same settings with more capacity
        bigger = {"n_estimators": trees * 2} if _warm_params(est) == "n_estimators" else {}
        if bigger:
            out.append(("same family, more trees", lambda X, y, e=est, b=bigger: clone(e).set_params(**b).fit(X, y)))

    RF = RandomForestClassifier if clf else RandomForestRegressor
    ET = ExtraTreesClassifier if clf else ExtraTreesRegressor
    HGB = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
    leaf = 5 if overfit else 1

    out.append((f"random forest {trees}", lambda X, y: RF(n_estimators=trees, min_samples_leaf=leaf, n_jobs=n_jobs, random_state=seed, **cw).fit(X, y)))
    if clf and imbalanced:
        out.append(("random forest, balanced subsample", lambda X, y: RF(n_estimators=trees, min_samples_leaf=leaf, n_jobs=n_jobs, random_state=seed,
                                                                         class_weight="balanced_subsample").fit(X, y)))
    if level >= 2:
        out.append((f"extra trees {trees}", lambda X, y: ET(n_estimators=trees, min_samples_leaf=leaf, n_jobs=n_jobs, random_state=seed, **cw).fit(X, y)))
        if not X_sparse and n_features <= HGB_MAX_FEATURES:
            out.append(("gradient boosting (hist)", lambda X, y: HGB(max_iter=trees, learning_rate=0.06, early_stopping=True, random_state=seed,
                                                                     **cw).fit(X, y)))
        lin = (lambda X, y: make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, C=1.0, **cw)).fit(X, y)) if clf \
            else (lambda X, y: make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)).fit(X, y))
        out.append(("linear model", lin))
    if level >= 3:
        if not X_sparse and n_features <= HGB_MAX_FEATURES:
            out.append(("gradient boosting, shallow+slow", lambda X, y: HGB(max_iter=trees * 2, learning_rate=0.03, max_depth=4, l2_regularization=1.0,
                                                                            early_stopping=True, random_state=seed, **cw).fit(X, y)))
        out.append(("random forest, shallow leaves", lambda X, y: RF(n_estimators=trees, min_samples_leaf=10, max_features="sqrt", n_jobs=n_jobs,
                                                                      random_state=seed, **cw).fit(X, y)))
    return out


# ---------------------------------------------------------------------------------------------------------------- main
def _dense_if_needed(X, need_dense: bool):
    return X.toarray() if (need_dense and sparse.issparse(X)) else X


def improve_model(model, df, target: str, task: str, *, features: Optional[List[str]] = None, data_is_new: bool = False,
                  budget: str = "balanced", seed: int = 42, only_if_weak: bool = False, max_seconds: float = 120.0,
                  n_jobs: int = 1) -> ImproveResult:
    """See the module docstring. Never raises for an unsuitable model/data: returns the original model and a report saying why."""
    started = time.time()
    report: Dict[str, Any] = {"status": "unchanged", "reason": "", "budget": budget, "data_is_new": bool(data_is_new), "tried": [],
                              "caveats": ["preprocessing is reused as it was fitted (unsupervised steps only)", "random split, no time ordering"]}
    try:
        return _improve(model, df, target, task, features, data_is_new, budget, seed, only_if_weak, max_seconds, n_jobs, started, report)
    except Unsupported as e:
        report.update(status="skipped", reason=str(e))
    except Exception as e:  # never take the platform down because of an exotic model
        report.update(status="skipped", reason=f"could not evaluate this model: {type(e).__name__}: {e}")
    report["seconds"] = round(time.time() - started, 2)
    return ImproveResult(model=model, improved=False, report=report)


def _improve(model, df, target, task, features, data_is_new, budget, seed, only_if_weak, max_seconds, n_jobs, started, report) -> ImproveResult:
    kind = _kind(task)
    prep, est_name, est = _unwrap(model)
    feats = _features(model, df, target, features)
    missing = [c for c in feats + [target] if c not in df.columns]
    if missing:
        raise Unsupported(f"the data lacks columns the model needs: {', '.join(map(str, missing[:5]))}")
    data = df[feats + [target]].dropna(subset=[target]).reset_index(drop=True)
    if len(data) < MIN_ROWS:
        raise Unsupported(f"only {len(data)} usable rows; at least {MIN_ROWS} are needed to tell an improvement from luck")
    y_all = data[target].to_numpy()
    classes_ok = True
    if kind == "classification":
        counts = {c: int((y_all == c).sum()) for c in np.unique(y_all)}
        if len(counts) < 2 or min(counts.values()) < 5:
            raise Unsupported("a class has fewer than 5 rows; the data cannot be split fairly")
        known = set(getattr(est, "classes_", getattr(model, "classes_", [])))
        if known and not set(counts) <= known:
            raise Unsupported(f"the data has labels the model has never seen: {sorted(map(str, set(counts) - known))[:5]}")
        classes_ok = (not known) or set(counts) == known
        imbalanced = min(counts.values()) / len(y_all) < IMBALANCE_SHARE
        strat = y_all
    else:
        imbalanced, strat = False, None

    try:
        X_all = prep.transform(data[feats]) if prep is not None else data[feats].to_numpy()
    except Exception as e:
        raise Unsupported(f"the existing preprocessing cannot transform this data ({type(e).__name__}: {e})")
    X_all = X_all.tocsr() if sparse.issparse(X_all) else np.asarray(X_all)
    is_sp, n_feat = sparse.issparse(X_all), X_all.shape[1]

    idx = np.arange(len(data))
    i_trva, i_te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    i_tr, i_va = train_test_split(i_trva, test_size=0.25, random_state=seed, stratify=None if strat is None else y_all[i_trva])
    X_tr, X_va, X_te, X_trva = X_all[i_tr], X_all[i_va], X_all[i_te], X_all[i_trva]
    y_tr, y_va, y_te, y_trva = y_all[i_tr], y_all[i_va], y_all[i_te], y_all[i_trva]
    report["rows"] = {"train": len(i_tr), "validation": len(i_va), "test": len(i_te)}
    report["metric"] = "balanced_accuracy" if kind == "classification" else "r2"
    base = _baseline(kind, y_te)
    report["baseline"] = round(base, 4)

    # reference: what we have to beat
    if data_is_new:
        ref_fit = None
        ref_val_est = est
        ref_note = "the model as it is (these rows are new to it)"
    else:
        ref_fit = lambda X, y: clone(est).fit(X, y)
        ref_val_est = ref_fit(_dense_if_needed(X_tr, _needs_dense(est)), y_tr)
        ref_note = "the same settings refitted on this split (the original has seen these rows)"
    report["reference"] = ref_note
    ref_va = _score(kind, y_va, ref_val_est.predict(_dense_if_needed(X_va, _needs_dense(est))))
    ref_tr = _score(kind, y_tr, ref_val_est.predict(_dense_if_needed(X_tr, _needs_dense(est))))
    diagnosis = []
    if ref_va - base < WEAK_MARGIN:
        diagnosis.append(f"barely better than the trivial baseline ({ref_va:.2f} vs {base:.2f})")
    if ref_tr - ref_va > OVERFIT_GAP:
        diagnosis.append(f"overfits: {ref_tr:.2f} on training rows vs {ref_va:.2f} on validation")
    if imbalanced:
        diagnosis.append("a rare class (under 25% of rows)")
    weak = (ref_va - base < WEAK_MARGIN) or (ref_tr - ref_va > OVERFIT_GAP)
    report.update(diagnosis=diagnosis, weak=bool(weak), validation_before=round(ref_va, 4))
    if only_if_weak and not report["weak"] and not data_is_new:
        report.update(status="unchanged", reason="the model is not weak (clearly above the baseline, no large overfit); not searched (only_if_weak)",
                      seconds=round(time.time() - started, 2))
        return ImproveResult(model=model, improved=False, report=report)

    # try candidates, rank on VALIDATION only
    cands = _candidates(est, kind, is_sp, n_feat, imbalanced, ref_tr - ref_va > OVERFIT_GAP, budget, seed, n_jobs, data_is_new, classes_ok)
    scored: List[Tuple[float, str, Callable]] = []
    for name, fit in cands:
        if time.time() - started > max_seconds:
            report["tried"].append({"name": name, "skipped": "time budget used up"})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = fit(X_tr, y_tr)
                v = _score(kind, y_va, m.predict(_dense_if_needed(X_va, _needs_dense(m))))
            scored.append((v, name, fit))
            report["tried"].append({"name": name, "validation": round(v, 4)})
        except Exception as e:
            report["tried"].append({"name": name, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    if not scored:
        raise Unsupported("no candidate could be fitted on this data")
    best_v, best_name, best_fit = max(scored, key=lambda t: t[0])
    report["winner_candidate"] = best_name
    if best_v <= ref_va:
        report.update(status="unchanged", reason=f"nothing beat the reference on validation (best: {best_name} {best_v:.3f} vs {ref_va:.3f})",
                      seconds=round(time.time() - started, 2))
        return ImproveResult(model=model, improved=False, report=report)

    # refit the winner on train+validation, then look at the untouched TEST rows once
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_est = best_fit(X_trva, y_trva)
        ref_est = ref_val_est if data_is_new else ref_fit(_dense_if_needed(X_trva, _needs_dense(est)), y_trva)
        pred_new = final_est.predict(_dense_if_needed(X_te, _needs_dense(final_est)))
        pred_ref = ref_est.predict(_dense_if_needed(X_te, _needs_dense(est)))
    s_new, s_ref = _score(kind, y_te, pred_new), _score(kind, y_te, pred_ref)
    gain, p = s_new - s_ref, _paired_bootstrap(kind, y_te, pred_ref, pred_new, seed)
    if kind == "regression":
        report["mae_before"], report["mae_after"] = round(float(mean_absolute_error(y_te, pred_ref)), 4), round(float(mean_absolute_error(y_te, pred_new)), 4)
    report.update(test_before=round(s_ref, 4), test_after=round(s_new, 4), gain=round(gain, 4), p_better=round(p, 3),
                  delivered_trained_on="train+validation; the test rows were used only for this accept/reject check")
    if gain < MIN_GAIN or p < MIN_CONFIDENCE:
        report.update(status="unchanged", reason=(f"{best_name} looked better on validation but the untouched test rows do not confirm it "
                                                  f"(gain {gain:+.3f}, better in {p:.0%} of resamples; needed gain >= {MIN_GAIN} and >= {MIN_CONFIDENCE:.0%})"),
                      seconds=round(time.time() - started, 2))
        return ImproveResult(model=model, improved=False, report=report)

    report["metrics_test"] = _standard_metrics(kind, y_te, pred_new, final_est, X_te, y_trva)
    steps = (list(copy.deepcopy(prep).steps) if prep is not None else []) + [(est_name, final_est)]
    new_model = Pipeline(steps)
    report.update(status="improved", reason=f"{best_name}: {s_ref:.3f} -> {s_new:.3f} on rows no decision used", seconds=round(time.time() - started, 2))
    return ImproveResult(model=new_model, improved=True, report=report)


def _needs_dense(est) -> bool:
    return isinstance(est, (HistGradientBoostingClassifier, HistGradientBoostingRegressor))

