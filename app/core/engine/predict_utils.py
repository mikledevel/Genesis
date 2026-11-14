"""Helpers for turning user-supplied feature values into model input.

Two registry formats exist side by side:

* ``input_format == "raw_dataframe"`` (models trained by DatasetTrainer from this version on):
  the saved artifact is a full sklearn Pipeline (imputation + one-hot encoding + model) that
  takes the ORIGINAL columns, so callers pass raw values - including categorical strings -
  and never have to reproduce encoding by hand.
* anything else (legacy models, LLM-generated pipelines): the model expects a plain numeric
  vector in the order of ``record["features"]``. That behaviour is unchanged.
"""
from typing import Any, Dict, List, Sequence
import math

import numpy as np

RAW_FORMAT = "raw_dataframe"


def is_raw_record(record: Dict) -> bool:
    return (record.get("metadata") or {}).get("input_format") == RAW_FORMAT


def coerce_categoricals(df, cat_cols: Sequence[str]):
    """Make categorical columns uniformly str (NaN stays NaN) so OneHotEncoder never sees a
    mix of ints and strings, which it cannot sort. Used identically at train and predict time."""
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(object).where(df[c].isna(), df[c].astype(str))
    return df


def _row_from_values(record: Dict, values: Dict[str, Any]) -> Dict[str, Any]:
    schema = (record.get("metadata") or {}).get("input_schema") or {}
    numeric = set(schema.get("numeric", []))
    dates = set(schema.get("datetime", []))
    texts = set(schema.get("text", []))
    row = {}
    for f in record.get("features", []):
        v = values[f]
        if v is None or isinstance(v, (dict, list)) or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"A value is required for {f}")
        if f in numeric:
            row[f] = _finite_number(v, f)
        elif f in dates:
            import pandas as pd
            if not isinstance(v, str) or pd.isna(pd.to_datetime(v, errors="coerce")):
                raise ValueError(f"A valid date is required for {f}")
            row[f] = v
        else:
            if f in texts and not isinstance(v, str):
                raise ValueError(f"Text is required for {f}")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                _finite_number(v, f)
            row[f] = str(v)
    return row


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"A number is required for {name}")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"A finite number is required for {name}") from error
    if not math.isfinite(number):
        raise ValueError(f"A finite number is required for {name}")
    return number


def build_model_input(record: Dict, values: Dict[str, Any]):
    """dict of {feature: value} -> model input. Raises KeyError for a missing feature and
    ValueError/TypeError for a value that doesn't fit its column type."""
    expected: List[str] = record.get("features", [])
    extra = set(values) - set(expected)
    if extra:
        raise ValueError("Unknown features: " + ", ".join(sorted(extra)))
    if is_raw_record(record):
        import pandas as pd
        return pd.DataFrame([_row_from_values(record, values)], columns=expected)
    return np.array([_finite_number(values[f], f) for f in expected]).reshape(1, -1)


def build_model_input_from_list(record: Dict, values: Sequence[Any]):
    """Positional variant used by the public predict endpoints (values in `features` order)."""
    expected: List[str] = record.get("features", [])
    # Same length check for every model kind (legacy numeric models used to skip it, so a wrong-length
    # list only failed later, inside model.predict, as a raw sklearn error).
    if expected and len(values) != len(expected):
        raise ValueError(f"Expected {len(expected)} values ({', '.join(expected)}), got {len(values)}")
    if is_raw_record(record):
        return build_model_input(record, dict(zip(expected, values)))
    return np.array([_finite_number(value, str(i)) for i, value in enumerate(values)]).reshape(1, -1)


def prediction_provenance(record):
    metadata = record.get("metadata") or {}
    metrics = {key: float(value) for key, value in record.get("metrics", {}).items()
               if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}
    limitations = ["Evaluation metrics describe the test data, not a guarantee for this individual prediction.",
                   "No causal explanation or feature attribution was computed for this prediction."]
    if not metrics:
        limitations.append("No evaluation metrics are recorded for this model.")
    if metadata.get("trust"):
        limitations.append("Model trust assessment: " + str(metadata["trust"])[:500])
    return {"model_version": record.get("version", 1), "model_family_id": record.get("family_id"),
            "task": record.get("task"), "evaluation_metrics": metrics, "limitations": limitations}
