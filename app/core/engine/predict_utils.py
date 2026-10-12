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
    row = {}
    for f in record.get("features", []):
        v = values[f]
        row[f] = float(v) if f in numeric else str(v)
    return row


def build_model_input(record: Dict, values: Dict[str, Any]):
    """dict of {feature: value} -> model input. Raises KeyError for a missing feature and
    ValueError/TypeError for a value that doesn't fit its column type."""
    expected: List[str] = record.get("features", [])
    if is_raw_record(record):
        import pandas as pd
        return pd.DataFrame([_row_from_values(record, values)], columns=expected)
    return np.array([float(values[f]) for f in expected]).reshape(1, -1)


def build_model_input_from_list(record: Dict, values: Sequence[Any]):
    """Positional variant used by the public predict endpoints (values in `features` order)."""
    expected: List[str] = record.get("features", [])
    if is_raw_record(record):
        if len(values) != len(expected):
            raise ValueError(f"Expected {len(expected)} values ({', '.join(expected)}), got {len(values)}")
        return build_model_input(record, dict(zip(expected, values)))
    return np.array(values, dtype=float).reshape(1, -1)
