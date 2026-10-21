"""Feature transformers that live INSIDE the saved model pipeline (so a raw row can be predicted),
plus the column-type detectors the trainer uses. Everything here is plain sklearn-style Python so it
pickles through app/core/safe_pickle.py (this module is on its allow-list)."""
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_MONTH_WORDS = r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"


def _parse(series, dayfirst: bool = False) -> "pd.Series":
    """Best-effort datetime parse (mixed formats allowed, unparseable -> NaT, tz-naive)."""
    try:
        d = pd.to_datetime(series, errors="coerce", format="mixed", utc=True, dayfirst=dayfirst)
    except (TypeError, ValueError):
        d = pd.to_datetime(series, errors="coerce", utc=True, dayfirst=dayfirst)
    return d.dt.tz_localize(None)


_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})")


def _infer_dayfirst(series) -> bool:
    """DD/MM/YYYY or MM/DD/YYYY? Pandas assumes month-first for ambiguous values, which silently turns
    3 May into 5 March for most of the world. Decide from the data: any first part > 12 means
    day-first, any second part > 12 means month-first; if every value is ambiguous, '.' (the
    Russian / European convention) means day-first and '/' keeps the month-first default.
    ISO dates (2024-05-03) are unaffected."""
    first_big = second_big = False
    sep = None
    for v in series.dropna().astype(str).head(2000):
        m = _NUMERIC_DATE.match(v)
        if not m:
            continue
        sep = sep or m.group(2)
        first_big |= int(m.group(1)) > 12
        second_big |= int(m.group(3)) > 12
    if first_big and not second_big:
        return True
    if second_big and not first_big:
        return False
    return sep == "."


class DateFeatures(BaseEstimator, TransformerMixin):
    """DataFrame of date columns -> numeric calendar features, per column:
    year, month, day, dayofweek, is_weekend, days (since 1970-01-01) and hour (only if the training
    data has a time of day). Unparseable / missing dates become NaN (a following imputer fills them)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = [str(c) for c in X.columns]
        self.use_hour_ = {}
        self.dayfirst_ = {}
        for c in X.columns:
            self.dayfirst_[str(c)] = _infer_dayfirst(X[c])
            d = _parse(X[c], self.dayfirst_[str(c)])
            self.use_hour_[str(c)] = bool(d.notna().any() and ((d.dt.hour != 0) | (d.dt.minute != 0)).any())
        self.feature_names_ = []
        for c in self.columns_:
            self.feature_names_ += [f"{c}_{p}" for p in ("year", "month", "day", "dayofweek", "is_weekend", "days")]
            if self.use_hour_[c]:
                self.feature_names_.append(f"{c}_hour")
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = []
        for c in X.columns:
            d = _parse(X[c], self.dayfirst_.get(str(c), False))
            cols += [d.dt.year, d.dt.month, d.dt.day, d.dt.dayofweek, (d.dt.dayofweek >= 5).astype(float).where(d.notna()),
                     (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0]
            if self.use_hour_.get(str(c)):
                cols.append(d.dt.hour)
        return np.column_stack([np.asarray(c, dtype=float) for c in cols]) if cols else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)


def detect_datetime_columns(df: "pd.DataFrame", cols: List[str], min_ratio: float = 0.9) -> List[str]:
    """Columns that are real datetimes, or text that parses as dates in >= 90% of a sample.
    Plain numbers ("20240101", "3.5") are never dates, and the text must look date-like."""
    out = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
            continue
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        sample = s.dropna().astype(str).head(300)
        if len(sample) < 5:
            continue
        if sample.str.fullmatch(r"\s*-?\d+([.,]\d+)?\s*").mean() > 0.5:
            continue
        if sample.str.contains(r"[-/:.]|" + _MONTH_WORDS, case=False, regex=True).mean() < 0.9:
            continue
        if _parse(sample).notna().mean() >= min_ratio:
            out.append(c)
    return out


def detect_text_columns(df: "pd.DataFrame", cols: List[str]) -> List[str]:
    """Free-text columns: multi-word, mostly unique values (reviews, comments, descriptions).
    Names ("John Smith", 2 words), codes/IDs (1 word) and low-cardinality labels stay categorical."""
    out = []
    for c in cols:
        s = df[c].dropna().astype(str)
        if len(s) < 20:
            continue
        avg_words = s.str.split().str.len().mean()
        avg_len = s.str.len().mean()
        unique_ratio = s.nunique() / len(s)
        if (avg_words >= 4 and unique_ratio > 0.3) or avg_len >= 60:
            out.append(c)
    return out


_ID_NAME = re.compile(r"(^|[_\s-])(id|idx|index|uuid|guid|no|num|number|код|номер|ид)([_\s-]|$)", re.IGNORECASE)


def looks_like_id(series: "pd.Series", name: str) -> bool:
    """A row identifier, not a measurement: (almost) every value unique AND the column behaves like a key -
    a string, an integer that counts up/down (a row number), or anything whose NAME says id/index/number.
    Continuous floats (avg_order_value) are never flagged on uniqueness alone."""
    n = len(series)
    if n < 50 or series.nunique(dropna=True) / n < 0.95:
        return False
    name_hit = bool(_ID_NAME.search(str(name))) or str(name).lower().startswith("unnamed")
    if pd.api.types.is_float_dtype(series):
        return name_hit
    if pd.api.types.is_integer_dtype(series):
        return name_hit or series.is_monotonic_increasing or series.is_monotonic_decreasing
    return True        # strings / objects with ~all-unique values
