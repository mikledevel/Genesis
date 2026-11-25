"""Версионированный снимок прогноза и автономный экспорт сохранённых чисел."""
import csv
from datetime import datetime, timezone
import io
import math
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Series(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    dates: list[datetime] = Field(min_length=1, max_length=10000)
    values: list[float] = Field(min_length=1, max_length=10000)
    lower: list[float] | None = None
    upper: list[float] | None = None

    @model_validator(mode="after")
    def aligned(self):
        arrays = [self.values] + [a for a in (self.lower, self.upper) if a is not None]
        if any(len(a) != len(self.dates) for a in arrays):
            raise ValueError("Forecast arrays must have matching lengths")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Both interval bounds are required")
        if self.lower is not None and any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Invalid interval bounds")
        if any(a >= b for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("Forecast dates must be increasing")
        return self


class Forecast(BaseModel):
    target: str = Field(min_length=1, max_length=1000)
    freq: str = Field(min_length=1, max_length=100)
    history: Series
    future: Series


class Snapshot(BaseModel):
    schema_version: Literal[1]
    model_id: str = Field(min_length=1, max_length=200)
    model_version: int = Field(ge=1)
    model_type: str = Field(min_length=1, max_length=200)
    generated_at: datetime
    date_column: str = Field(min_length=1, max_length=1000)
    aggregation: Literal["sum", "mean", "last", "min", "max"]
    forecast: Forecast
    metrics: dict[str, Any]
    info: dict[str, Any]

    @model_validator(mode="after")
    def complete(self):
        future = self.forecast.future
        if future.lower is None or self.metrics.get("horizon") != len(future.dates):
            raise ValueError("Incomplete forecast snapshot")
        for name in ("mae_test", "baseline_mae_test"):
            value = self.metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing forecast evaluation")
        coverage = self.metrics.get("coverage_80_test")
        if coverage is not None and (not isinstance(coverage, (int, float)) or not 0 <= coverage <= 1):
            raise ValueError("Invalid interval coverage")
        return self


def create_snapshot(result, model_id, model_version, aggregation):
    history, future = result["history"], result["forecast"]
    def dates(values):
        return [pd.Timestamp(d).isoformat() for d in values]
    value = {"schema_version": 1, "model_id": model_id, "model_version": model_version,
             "model_type": result["model_type"], "generated_at": datetime.now(timezone.utc).isoformat(),
             "date_column": result["date_col"], "aggregation": aggregation,
             "forecast": {"target": result["target"], "freq": result["info"]["freq"],
                          "history": {"dates": dates(history.index), "values": list(map(float, history.values))},
                          "future": {"dates": dates(future["date"]), "values": list(map(float, future["forecast"])),
                                     "lower": list(map(float, future["lower"])), "upper": list(map(float, future["upper"]))}},
             "metrics": result["metrics"], "info": result["info"]}
    return Snapshot.model_validate(value).model_dump(mode="json")


def render_snapshot(value, lang, output_format):
    snapshot = Snapshot.model_validate(value)
    forecast = snapshot.forecast
    future = forecast.future
    if lang not in ("en", "ru") or output_format not in ("html", "csv"):
        raise ValueError("Unsupported report options")
    if output_format == "csv":
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(("date", "forecast", "lower", "upper"))
        writer.writerows((d.isoformat(), v, lo, hi) for d, v, lo, hi in
                         zip(future.dates, future.values, future.lower, future.upper))
        return "\ufeff" + output.getvalue()
    from app.core.report import render_forecast_result
    result = {"target": forecast.target, "date_col": snapshot.date_column,
              "horizon": len(future.dates), "metrics": snapshot.metrics, "info": snapshot.info,
              "history": pd.Series(forecast.history.values, index=pd.to_datetime(forecast.history.dates)),
              "forecast": pd.DataFrame({"date": pd.to_datetime(future.dates), "forecast": future.values,
                                        "lower": future.lower, "upper": future.upper})}
    provenance = (f"Модель: {snapshot.model_id}, версия {snapshot.model_version}." if lang == "ru" else
                  f"Model: {snapshot.model_id}, version {snapshot.model_version}.")
    return render_forecast_result(result, lang=lang, agg=snapshot.aggregation, notes=[provenance],
                                  generated_on=snapshot.generated_at.isoformat())["html"]
