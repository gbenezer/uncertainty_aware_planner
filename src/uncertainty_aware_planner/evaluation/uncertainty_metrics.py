from typing import TypedDict

import numpy as np
import pandas as pd


class MetricResult(TypedDict):
    value: float
    n: int  # number of rows the metric was computed over


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """helper function to check if the input DataFrame has the required
    columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def prediction_interval_coverage_fraction(
    actual_and_interval_df: pd.DataFrame,
    conformal_intervals: bool = False,
) -> MetricResult:
    # assumes CONTRACTS.md imputed_hourly columns exist
    lo, hi = ("q025_cal", "q975_cal") if conformal_intervals else ("q025", "q975")

    _require_columns(actual_and_interval_df, ["actual_demand", lo, hi])

    # Keep only rows with a real actual value and valid, non-degenerate quantiles
    df = actual_and_interval_df[["actual_demand", lo, hi]].dropna()
    df = df[~np.isclose(df[lo], df[hi])]

    if df.empty:
        return {"value": float("nan"), "n": 0}  # nothing to evaluate

    covered = df["actual_demand"].between(df[lo], df[hi])  # inclusive on both ends
    return {"value": float(covered.mean()), "n": len(df)}


def mean_prediction_interval_width(
    interval_df: pd.DataFrame,
    include_stockout: bool = False,
    conformal_intervals: bool = False,
) -> MetricResult:
    # assumes CONTRACTS.md imputed_hourly columns exist
    lo, hi = ("q025_cal", "q975_cal") if conformal_intervals else ("q025", "q975")

    _require_columns(interval_df, ["actual_demand", lo, hi])

    df = interval_df

    if not include_stockout:
        # intervals to be evaluated need to have non-null actual values
        df = df[df["actual_demand"].notna()]

    # valid, non-degenerate quantiles (same exclusion as the coverage metric)
    df = df[[lo, hi]].dropna()
    df = df[~np.isclose(df[lo], df[hi])]

    if df.empty:
        return {"value": float("nan"), "n": 0}  # nothing to evaluate

    return {"value": float((df[hi] - df[lo]).mean()), "n": len(df)}