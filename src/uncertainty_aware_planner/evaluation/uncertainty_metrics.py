"""Prediction-interval metrics: coverage (PICP) and mean width (MPIW).

Both return a dict with the same keys whether or not any rows qualify.
"""

from typing import Dict

import numpy as np
import pandas as pd

from uncertainty_aware_planner.utils.util_functions import _require_columns


def _interval_columns(conformal_intervals: bool) -> tuple:
    # assumes CONTRACTS.md imputed_hourly/forecast columns exist
    return ("q025_cal", "q975_cal") if conformal_intervals else ("q025", "q975")


def _usable_intervals(
    df: pd.DataFrame, lo: str, hi: str, require_actual: bool
) -> pd.DataFrame:
    """Rows with non-null, non-degenerate bounds (and an actual, if required)."""
    _require_columns(df, ["actual_demand", lo, hi])

    sub = df[["actual_demand", lo, hi]]
    if require_actual:
        sub = sub[sub["actual_demand"].notna()]
    sub = sub.dropna(subset=[lo, hi])
    return sub[~np.isclose(sub[lo], sub[hi])]


def prediction_interval_coverage_fraction(
    actual_and_interval_df: pd.DataFrame,
    conformal_intervals: bool = False,
) -> Dict:
    lo, hi = _interval_columns(conformal_intervals)
    df = _usable_intervals(actual_and_interval_df, lo, hi, require_actual=True)

    covered = df["actual_demand"].between(df[lo], df[hi])  # inclusive on both ends
    return {"value": float(covered.mean()), "n_picp": len(df)}  # NaN, 0 when empty


def mean_prediction_interval_width(
    interval_df: pd.DataFrame,
    include_stockout: bool = False,
    conformal_intervals: bool = False,
) -> Dict:
    lo, hi = _interval_columns(conformal_intervals)
    df = _usable_intervals(interval_df, lo, hi, require_actual=not include_stockout)

    return {
        "value": float((df[hi] - df[lo]).mean()),
        "n_mpiw": len(df),
    }  # NaN, 0 when empty
