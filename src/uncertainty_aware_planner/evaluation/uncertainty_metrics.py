"""Prediction-interval metrics: coverage (PICP) and mean width (MPIW).

Coverage functions also report the coverage gap (PICP - nominal coverage;
positive = over-covering, negative = under-covering).

Width functions also report the median and (sample, ddof=1) standard
deviation of the interval width.

Both the aggregate and the per-group functions return the same keys whether or
not any rows qualify (value=NaN, n=0).
"""

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from uncertainty_aware_planner.utils.util_functions import _require_columns

GroupCols = Union[str, Sequence[str]]
DEFAULT_GROUP_COLS = ("store_id", "product_id")
NOMINAL_COVERAGE = 0.95  # q025..q975


def _interval_columns(conformal_intervals: bool) -> tuple:
    # assumes CONTRACTS.md imputed_hourly/forecast columns exist
    return ("q025_cal", "q975_cal") if conformal_intervals else ("q025", "q975")


def _usable_intervals(
    df: pd.DataFrame,
    lo: str,
    hi: str,
    require_actual: bool,
    keep: Sequence[str] = (),
) -> pd.DataFrame:
    """Rows with non-null, non-degenerate bounds (and an actual, if required).

    `keep` lists extra columns (e.g. group keys) to carry through.
    """
    keep = list(keep)
    _require_columns(df, keep + ["actual_demand", lo, hi])

    sub = df[keep + ["actual_demand", lo, hi]]
    if require_actual:
        sub = sub[sub["actual_demand"].notna()]
    sub = sub.dropna(subset=[lo, hi])
    return sub[~np.isclose(sub[lo], sub[hi])]


def prediction_interval_coverage_fraction(
    actual_and_interval_df: pd.DataFrame,
    conformal_intervals: bool = False,
    nominal_coverage: float = NOMINAL_COVERAGE,
) -> Dict:
    lo, hi = _interval_columns(conformal_intervals)
    df = _usable_intervals(actual_and_interval_df, lo, hi, require_actual=True)

    covered = df["actual_demand"].between(df[lo], df[hi])  # inclusive on both ends
    picp = float(covered.mean())
    return {
        "value": picp,
        "coverage_gap": picp - nominal_coverage,
        "n_picp": len(df),
    }  # NaN, 0 when empty


def mean_prediction_interval_width(
    interval_df: pd.DataFrame,
    include_stockout: bool = False,
    conformal_intervals: bool = False,
) -> Dict:
    lo, hi = _interval_columns(conformal_intervals)
    df = _usable_intervals(interval_df, lo, hi, require_actual=not include_stockout)

    width = df[hi] - df[lo]
    return {
        "value": float(width.mean()),
        "median": float(width.median()),
        "std": float(width.std()),  # sample std (ddof=1); NaN when n < 2
        "n_mpiw": len(df),
    }  # NaN, 0 when empty


# --------------------------------------------------------------------------
# Per-group (default: store_id x product_id) versions
# --------------------------------------------------------------------------


def _as_group_list(group_cols: GroupCols) -> list:
    return [group_cols] if isinstance(group_cols, str) else list(group_cols)


def _per_group_metric(
    df: pd.DataFrame,
    group_cols: list,
    lo: str,
    hi: str,
    require_actual: bool,
    row_value: Callable[[pd.DataFrame], pd.Series],
    value_name: str,
    n_name: str,
    extra_stats: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Mean of a per-row quantity over usable rows, for every group in `df`.

    `extra_stats` maps output column name -> pandas agg name (e.g. "median").

    Every group present in the input appears in the output; groups with no
    usable rows get value=NaN and n=0 (mirrors the aggregate functions).
    """
    _require_columns(df, group_cols)
    keys = df[group_cols].drop_duplicates().reset_index(drop=True)

    usable = _usable_intervals(df, lo, hi, require_actual, keep=group_cols).copy()
    usable["_v"] = row_value(usable)

    extra_stats = extra_stats or {}
    agg = (
        usable.groupby(group_cols, dropna=False)["_v"]
        .agg(**{value_name: "mean", **extra_stats, n_name: "size"})
        .reset_index()
    )

    out = keys.merge(agg, on=group_cols, how="left")
    out[n_name] = out[n_name].fillna(0).astype(int)
    for col in [value_name, *extra_stats]:
        out[col] = out[col].astype(float)
    return out


def prediction_interval_coverage_fraction_by_group(
    actual_and_interval_df: pd.DataFrame,
    group_cols: GroupCols = DEFAULT_GROUP_COLS,
    conformal_intervals: bool = False,
    nominal_coverage: float = NOMINAL_COVERAGE,
) -> pd.DataFrame:
    """PICP per group. Columns: *group_cols, picp, coverage_gap, n_picp."""
    lo, hi = _interval_columns(conformal_intervals)
    out = _per_group_metric(
        actual_and_interval_df,
        _as_group_list(group_cols),
        lo,
        hi,
        require_actual=True,
        row_value=lambda d: d["actual_demand"].between(d[lo], d[hi]).astype(float),
        value_name="picp",
        n_name="n_picp",
    )
    out.insert(out.columns.get_loc("picp") + 1, "coverage_gap", out["picp"] - nominal_coverage)
    return out


def mean_prediction_interval_width_by_group(
    interval_df: pd.DataFrame,
    group_cols: GroupCols = DEFAULT_GROUP_COLS,
    include_stockout: bool = False,
    conformal_intervals: bool = False,
) -> pd.DataFrame:
    """Width stats per group. Columns: *group_cols, mpiw, median_piw, std_piw, n_mpiw."""
    lo, hi = _interval_columns(conformal_intervals)
    return _per_group_metric(
        interval_df,
        _as_group_list(group_cols),
        lo,
        hi,
        require_actual=not include_stockout,
        row_value=lambda d: d[hi] - d[lo],
        value_name="mpiw",
        n_name="n_mpiw",
        extra_stats={"median_piw": "median", "std_piw": "std"},
    )


def interval_metrics_by_group(
    df: pd.DataFrame,
    group_cols: GroupCols = DEFAULT_GROUP_COLS,
    include_stockout: bool = False,
    conformal_intervals: bool = False,
    nominal_coverage: float = NOMINAL_COVERAGE,
) -> pd.DataFrame:
    """PICP and MPIW side by side, one row per group.

    Columns: *group_cols, picp, coverage_gap, n_picp, mpiw, median_piw,
    std_piw, n_mpiw.

    PICP always requires an actual; MPIW only does when include_stockout is
    False, so n_picp and n_mpiw can differ when include_stockout=True.
    """
    cols = _as_group_list(group_cols)
    picp = prediction_interval_coverage_fraction_by_group(
        df,
        cols,
        conformal_intervals=conformal_intervals,
        nominal_coverage=nominal_coverage,
    )
    mpiw = mean_prediction_interval_width_by_group(
        df,
        cols,
        include_stockout=include_stockout,
        conformal_intervals=conformal_intervals,
    )
    return picp.merge(mpiw, on=cols, how="outer")