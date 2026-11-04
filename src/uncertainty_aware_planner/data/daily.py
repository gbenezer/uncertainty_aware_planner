"""Step 6: demand_daily (CONTRACTS.md 2.4) from the cleaned data plus one imputed_hourly run.
Step 8: planner edit_log overlay (CONTRACTS.md 2.9). The source data and logs are never modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cube import COVARIATES, HIERARCHY, Cube
from .splits import final_split_labels

COLUMNS = (
    HIERARCHY
    + [
        "date",
        "imputation_run_id",
        "demand_observed",
        "demand_recovered",
        "demand_total",
        "stock_hour6_22_cnt",
    ]
    + COVARIATES
    + ["forecast_fold", "forecast_split"]
)


def recovered_demand(cube: Cube, imputed_hourly: pd.DataFrame) -> np.ndarray:
    """Daily sum of estimated demand over real stockout hours, shape (n_series, n_days).

    0 on days without stockout hours; NaN ("not computed") on days whose stockout hours were not imputed.
    Rows with a non-null actual_demand are masked evaluation hours and are ignored.
    """
    rows = imputed_hourly[imputed_hourly["actual_demand"].isna()]
    timestamps = pd.to_datetime(rows["datetime"])
    series = cube.series_index(rows["store_id"], rows["product_id"])
    day = ((timestamps.dt.normalize() - pd.Timestamp(cube.dates[0])).dt.days).to_numpy()
    hour = timestamps.dt.hour.to_numpy()
    if ((day < 0) | (day >= cube.n_days)).any():
        raise ValueError("imputed_hourly has rows outside the data range")
    estimate = rows["estimated_demand"].to_numpy(dtype=float)
    if np.isnan(estimate).any() or (estimate < 0).any():
        raise ValueError(
            "estimated_demand must be non-null and >= 0 (CONTRACTS.md 2.3)"
        )
    if not cube.stockout[series, day, hour].all():
        raise ValueError("imputed_hourly fills hours that are not stockout hours")

    filled = np.zeros(cube.stockout.shape, dtype=float)
    covered = np.zeros(cube.stockout.shape, dtype=bool)
    flat = np.ravel_multi_index((series, day, hour), cube.stockout.shape)
    if len(np.unique(flat)) != len(flat):
        raise ValueError("imputed_hourly has more than one row for the same hour")
    filled[series, day, hour] = estimate
    covered[series, day, hour] = True

    n_stockout = cube.stockout.sum(axis=-1)
    n_covered = covered.sum(axis=-1)
    if ((n_covered > 0) & (n_covered < n_stockout)).any():
        raise ValueError("some days have only part of their stockout hours imputed")
    return np.where(
        n_stockout == 0,
        0.0,
        np.where(n_covered == n_stockout, filled.sum(axis=-1), np.nan),
    )


def build_demand_daily(
    cube: Cube,
    imputed_hourly: pd.DataFrame,
    imputation_run_id: str,
    splits_cfg: dict,
    edit_log: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return demand_daily and the edit_ids applied (empty without an edit log)."""
    observed = np.where(cube.stockout, 0.0, cube.sales).sum(axis=-1).reshape(-1)
    recovered = recovered_demand(cube, imputed_hourly).reshape(-1)

    daily = cube.daily
    out = daily[HIERARCHY].copy()
    out["date"] = daily["date"]
    out["imputation_run_id"] = imputation_run_id
    out["demand_observed"] = observed
    out["demand_recovered"] = recovered
    out["demand_total"] = observed + recovered
    out["stock_hour6_22_cnt"] = daily["stock_hour6_22_cnt"].astype("int32")
    for column in COVARIATES:
        out[column] = daily[column]
    out["holiday_flag"] = out["holiday_flag"].astype("int32")
    out["activity_flag"] = out["activity_flag"].astype("int32")
    out["forecast_fold"] = pd.array([pd.NA] * len(out), dtype="UInt8")
    out["forecast_split"] = final_split_labels(splits_cfg, out["date"])
    out = out[COLUMNS]

    if edit_log is None or edit_log.empty:
        return out, []
    return apply_edit_log(out, edit_log)


def apply_edit_log(
    demand_daily: pd.DataFrame, edit_log: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Overlay planner edits on a copy of demand_daily. The newest edit per (store, product, date, field) wins.

    field "demand" replaces demand_total; a covariate name replaces that covariate.
    """
    editable = {"demand": "demand_total", **{c: c for c in COVARIATES}}
    if edit_log["reason"].fillna("").str.strip().eq("").any():
        raise ValueError("every edit needs a non-empty reason (CONTRACTS.md 2.9)")
    unknown = set(edit_log["field"]) - set(editable)
    if unknown:
        raise ValueError(
            f"edit_log has fields that cannot be edited: {sorted(unknown)}"
        )

    keys = ["store_id", "product_id", "date", "field"]
    latest = (
        edit_log.assign(date=pd.to_datetime(edit_log["date"]))
        .sort_values("edited_at", kind="stable")
        .drop_duplicates(keys, keep="last")
    )
    out = demand_daily.copy()
    position = pd.Series(
        np.arange(len(out)),
        index=pd.MultiIndex.from_arrays(
            [out["store_id"], out["product_id"], pd.to_datetime(out["date"])]
        ),
    )
    rows = position.reindex(
        pd.MultiIndex.from_frame(latest[["store_id", "product_id", "date"]])
    )
    if rows.isna().any():
        raise ValueError(
            f"{int(rows.isna().sum())} edits refer to store-product-days that do not exist"
        )
    for field, column in editable.items():
        chosen = (latest["field"] == field).to_numpy()
        if chosen.any():
            out.iloc[
                rows.to_numpy()[chosen].astype(int), out.columns.get_loc(column)
            ] = latest["new_value"].to_numpy()[chosen]
    return out, latest["edit_id"].astype(str).tolist()
