"""Quantile forecasts -> the CONTRACTS.md 2.5 `forecast` table. Architecture-agnostic: any model that produces
an (n_samples, horizon, n_quantiles) array for a pytorch-forecasting dataset can use it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.cube import HIERARCHY

COLUMNS = HIERARCHY + [
    "date",
    "imputation_run_id",
    "forecast_model_id",
    "forecast_seed",
    "forecast_fold",
    "forecast_split",
    "forecast_run_id",
    "origin_date",
    "horizon",
    "target_date",
    "q025",
    "q975",
    "estimated_demand",
    "actual_demand",
    "q025_cal",
    "q975_cal",
    "qhat95",
]
LOWER, UPPER = 0.025, 0.975


def quantile_column(level: float) -> str:
    """CONTRACTS.md naming: 3-digit per-mille, e.g. 0.025 -> q025."""
    return f"q{round(level * 1000):03d}"


def run_id(model_id: str, seed: int) -> str:
    return f"{model_id}-{seed}"


def long_quantiles(
    q: np.ndarray,
    index: pd.DataFrame,
    decoder_lengths: np.ndarray,
    quantiles: list[float],
    start_date: date,
) -> tuple[pd.DataFrame, dict]:
    """One row per (series, origin, horizon) with every quantile column, monotone and clipped at 0.

    `index` holds the group ids and the time_idx of the first forecast day of each sample (pytorch-forecasting
    `return_index=True`); time_idx counts days from `start_date`.
    """
    n, horizon, n_q = q.shape
    if n_q != len(quantiles):
        raise ValueError(f"{n_q} quantile outputs but {len(quantiles)} quantile levels")
    order = np.argsort(quantiles)
    q = q[..., order]
    levels = [quantiles[i] for i in order]
    valid = np.arange(horizon)[None, :] < np.asarray(decoder_lengths)[:, None]

    crossing = (np.diff(q, axis=-1) < 0).any(axis=-1)[valid]
    q = np.sort(q, axis=-1)
    negative = (q < 0)[valid]
    q = np.clip(q, 0.0, None)

    rows = index.loc[index.index.repeat(horizon)].reset_index(drop=True)
    rows["horizon"] = np.tile(np.arange(1, horizon + 1), n).astype("uint8")
    start = pd.Timestamp(start_date)
    first_day = start + pd.to_timedelta(rows["time_idx"].astype("int64"), unit="D")
    rows["origin_date"] = (first_day - pd.Timedelta(days=1)).dt.date
    rows["target_date"] = (
        first_day + pd.to_timedelta(rows["horizon"].astype("int64") - 1, unit="D")
    ).dt.date
    for k, level in enumerate(levels):
        rows[quantile_column(level)] = q[..., k].reshape(-1)
    rows = rows[valid.reshape(-1)].drop(columns="time_idx").reset_index(drop=True)
    for column in ("store_id", "product_id"):
        rows[column] = rows[column].astype("int64")

    stats = {
        "n_rows": len(rows),
        "quantile_crossing_rate": float(crossing.mean()) if crossing.size else 0.0,
        "negative_clip_rate": float(negative.mean()) if negative.size else 0.0,
    }
    return rows, stats


def to_forecast_table(
    quantile_rows: pd.DataFrame,
    demand_daily: pd.DataFrame,
    model_id: str,
    seed: int,
    fold: int | None,
    split: str,
) -> pd.DataFrame:
    """Contract-exact table from `long_quantiles` rows. Hierarchy, imputation_run_id and actual_demand come
    from demand_daily; the conformal columns stay null until uq/ fills them.
    """
    lower, upper = quantile_column(LOWER), quantile_column(UPPER)
    missing = {lower, upper} - set(quantile_rows.columns)
    if missing:
        raise ValueError(
            f"the forecast table needs quantiles {LOWER} and {UPPER}; missing {sorted(missing)}"
        )

    daily = demand_daily.assign(
        target_date=pd.to_datetime(demand_daily["date"]).dt.date
    )
    hierarchy = daily[HIERARCHY].drop_duplicates(["store_id", "product_id"])
    actuals = daily[
        ["store_id", "product_id", "target_date", "imputation_run_id", "demand_total"]
    ]

    out = quantile_rows.merge(
        hierarchy, on=["store_id", "product_id"], how="left", validate="many_to_one"
    )
    out = out.merge(
        actuals,
        on=["store_id", "product_id", "target_date"],
        how="left",
        validate="many_to_one",
    )
    if out["city_id"].isna().any():
        raise ValueError("forecast rows for series that are not in demand_daily")
    out["imputation_run_id"] = out["imputation_run_id"].fillna(
        daily["imputation_run_id"].iloc[0]
    )

    out["date"] = out["target_date"]
    out["forecast_model_id"] = model_id
    out["forecast_seed"] = pd.array([seed] * len(out), dtype="UInt32")
    out["forecast_fold"] = pd.array([fold] * len(out), dtype="UInt8")
    out["forecast_split"] = split
    out["forecast_run_id"] = run_id(model_id, seed)
    out["q025"] = out[lower].astype("float64")
    out["q975"] = out[upper].astype("float64")
    out["estimated_demand"] = (out["q025"] + out["q975"]) / 2
    out["actual_demand"] = out["demand_total"].astype("float64")
    for column in ("q025_cal", "q975_cal", "qhat95"):
        out[column] = np.nan
    for column in HIERARCHY:
        out[column] = out[column].astype("int64")
    out = out[COLUMNS].sort_values(["store_id", "product_id", "origin_date", "horizon"])
    return out.reset_index(drop=True)
