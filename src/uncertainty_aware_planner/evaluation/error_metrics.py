import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from uncertainty_aware_planner.utils.constants import (
    IMPUTATION_WINDOW_DURATION,
    NUMBER_HOURS_IN_DAY,
    OPEN_HOURS,
)
from uncertainty_aware_planner.data.helper_functions import get_series_hourly
from uncertainty_aware_planner.utils.util_functions import _require_columns, _valid_day_rows

# function constants
SERIES_KEYS = ["store_id", "product_id"]
DAY_KEYS = ["store_id", "product_id", "date"]


def calculate_imputation_percent_errors(
    eval_pypots,
    imputation_point_estimate,
):

    # initialize the proper shape for resizing
    # as these series have hours outside of the open hours
    n_series = eval_pypots["X"].shape[0]
    shape = (n_series, IMPUTATION_WINDOW_DURATION, NUMBER_HOURS_IN_DAY)

    # create a helper function that takes the defined shape,
    # reshapes the array, and takes the open hours from the day
    def open_hours(a):
        return a.reshape(shape)[..., OPEN_HOURS]

    # extract the components that matter from the censored and masked,
    # censored, and imputed NumPy arrays with NaN
    X = open_hours(eval_pypots["X"][..., 0])
    X_ori = open_hours(eval_pypots["X_ori"][..., 0])
    imputed = open_hours(imputation_point_estimate[..., 0])

    # separately assess masked-ness and true stockout
    masked = np.isnan(X) & ~np.isnan(X_ori)
    stockout = np.isnan(X_ori)

    # get the indices of the valid days
    day_valid = (masked.sum(axis=2) > 0) & (stockout.sum(axis=2) == 0)

    # convert original stockout back from NaN to 0.0
    # to get all hourly differences
    truth = np.nan_to_num(X_ori)
    hourly_diff = imputed - truth

    # select the appropriate days
    hours = np.broadcast_to(day_valid[..., None], hourly_diff.shape)

    # sum across all the relevant hours
    total_diff = hourly_diff[hours].sum()

    # sum absolute values across all relevant hours
    total_abs_diff = np.abs(hourly_diff)[hours].sum()

    # sum all sales over relevant hours
    total_actual = truth[hours].sum()

    # sample size is the number of valid hours
    n = int(hours.sum())

    if total_actual == 0:
        return {"wape": np.nan, "wpe": np.nan, "n": n}

    return {
        "wape": float(total_abs_diff / total_actual),
        "wpe": float(total_diff / total_actual),
        "n": n,
    }


def calculate_imputation_errors_df(
    imputed_hourly: pd.DataFrame,
    estimate_col: str = "estimated_demand",
) -> dict:
    """Hourly WAPE / WPE of imputed demand on masked in-stock hours.

    imputed_hourly: the `imputed_hourly` table for a single run
        (one model / seed / fold / split).
    estimate_col: "estimated_demand" or a calibrated point estimate.
    """
    # keep only open hours
    hour = pd.to_datetime(imputed_hourly["datetime"]).dt.hour
    in_open_hours = hour.between(OPEN_HOURS.start, OPEN_HOURS.stop, inclusive="left")

    df = _valid_day_rows(imputed_hourly[in_open_hours])

    # on valid days every row is a masked hour with ground truth
    diff = df[estimate_col] - df["actual_demand"]
    total_actual = df["actual_demand"].sum()
    n = len(df)

    if total_actual == 0:
        return {"wape": np.nan, "wpe": np.nan, "n": n}

    return {
        "wape": float(diff.abs().sum() / total_actual),
        "wpe": float(diff.sum() / total_actual),
        "n": n,
    }


def mean_absolute_scaled_error(
    imputed_df: pd.DataFrame,
    lag_hours: int = 24,
    cache_dir: Optional[Union[str, Path]] = None,
    series_hourly: Optional[pd.DataFrame] = None,
) -> dict:
    """MASE of `estimated_demand` against `actual_demand` on masked evaluation rows.

    The scale for each store-product series is the mean absolute difference
    between in-stock sales at hour t and hour t - lag_hours, using only pairs
    where both hours were observed. The masked evaluation hours themselves are
    excluded from the scale, so no evaluation truth leaks into the denominator.
    Pass `series_hourly` to avoid reloading the dataset on every call.
    """
    keys = list(SERIES_KEYS)
    _require_columns(imputed_df, ["actual_demand", "estimated_demand", "datetime", *keys])

    if series_hourly is None:
        series_hourly = get_series_hourly(cache_dir) if cache_dir is not None else get_series_hourly()
    _require_columns(series_hourly, [*keys, "datetime", "is_stockout", "scaled_sales_quantity"])

    # evaluation rows: masked in-stock hours with a known truth
    df = imputed_df.loc[
        imputed_df["actual_demand"].notna(),
        [*keys, "datetime", "actual_demand", "estimated_demand"],
    ].dropna()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # observed in-stock hours only, restricted to the series being evaluated
    obs = (
        series_hourly.loc[~series_hourly["is_stockout"], [*keys, "datetime", "scaled_sales_quantity"]]
        .rename(columns={"scaled_sales_quantity": "sales"})
        .merge(df[keys].drop_duplicates(), on=keys)
    )
    obs["datetime"] = pd.to_datetime(obs["datetime"])

    # anti-join: drop the masked evaluation hours from the scale data
    obs = obs.merge(
        df[[*keys, "datetime"]].drop_duplicates().assign(_eval=True),
        on=[*keys, "datetime"],
        how="left",
    )
    obs = obs[obs["_eval"].isna()].drop(columns="_eval")

    # naive-forecast scale, matched on timestamps so gaps in a series are
    # skipped instead of being bridged by a positional shift
    lagged = obs.assign(datetime=obs["datetime"] + pd.Timedelta(hours=lag_hours))
    pairs = obs.merge(lagged, on=[*keys, "datetime"], suffixes=("", "_lag"))
    scale = (
        pairs.assign(abs_diff=(pairs["sales"] - pairs["sales_lag"]).abs())
        .groupby(keys, as_index=False)["abs_diff"].mean()
        .rename(columns={"abs_diff": "scale"})
    )

    df = df.merge(scale, on=keys, how="left")

    # series with no usable lag pairs, or a constant series (scale 0), have no valid scale
    valid = df["scale"].notna() & (df["scale"] > 0)
    n_dropped = int((~valid).sum())
    df = df[valid]

    if df.empty:
        return {"value": float("nan"), "n": 0, "n_dropped_no_scale": n_dropped}

    scaled_err = (df["estimated_demand"] - df["actual_demand"]).abs() / df["scale"]
    return {
        "value": float(scaled_err.mean()),
        "n": len(df),
        "n_dropped_no_scale": n_dropped,
    }