"""Imputation error metrics: percent errors (WAPE / WPE) and MASE.

Every metric returns a dict with the same keys whether or not the slice it was
given is empty, so results can be flattened into a fixed set of columns.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.helper_functions import get_series_hourly
from uncertainty_aware_planner.utils.constants import (
    IMPUTATION_WINDOW_DURATION,
    NUMBER_HOURS_IN_DAY,
    OPEN_HOURS,
    SERIES_KEYS,
)
from uncertainty_aware_planner.utils.util_functions import (
    _require_columns,
    _valid_day_rows,
)


# ------------------------------------------------------------ percent errors
def _percent_error_summary(total_abs_diff, total_diff, total_actual, n) -> dict:
    """WAPE / WPE as a dict; NaN when there is no actual demand to divide by."""
    if total_actual == 0:
        return {"wape": np.nan, "wpe": np.nan, "n_pe": n}
    return {
        "wape": float(total_abs_diff / total_actual),
        "wpe": float(total_diff / total_actual),
        "n_pe": n,
    }


def _in_open_hours(datetimes: pd.Series) -> pd.Series:
    """ takes a pandas Series of datetime information (either as str or pd.Timestamp),
    extracts the hour information, and generates a boolean pandas Series indicating 
    whether the row data was collected during the open hours of the store.
    """
    hour = pd.to_datetime(datetimes).dt.hour
    return hour.between(OPEN_HOURS.start, OPEN_HOURS.stop, inclusive="left")


def calculate_imputation_percent_errors(eval_pypots, imputation_point_estimate) -> dict:
    """WAPE / WPE from PyPOTS arrays, over open hours of valid days."""
    n_series = eval_pypots["X"].shape[0]
    shape = (n_series, IMPUTATION_WINDOW_DURATION, NUMBER_HOURS_IN_DAY)

    # series carry hours outside opening times, so reshape to days x hours
    # and keep only the open hours with an internal helper function
    def open_hours(a):
        return a.reshape(shape)[..., OPEN_HOURS]

    # extract the components that matter from the censored and masked,
    # censored, and imputed NumPy arrays with NaN
    X = open_hours(eval_pypots["X"][..., 0])
    X_ori = open_hours(eval_pypots["X_ori"][..., 0])
    imputed = open_hours(imputation_point_estimate)

    # masked = hidden by the eval masks; stockout = NaN in the original data
    masked = np.isnan(X) & ~np.isnan(X_ori)
    stockout = np.isnan(X_ori)

    # a valid day has at least one masked hour and no stockout hours
    day_valid = (masked.sum(axis=2) > 0) & (stockout.sum(axis=2) == 0)
    valid_hours = np.broadcast_to(day_valid[..., None], imputed.shape)

    # stockouts only occur on invalid days, so zero-filling them is harmless
    truth = np.nan_to_num(X_ori)
    diff = (imputed - truth)[valid_hours]

    return _percent_error_summary(
        np.abs(diff).sum(), diff.sum(), truth[valid_hours].sum(), int(valid_hours.sum())
    )


def _open_hour_sales_on_valid_days(valid_rows, series_hourly) -> tuple[float, int]:
    """Private helper function that outputs the tuple (total sales, hour count) tuple
    over every open hour of the (series, day) pairs present in `valid_rows`, 
    observed hours included."""
    keys = list(SERIES_KEYS) # technically already a list, but it's a harmless no-op

    # filter out unique [store_id, product_id, date] combinations
    # assumes that valid_rows contains a 'date' column that is either a datetime or
    # can be converted to one, and renames it to 'day'
    valid_days = valid_rows[keys].assign(day=pd.to_datetime(valid_rows["date"]))
    valid_days = valid_days.drop_duplicates()

    # valid days have no open-hour stockouts, so this is exactly the masked
    # hours plus the observed hours

    # selects the proper columns from the full dataset series_hourly
    # pandas DataFrame, then selects only those rows during the
    # stores open hours
    sh = series_hourly[[*keys, "datetime", "scaled_sales_quantity"]]
    sh = sh[_in_open_hours(sh["datetime"])]

    # set the hour component to midnight with .normalize(), then
    # inner join (pandas default) on the valid days
    sh = sh.assign(day=sh["datetime"].dt.normalize()).merge(
        valid_days, on=[*keys, "day"]
    )

    # return the sum of actual masked and observed demand, along with
    # the sample size
    return float(sh["scaled_sales_quantity"].sum()), len(sh)


def calculate_imputation_errors_df(
    imputed_hourly: pd.DataFrame,
    estimate_col: str = "estimated_demand",
    include_observed: bool = False,
    series_hourly: Optional[pd.DataFrame] = None,
) -> dict:
    """Hourly WAPE / WPE of imputed demand on masked in-stock hours.

    imputed_hourly: the `imputed_hourly` table for a single run
        (one model / seed / fold / split).
    estimate_col: "estimated_demand" or a calibrated point estimate.
    include_observed: emulate `calculate_imputation_percent_errors`, which
        scores every open hour of a valid day. Observed hours were visible to
        the model, so they add true sales to the denominator and sample size
        but nothing to the error. Requires `series_hourly`.
    series_hourly: full true hourly table; only used when include_observed=True.
    """
    if include_observed and series_hourly is None:
        raise ValueError("include_observed=True requires series_hourly")

    # filter to rows from days that have at least one masked hour 
    # and no true stockout.
    df = _valid_day_rows(imputed_hourly[_in_open_hours(imputed_hourly["datetime"])])

    # on valid days every row is a masked hour with ground truth
    # so calculate residuals, and then sum them
    diff = df[estimate_col] - df["actual_demand"]
    total_actual, n = df["actual_demand"].sum(), len(df)

    # if the zero-error observed data should be included (inflating the denominator and
    # reducing WAPE/WPE artificially, but matching dataset author convention), use the
    # specialized helper function to calculate the total instead
    if include_observed:
        total_actual, n = _open_hour_sales_on_valid_days(df, series_hourly)

    # use the helper function to map the appropriate numbers to an output dictionary
    return _percent_error_summary(diff.abs().sum(), diff.sum(), total_actual, n)


# ---------------------------------------------------------------------- MASE
def _ase_column(lag_hours: int) -> str:
    """Simple helper function giving the proper column name for a given
    absolute scaled error lag.""" 
    return f"ase_{lag_hours}hr"


def _observed_hours(
    eval_rows: pd.DataFrame, series_hourly: pd.DataFrame
) -> pd.DataFrame:
    """Observed in-stock hours of the evaluated series, minus the evaluation hours.

    Dropping the evaluation hours keeps their ground truth out of the scale.
    Independent of the lag, so it is built once and reused for every lag.
    """
    keys = list(SERIES_KEYS)

    obs = (
        # select the rows in the full ground truth dataset that are not stockout,
        # and only retrieve the relevant columns
        series_hourly.loc[
            ~series_hourly["is_stockout"], [*keys, "datetime", "scaled_sales_quantity"]
        ]
        # rename the long confusing name to just sales, with the data context implying
        # global scaling and not absolute units, then inner join (pandas default) on
        # the evaluation rows
        .rename(columns={"scaled_sales_quantity": "sales"})
        .merge(eval_rows[keys].drop_duplicates(), on=keys)
    )
    # convert the datetime to an official pandas Timestamp Series if not already
    obs["datetime"] = pd.to_datetime(obs["datetime"])

    # indirect anti-join observed data on (series, hour) to remove the hours 
    # that are evaluated from the scale calculations downstream to prevent 
    # information leakage
    eval_hours = eval_rows[[*keys, "datetime"]].drop_duplicates().assign(_eval=True)
    obs = obs.merge(eval_hours, on=[*keys, "datetime"], how="left")
    return obs[obs["_eval"].isna()].drop(columns="_eval")


def _naive_scale(obs: pd.DataFrame, lag_hours: int) -> pd.Series:
    """Per-series mean |sales_t - sales_{t - lag}|, indexed by SERIES_KEYS
    ([store_id, product_id]).

    Pairs are matched on timestamps, so gaps in a series are skipped instead of
    being bridged by a positional shift.
    """
    keys = list(SERIES_KEYS) # again, another harmless no-op

    # create a view with an overwritten datetime that is ahead in time by lag_hours
    lagged = obs.assign(datetime=obs["datetime"] + pd.Timedelta(hours=lag_hours))

    # inner join original data with the data that has advanced time, removing
    # rows without appropriate pairs; sales_lag is the scaled sales figure that
    # has the advanced time equal to the sales timestamp, recovering the appropriate
    # time lagged information
    pairs = obs.merge(lagged, on=[*keys, "datetime"], suffixes=("", "_lag"))

    # calculate the absolute backwards difference
    abs_diff = (pairs["sales"] - pairs["sales_lag"]).abs()

    # group by [store_id, product_id], then calculate mean and rename it 'scale'
    return abs_diff.groupby([pairs[k] for k in keys]).mean().rename("scale")


def scaled_absolute_errors(
    imputed_df: pd.DataFrame,
    lags: Sequence[int] = (24,),
    series_hourly: Optional[pd.DataFrame] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Per-row absolute scaled errors of `estimated_demand` vs `actual_demand`.

    Compute this once on the full imputed frame, then slice it by row index to
    get any subset's MASE with `summarize_scaled_errors`. Every subset then
    shares one scale per series, and that scale excludes all evaluation hours
    (not just the subset's), so no evaluation truth leaks into it.

    Returns a frame indexed like `imputed_df` (whose index must be unique) with
        evaluated:     True for masked in-stock rows with an estimate
        ase_{lag}hr:   |estimate - actual| / scale for each lag; NaN where the
                       row was not evaluated or its series has no valid scale
                       (no usable lag pairs, or a constant series)
    """
    keys = list(SERIES_KEYS) # another no-op

    # checking if the columns necessary for calculations exist and raising a
    # KeyError if not
    _require_columns(
        imputed_df, ["actual_demand", "estimated_demand", "datetime", *keys]
    )

    # checking for uniqueness
    if not imputed_df.index.is_unique:
        raise ValueError("imputed_df must have a unique index")

    # initialize the full ground truth hourly dataset as efficiently as possible
    if series_hourly is None:
        series_hourly = (
            get_series_hourly(cache_dir)
            if cache_dir is not None
            else get_series_hourly()
        )
    _require_columns(
        series_hourly, [*keys, "datetime", "is_stockout", "scaled_sales_quantity"]
    )

    # evaluation rows: masked in-stock hours with a known truth

    # select the rows where there exists an actual_demand entry,
    # then drop all rows that have NaN in other required fields
    ev = imputed_df.loc[
        imputed_df["actual_demand"].notna(),
        [*keys, "datetime", "actual_demand", "estimated_demand"],
    ].dropna()

    # convert the datetime to a pandas Timestamp Series if it's not already
    ev = ev.assign(datetime=pd.to_datetime(ev["datetime"]))

    # calculate the residuals and their absolute magnitude
    abs_err = (ev["estimated_demand"] - ev["actual_demand"]).abs()

    # filter the relevant ground truth observed hours that were not masked
    # for calculating the relevant denominator without information leakage
    obs = _observed_hours(ev, series_hourly)

    # create a DataFrame with a boolean Series denoting which rows were and
    # were not evaluated for the scaled absolute error numerator
    out = pd.DataFrame(
        {"evaluated": imputed_df.index.isin(ev.index)}, index=imputed_df.index
    )

    # across all requested lags to calculate
    for lag in lags:

        # inner join the evaluated hours with their 
        # appropriate store-product mean
        # then extract those means
        scale = ev.join(
            # generates an appropriate set of all store-product means 
            # for a particular lag
            _naive_scale(obs, lag), 
            on=keys
            )["scale"]
        # NaN scale (no pairs) and zero scale (constant series) are both invalid
        out[_ase_column(lag)] = (abs_err / scale).where(scale > 0)
    return out


def summarize_scaled_errors(errors: pd.DataFrame, lag_hours: int = 24) -> dict:
    """MASE and counts for (a slice of) the output of `scaled_absolute_errors`."""
    # errors is assumed a DataFrame with absolute scaled errors as defined in the
    # `scaled_absolute_errors` function docstring
    scaled = errors[_ase_column(lag_hours)].dropna()
    return {
        "value": float(scaled.mean()),  # NaN when empty
        "n_mase": len(scaled),
        "n_mase_dropped_no_scale": int(errors["evaluated"].sum()) - len(scaled),
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

    For several lags or several subsets of one frame, call
    `scaled_absolute_errors` once and `summarize_scaled_errors` per slice.
    """
    errors = scaled_absolute_errors(imputed_df, [lag_hours], series_hourly, cache_dir)
    return summarize_scaled_errors(errors, lag_hours)
