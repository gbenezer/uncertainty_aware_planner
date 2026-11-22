# file to store developer-friendly helper functions
from typing import Optional, Union
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from uncertainty_aware_planner.utils.constants import (
    DAILY_METADATA_COLS,
    DAY_KEYS,
    SERIES_KEYS,
)


def _hidden_hours(dataset, index):
    """Positions and ground-truth columns for every hidden hour a sample owns."""

    # model input: NaN wherever an hour is hidden, either by a real stockout
    # (censored) or by the eval masks (masked)
    x = dataset["X"][..., 0]

    # get the sales that are censored and not masked, if given,
    # otherwise default to the base dataset
    truth = dataset.get("X_ori", dataset["X"])[..., 0]

    # define an anonymous function to move from an input column name
    # to the column in the index dataframe, to a numpy element, and
    # finally convert it to a datetime in terms of hours
    hours = lambda col: pd.to_datetime(index[col]).to_numpy().astype("datetime64[h]")

    # in the numpy array, the store-product series is the row
    # and the hour is the column, so this gets the entries that
    # are censored or masked
    sample, step = np.nonzero(np.isnan(x))

    # window start (in hours) plus the step offset gives each hour's
    # absolute timestamp
    stamp = hours("start_date")[sample] + step.astype("timedelta64[h]")

    # not sure, something about removing possible duplicates
    own = stamp >= hours("keep_start")[sample]
    sample, step, stamp = sample[own], step[own], stamp[own]

    actual = truth[sample, step]  # NaN at real stockouts, true sales at masked hours
    base = pd.DataFrame(
        {
            "store_id": index["store_id"].to_numpy()[sample],
            "product_id": index["product_id"].to_numpy()[sample],
            "datetime": stamp.astype("datetime64[us]"),
            "actual_demand": actual,
            "is_stockout": np.isnan(actual),  # true censoring: real stockout hour
            "is_masked": ~np.isnan(
                actual
            ),  # artificial censoring: hidden by eval masks
        }
    )
    # add date metadata to all tables deriving from this function
    # (same columns and dtypes as get_series_hourly)
    dt = base["datetime"].dt
    base = base.assign(
        date=dt.normalize().astype("datetime64[us]"),
        hour_idx=dt.hour.astype("int8"),
        day_of_week=dt.day_of_week.astype("int32"),
        weekday=dt.day_of_week < 5,
    )
    # derive time from datetime - date, as get_series_hourly does (datetime = date + time)
    base["time"] = (base["datetime"] - base["date"]).astype("timedelta64[s]")
    return base, sample, step


def _estimate(imputation, sample, step, clip_demand):
    # extract the imputation data
    # at the censored and masked times across relevant series
    est = imputation[sample, step]

    # if requested, clip the values to zero
    # given negative demand is meaningless
    return np.clip(est, 0, None) if clip_demand else est


def to_imputed_hourly(dataset, index, imputation, clip_demand: bool = True):
    # get the base dataframe, the indices of valid series,
    # and the indices of valid hours
    base, sample, step = _hidden_hours(dataset, index)

    # add the estimated demand to the base dataframe
    return base.assign(
        estimated_demand=_estimate(imputation, sample, step, clip_demand)
    )


def to_imputed_hourly_interval(dataset, index, imputation, clip_demand: bool = True):

    assert imputation.shape[-1] == 2, (
        f"expected (lower, upper) quantiles as last dimension, got {imputation.shape}"
    )

    # get the base dataframe, the indices of valid series,
    # and the indices of valid hours
    base, sample, step = _hidden_hours(dataset, index)

    # create an anonymous function to get the other quantiles
    q = lambda k: _estimate(imputation[..., 0, k], sample, step, clip_demand)

    # construct the demand estimate given there are only intervals
    demand_estimate = (q(0) + q(1)) * 0.5

    # assign the proper values to the proper column names and return
    return base.assign(q025=q(0), estimated_demand=demand_estimate, q975=q(1))


def to_imputed_hourly_interval_median(
    dataset, index, imputation, clip_demand: bool = True
):

    assert imputation.shape[-1] == 3, (
        f"expected (lower, median, upper) quantiles as last dimension, got {imputation.shape}"
    )

    # get the base dataframe, the indices of valid series,
    # and the indices of valid hours
    base, sample, step = _hidden_hours(dataset, index)

    # create an anonymous function to get the other quantiles
    q = lambda k: _estimate(imputation[..., 0, k], sample, step, clip_demand)

    # assign the proper values to the proper column names and return
    return base.assign(q025=q(0), estimated_demand=q(1), q975=q(2))


def get_series_hourly(
    cache_dir: Optional[Union[str, Path]] = "data/hf_cache",
) -> pd.DataFrame:
    """Circumvents the majority of the data pipeline machinery to get the required `series_hourly`
    dataframe from CONTRACTS.md because it is non-obvious how to best do so from a Cube object"""

    # load the dataset
    ds = load_dataset(
        "Dingdong-Inc/FreshRetailNet-50K", cache_dir=cache_dir
    )  # load once

    # initialize a list for the dataframes
    frames = []

    # initialize a constant for the number of hours in the day
    n_hours = 24

    for split in ("train", "eval"):
        # get the proper dataset and its size
        d = ds[split]
        n = len(d)

        # Everything except the two list columns
        scalar_cols = [
            c for c in d.column_names if c not in ("hours_sale", "hours_stock_status")
        ]
        base = d.select_columns(scalar_cols).to_pandas()

        # List columns -> (n, 24) NumPy arrays directly
        sales = np.asarray(d["hours_sale"], dtype=np.float32)
        stock = np.asarray(d["hours_stock_status"], dtype=np.int8)

        # Repeat each scalar row 24 times (equivalent to explode), then attach flattened arrays
        out = base.loc[base.index.repeat(n_hours)].reset_index(drop=True)
        hour_idx = np.tile(np.arange(n_hours, dtype=np.int8), n)

        out["hour_idx"] = hour_idx
        out["scaled_sales_quantity"] = sales.ravel()
        out["is_stockout"] = stock.ravel().astype(bool)

        # Vectorized datetime: parse dates once, add hour offsets
        out["date"] = pd.to_datetime(out["dt"])
        out["time"] = pd.to_timedelta(out["hour_idx"], unit="h")
        out["datetime"] = out["date"] + out["time"]
        out["day_of_week"] = out["datetime"].dt.day_of_week
        out["weekday"] = out["day_of_week"] < 5

        frames.append(out)

    # joining the dataframes
    output_df = pd.concat(frames, ignore_index=True)

    # doing some preprocessing

    # making boolean flags boolean
    output_df["holiday_flag"] = output_df["holiday_flag"].astype(bool)
    output_df["activity_flag"] = output_df["activity_flag"].astype(bool)

    # renaming columns for clarity
    output_df = output_df.rename(
        columns={
            "sale_amount": "scaled_daily_demand",
            "stock_hour6_22_cnt": "number_stockout_open_hours",
        }
    )

    # creating the raw_data_actual_demand as the scaled_sales_quantity where it exists
    # and NaN where stockout occurs
    output_df["raw_data_actual_demand"] = output_df["scaled_sales_quantity"].where(
        ~output_df["is_stockout"], other=np.nan
    )

    return output_df


def add_series_metadata(
    imputed: pd.DataFrame,
    series_hourly: pd.DataFrame,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Left-join day-level metadata from `series_hourly` onto an imputed hourly frame.

    Joins on (store_id, product_id, date). Row order and count of `imputed` are
    preserved, and the merge raises if any row finds no match or if the lookup
    table has duplicate keys.
    """
    columns = list(DAILY_METADATA_COLS if columns is None else columns)

    missing = [c for c in columns if c not in series_hourly.columns]
    if missing:
        raise KeyError(f"columns not in series_hourly: {missing}")

    clash = [c for c in columns if c in imputed.columns]
    if clash:
        raise ValueError(f"columns already in imputed frame: {clash}")

    # every metadata column is constant within a day, so keeping hour 0
    # gives exactly one row per store-product-day without a drop_duplicates
    # pass over the full hourly table
    meta = series_hourly.loc[series_hourly["hour_idx"] == 0, DAY_KEYS + columns]

    out = imputed.merge(
        meta,
        on=DAY_KEYS,
        how="left",
        validate="many_to_one",
        indicator="_merge",
    )

    n_unmatched = int((out["_merge"] != "both").sum())
    if n_unmatched:
        raise ValueError(f"{n_unmatched} rows in imputed frame had no metadata match")

    return out.drop(columns="_merge")


def filter_by_mean_demand(
    imputed_df: pd.DataFrame,
    series_hourly: pd.DataFrame,
    group: str = "high",
    threshold: float = 1 / 24,  # hourly mean; equivalent to 1 on a daily scale
    mean_source: str = "full",
    return_means: bool = False,
):
    """
    Keep only the series (store_id, product_id) in imputed_df whose mean
    hourly actual demand is >= threshold ("high") or < threshold ("low").

    mean_source:
        "full"     -> mean of series_hourly.raw_data_actual_demand over the entire
                      dataset (train + eval). NaN at stockouts, so stockouts are skipped.
        "internal" -> mean of imputed_df.actual_demand. Only rows in imputed_df
                      count, and NaN stockout rows are skipped, so this is effectively
                      the mean over masked hours only.
    """
    if group not in {"high", "low"}:
        raise ValueError("group must be 'high' or 'low'")

    if mean_source == "full":
        src, col = series_hourly, "raw_data_actual_demand"
    elif mean_source == "internal":
        src, col = imputed_df, "actual_demand"
    else:
        raise ValueError("mean_source must be 'full' or 'internal'")

    # select only the needed columns before casting: series_hourly is very large
    sub = src[SERIES_KEYS + [col]]
    means = (
        sub[col]
        .astype("float64")  # avoid float32 accumulation error near the threshold
        .groupby([sub["store_id"], sub["product_id"]])
        .mean()  # skips NaN
        .rename("mean_actual_demand")
    )

    keep = means[means >= threshold] if group == "high" else means[means < threshold]

    row_keys = pd.MultiIndex.from_frame(imputed_df[SERIES_KEYS])
    filtered = imputed_df[row_keys.isin(keep.index)]

    return (filtered, means) if return_means else filtered


def import_inference_table(file_path: Union[Path, str]):
    """reconstruct an inference dataframe generated by one of the above
    constructors that has been saved as a CSV file"""

    if isinstance(file_path, str):
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError("The file needs to exist; check for typos.")

    # import the file
    df = pd.read_csv(file_path, index_col=0)

    # use the initial datetime to reconstruct the other temporal information
    df["datetime"] = pd.to_datetime(df["datetime"], unit="us")
    df["date"] = df["datetime"].dt.normalize().astype("datetime64[us]")
    df["time"] = (df["datetime"] - df["date"]).astype("timedelta64[s]")

    # convert to smaller datatypes
    df["hour_idx"] = df["hour_idx"].astype("int8")  # 0 to 24
    df["day_of_week"] = df["day_of_week"].astype("int8")  # 0 to 6

    return df
