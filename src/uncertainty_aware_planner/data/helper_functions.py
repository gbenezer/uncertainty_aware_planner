# file to store developer-friendly helper functions
from typing import Optional, Union
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets

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
    base = pd.DataFrame({
        "store_id": index["store_id"].to_numpy()[sample],
        "product_id": index["product_id"].to_numpy()[sample],
        "datetime": stamp.astype("datetime64[s]"),
        "actual_demand": actual,
        "is_stockout": np.isnan(actual),   # true censoring: real stockout hour
        "is_masked": ~np.isnan(actual),    # artificial censoring: hidden by eval masks
    })
    # add date metadata to all tables deriving from this function
    dt = base["datetime"].dt
    base = base.assign(
        date=dt.date,
        hour=dt.time,
        day_of_week=dt.day_of_week,
        weekday=dt.day_of_week < 5,  # Saturday is 5, Sunday is 6
    )
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
    return base.assign(estimated_demand=_estimate(imputation, sample, step, clip_demand))


def to_imputed_hourly_interval(dataset, index, imputation, clip_demand: bool = True):

    assert imputation.shape[-1] == 2, f"expected (lower, upper) quantiles as last dimension, got {imputation.shape}"
    
    # get the base dataframe, the indices of valid series,
    # and the indices of valid hours
    base, sample, step = _hidden_hours(dataset, index)

    # create an anonymous function to get the other quantiles
    q = lambda k: _estimate(imputation[..., 0, k], sample, step, clip_demand)

    # construct the demand estimate given there are only intervals
    demand_estimate = (q(0) + q(1)) * 0.5

    # assign the proper values to the proper column names and return
    return base.assign(q025=q(0), estimated_demand=demand_estimate, q975=q(1))

def to_imputed_hourly_interval_median(dataset, index, imputation, clip_demand: bool = True):

    assert imputation.shape[-1] == 3, f"expected (lower, median, upper) quantiles as last dimension, got {imputation.shape}"
    
    # get the base dataframe, the indices of valid series,
    # and the indices of valid hours
    base, sample, step = _hidden_hours(dataset, index)

    # create an anonymous function to get the other quantiles
    q = lambda k: _estimate(imputation[..., 0, k], sample, step, clip_demand)

    # assign the proper values to the proper column names and return
    return base.assign(q025=q(0), estimated_demand=q(1), q975=q(2))

def get_series_hourly(cache_dir: Optional[Union[str, Path]] = "data/hf_cache") -> pd.DataFrame:
    """Circumvents the majority of the data pipeline machinery to get the required `series_hourly`
    dataframe from CONTRACTS.md because it is non-obvious how to best do so from a Cube object"""

    # load the dataset
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", cache_dir=cache_dir)  # load once

    # initialize a list for the dataframes
    frames = []

    # initialize a constant for the number of hours in the day
    n_hours = 24


    for split in ("train", "eval"):

        # get the proper dataset and its size
        d = ds[split]
        n = len(d)

        # Everything except the two list columns
        scalar_cols = [c for c in d.column_names if c not in ("hours_sale", "hours_stock_status")]
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
        out["datetime"] = out["date"] + pd.to_timedelta(out["hour_idx"], unit="h")
        out["time"] = out["datetime"].dt.time
        out["day_of_week"] = out["datetime"].dt.day_of_week
        out["weekday"] = out["day_of_week"] < 5

        frames.append(out)

    return pd.concat(frames, ignore_index=True)