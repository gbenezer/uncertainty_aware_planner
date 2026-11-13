# file to store developer-friendly helper functions
from typing import Optional, Union
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets


def to_imputed_hourly(dataset, index, imputation):
    """imputed_hourly rows (the columns `daily` reads) for every hidden hour a sample owns."""
    x = dataset["X"][..., 0]
    truth = dataset.get("X_ori", dataset["X"])[..., 0]
    sample, step = np.nonzero(np.isnan(x))
    start = pd.to_datetime(index["start_date"]).to_numpy().astype("datetime64[h]")
    keep = pd.to_datetime(index["keep_start"]).to_numpy().astype("datetime64[h]")
    stamp = start[sample] + step.astype("timedelta64[h]")
    own = stamp >= keep[sample]
    sample, step, stamp = sample[own], step[own], stamp[own]
    return pd.DataFrame(
        {
            "store_id": index["store_id"].to_numpy()[sample],
            "product_id": index["product_id"].to_numpy()[sample],
            "datetime": stamp.astype("datetime64[s]"),
            "estimated_demand": np.clip(imputation[sample, step], 0, None),
            "actual_demand": truth[
                sample, step
            ],  # NaN at real stockouts, true sales at masked hours
            # true censoring: a real stockout hour, no ground truth
            "is_stockout": np.isnan(truth[sample, step]),
            # artificial censoring: an in-stock hour hidden by the eval masks
            "is_masked": ~np.isnan(truth[sample, step]),
        }
    )

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

        frames.append(out)

    return pd.concat(frames, ignore_index=True)