# file to store developer-friendly helper functions

from pathlib import Path

import numpy as np
import pandas as pd


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
