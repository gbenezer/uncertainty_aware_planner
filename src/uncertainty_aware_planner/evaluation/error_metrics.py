import numpy as np
import pandas as pd

from uncertainty_aware_planner.utils.constants import (
    IMPUTATION_WINDOW_DURATION,
    NUMBER_HOURS_IN_DAY,
    OPEN_HOURS,
)


def calculate_imputation_percent_errors(
    eval_pypots,
    imputation_point_estimate,
    daily_pool: bool = True,
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

    # if the values are to be pooled by day then calculated
    if daily_pool:
        # sum hour values across the days for residuals
        daily_diff = hourly_diff.sum(axis=2)

        # only take the days that are valid (has masking but no stockout)
        # when getting the signed difference
        total_diff = daily_diff[day_valid].sum()

        # get the sum of absolute values of daily differences
        total_abs_diff = np.abs(daily_diff)[day_valid].sum()

        # get the total sales over all the relevant days
        total_actual = truth.sum(axis=2)[day_valid].sum()

        # sample size is the number of valid days
        n = int(day_valid.sum())

    else:
        # same population of days as the pooled metric
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