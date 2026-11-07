import numpy as np
import pandas as pd

from uncertainty_aware_planner.utils.constants import IMPUTATION_WINDOW_DURATION, NUMBER_HOURS_IN_DAY, OPEN_HOURS

def evaluation_mnar(
    eval_pypots, 
    imputation, 
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
    X       = open_hours(eval_pypots["X"][..., 0])
    X_ori   = open_hours(eval_pypots["X_ori"][..., 0])
    imputed = open_hours(imputation[..., 0])

    # separately assess masked-ness and true stockout
    masked   = np.isnan(X) & ~np.isnan(X_ori)
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
        daily_diff   = hourly_diff.sum(axis=2)

        # only take the days that are valid (has masking but no stockout)
        # when getting the signed difference
        total_diff   = daily_diff[day_valid].sum()

        # get the sum of absolute values of daily differences
        total_abs_diff    = np.abs(daily_diff)[day_valid].sum()

        # get the total sales over all the relevant days
        total_actual = truth.sum(axis=2)[day_valid].sum()

    else:

        # same population of days as the pooled metric
        hours = np.broadcast_to(day_valid[..., None], hourly_diff.shape)
        
        # sum across all the relevant hours
        total_diff   = hourly_diff[hours].sum()

        # sum absolute values across all relevant hours
        total_abs_diff    = np.abs(hourly_diff)[hours].sum()

        # sum all sales over relevant hours
        total_actual = truth[hours].sum()

    if total_actual == 0:
        return {"wape": np.nan, "wpe": np.nan}

    return {"wape": total_abs_diff / total_actual,
            "wpe":  total_diff / total_actual}

def calculate_imputation_percent_errors(
    to_imputed_hourly_df: pd.DataFrame,
    indicators_included: bool = False,
    daily_pool: bool = True,
    window: tuple = (6, 22),  # hours [6, 22), as in the authors' code
    include_outside_window: bool = True,  # True: authors' behavior
):

    # copy the dataframe to prevent mutation
    df = to_imputed_hourly_df.copy()

    # before anything, if there are issues with the imputed values,
    # raise an error
    if df["estimated_demand"].isna().any():
        raise ValueError("estimated_demand contains NaN values")

    # define the groupby keys
    keys = ["store_id", "product_id", "date"]

    # construct datetime if necessary, extract the date,
    # and assess if the hour is in store hours
    dt = pd.to_datetime(df["datetime"])
    df["date"] = dt.dt.date
    df["_in_window"] = dt.dt.hour.between(window[0], window[1] - 1)

    # day-level validity, computed BEFORE dropping rows
    valid_days = None

    # if pooling of the residuals is requested and there are also indicator
    # variables of true and artificial censoring, filter out the hours that
    # the authors of the dataset do
    if daily_pool and indicators_included:
        # adding 0/1 flags to the DataFrame by hour
        # and calculating the total hours for each store-product-day
        flags = (
            df.assign(
                # 0/1 indicator of stockouts during open hours
                stockout_w=(df["is_stockout"].astype(bool) & df["_in_window"]).astype(
                    int
                ),
                # 0/1 indicator of artificial masking during open hours
                masked_w=(df["is_masked"].astype(bool) & df["_in_window"]).astype(int),
            )
            .groupby(keys)[["stockout_w", "masked_w"]]
            .sum()
        )

        # getting the indices of the grouped dataframe where there is masking during
        # the work day but no stockout
        valid_days = flags.index[(flags["masked_w"] > 0) & (flags["stockout_w"] == 0)]

    # dropping the NA values the actual_demand column after assessing the valid days
    # if necessary
    df = df.dropna(subset=["actual_demand"])

    # if we want a more stringent calculation of imputation error
    # only include evaluation of hours within the workday
    if not include_outside_window:
        # score only the window: outside hours leave both numerator and denominator
        df = df[df["_in_window"]].copy()

    # when outside hours are kept, the prediction equals the original there,
    # so they contribute zero error but still count toward actual demand
    df["demand_difference"] = np.where(
        df["_in_window"], df["estimated_demand"] - df["actual_demand"], 0.0
    )

    # if pooling is requested, calculate across store-product-day
    if daily_pool:
        daily = df.groupby(keys)[["actual_demand", "demand_difference"]].sum()

        # if not all groups are valid, only select the valid days
        if valid_days is not None:
            daily = daily.loc[daily.index.intersection(valid_days)]

        err, actual = daily["demand_difference"], daily["actual_demand"]

    # otherwise calculate on a per-hour basis
    else:
        err, actual = df["demand_difference"], df["actual_demand"]

    # calculate total actual demand for the denominator
    total = actual.sum()

    # if there's some weird shit going on, return NaN
    if total == 0:
        return {"wape": np.nan, "wpe": np.nan}

    # otherwise return the proper calculated values
    return {"wape": err.abs().sum() / total, "wpe": err.sum() / total}