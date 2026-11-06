import numpy as np
import pandas as pd

from uncertainty_aware_planner.utils.constants import IMPUTATION_WINDOW_DURATION, NUMBER_HOURS_IN_DAY, OPEN_HOURS

def evaluation_mnar(eval_pypots, model_output, daily_pool: bool = True):
    
    # initialize the proper shape for resizing
    n_series = eval_pypots["X"].shape[0]
    shape = (n_series, IMPUTATION_WINDOW_DURATION, NUMBER_HOURS_IN_DAY)

    # create a helper function that takes the defined shape,
    # reshapes the array, and takes the open hours from the day
    # (this is to avoid censoring NaN in X_ori)
    def open_hours(a):
        return a.reshape(shape)[..., OPEN_HOURS]

    # extract the components that matter from the censored and masked,
    # censored, and imputed NumPy arrays with NaN
    X       = open_hours(eval_pypots["X"][..., 0])
    X_ori   = open_hours(eval_pypots["X_ori"][..., 0])
    imputed = open_hours(model_output["imputation"][..., 0])

    # separately assess masked-ness and true stockout
    masked   = np.isnan(X) & ~np.isnan(X_ori)
    stockout = np.isnan(X_ori)

    # calculate the difference between imputed and ground truth hours
    hourly_diff = np.where(masked, imputed - X_ori, 0.0)

    # if the values are to be pooled by day then calculated
    if daily_pool:

        # create an indicator variable that denotes days with masking but
        # no stockout
        day_valid = (masked.sum(axis=2) > 0) & (stockout.sum(axis=2) == 0)

        # sum hour values across the days for residuals
        daily_diff   = hourly_diff.sum(axis=2)

        # fill the true censor data with zeros, and sum together true values
        # across all open hours
        daily_actual = np.nan_to_num(X_ori).sum(axis=2)   # open-hour sales for the day

        # take the totals only across the valid days
        total_diff     = daily_diff[day_valid].sum()
        total_abs_diff = np.abs(daily_diff)[day_valid].sum()
        total_actual   = daily_actual[day_valid].sum()
    else:

        # sum across all the possible hours
        total_diff     = hourly_diff.sum()
        total_abs_diff = np.abs(hourly_diff).sum()
        total_actual   = np.where(masked, X_ori, 0.0).sum()

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