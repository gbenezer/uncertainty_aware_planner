import numpy as np
import pandas as pd

# def calculate_imputation_percent_errors(
#     to_imputed_hourly_df: pd.DataFrame,
#     indicators_included: bool = False,
#     daily_pool: bool = True
# ):

#     # assumes at least the columns "store_id", "product_id", "datetime",
#     # "estimated_demand", "actual_demand" exist (as per the `to_imputed_hourly` function)

#     # drop any rows where there is no actual demand, as those cannot be evaluated
#     df = to_imputed_hourly_df.dropna(subset=["actual_demand"]).copy()

#     # get the residual
#     df["demand_difference"] = df["estimated_demand"] - df["actual_demand"]

#     # if it is a pandas Timestamp, extract the date
#     if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
#         df["date"] = df["datetime"].date

#     # otherwise it's a string, split by the whitespace, and take the date portion
#     else:
#         df["date"] = df["datetime"].map(lambda dt: dt.split()[0])

#     # if pooling by day is requested (as performed by the authors of the dataset)
#     # then group by the date and indices

#     if daily_pool:

#         # if there are indicator variables (0/1 or True/False), only take days with masked values
#         # but no true stockouts
#         if indicators_included:
#             summary_df = df.groupby(["store_id", "product_id", "date"])[["actual_demand", "demand_difference", "is_stockout", "is_masked"]].sum()

#             # extract proper days
#             day_has_stockout = summary_df["is_stockout"] > 0
#             day_has_masked_values = summary_df["is_masked"] > 0
#             boolean_series = day_has_masked_values & ~day_has_stockout
#             summary_df = summary_df[boolean_series]

#             # calculate metrics
#             wape = summary_df["demand_difference"].abs().sum() / summary_df["actual_demand"].sum()
#             wpe = summary_df["demand_difference"].sum() / summary_df["actual_demand"].sum()

#         # otherwise assume all elements have no true stockouts and all actual/estimated pairs correspond to
#         # evaluation data
#         else:
#             summary_df = df.groupby(["store_id", "product_id", "date"])[["actual_demand", "demand_difference"]].sum()

#             # calculate metrics
#             wape = summary_df["demand_difference"].abs().sum() / summary_df["actual_demand"].sum()
#             wpe = summary_df["demand_difference"].sum() / summary_df["actual_demand"].sum()

#     # otherwise do the calculation on a more granular hourly basis
#     else:

#         # calculate metrics
#         wape = df["demand_difference"].abs().sum() / df["actual_demand"].sum()
#         wpe = df["demand_difference"].sum() / df["actual_demand"].sum()

#     return {
#         "wape": wape,
#         "wpe" : wpe
#     }


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


def calculate_imputation_mase():
    pass


# def imputation_metric_calculation(
#     imputed_hourly_dataframe: pd.DataFrame,
#     calculate_mase: bool = False,
#     time_col: str = "datetime",
# ):
#     keys = ["store_id", "product_id"]

#     # reference only; the original dataframe is never modified
#     df = imputed_hourly_dataframe

#     # select only those entries with actual demand, which are the only ones
#     # that can be used to calculate evaluation metrics
#     evaluation_dataframe = df[df["actual_demand"].notna()].copy()

#     # if the estimated demand doesn't exist, breaking the function contract,
#     # raise a ValueError
#     if evaluation_dataframe["estimated_demand"].isna().any():
#         raise ValueError("estimated_demand contains NaN values")

#     # calculate demand residual
#     evaluation_dataframe["demand_difference"] = (
#         evaluation_dataframe["estimated_demand"] - evaluation_dataframe["actual_demand"]
#     )

#     # total demand across all store-product time series present
#     actual_sum = evaluation_dataframe["actual_demand"].sum()

#     # metrics that are valid for series with gaps
#     wpe = evaluation_dataframe["demand_difference"].sum() / actual_sum if actual_sum else np.nan
#     wape = evaluation_dataframe["demand_difference"].abs().sum() / actual_sum if actual_sum else np.nan

#     results = {"wape": wape, "wpe": wpe}

#     if calculate_mase:
#         # Reindex each store-product series onto a complete hourly grid
#         # (spanning that series' own first to last timestamp). Absent hours
#         # become NaN rows, so diff() only yields a value where two
#         # consecutive hours both have actual demand, i.e. a true one-step
#         # naive error. Requires datetime-typed, hour-aligned, unique
#         # timestamps within each series.
#         full_actual = (
#             df.set_index(time_col)
#               .groupby(keys)["actual_demand"]
#               .resample("h")
#               .asfreq()
#         )
#         naive_error = full_actual.groupby(level=keys).diff()

#         # NaN entries (series starts and anything touching a gap) are
#         # skipped by mean(), so only genuine one-step differences count
#         naive_mae = naive_error.abs().mean()
#         mae = evaluation_dataframe["demand_difference"].abs().mean()
#         results["mase"] = mae / naive_mae if naive_mae else np.nan

#     return results
