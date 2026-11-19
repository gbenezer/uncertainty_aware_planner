from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.anova import anova_lm
import re

# constants
RESULTS_DIR = Path("results/imputation/pretrained_model_eval/")
SUMMARY_DIR = RESULTS_DIR / "summaries"
PRETRAINED_STEM = "pretrained_model_eval_py_execution_1"
RUN_PATH = RESULTS_DIR / f"{PRETRAINED_STEM}_runs.csv"
METRIC_PATH = RESULTS_DIR / f"{PRETRAINED_STEM}_metrics.csv"
RUN_GROUP_COLUMNS = ["num_quantiles", "model_type"]
METRIC_CATEGORICAL_COLUMNS = ["day_type", "volume", "metric"]
METRIC_SUMMARY_COLUMNS = [
    "day_type",
    "volume",
    "model_type",
    "num_quantiles",
]  # specified order
SUM_REGEX = re.compile("n_samples|timed_runs")
FORMULA = 'value ~ day_type * volume + model_type + num_quantiles'
FULL_LINEAR_FORMULA = 'value ~ day_type + volume + model_type + num_quantiles'
SECOND_ORDER_FORMULA = 'value ~ day_type * volume + model_type * num_quantiles'
NO_DAY_FORMULA = 'value ~ volume + model_type + num_quantiles'
NO_VOLUME_FORMULA = 'value ~ day_type + model_type + num_quantiles'
NO_QUANTILE_FORMULA = 'value ~ day_type + volume + model_type'
NO_MODEL_FORMULA = 'value ~ day_type + volume + num_quantiles'

# data import
run_df = pd.read_csv(RUN_PATH)
metric_df = pd.read_csv(METRIC_PATH)

# data-dependent constants
RUN_DATA_COLUMNS = [s for s in run_df.columns if s.startswith("latency_")]
RUN_DATA_AGGREGATION_MAP = {
    k: "median" if SUM_REGEX.search(k) is None else "sum" for k in RUN_DATA_COLUMNS
}
RUN_SUMMARY_RENAME_MAP = {
    k: f"median_{k}" if SUM_REGEX.search(k) is None else f"sum_{k}"
    for k in RUN_DATA_COLUMNS
}


# data preprocessing
run_df["num_quantiles"] = run_df["num_quantiles"].fillna(1).astype(int).astype("category")
run_df["model_type"] = run_df["model_type"].astype("category")
metric_df[METRIC_CATEGORICAL_COLUMNS] = metric_df[METRIC_CATEGORICAL_COLUMNS].astype(
    "category"
)
metric_df = metric_df.drop(columns="n_dropped")


# run latency summary table
run_data_subset = RUN_GROUP_COLUMNS + RUN_DATA_COLUMNS
run_data_subset_df = run_df[run_data_subset]
run_data_summary_df = (
    run_data_subset_df.groupby(RUN_GROUP_COLUMNS)
    .agg(RUN_DATA_AGGREGATION_MAP)
    .rename(columns=RUN_SUMMARY_RENAME_MAP)
)
# run_data_summary_df.to_csv(
#     (SUMMARY_DIR / "pretrained_model_eval_py_execution_1_latency_summary.csv")
# )

# run metadata incorporation
metric_df = metric_df.merge(
    right=run_df[["run_id", "num_quantiles", "model_type"]],
    how="left",
    on="run_id",
    suffixes=("_metric", "_run"),
    validate="many_to_one",
).drop(columns="run_id")

# filtration and summarization of the metrics across relevant dimensions
for metric_type in metric_df["metric"].cat.categories:
    
    # filter the dataframe
    filtered_df = metric_df[metric_df["metric"] == metric_type]

    # if metric_type in ["mpiw", "picp"]:
    #     filtered_df = filtered_df.dropna()

    # # removing aggregated values that may cause collinearity
    # # because of dependent information
    # filtered_df = filtered_df[filtered_df["day_type"] != "overall"]
    # filtered_df = filtered_df[filtered_df["volume"] != "all"]

    # typical models
    y, X = dmatrices(FORMULA, data=filtered_df, return_type='dataframe')
    y_lm, X_lm = dmatrices(FULL_LINEAR_FORMULA, data=filtered_df, return_type='dataframe')
    y_so, X_so = dmatrices(SECOND_ORDER_FORMULA, data=filtered_df, return_type='dataframe')

    # impact of variable subtraction
    y_nd, X_nd = dmatrices(NO_DAY_FORMULA, data=filtered_df, return_type='dataframe')
    y_nv, X_nv = dmatrices(NO_VOLUME_FORMULA, data=filtered_df, return_type='dataframe')
    y_nq, X_nq = dmatrices(NO_QUANTILE_FORMULA, data=filtered_df, return_type='dataframe')
    y_nm, X_nm = dmatrices(NO_MODEL_FORMULA, data=filtered_df, return_type='dataframe')

    # OLS model definitions and fitting
    model = sm.OLS(y, X).fit()
    linear_model = sm.OLS(y_lm, X_lm).fit()
    second_order_model = sm.OLS(y_so, X_so).fit()
    no_day_model = sm.OLS(y_nd, X_nd).fit()
    no_volume_model = sm.OLS(y_nv, X_nv).fit()
    no_quantile_model = sm.OLS(y_nq, X_nq).fit()
    no_model_model = sm.OLS(y_nm, X_nm).fit()

    # output to stdout
    print(metric_type)

    print("model vs. linear")
    print(anova_lm(linear_model, model))
    print("second order vs. linear")
    print(anova_lm(linear_model, second_order_model))

    print("impact of day")
    print(anova_lm(no_day_model, linear_model))
    print("impact of volume")
    print(anova_lm(no_volume_model, linear_model))
    print("impact of quantile number")
    print(anova_lm(no_quantile_model, linear_model))
    print("impact of model type")
    print(anova_lm(no_model_model, linear_model))
    print("full linear model")
    print(linear_model.summary())
    print("linear model without day")
    print(no_day_model.summary())

    # # remove NaN from baseline entries
    # if metric_type in ["mpiw", "picp"]:
    #     filtered_df = filtered_df.dropna()
    #     summary_df = filtered_df.groupby(METRIC_SUMMARY_COLUMNS).agg(
    #         **{
    #             f"{metric_type}_mean": pd.NamedAgg(column="value", aggfunc="mean"),
    #             f"{metric_type}_median": pd.NamedAgg(column="value", aggfunc="median"),
    #             f"{metric_type}_std": pd.NamedAgg(column="value", aggfunc="std"),
    #             f"{metric_type}_total_group_n": pd.NamedAgg(column="n", aggfunc="sum"),
    #             f"{metric_type}_mean_group_n": pd.NamedAgg(column="n", aggfunc="mean"),
    #             f"{metric_type}_median_group_n": pd.NamedAgg(column="n", aggfunc="median"),
    #             f"{metric_type}_std_group_n": pd.NamedAgg(column="n", aggfunc="std"),
    #         }
    #     ).reset_index()

    # else:
    #     summary_df = filtered_df.groupby(METRIC_SUMMARY_COLUMNS).agg(
    #         **{
    #             f"{metric_type}_mean": pd.NamedAgg(column="value", aggfunc="mean"),
    #             f"{metric_type}_median": pd.NamedAgg(column="value", aggfunc="median"),
    #             f"{metric_type}_std": pd.NamedAgg(column="value", aggfunc="std"),
    #             f"{metric_type}_group_n_per_seed": pd.NamedAgg(column="n", aggfunc="median"),
    #             f"{metric_type}_total_group_n": pd.NamedAgg(column="n", aggfunc="sum"),
    #         }
    #     ).reset_index()

    # file_name = PRETRAINED_STEM + f"_{metric_type}_summary.csv"
    # file_path = SUMMARY_DIR / file_name
    # summary_df.to_csv(file_path)