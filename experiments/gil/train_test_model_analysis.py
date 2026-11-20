from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.anova import anova_lm
import plotly.express as px
import re

# constants
RESULTS_DIR = Path("results/imputation/cv_folds/")
FIGURE_DIR = RESULTS_DIR / "figures"
SUMMARY_DIR = RESULTS_DIR / "summaries"
PRETRAINED_STEM = "cv_folds"
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
PERCENT_METRICS = ["picp", "wape", "wpe"]
METRIC_TYPE_NAME_MAP = {
    "mpiw": "Mean Prediction<br>Interval Width",
    "picp": "Prediction Interval<br>Coverage Percentage",
    "wape": "Weighted Absolute<br>Percent Error",
    "wpe": "Weighted<br>Percent Error",
    "mase_1hr": "Mean Absolute<br>Scaled Error<br>1 Hour",
    "mase_24hr": "Mean Absolute<br>Scaled Error<br>1 Day",
    "mase_168hr": "Mean Absolute<br>Scaled Error<br>1 Week",
}
FORMULA = "value ~ day_type * volume + model_type + num_quantiles"
FULL_LINEAR_FORMULA = "value ~ day_type + volume + model_type + num_quantiles"
SECOND_ORDER_FORMULA = "value ~ day_type * volume + model_type * num_quantiles"
LINEAR_FORMULA_WITH_MODEL_QUANTILE_INTERACTION = (
    "value ~ day_type + volume + model_type * num_quantiles"
)
NO_DAY_FORMULA = "value ~ volume + model_type + num_quantiles"
NO_VOLUME_FORMULA = "value ~ day_type + model_type + num_quantiles"
NO_QUANTILE_FORMULA = "value ~ day_type + volume + model_type"
NO_MODEL_FORMULA = "value ~ day_type + volume + num_quantiles"
FULL_LINEAR_FORMULA_WITH_N = (
    "value ~ day_type + volume + model_type + num_quantiles + n"
)
LINEAR_FORMULA_WITH_N_AND_MQ = (
    "value ~ day_type + volume + model_type * num_quantiles + n"
)

# data import
run_df = pd.read_csv(RUN_PATH)
metric_df = pd.read_csv(METRIC_PATH)

# data-dependent constants
RUN_DATA_COLUMNS = [s for s in run_df.columns if s.startswith("latency_")] + [
    "train_seconds",
    "eval_seconds",
]
RUN_DATA_AGGREGATION_MAP = {
    k: "median" if SUM_REGEX.search(k) is None else "sum" for k in RUN_DATA_COLUMNS
}
RUN_SUMMARY_RENAME_MAP = {
    k: f"median_{k}" if SUM_REGEX.search(k) is None else f"sum_{k}"
    for k in RUN_DATA_COLUMNS
}
RUN_DATA_FIGURE_MAP = {
    "latency_total_s_mean": ("Inference Latency<br>Mean, Seconds", True),
    "latency_total_s_std": ("Inference Latency<br>Standard Deviation, Seconds", True),
    "latency_total_s_median": ("Inference Latency<br>Median, Seconds", True),
    "latency_peak_gpu_mem_mb": ("Inference Peak<br>GPU Memory, MB", False),
    "train_seconds": ("Training Time<br>Seconds", False),
    "eval_seconds": ("Evaluation Time<br>Seconds", False),
}


# data preprocessing
run_df["num_quantiles"] = (
    run_df["num_quantiles"].fillna(1).astype(int).astype("category")
)
run_df["model_type"] = run_df["model_type"].astype("category")
metric_df[METRIC_CATEGORICAL_COLUMNS] = metric_df[METRIC_CATEGORICAL_COLUMNS].astype(
    "category"
)
metric_df = metric_df.drop(columns="n_dropped")
MODEL_TYPE_RENAME_MAP = {
    k: k.split("_")[0] + " " + k.split("_")[2]
    for k in run_df["model_type"].cat.categories
}
run_df["model_type"] = run_df["model_type"].cat.rename_categories(MODEL_TYPE_RENAME_MAP)

# run latency summary table
run_data_subset = RUN_GROUP_COLUMNS + RUN_DATA_COLUMNS
run_data_subset_df = run_df[run_data_subset]
run_data_summary_df = (
    run_data_subset_df.groupby(RUN_GROUP_COLUMNS)
    .agg(RUN_DATA_AGGREGATION_MAP)
    .rename(columns=RUN_SUMMARY_RENAME_MAP)
)

# run latency figures
for key, value in RUN_DATA_FIGURE_MAP.items():
    run_fig = px.violin(
        run_data_subset_df,
        x="model_type",
        y=key,
        box=True,
        points="all",
        color="num_quantiles",
        labels={
            key: value[0],
            "model_type": "Model Type",
            "num_quantiles": "Number of<br>Estimated Quantiles",
        },
        category_orders={"num_quantiles": [1, 2, 3]},
        template="seaborn",
    )
    if value[1]:
        run_fig.update_yaxes(range=[0, None])
    file_name = f"{key}_violin_plot.html"
    file_path = FIGURE_DIR / file_name
    run_fig.write_html(file_path)

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


def create_violin_plots(df, metric_str: str, array: bool = True):

    filtered_df = df[df["metric"] == metric_str]

    if metric_str in PERCENT_METRICS:
        filtered_df["value"] = filtered_df["value"] * 100

    if array:
        fig = px.violin(
            filtered_df,
            x="model_type",
            y="value",
            box=True,
            points="all",
            facet_row="day_type",
            facet_col="volume",
            color="num_quantiles",
            labels={
                "value": METRIC_TYPE_NAME_MAP[metric_str],
                "model_type": "Model Type",
                "day_type": "Part of Week",
                "volume": "Sales Volume",
                "num_quantiles": "Number of<br>Estimated Quantiles",
            },
            category_orders={"num_quantiles": [1, 2, 3]},
            template="seaborn",
        )
    else:
        all_days = filtered_df["day_type"] == "overall"
        all_sales = filtered_df["volume"] == "all"
        filtered_df = filtered_df[all_days & all_sales]
        fig = px.violin(
            filtered_df,
            x="model_type",
            y="value",
            box=True,
            points="all",
            color="num_quantiles",
            labels={
                "value": METRIC_TYPE_NAME_MAP[metric_str],
                "model_type": "Model Type",
                "num_quantiles": "Number of<br>Estimated Quantiles",
            },
            category_orders={"num_quantiles": [1, 2, 3]},
            template="seaborn",
        )

    if metric_str == "picp":
        fig.add_hline(
            y=80,
            line_dash="dot",
            line_width=2,
            annotation_text="Empirical Coverage Specification",
            annotation_position="bottom right",
        )

        fig.update_shapes(
            layer="above",
        )

    if metric_str.startswith("mase"):
        fig.add_hline(
            y=1,
            line_dash="dot",
            line_width=2,
            annotation_text="Naive Forecast Performance",
            annotation_position="top right",
        )

        fig.update_shapes(
            layer="above",
        )

    return fig


# filtration and summarization of the metrics across relevant dimensions
for metric_type in metric_df["metric"].cat.categories:
    metric_figure = create_violin_plots(df=metric_df, metric_str=metric_type)
    file_name = f"{metric_type}_violin_plot_array.html"
    file_path = FIGURE_DIR / file_name
    metric_figure.write_html(file_path)

    metric_figure = create_violin_plots(
        df=metric_df, metric_str=metric_type, array=False
    )
    file_name = f"{metric_type}_overall_violin_plot.html"
    file_path = FIGURE_DIR / file_name
    metric_figure.write_html(file_path)

    # filter the dataframe
    filtered_df = metric_df[metric_df["metric"] == metric_type]

    if metric_type in ["mpiw", "picp"]:
        filtered_df = filtered_df.dropna()

    # removing aggregated values that may cause collinearity
    # because of dependent information
    filtered_df = filtered_df[filtered_df["day_type"] != "overall"]
    filtered_df = filtered_df[filtered_df["volume"] != "all"]

    # typical models
    y, X = dmatrices(FORMULA, data=filtered_df, return_type="dataframe")
    y_lm, X_lm = dmatrices(
        FULL_LINEAR_FORMULA, data=filtered_df, return_type="dataframe"
    )
    y_mq, X_mq = dmatrices(
        LINEAR_FORMULA_WITH_MODEL_QUANTILE_INTERACTION,
        data=filtered_df,
        return_type="dataframe",
    )
    y_so, X_so = dmatrices(
        SECOND_ORDER_FORMULA, data=filtered_df, return_type="dataframe"
    )

    # impact of variable subtraction (and addition)
    y_nd, X_nd = dmatrices(NO_DAY_FORMULA, data=filtered_df, return_type="dataframe")
    y_nv, X_nv = dmatrices(NO_VOLUME_FORMULA, data=filtered_df, return_type="dataframe")
    y_nq, X_nq = dmatrices(
        NO_QUANTILE_FORMULA, data=filtered_df, return_type="dataframe"
    )
    y_nm, X_nm = dmatrices(NO_MODEL_FORMULA, data=filtered_df, return_type="dataframe")
    y_wn, X_wn = dmatrices(
        FULL_LINEAR_FORMULA_WITH_N, data=filtered_df, return_type="dataframe"
    )
    y_wnmq, X_wnmq = dmatrices(
        LINEAR_FORMULA_WITH_N_AND_MQ, data=filtered_df, return_type="dataframe"
    )

    # OLS model definitions and fitting
    model = sm.OLS(y, X).fit()
    linear_model = sm.OLS(y_lm, X_lm).fit()
    second_order_model = sm.OLS(y_so, X_so).fit()
    mq_model = sm.OLS(y_mq, X_mq).fit()
    no_day_model = sm.OLS(y_nd, X_nd).fit()
    no_volume_model = sm.OLS(y_nv, X_nv).fit()
    no_quantile_model = sm.OLS(y_nq, X_nq).fit()
    no_model_model = sm.OLS(y_nm, X_nm).fit()
    linear_model_with_n = sm.OLS(y_wn, X_wn).fit()
    linear_model_with_n_and_mq = sm.OLS(y_wnmq, X_wnmq).fit()
    

    # output to stdout
    print(metric_type)

    print("Interaction Terms")
    print("model with day/volume interaction vs. linear")
    print(anova_lm(linear_model, model))
    print("model with model/quantile interaction vs. linear")
    print(anova_lm(linear_model, mq_model))
    print("model with model/quantile interaction and n vs. linear without n")
    print(anova_lm(linear_model, linear_model_with_n_and_mq))
    print("Model with day/volume and model/quantile interactions vs. linear")
    print(anova_lm(linear_model, second_order_model))

    print("Variable Exclusion")
    print("impact of day")
    print(anova_lm(no_day_model, linear_model))
    print("impact of volume")
    print(anova_lm(no_volume_model, linear_model))
    print("impact of quantile number")
    print(anova_lm(no_quantile_model, linear_model))
    print("impact of model type")
    print(anova_lm(no_model_model, linear_model))
    print("impact of adding n")
    print(anova_lm(linear_model, linear_model_with_n))

    print("Linear model + model/quantile interaction")
    print(mq_model.summary())
    print("Linear model + model/quantile interaction + n")
    print(linear_model_with_n_and_mq.summary())

    # remove NaN from baseline entries
    if metric_type in ["mpiw", "picp"]:
        filtered_df = filtered_df.dropna()
        summary_df = (
            filtered_df.groupby(METRIC_SUMMARY_COLUMNS)
            .agg(
                **{
                    f"{metric_type}_mean": pd.NamedAgg(column="value", aggfunc="mean"),
                    f"{metric_type}_median": pd.NamedAgg(
                        column="value", aggfunc="median"
                    ),
                    f"{metric_type}_std": pd.NamedAgg(column="value", aggfunc="std"),
                    f"{metric_type}_total_group_n": pd.NamedAgg(
                        column="n", aggfunc="sum"
                    ),
                    f"{metric_type}_mean_group_n": pd.NamedAgg(
                        column="n", aggfunc="mean"
                    ),
                    f"{metric_type}_median_group_n": pd.NamedAgg(
                        column="n", aggfunc="median"
                    ),
                    f"{metric_type}_std_group_n": pd.NamedAgg(
                        column="n", aggfunc="std"
                    ),
                }
            )
            .reset_index()
        )

    else:
        summary_df = (
            filtered_df.groupby(METRIC_SUMMARY_COLUMNS)
            .agg(
                **{
                    f"{metric_type}_mean": pd.NamedAgg(column="value", aggfunc="mean"),
                    f"{metric_type}_median": pd.NamedAgg(
                        column="value", aggfunc="median"
                    ),
                    f"{metric_type}_std": pd.NamedAgg(column="value", aggfunc="std"),
                    f"{metric_type}_group_n_per_seed": pd.NamedAgg(
                        column="n", aggfunc="median"
                    ),
                    f"{metric_type}_total_group_n": pd.NamedAgg(
                        column="n", aggfunc="sum"
                    ),
                }
            )
            .reset_index()
        )

    file_name = PRETRAINED_STEM + f"_{metric_type}_summary.csv"
    file_path = SUMMARY_DIR / file_name
    summary_df.to_csv(file_path)
