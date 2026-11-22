from pathlib import Path
from typing import Union
import pandas as pd
import plotly.graph_objects as go
import uncertainty_aware_planner.data.helper_functions as help_fun
import uncertainty_aware_planner.evaluation.imputation_evaluation_helpers as ieh
import uncertainty_aware_planner.evaluation.error_metrics as em
import uncertainty_aware_planner.evaluation.uncertainty_metrics as um


# constants that can be modulated
INFERENCE_DATA_DIR = Path("results/imputation/pretrained_model_eval/model_outputs")
INFERENCE_FIGURE_DIR = Path("results/imputation/pretrained_model_eval/figures")
FILE_STEM = "pretrained_model_inference_output"
SEEDS = [86400, 4, 42]
THRESHOLD = 1 / 24
MEAN_SOURCE = "full"
MODEL_TYPE_QUANTILE_NUMBERS = [
    ("DLinear_Imputation_Baseline", None),
    ("DLinear_Imputation_Quantile", 2),
    ("DLinear_Imputation_Quantile", 3),
    ("TimesNet_Imputation_Quantile", 3),
]
# SERIES_HOURLY = ieh.load_series_hourly("data/hf_cache")
UNCERTAINTY_COLUMNS = [
    "store_id",
    "product_id",
    "picp",
    "coverage_gap",
    "n_picp",
    "mpiw",
    "median_piw",
    "std_piw",
    "n_mpiw",
]


# extract the worst series
TEST_SEED = 86400
TEST_MODEL_TYPE = "DLinear_Imputation_Quantile"
TEST_NUM_QUANTILES = 3

file_name = (
    FILE_STEM
    + f"_seed_{TEST_SEED}_{TEST_MODEL_TYPE}_{TEST_NUM_QUANTILES}_quantiles.csv"
)
file_path = INFERENCE_DATA_DIR / file_name

# load the appropriate data
inference_df = help_fun.import_inference_table(file_path)

# get the series of interest
worst_mase_store = 878
worst_mase_product = 51
worst_mase_boolean = (inference_df["store_id"] == worst_mase_store) & (
    inference_df["product_id"] == worst_mase_product
)

worst_picp_store = 822
worst_picp_product = 694
worst_picp_boolean = (inference_df["store_id"] == worst_picp_store) & (
    inference_df["product_id"] == worst_picp_product
)

worst_mpiw_store = 508
worst_mpiw_product = 267
worst_mpiw_boolean = (inference_df["store_id"] == worst_mpiw_store) & (
    inference_df["product_id"] == worst_mpiw_product
)

worst_mase_series_df = inference_df[worst_mase_boolean]
worst_picp_series_df = inference_df[worst_picp_boolean]
worst_mpiw_series_df = inference_df[worst_mpiw_boolean]

# bad practice, but copy-pasting a useful function from another script

# slideshow figure styling
BASE_FONT_SIZE = 18  # tick labels, legend entries, facet labels, annotations
AXIS_TITLE_FONT_SIZE = 22
LEGEND_TITLE_FONT_SIZE = 20


def apply_slide_style(fig, height: int | None = None):
    """Enlarge all text in a figure so it is legible in a slideshow."""
    fig.update_layout(
        font=dict(size=BASE_FONT_SIZE),
        legend=dict(
            font=dict(size=BASE_FONT_SIZE),
            title=dict(font=dict(size=LEGEND_TITLE_FONT_SIZE)),
        ),
        margin=dict(l=100, r=40, t=40, b=80),
    )
    fig.update_xaxes(
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=BASE_FONT_SIZE),
        tickangle=-45,
    )
    fig.update_yaxes(
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=BASE_FONT_SIZE),
    )
    # covers facet labels and hline annotations
    fig.update_annotations(font=dict(size=BASE_FONT_SIZE))
    if height is not None:
        fig.update_layout(height=height)
    return fig


def create_time_series_demand_figure(inference_df, store_id, product_id):

    fig = go.Figure()

    datetime = inference_df["datetime"]
    actual_demand = inference_df["actual_demand"]
    estimated_demand = inference_df["estimated_demand"]
    lower_quantile = inference_df["q025"]
    upper_quantile = inference_df["q975"]

    fig.add_trace(
        go.Scatter(
            x=datetime,
            y=actual_demand,
            name="Actual Demand",
            line=dict(color="black", width=4, dash="solid"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=datetime,
            y=estimated_demand,
            name="Estimated Demand",
            line=dict(color="black", width=4, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=datetime,
            y=lower_quantile,
            name="Lower Quantile (2.5%) Estimate",
            line=dict(color="black", width=4, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=datetime,
            y=upper_quantile,
            name="Upper Quantile (97.5%) Estimate",
            line=dict(color="black", width=4, dash="dot"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Calibration Week Demand Time Series<br>For Store {store_id}, Product {product_id}"
        ),
        xaxis=dict(title=dict(text="Datetime")),
        yaxis=dict(title=dict(text="Scaled Sales<br>Demand")),
    )
    return fig


worst_mase_figure = apply_slide_style(
    create_time_series_demand_figure(
        inference_df=worst_mase_series_df,
        store_id=worst_mase_store,
        product_id=worst_mase_product,
    )
)
worst_mase_figure.write_html((INFERENCE_FIGURE_DIR / "worst_mase_figure_example.html"))

worst_picp_figure = apply_slide_style(
    create_time_series_demand_figure(
        inference_df=worst_picp_series_df,
        store_id=worst_picp_store,
        product_id=worst_picp_product,
    )
)
worst_picp_figure.write_html((INFERENCE_FIGURE_DIR / "worst_picp_figure_example.html"))

worst_mpiw_figure = apply_slide_style(
    create_time_series_demand_figure(
        inference_df=worst_mpiw_series_df,
        store_id=worst_mpiw_store,
        product_id=worst_mpiw_product,
    )
)
worst_mpiw_figure.write_html((INFERENCE_FIGURE_DIR / "worst_mpiw_figure_example.html"))
