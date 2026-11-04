
from pathlib import Path

import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Uncertainty-Aware Demand Planning",
    layout="wide",
)

st.title("Uncertainty-Aware Demand Planning")
st.caption("FreshRetailNet-50K | Processed Daily Demand")

# Dataset path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_daily"
    / "imp-none-censored.parquet"
)

# Load processed daily data
@st.cache_data(show_spinner="Loading processed demand data...")
def load_data():
    data = pd.read_parquet(
        DATA_PATH,
        columns=[
            "store_id",
            "product_id",
            "date",
            "demand_observed",
            "demand_recovered",
            "demand_total",
            "stock_hour6_22_cnt",
            "discount",
            "holiday_flag",
            "activity_flag",
            "avg_temperature",
            "avg_humidity",
        ],
    )

    data["date"] = pd.to_datetime(data["date"])
    return data


if not DATA_PATH.exists():
    st.error(f"Processed dataset not found: {DATA_PATH}")
    st.stop()

df = load_data()

# Store and product selection
col1, col2 = st.columns(2)

with col1:
    store_id = st.selectbox(
        "Select Store",
        sorted(df["store_id"].unique()),
    )

store_df = df[df["store_id"] == store_id]

with col2:
    product_id = st.selectbox(
        "Select Product",
        sorted(store_df["product_id"].unique()),
    )

# Filter selected product
filtered = store_df[
    store_df["product_id"] == product_id
].copy()

filtered = filtered.sort_values("date")

if filtered.empty:
    st.warning("No data available for this product.")
    st.stop()

# Product information
st.divider()
st.subheader(f"Store {store_id} | Product {product_id}")

total_days = filtered["date"].nunique()
average_demand = filtered["demand_total"].mean()
zero_demand_days = (filtered["demand_observed"] == 0).sum()

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Days of Data", f"{total_days:,}")

with col2:
    st.metric(
        "Average Daily Demand (Scaled)",
        f"{average_demand:.2f}",
    )

with col3:
    st.metric(
        "Zero Observed-Demand Days",
        f"{zero_demand_days:,}",
    )

# Dashboard tabs
tab1, tab2, tab3 = st.tabs(
    ["Demand Chart", "Data Overrides", "Covariates"]
)

# Demand chart
with tab1:
    st.subheader("Historical Daily Demand")

    chart_data = filtered[
        ["date", "demand_observed", "demand_total"]
    ].rename(
        columns={
            "demand_observed": "Observed Demand",
            "demand_total": "No-Recovery Baseline",
        }
    )

    st.line_chart(
        chart_data,
        x="date",
        y=["Observed Demand", "No-Recovery Baseline"],
        x_label="Date",
        y_label="Scaled Demand",
        height=400,
    )

    st.info(
        "This chart currently uses the no-recovery baseline. "
        "The baseline keeps recorded sales unchanged during "
        "stockouts, so the two lines may overlap. "
        "Recovered demand and prediction intervals will be "
        "added when the trained models are available."
    )

# Data overrides
with tab2:
    st.subheader("Manual Demand Corrections")

    st.info(
        "Manual editing will be connected to the team's "
        "data pipeline. Corrections will be saved with "
        "a reason and used in a future forecast run."
    )

# Covariates
with tab3:
    st.subheader("Discount History")

    st.line_chart(
        filtered,
        x="date",
        y="discount",
        x_label="Date",
        y_label="Discount",
    )

    st.subheader("Temperature History")

    st.line_chart(
        filtered,
        x="date",
        y="avg_temperature",
        x_label="Date",
        y_label="Average Temperature",
    )

    st.subheader("Humidity History")

    st.line_chart(
        filtered,
        x="date",
        y="avg_humidity",
        x_label="Date",
        y_label="Average Humidity",
    )

    st.subheader("Holiday and Activity Indicators")

    st.line_chart(
        filtered,
        x="date",
        y=["holiday_flag", "activity_flag"],
    )
