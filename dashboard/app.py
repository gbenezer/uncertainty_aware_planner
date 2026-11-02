
from pathlib import Path

import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Uncertainty-Aware Demand Planning",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Uncertainty-Aware Demand Planning")
st.caption("FreshRetailNet-50K | Historical Sales")

# Dataset path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data" / "train.parquet"


# Load dataset
@st.cache_data(show_spinner="Loading sales data...")
def load_data():
    data = pd.read_parquet(
        DATA_PATH,
        columns=[
            "store_id",
            "product_id",
            "dt",
            "sale_amount",
            "stock_hour6_22_cnt",
        ],
    )

    data["dt"] = pd.to_datetime(data["dt"])
    return data


if not DATA_PATH.exists():
    st.error(f"Dataset not found. Expected file: {DATA_PATH}")
    st.stop()

df = load_data()

# Store and product selection
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    store_id = st.selectbox(
        "Select Store",
        options=sorted(df["store_id"].unique()),
        key="store_id",
        width="stretch",
    )

store_df = df[df["store_id"] == store_id]

with col2:
    product_id = st.selectbox(
        "Select Product",
        options=sorted(store_df["product_id"].unique()),
        key="product_id",
        width="stretch",
    )

# Filter selected product data
filtered = store_df[
    store_df["product_id"] == product_id
].copy()

filtered = filtered.sort_values("dt")

if filtered.empty:
    st.warning("No sales data available for this store-product combination.")
    st.stop()

# Product information
st.divider()
st.subheader(f"Store {store_id} | Product {product_id}")

total_days = len(filtered)
average_sales = filtered["sale_amount"].mean()
zero_sales_days = (filtered["sale_amount"] == 0).sum()

# Summary metrics
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Days of Sales Data",
        f"{total_days:,}",
    )

with metric2:
    st.metric(
        "Average Daily Sales (Scaled)",
        f"{average_sales:.2f}",
    )

with metric3:
    st.metric(
        "Zero-Sales Days",
        f"{zero_sales_days:,}",
    )

# Historical sales chart
st.subheader("Historical Daily Sales")

st.line_chart(
    filtered,
    x="dt",
    y="sale_amount",
    x_label="Date",
    y_label="Scaled Sales",
    width="stretch",
    height=400,
)

st.caption(
    "Sales values are scaed. "
    "They do not represent actual numbers of items sold."
)
