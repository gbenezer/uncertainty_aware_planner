from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Demand Planning Dashboard",
    layout="wide",
)

st.title("Uncertainty-Aware Demand Planning")
st.caption("FreshRetailNet-50K | Historical Sales")

# Locate dataset from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "data" / "train.parquet"


@st.cache_data
def load_data():
    return pd.read_parquet(
        DATA_PATH,
        columns=[
            "store_id",
            "product_id",
            "dt",
            "sale_amount",
            "stock_hour6_22_cnt",
        ],
    )


df = load_data()


# Create two equal-width columns
col1, col2 = st.columns(2, gap="medium")

with col1:
    store_id = st.selectbox(
        "Select Store",
        sorted(df["store_id"].unique()),
    )

# Filter products based on selected store
store_df = df[df["store_id"] == store_id]

with col2:
    product_id = st.selectbox(
        "Select Product",
        sorted(store_df["product_id"].unique()),
    )

# Filter complete history of selected product
filtered = store_df[store_df["product_id"] == product_id].copy()

filtered["dt"] = pd.to_datetime(filtered["dt"])
filtered = filtered.sort_values("dt")

st.subheader("Historical Daily Sales")

st.line_chart(
    filtered,
    x="dt",
    y="sale_amount",
)

st.subheader("Sales Records")

st.dataframe(
    filtered[["dt", "sale_amount", "stock_hour6_22_cnt"]],
    use_container_width=True,
)
