"""Create shared low-sales and high-sales product labels."""

from pathlib import Path

import pandas as pd
import yaml

DATA_PATH = Path("data/processed/demand_daily/imp-none-censored.parquet")
SPLITS_PATH = Path("configs/splits.yaml")
OUTPUT_PATH = Path("data/processed/slices.parquet")

KEYS = ["store_id", "product_id"]


def main():
    with SPLITS_PATH.open("r") as file:
        config = yaml.safe_load(file)

    # Use the earliest validation cutoff to prevent future-data leakage.
    first_cutoff = pd.Timestamp(min(config["folds"]["cutoffs"]))
    train_days = config["folds"]["train_days"]

    start_date = first_cutoff - pd.Timedelta(days=train_days)
    end_date = first_cutoff - pd.Timedelta(days=1)

    print(f"Slice history: {start_date.date()} to {end_date.date()}")

    df = pd.read_parquet(
        DATA_PATH,
        columns=[
            "store_id",
            "product_id",
            "date",
            "demand_total",
            "forecast_split",
        ],
    )

    df["date"] = pd.to_datetime(df["date"])

    history = df.loc[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
        & (df["forecast_split"] == "train")
    ].copy()

    if history.empty:
        raise ValueError("No training records found")

    if history["demand_total"].isna().any():
        raise ValueError("Historical demand contains missing values")

    if history.duplicated(KEYS + ["date"]).any():
        raise ValueError("Duplicate store-product-day records found")

    slices = history.groupby(KEYS, as_index=False).agg(
        mean_scaled_demand=("demand_total", "mean"),
        history_days=("demand_total", "size"),
    )

    slices["slice"] = "high_sales"
    slices.loc[
        slices["mean_scaled_demand"] < 1,
        "slice",
    ] = "low_sales"

    if slices.duplicated(KEYS).any():
        raise ValueError("Duplicate store-product pairs")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    slices.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nTotal store-product pairs: {len(slices):,}")
    print("\nSlice distribution:")
    print(slices["slice"].value_counts().to_string())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
