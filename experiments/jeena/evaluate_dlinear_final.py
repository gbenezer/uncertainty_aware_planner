"""Evaluate final DLinear forecasts and calibrated intervals."""

from pathlib import Path

import numpy as np
import pandas as pd

FORECAST_PATH = Path(
    "data/processed/forecast/imp-none-censored/fc-dlinear-quantile-2025/calibrated/final_test.parquet"
)

SLICES_PATH = Path("data/processed/slices.parquet")

OUTPUT_PATH = Path("results/forecasting/dlinear_final_test_metrics.csv")


def calculate_metrics(df):
    y = df["actual_demand"].to_numpy()
    pred = df["estimated_demand"].to_numpy()

    lower = df["q025"].to_numpy()
    upper = df["q975"].to_numpy()

    lower_cal = df["q025_cal"].to_numpy()
    upper_cal = df["q975_cal"].to_numpy()

    denominator = np.abs(y).sum()

    return {
        "rows": len(df),
        "wape": (
            np.abs(y - pred).sum() / denominator * 100 if denominator > 0 else np.nan
        ),
        "picp_before": np.mean((y >= lower) & (y <= upper)) * 100,
        "picp_after": np.mean((y >= lower_cal) & (y <= upper_cal)) * 100,
        "mpiw_before": np.mean(upper - lower),
        "mpiw_after": np.mean(upper_cal - lower_cal),
    }


def main():
    df = pd.read_parquet(FORECAST_PATH)

    slices = pd.read_parquet(
        SLICES_PATH,
        columns=["store_id", "product_id", "slice"],
    )

    keys = ["store_id", "product_id"]

    if slices.duplicated(keys).any():
        raise ValueError("Duplicate store-product slice labels")

    df = df.merge(
        slices,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    if df["slice"].isna().any():
        raise ValueError("Missing slice labels")

    required = [
        "actual_demand",
        "estimated_demand",
        "q025",
        "q975",
        "q025_cal",
        "q975_cal",
    ]

    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError("Missing or non-finite forecast values")

    if (df["q025"] > df["q975"]).any() or (df["q025_cal"] > df["q975_cal"]).any():
        raise ValueError("Invalid prediction intervals")

    results = []

    for group in ["all", "low_sales", "high_sales"]:
        subset = df if group == "all" else df.loc[df["slice"] == group]

        if subset.empty:
            raise ValueError(f"No rows for {group}")

        metrics = calculate_metrics(subset)
        results.append({"slice": group, **metrics})

        print(f"\n{group}")
        print(f"Rows: {metrics['rows']:,}")
        print(f"WAPE: {metrics['wape']:.2f}%")
        print(f"PICP before: {metrics['picp_before']:.2f}%")
        print(f"PICP after: {metrics['picp_after']:.2f}%")
        print(f"MPIW before: {metrics['mpiw_before']:.4f}")
        print(f"MPIW after: {metrics['mpiw_after']:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
