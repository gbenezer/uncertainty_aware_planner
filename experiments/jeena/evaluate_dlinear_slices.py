"""Evaluate DLinear Quantile forecasts by sales-volume slice."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FORECAST_DIR = Path(
    "data/processed/forecast/imp-none-censored/fc-dlinear-quantile-2025"
)
SLICES_PATH = Path("data/processed/jeena_slices_experimental.parquet")
OUTPUT_PATH = Path("results/forecasting/dlinear_ma7_slice_evaluation.csv")

KEYS = ["store_id", "product_id"]
GROUPS = ["all", "low_sales", "high_sales"]


def empty_stats():
    return {
        "rows": 0,
        "error": 0.0,
        "actual_sum": 0.0,
        "covered": 0,
        "width_sum": 0.0,
    }


def update(stats, df):
    actual = df["actual_demand"].to_numpy()
    pred = df["estimated_demand"].to_numpy()
    lower = df["q025"].to_numpy()
    upper = df["q975"].to_numpy()

    valid = (
        np.isfinite(actual)
        & np.isfinite(pred)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (lower <= upper)
    )

    actual = actual[valid]
    pred = pred[valid]
    lower = lower[valid]
    upper = upper[valid]

    stats["rows"] += len(actual)
    stats["error"] += np.abs(actual - pred).sum()
    stats["actual_sum"] += np.abs(actual).sum()
    stats["covered"] += np.count_nonzero((actual >= lower) & (actual <= upper))
    stats["width_sum"] += (upper - lower).sum()


def metrics(stats):
    n = stats["rows"]

    return {
        "rows": n,
        "wape": (
            100 * stats["error"] / stats["actual_sum"]
            if stats["actual_sum"] > 0
            else float("nan")
        ),
        "picp": (100 * stats["covered"] / n if n else float("nan")),
        "mpiw": (stats["width_sum"] / n if n else float("nan")),
    }


def main():
    slices = pd.read_parquet(
        SLICES_PATH,
        columns=KEYS + ["slice"],
    )

    if slices.duplicated(KEYS).any():
        raise ValueError("Duplicate slice keys")

    results = []
    pooled = {group: empty_stats() for group in GROUPS}

    for fold in range(1, 7):
        path = FORECAST_DIR / f"fold{fold}.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        print(f"\nEvaluating Fold {fold}...", flush=True)

        fold_stats = {group: empty_stats() for group in GROUPS}

        parquet = pq.ParquetFile(path)

        for batch in parquet.iter_batches(
            batch_size=100_000,
            columns=KEYS
            + [
                "actual_demand",
                "estimated_demand",
                "q025",
                "q975",
            ],
        ):
            df = batch.to_pandas()

            df = df.merge(
                slices,
                on=KEYS,
                how="left",
                validate="many_to_one",
            )

            if df["slice"].isna().any():
                raise ValueError(f"Fold {fold}: some products lack slice labels")

            for group in GROUPS:
                subset = df if group == "all" else df.loc[df["slice"] == group]

                update(fold_stats[group], subset)
                update(pooled[group], subset)

        for group in GROUPS:
            result = metrics(fold_stats[group])
            results.append(
                {
                    "fold": fold,
                    "slice": group,
                    **result,
                }
            )

            print(
                f"{group}: "
                f"WAPE={result['wape']:.2f}%, "
                f"PICP={result['picp']:.2f}%, "
                f"MPIW={result['mpiw']:.4f}"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fold",
                "slice",
                "rows",
                "wape",
                "picp",
                "mpiw",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nOverall results (pooled across folds):")

    for group in GROUPS:
        result = metrics(pooled[group])

        print(
            f"{group}: "
            f"WAPE={result['wape']:.2f}%, "
            f"PICP={result['picp']:.2f}%, "
            f"MPIW={result['mpiw']:.4f}"
        )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
