"""Evaluate PICP and MPIW for DLinear Quantile forecasts.

Example:
    uv run python experiments/jeena/evaluate_dlinear_intervals.py

Reads saved forecast Parquet files and calculates interval metrics.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_FORECAST_DIR = Path("data/processed/forecast/fc-dlinear-quantile-2025")

DEFAULT_OUTPUT = Path("results/forecasting/dlinear_interval_evaluation.csv")


def calculate_metrics(path: Path) -> dict:
    """Calculate interval coverage and width in batches."""

    covered = 0
    total = 0
    width_sum = 0.0

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["actual_demand", "q025", "q975"],
    ):
        actual = batch.column(0).to_numpy()
        lower = batch.column(1).to_numpy()
        upper = batch.column(2).to_numpy()

        valid = (
            np.isfinite(actual)
            & np.isfinite(lower)
            & np.isfinite(upper)
            & (lower <= upper)
        )

        actual = actual[valid]
        lower = lower[valid]
        upper = upper[valid]

        covered += np.count_nonzero((actual >= lower) & (actual <= upper))

        width_sum += np.sum(upper - lower)
        total += len(actual)

    if total == 0:
        raise ValueError(f"No valid evaluation rows in {path}")

    return {
        "rows": total,
        "picp": round(covered / total * 100, 4),
        "mpiw": round(width_sum / total, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=DEFAULT_FORECAST_DIR,
        help="Folder containing fold1.parquet, fold2.parquet, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    results = []

    for fold in range(1, 7):
        path = args.forecast_dir / f"fold{fold}.parquet"

        if not path.exists():
            print(f"Skipping Fold {fold}: file not found")
            continue

        metrics = calculate_metrics(path)

        result = {
            "fold": fold,
            **metrics,
        }
        results.append(result)

        print(
            f"Fold {fold}: "
            f"PICP = {metrics['picp']:.2f}%, "
            f"MPIW = {metrics['mpiw']:.4f}, "
            f"Rows = {metrics['rows']:,}"
        )

    if not results:
        raise FileNotFoundError(f"No forecast files found in {args.forecast_dir}")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["fold", "rows", "picp", "mpiw"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output}")

    print(f"Average PICP: {np.mean([r['picp'] for r in results]):.2f}%")
    print(f"Average MPIW: {np.mean([r['mpiw'] for r in results]):.4f}")


if __name__ == "__main__":
    main()
