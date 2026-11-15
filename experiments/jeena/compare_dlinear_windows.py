"""Compare DLinear moving-average windows across validation folds."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EXPERIMENTS = {
    "MA7": {
        "config": "configs/forecasting_models/dlinear_forecasting_quantile.yaml",
        "run_id": "fc-dlinear-quantile-2025",
    },
    "MA29": {
        "config": "configs/forecasting_models/dlinear_forecasting_quantile_ma29.yaml",
        "run_id": "dlinear_forecasting_quantile_ma29.yaml-2025",
    },
}

DATA_PATH = "data/processed/demand_daily/imp-none-censored.parquet"

OUTPUT_DIR = Path("data/processed/forecast")
RESULTS_FILE = Path("results/forecasting/dlinear_ma_comparison.csv")


def calculate_wape(path):
    """Calculate WAPE in batches to reduce memory use."""

    total_error = 0.0
    total_actual = 0.0
    total_rows = 0

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=100_000,
        columns=["actual_demand", "estimated_demand"],
    ):
        actual = batch.column(0).to_numpy()
        predicted = batch.column(1).to_numpy()

        valid = np.isfinite(actual) & np.isfinite(predicted)

        actual = actual[valid]
        predicted = predicted[valid]

        total_error += np.abs(actual - predicted).sum()
        total_actual += np.abs(actual).sum()
        total_rows += len(actual)

    wape = total_error / total_actual * 100 if total_actual > 0 else float("nan")

    return wape, total_rows


def main():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for fold in range(1, 7):
        for name, experiment in EXPERIMENTS.items():
            output = OUTPUT_DIR / experiment["run_id"] / f"fold{fold}.parquet"

            if output.exists():
                print(f"Using existing {name} Fold {fold}", flush=True)
            else:
                print(f"Running {name} Fold {fold}", flush=True)

                command = [
                    sys.executable,
                    "-m",
                    "uncertainty_aware_planner.pipeline.forecast",
                    "--config",
                    experiment["config"],
                    "--demand-daily",
                    DATA_PATH,
                    "--seed",
                    "2025",
                    "--fold",
                    str(fold),
                ]

                subprocess.run(command, check=True)

            wape, rows = calculate_wape(output)

            results.append(
                {
                    "model": name,
                    "fold": fold,
                    "wape": round(wape, 4),
                    "rows": rows,
                }
            )

            print(
                f"{name} Fold {fold}: WAPE = {wape:.2f}%",
                flush=True,
            )

            with RESULTS_FILE.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["model", "fold", "wape", "rows"],
                )
                writer.writeheader()
                writer.writerows(results)

    print("\nAverage WAPE across folds:")

    for name in EXPERIMENTS:
        values = [row["wape"] for row in results if row["model"] == name]

        print(f"{name}: {np.mean(values):.2f}%")


if __name__ == "__main__":
    main()
