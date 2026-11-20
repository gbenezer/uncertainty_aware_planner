"""Compare DLinear moving-average windows using official metrics."""

from pathlib import Path

import pandas as pd

METRICS_PATH = Path("results/forecasting/imp-none-censored/project/metrics.csv")

MODELS = {
    "MA7": "fc-dlinear-quantile",
    # Existing forecasts used the old model ID.
    "MA29": "dlinear_forecasting_quantile_ma29.yaml",
}

SEED = 2025


def main():
    df = pd.read_csv(METRICS_PATH)

    results = df.loc[
        (df["model_id"].isin(MODELS.values()))
        & (df["metric"] == "wape")
        & (df["slice"] == "all")
        & (df["split"] == "val")
        & (df["point"] == "q500")
        & (df["protocol"] == "project")
        & (df["imputation_run_id"] == "imp-none-censored")
        & (df["seed"] == SEED)
        & (df["calibrated"] == False)
    ].copy()

    if results.empty:
        raise ValueError("No matching validation metrics found")

    results["model"] = results["model_id"].map(
        {model_id: name for name, model_id in MODELS.items()}
    )

    results["wape_percent"] = results["value"] * 100

    comparison = results.pivot(
        index="fold",
        columns="model",
        values="wape_percent",
    )

    expected_folds = set(range(1, 7))

    if (
        set(comparison.index) != expected_folds
        or set(comparison.columns) != set(MODELS)
        or comparison.isna().any().any()
    ):
        raise ValueError("Expected six complete folds for MA7 and MA29")

    print("\nWAPE by validation fold (%):")
    print(comparison.round(2).to_string())

    print("\nAverage CV WAPE (%):")
    print(comparison.mean().round(2).to_string())

    output = Path("results/forecasting/dlinear_ma_comparison_official.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output)

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
