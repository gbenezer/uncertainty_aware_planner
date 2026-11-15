
"""Apply split conformal quantile calibration to DLinear forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_DIR = Path(
    "data/processed/forecast/fc-dlinear-quantile-2025"
)

CALIB_PATH = FORECAST_DIR / "final_calib.parquet"
TEST_PATH = FORECAST_DIR / "final_test.parquet"

OUTPUT_PATH = (
    FORECAST_DIR / "calibrated" / "final_test.parquet"
)

ALPHA = 0.05


def main():
    # 1. Load calibration predictions
    calib = pd.read_parquet(CALIB_PATH)

    required = ["actual_demand", "q025", "q975"]

    if calib[required].isna().any().any():
        raise ValueError("Calibration data contains missing values")

    # 2. Calculate conformal nonconformity scores
    scores = np.maximum(
        calib["q025"].to_numpy() - calib["actual_demand"].to_numpy(),
        calib["actual_demand"].to_numpy() - calib["q975"].to_numpy(),
    )

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite calibration scores")

    # 3. Calculate finite-sample conformal quantile
    n = len(scores)
    if n == 0:
        raise ValueError("No calibration samples")

    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    qhat95 = (
        np.partition(scores, k - 1)[k - 1]
        if k <= n
        else float("inf")
    )

    print(f"Calibration rows: {n:,}")
    print(f"Conformal qhat95: {qhat95:.6f}")

    # 4. Apply correction to test forecasts
    test = pd.read_parquet(TEST_PATH)

    if test[["q025", "q975"]].isna().any().any():
        raise ValueError("Test predictions contain missing bounds")

    test["qhat95"] = qhat95
    test["q025_cal"] = np.maximum(
        0.0, test["q025"] - qhat95
    )
    test["q975_cal"] = test["q975"] + qhat95

    if (test["q025_cal"] > test["q975_cal"]).any():
        raise ValueError("Invalid calibrated intervals")

    # 5. Save calibrated forecasts separately
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test.to_parquet(OUTPUT_PATH, index=False)

    print(f"Calibrated test rows: {len(test):,}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
