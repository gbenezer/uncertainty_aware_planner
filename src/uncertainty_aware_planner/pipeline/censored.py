"""A "no recovery" imputed_hourly run: every real stockout hour keeps the sales recorded in that hour.

demand_daily built from it has demand_total equal to the raw daily sales, which is what the dataset authors'
censored baselines train on (TFT 31.75% WAPE, DLinear 30.60%). Use it to check forecasting models against the
paper before an imputation run exists, and as the "no recovery" comparison afterwards.

    python -m uncertainty_aware_planner.pipeline.censored
    python -m uncertainty_aware_planner.data daily \
        --imputed data/processed/imputed_hourly/imp-none-censored.parquet --run-id imp-none-censored
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.config import load_config, resolve
from uncertainty_aware_planner.data.cube import Cube, load_cube

RUN_ID = "imp-none-censored"


def censored_imputed_hourly(cube: Cube) -> pd.DataFrame:
    """The five imputed_hourly columns demand_daily reads, one row per real stockout hour."""
    series, day, hour = np.nonzero(cube.stockout)
    stamps = cube.dates[day].astype("datetime64[h]") + hour.astype("timedelta64[h]")
    recorded = np.nan_to_num(cube.sales[series, day, hour], nan=0.0)
    return pd.DataFrame(
        {
            "store_id": cube.series["store_id"].to_numpy()[series],
            "product_id": cube.series["product_id"].to_numpy()[series],
            "datetime": stamps.astype("datetime64[s]"),
            "estimated_demand": np.clip(recorded, 0.0, None),
            "actual_demand": np.nan,
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rows = censored_imputed_hourly(load_cube(cfg))
    output = (
        resolve(cfg["paths"]["processed_dir"]) / "imputed_hourly" / f"{RUN_ID}.parquet"
    )
    if output.exists():
        raise FileExistsError(f"{output} exists; delete it to rebuild")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output, index=False)
    print(f"wrote {output}: {len(rows)} stockout hours")


if __name__ == "__main__":
    main()
