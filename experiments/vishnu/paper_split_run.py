"""TFT trained on all 90 training days (03-28..06-25), as the dataset authors do, and scored on the test week.

The project's final split holds 06-19..06-25 out for conformal calibration. This run measures what that costs
against the paper's censored TFT (31.75% WAPE). It changes only the final training window, in memory; outputs go
to separate *_paper_split folders so they never mix with project runs.

    python experiments/vishnu/paper_split_run.py --seed 2025 \
        --demand-daily data/processed/demand_daily/imp-none-censored.parquet
    python experiments/vishnu/paper_wape.py --demand-daily data/processed/demand_daily/imp-none-censored.parquet \
        --quantiles data/runs/forecasting_paper_split/fc-tft-quantile-2025/final/quantiles_test.parquet
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from uncertainty_aware_planner.data.config import load_config, load_yaml, resolve
from uncertainty_aware_planner.pipeline.forecast import run

CONFIG = "configs/forecasting_models/temporal_fusion_transformer.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demand-daily", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    model_cfg["output_dir"] = model_cfg["output_dir"] + "_paper_split"
    model_cfg["run_dir"] = model_cfg["run_dir"] + "_paper_split"
    data_cfg = load_config(model_cfg["data_config"])
    final = data_cfg["splits"]["final"]
    final["train"] = [
        final["train"][0],
        final["calib"][1],
    ]  # train through the calibration week

    demand = pd.read_parquet(resolve(args.demand_daily))
    log = run(
        model_cfg,
        data_cfg,
        demand,
        args.seed,
        eval_parts=["test"],
        demand_daily_path=args.demand_daily,
    )
    print(json.dumps({"run_id": log["run_id"], **log["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
