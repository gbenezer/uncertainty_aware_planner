"""Paper-style WAPE / WPE for a forecasting run, to compare with the dataset authors' published numbers.

Matches demand_forecasting/TFT/predictTFT.py: the point forecast is the median (q500), only days with no
stockout between 06:00 and 22:00 are scored, and the reported number is the MEAN OF THE 7 DAILY values (not
one pooled WAPE). Groups follow the authors' psd slices (mean daily training demand >= 1 / < 1).
This is a sanity check for model code, not the project's evaluation (that lives in evaluation/).

    python experiments/vishnu/paper_wape.py \
        --quantiles data/runs/forecasting/fc-tft-quantile-2025/final/quantiles_test.parquet \
        --demand-daily data/processed/demand_daily/imp-none-censored.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd


def paper_metrics(quantiles: pd.DataFrame, demand_daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["store_id", "product_id"]
    daily = demand_daily.assign(
        target_date=pd.to_datetime(demand_daily["date"]).dt.date
    )
    first_day = quantiles["target_date"].min()
    psd = (
        daily[daily["target_date"] < first_day]
        .groupby(keys)["demand_total"]
        .mean()
        .rename("psd")
    )
    scored = quantiles.merge(
        daily[keys + ["target_date", "demand_total", "stock_hour6_22_cnt"]],
        on=keys + ["target_date"],
    ).join(psd, on=keys)
    scored = scored[scored["stock_hour6_22_cnt"] == 0]
    scored["error"] = scored["q500"] - scored["demand_total"]

    rows = []
    for group, mask in {
        "psd>=0": scored["psd"] >= 0,
        "psd>=1": scored["psd"] >= 1,
        "psd<1": scored["psd"] < 1,
    }.items():
        per_day = scored[mask].groupby("target_date")
        actual = per_day["demand_total"].apply(lambda s: s.abs().sum())
        wape = per_day["error"].apply(lambda s: s.abs().sum()) / actual
        wpe = per_day["error"].sum() / actual
        rows.append(
            {
                "group": group,
                "wape": wape.mean(),
                "wpe": wpe.mean(),
                "days": len(wape),
                "rows": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quantiles", required=True, help="quantiles_<part>.parquet from a run"
    )
    parser.add_argument("--demand-daily", required=True)
    args = parser.parse_args()
    result = paper_metrics(
        pd.read_parquet(args.quantiles), pd.read_parquet(args.demand_daily)
    )
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
