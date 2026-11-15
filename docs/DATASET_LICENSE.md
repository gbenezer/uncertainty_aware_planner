# Dataset license and attribution

This project uses **FreshRetailNet-50K** by Dingdong Ltd. and collaborators, licensed under the
**Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

| | |
|---|---|
| Dataset | FreshRetailNet-50K |
| Source | <https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K> |
| Version used | Hub revision `c374e85975c82816c495acfa904cc55ab14a47d3`, dataset card version 1.0 (05/08/2025), pinned in `configs/data.yaml` |
| File hashes | SHA-256 of `data/train.parquet` and `data/eval.parquet` in `configs/data.yaml`; `load` stops if a download differs |
| License | CC BY 4.0, summary at <https://creativecommons.org/licenses/by/4.0/>, legal code at <https://creativecommons.org/licenses/by/4.0/legalcode> |
| Paper | Y. Wang et al., *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail*, arXiv:2505.16319, doi:10.48550/arXiv.2505.16319 |
| Baseline code | <https://github.com/Dingdong-Inc/frn-50k-baseline> |

## What the license allows and requires

CC BY 4.0 allows sharing and adapting the data for any purpose, provided that we:

- credit the creators (above),
- link to the license (above),
- say whether we changed the data (next section),
- add no terms or technical measures that restrict others from doing what the license permits.

The data is provided as-is, without warranties. The legal code linked above is the binding text.

## What we keep in this repository

- **No raw or derived FreshRetailNet-50K data is committed.** `data/` is gitignored, and each machine downloads
  the pinned revision with `uv run python -m uncertainty_aware_planner.data load`.
- `fixtures/raw_series/` holds **synthetic** tables with the same schema, made by
  `fixtures/synthetic_raw_series.py`. They contain no FreshRetailNet-50K values.
- `results/` holds only summary metrics computed from the data.

## Changes we make to the data

These changes happen in memory or in gitignored files under `data/`, not in the source files.

1. **Discount cleaning** (`configs/data.yaml` `cleaning`, `data/clean.py`):
   - A discount of 0, a missing-value code, is set to 1.0 (no discount).
   - Values above 1.0 are capped at 1.0.
2. **Stockout recovery:** sales in stockout hours are replaced by model estimates (Feature 1). The estimates are
   stored separately (`data/processed/imputed_hourly/`), next to the recorded sales.
3. **Daily aggregation:** hourly sales, recorded or recovered, are summed to daily demand for forecasting
   (`data/daily.py`).
4. **Evaluation masks:** some in-stock hours are hidden on purpose so imputation can be scored (`data/masks.py`).
5. **Model inputs:** covariates are rescaled within each window, as in the dataset authors' code
   (`data/pypots_arrays.py`).

We add no outside data and make no attempt to identify the anonymized store, product or city codes.
