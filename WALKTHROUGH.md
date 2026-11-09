# Modeling and code walkthrough: fixes

This file lists the fixes made after the baseline and pipeline walkthrough, each with its commit, and then the
status of each submission checklist item.

## Fixes

| # | Fix | Commit(s) |
|---|---|---|
| 1 | **Agreed in the meeting: comments in `pyproject.toml`** explaining each pin: Python 3.12.3 (PyPOTS / `tokenizers` install under uv), torch `<2.9` (Explorer's CUDA 12.8 and V100 GPUs), the CUDA wheel index for Linux/Windows, and the pytest import paths. | `0de0abe` added them. `91e2b96` restored the ones a merge had dropped and added the last two. `e1f1d13` removed a duplicate left by the merge. |
| 2 | **One command runs the baseline on the smoke-test subset**: `uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke`. It runs on CPU in about 10 s, on the committed synthetic fixtures (no download), and checks the environment, data hash and split during the run. | `cfe9d50` |
| 3 | **Baseline result logged with commit, config, data hash and hardware**: `results/imputation/baseline_log.csv`. It holds the official full-data rows and the smoke reference rows that each smoke run is compared against. | `45939fa` |
| 4 | **Repository hygiene**: the committed `.DS_Store` is untracked and `.DS_Store` is gitignored. | `48ec7aa` |
| 5 | **Dataset license terms in the repository**: `docs/DATASET_LICENSE.md` (CC BY 4.0: source, pinned revision, attribution, and the changes we make). | `e6dfc86` |
| 6 | **README additions**: a new "Reproducing the baseline" section, the smoke command in the Quick start, how to get the same data back after deleting `data/`, and a "Leakage checks" table. References to files that do not exist (`contracts/schemas.py`, `docs/datasheet.md`, `CODEOWNERS`) are fixed. | `6179453` |
| 7 | **CI format check passes**: five files are reformatted with `ruff format`. This is a formatting-only change: each file's Python AST is identical before and after. | `79c8683` |

## Baseline

**What it is.** The non-quantile DLinear imputation model, with the loss and architecture from the dataset
paper repository (Feature Cards 1.1). It uses the same split and metric as the planned quantile models: the
six rolling folds in `configs/splits.yaml`, scored with daily-pooled WAPE and WPE by
`evaluation/error_metrics.py`.

**The numbers.** Both rows come from `experiments/gil/data/seventh_dlinear_baseline_experiment.csv`, at commit
`5709f5b`, 9 seeds × 6 folds, and are logged in `results/imputation/baseline_log.csv`:

| Variant | WAPE, mean ± sd | WPE, mean ± sd |
|---|---|---|
| **not_leaked**, the official number: fit on the fold's train window | **45.69% ± 6.97** | −34.02% ± 12.56 |
| leaked, reference only: fit on the val window's visible hours | 42.07% ± 8.05 | −26.01% ± 17.95 |

**Data version.** Dataset revision `c374e85…` with raw-file SHA-256 `6706832d…` (train) and `1b118840…`
(eval). The commit that produced the number pinned that revision. The SHA-256 check in `load` came later
(`5270b79`), and every run now performs it.

**Run it twice.** The smoke run gives identical numbers on repeat: seeds are set before every model build and
torch runs in deterministic mode. Each run is compared with the logged reference and exits with status 1 if it
differs.

On the full data, same-seed reruns on Explorer's GPUs have repeated exactly. Other hardware takes a different
training path. Seed 86400, fold 1, re-run on a laptop CPU from the current code:

| Variant | Laptop CPU | Logged on GPU |
|---|---|---|
| not_leaked | 42.93% | 42.31% |
| leaked | 49.83% | 43.25% |

Both differences are within that fold's spread across seeds (34.6% to 67.4% for not_leaked). Full-data numbers
are compared on Explorer GPUs only.

## Leakage

Every risk is named, with the code that handles it, in the README under "Leakage checks". In short:
- **Handled:** the folds, the calibration week and the test week never enter training, and hidden hours'
  true values never reach `fit`.
- **Logged separately:** the `leaked` variant, which is never the official number.
- **Open, not leakage:** folds overlap in time, so the ± across folds understates uncertainty.
- **Open risk:** the forecasting models get the observed weather of the forecast week as a known-future input,
  as the dataset authors do.

## Submission checklist

- [x] The instructor has read access to the repository.
- [x] **One baseline command in the README** (Quick start and "Reproducing the baseline"):
  `uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke`.
- [x] **Environment is committed:** `.python-version` (3.12.3), `pyproject.toml` and `uv.lock`.
- [x] **Seeds are set in code that runs every time:** `utils/util_functions.py` `set_seed()` runs before every
  model build in `pipeline/baseline_smoke.py`, `experiments/gil/data_pipeline_experimentation.py` and
  `pipeline/forecast.py`.
- [x] **Data version recorded:** `configs/data.yaml` holds the source, pinned revision, retrieval date and the
  SHA-256 of both files. `load` checks them; the smoke run checks the fixtures against
  `configs/baseline_smoke.yaml`.
- [x] **Split committed:** `configs/splits.yaml`, with six cut-offs drawn once (seed 6170) and fixed, plus the
  final train, calibration and test windows. The run checks it with `data/splits.py` `validate_splits`.
- [x] **Baseline logged** with commit, config, data hash and hardware: `results/imputation/baseline_log.csv`.
- [x] **Smoke test under 5 minutes, no GPU, no credentials:** about 10 s on an Apple M3 laptop CPU.
- [x] **No secrets, credentials or restricted data** in the repository or its history. No raw data is
  committed (`data/` is gitignored), and the largest file in history is a 2.3 MB project PDF.
- [x] A teammate who did not write the pipeline cloned the repository fresh and ran the README command.
- [x] **Licensed public dataset:** license terms are in `docs/DATASET_LICENSE.md`.
