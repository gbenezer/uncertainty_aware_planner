# Running on Explorer (SLURM)

Laptop for tests and small samples; Explorer for full runs. Every script here is submitted **from the repo root** (logs and relative paths assume it).

| File | Use |
|---|---|
| `env.sh` | Shared job setup, sourced by every script: caches on `/scratch`, `PYTHONPATH`, `.venv`, prints host / commit / GPU / torch. |
| `setup_env.sbatch` | One-time `uv sync` on a GPU node + CUDA check. Re-run when `pyproject.toml` or `uv.lock` changes. |
| `cpu_job.sbatch` | Any command on a CPU node (data steps). |
| `gpu_job.sbatch` | Any command on one GPU, with auto-requeue before the time limit. |
| `array_job.sbatch` | A seed × fold grid as one job array (appends `--seed` / `--fold`). |

Logs go to `slurm/job_logs/<job-name>.<job-id>.out` (gitignored).

## 1. One-time setup

```bash
ssh <user>@login.explorer.northeastern.edu
git clone https://github.com/gbenezer/uncertainty_aware_planner && cd uncertainty_aware_planner
sinfo -p gpu,short -o "%P %l %G"      # check partition names, time limits, GPU types
sbatch slurm/setup_env.sbatch         # installs uv if missing, uv sync, checks CUDA
```

The scripts run on Explorer (Linux), so the laptop OS only matters for connecting. **From Windows:** run the same `ssh` command in PowerShell or Windows Terminal (OpenSSH ships with Windows 10/11), or use WSL or VS Code Remote-SSH. Clone and submit on Explorer itself rather than copying scripts over from a Windows checkout; `.gitattributes` keeps `*.sh` / `*.sbatch` on LF line endings in any case.

The setup log must show `cuda True` and a GPU name. torch is pinned to 2.8 (cu128) because torch ≥ 2.9 drops the V100 (compute capability 7.0). If a partition or limit in `sinfo` differs from the `#SBATCH` lines, override at submit time (`sbatch --partition=... --time=...`) rather than editing the scripts for one run.

Caches (`HF_HOME`, `HF_DATASETS_CACHE`, `UV_CACHE_DIR`) live under `/scratch/$USER/uap`; set `UAP_SCRATCH` to move them. The dataset itself stays in `data/` as configured in `configs/data.yaml`.

## 2. Submitting

```bash
# data steps (CPU)
sbatch slurm/cpu_job.sbatch python -m uncertainty_aware_planner.data load
sbatch --mem=64G slurm/cpu_job.sbatch python -m uncertainty_aware_planner.data splits

# any GPU command
sbatch slurm/gpu_job.sbatch python path/to/script.py --arg value

# 3 seeds x (6 folds + final) = 21 tasks: one seed (7 tasks) per submit, 3 at a time
sbatch --array=0-6%3 slurm/array_job.sbatch "2025 2026 2027" "1 2 3 4 5 6 final" \
    python -m <module> <args...>   # then --array=7-13%3 and --array=14-20%3
```

**Limits (QOS `gpu`):** at most 8 jobs submitted and 4 running per user, and every array task counts as a job. `sbatch` rejects a larger array outright (`QOSMaxSubmitJobPerUserLimit`), so submit a grid in chunks with the same arguments, for example one seed (7 tasks) at a time: `--array=0-6`, then `--array=7-13`, then `--array=14-20`. `sacctmgr show qos format=name,maxsubmitpu,maxjobspu` shows the current limits.

Name jobs with `--job-name` so the logs are easy to find, e.g. `sbatch --job-name=tft_f3 slurm/gpu_job.sbatch ...`.

## 3. Monitoring

```bash
squeue -u $USER                              # queued / running
tail -f slurm/job_logs/<name>.<id>.out       # live log
sacct -j <id> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode   # after it ends
scancel <id>                                 # cancel (an array: scancel <array-id>)
seff <id>                                    # CPU / memory efficiency, to right-size --mem and --time
```

Interactive GPU shell for debugging (name it `interactive` so Lightning does not treat it as a batch job):

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=interactive --pty bash
source slurm/env.sh
```

## 4. Requeue and resume

`gpu_job.sbatch` and `array_job.sbatch` ask SLURM for `SIGUSR1` 90 s before the time limit. Training code that uses a Lightning `Trainer` with `SLURMEnvironment` saves `hpc_ckpt_*.ckpt` in its `default_root_dir` and requeues itself; the requeued job resumes from that checkpoint. This only works when the command is launched with `srun` (the scripts do this) and the run directory is the same on restart. Non-Lightning scripts are simply stopped at the time limit.

## 5. TFT runs

`demand_daily` comes from the data stage (`python -m uncertainty_aware_planner.data daily ...`, see `src/uncertainty_aware_planner/data/README.md`). Outputs: `data/processed/forecast/<run_id>/*.parquet` (CONTRACTS.md 2.5) and `data/runs/forecasting/<run_id>/<fold>/` (checkpoint, logs, all quantiles, `run_log.json`).

```bash
CFG=configs/forecasting_models/temporal_fusion_transformer.yaml
DEMAND=data/processed/demand_daily/<imputation-run-id>.parquet
FORECAST="python -m uncertainty_aware_planner.pipeline.forecast --config $CFG --demand-daily $DEMAND"

# 1. smoke test: tiny model, 2 batches, on CPU (--smoke forces CPU); checks data loading and outputs
sbatch --job-name=tft_smoke --mem=64G --time=00:30:00 slurm/cpu_job.sbatch $FORECAST --seed 2025 --fold 1 --smoke

# 2. one fold, full settings: check runtime and memory with seff before launching the grid
sbatch --job-name=tft_f1 slurm/gpu_job.sbatch $FORECAST --seed 2025 --fold 1

# 3. final model: forecasts the calibration and test weeks from one checkpoint (calib alone: --eval-parts calib)
sbatch --job-name=tft_final slurm/gpu_job.sbatch $FORECAST --seed 2025

# 4. full grid: 3 seeds x (6 folds + final) = tasks 0-20, one seed per submit (QOS limit, see section 2)
sbatch --job-name=tft_grid --array=0-6%3 slurm/array_job.sbatch "2025 2026 2027" "1 2 3 4 5 6 final" $FORECAST
# when those finish: the same command with --array=7-13%3, then --array=14-20%3
```

On a laptop, run the same command with `--smoke` (CPU); the TFT config's `smoke` block sets the tiny model.

### Paper check (no imputation yet)

The authors' censored TFT scores 31.75% WAPE on the test week. To reproduce it before an imputation run exists, build a `demand_daily` that keeps the recorded sales in stockout hours, train the final model, and score the median the way the paper does:

```bash
sbatch --job-name=censored --mem=64G slurm/cpu_job.sbatch bash -c "python -m uncertainty_aware_planner.pipeline.censored && \
    python -m uncertainty_aware_planner.data daily --imputed data/processed/imputed_hourly/imp-none-censored.parquet --run-id imp-none-censored"
sbatch --job-name=tft_paper slurm/gpu_job.sbatch python -m uncertainty_aware_planner.pipeline.forecast --config $CFG \
    --demand-daily data/processed/demand_daily/imp-none-censored.parquet --seed 2025 --eval-parts calib test
python experiments/vishnu/paper_wape.py --demand-daily data/processed/demand_daily/imp-none-censored.parquet \
    --quantiles data/runs/forecasting/fc-tft-quantile-2025/final/quantiles_test.parquet
```

Expect a WAPE near the paper's, not equal to it: we train on 03-28..06-18 (the calibration week is held out) where the authors train on all 90 days, and our quantiles are 0.025/0.5/0.975 instead of 0.1..0.9.
