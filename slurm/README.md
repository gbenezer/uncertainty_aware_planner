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

The setup log must show `cuda True` and a GPU name. torch is pinned to 2.8 (cu128) because torch ≥ 2.9 drops the V100 (compute capability 7.0). If a partition or limit in `sinfo` differs from the `#SBATCH` lines, override at submit time (`sbatch --partition=... --time=...`) rather than editing the scripts for one run.

Caches (`HF_HOME`, `HF_DATASETS_CACHE`, `UV_CACHE_DIR`) live under `/scratch/$USER/uap`; set `UAP_SCRATCH` to move them. The dataset itself stays in `data/` as configured in `configs/data.yaml`.

## 2. Submitting

```bash
# data steps (CPU)
sbatch slurm/cpu_job.sbatch python -m uncertainty_aware_planner.data load
sbatch --mem=64G slurm/cpu_job.sbatch python -m uncertainty_aware_planner.data splits

# any GPU command
sbatch slurm/gpu_job.sbatch python path/to/script.py --arg value

# 3 seeds x (6 folds + final) = 21 tasks, at most 4 at a time
sbatch --array=0-20%4 slurm/array_job.sbatch "2025 2026 2027" "1 2 3 4 5 6 final" \
    python -m <module> <args...>
```

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
