#!/bin/bash
# One experiment end to end on a CPU node: data, imputation, demand_daily, forecasts and metrics
# (src/uncertainty_aware_planner/pipeline/run.py). Arguments after the experiment file pass through.
#   sbatch slurm/run_experiment.sbatch configs/experiments/dlinear_baseline_censored.yaml [--force] [--metrics-only]
#   sbatch slurm/run_experiment.sbatch configs/experiments/tft_censored.yaml --stages evaluate
# Override resources on the command line, e.g. sbatch --mem=64G --time=12:00:00 slurm/run_experiment.sbatch ...
#SBATCH --job-name=uap_experiment
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/job_logs/%x.%j.out
#SBATCH --error=slurm/job_logs/%x.%j.out

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch slurm/run_experiment.sbatch configs/experiments/<name>.yaml [pipeline options...]" >&2
    exit 2
fi

source slurm/env.sh
echo "cmd:    python -m uncertainty_aware_planner.pipeline.run --experiment $*"
srun python -m uncertainty_aware_planner.pipeline.run --experiment "$@"
