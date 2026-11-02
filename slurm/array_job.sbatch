#!/bin/bash
# Seed x fold grid as one job array. Each task runs the command with "--seed S" and, unless the fold is
# "final", "--fold F" appended. Seeds and folds are arguments, never defaults in this file.
#
#   sbatch --array=0-6%3 slurm/array_job.sbatch "2025 2026 2027" "1 2 3 4 5 6 final" \
#       python -m uncertainty_aware_planner.pipeline.forecast --config configs/forecasting_models/temporal_fusion_transformer.yaml \
#       --demand-daily data/processed/demand_daily/<run-id>.parquet
#
# Task i runs seed SEEDS[i / n_folds] and fold FOLDS[i % n_folds]; size --array as 0..(n_seeds*n_folds - 1).
# "%3" caps how many tasks run at once. Explorer allows 8 submitted GPU jobs per user (array tasks count),
# so submit a big grid in chunks (--array=0-6, 7-13, 14-20). Use --array=7 (for example) to rerun one task.
#SBATCH --job-name=uap_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/job_logs/%x.%A_%a.out
#SBATCH --error=slurm/job_logs/%x.%A_%a.out

if [[ $# -lt 3 ]]; then
    echo 'usage: sbatch --array=0-N slurm/array_job.sbatch "<seeds>" "<folds>" <command> [args...]' >&2
    exit 2
fi

read -r -a SEEDS <<< "$1"
read -r -a FOLDS <<< "$2"
shift 2

TASK="${SLURM_ARRAY_TASK_ID:?submit with --array}"
N_TASKS=$(( ${#SEEDS[@]} * ${#FOLDS[@]} ))
if (( TASK >= N_TASKS )); then
    echo "array task $TASK is out of range: ${#SEEDS[@]} seeds x ${#FOLDS[@]} folds = $N_TASKS tasks" >&2
    exit 2
fi
SEED="${SEEDS[TASK / ${#FOLDS[@]}]}"
FOLD="${FOLDS[TASK % ${#FOLDS[@]}]}"

ARGS=(--seed "$SEED")
if [[ "$FOLD" != "final" ]]; then
    ARGS+=(--fold "$FOLD")
fi

source slurm/env.sh
echo "seed:   $SEED  fold: $FOLD"
echo "cmd:    $* ${ARGS[*]}"
echo "restarts: ${SLURM_RESTART_COUNT:-0}"
srun "$@" "${ARGS[@]}"
