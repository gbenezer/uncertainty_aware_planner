#!/bin/bash
# Generic single-GPU job: runs the command given after the script name.
#   sbatch slurm/gpu_job.sbatch python -m uncertainty_aware_planner.pipeline.forecast --config ... --seed 2025
#
# Requeue: 90 s before the time limit SLURM sends SIGUSR1. Lightning (detected through srun) saves
# hpc_ckpt_*.ckpt and requeues the job; the requeued job resumes from it. --open-mode=append keeps both
# attempts in one log.
#SBATCH --job-name=uap_gpu
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
#SBATCH --output=slurm/job_logs/%x.%j.out
#SBATCH --error=slurm/job_logs/%x.%j.out

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch slurm/gpu_job.sbatch <command> [args...]" >&2
    exit 2
fi

source slurm/env.sh
echo "cmd:    $*"
echo "restarts: ${SLURM_RESTART_COUNT:-0}"
srun "$@"  # srun (not plain python) so Lightning detects SLURM and handles SIGUSR1
