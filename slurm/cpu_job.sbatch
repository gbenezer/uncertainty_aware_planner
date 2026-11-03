#!/bin/bash
# Generic CPU job: runs the command given after the script name.
#   sbatch slurm/cpu_job.sbatch python -m uncertainty_aware_planner.data load
# Override resources on the command line, e.g. sbatch --mem=64G --time=02:00:00 slurm/cpu_job.sbatch ...
#SBATCH --job-name=uap_cpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/job_logs/%x.%j.out
#SBATCH --error=slurm/job_logs/%x.%j.out

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch slurm/cpu_job.sbatch <command> [args...]" >&2
    exit 2
fi

source slurm/env.sh
echo "cmd:    $*"
srun "$@"
