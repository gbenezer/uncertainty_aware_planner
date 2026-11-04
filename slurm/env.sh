#!/bin/bash
# Shared job environment. Every sbatch script in slurm/ sources this file; source it yourself in an
# interactive session (srun --pty ... bash) to get the same setup.
#
# Large caches go to /scratch, not $HOME (home quota is small). Override UAP_SCRATCH to move them.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
cd "$REPO_ROOT"

export UAP_SCRATCH="${UAP_SCRATCH:-/scratch/$USER/uap}"
export HF_HOME="$UAP_SCRATCH/hf_home"
export HF_DATASETS_CACHE="$UAP_SCRATCH/hf_datasets"
export UV_CACHE_DIR="$UAP_SCRATCH/uv_cache"
export PATH="$HOME/.local/bin:$PATH"  # default uv install location
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1  # print() shows up in the job log immediately
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR"

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "No .venv in $REPO_ROOT; run: sbatch slurm/setup_env.sbatch" >&2
fi

echo "job:    ${SLURM_JOB_NAME:-interactive} ${SLURM_JOB_ID:-} ${SLURM_ARRAY_TASK_ID:+task $SLURM_ARRAY_TASK_ID}"
echo "host:   $(hostname)"
echo "repo:   $REPO_ROOT @ $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"
echo "start:  $(date -Is)"
if command -v nvidia-smi >/dev/null && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
if command -v python >/dev/null; then
    python -c "import sys, torch; print(f'python: {sys.version.split()[0]}  torch: {torch.__version__}  cuda: {torch.cuda.is_available()}')" || true
fi
