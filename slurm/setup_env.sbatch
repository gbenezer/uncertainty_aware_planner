#!/bin/bash
# One-time environment setup on a GPU node, so the CUDA torch build is checked on the hardware it runs on.
#   sbatch slurm/setup_env.sbatch
# Re-run after pyproject.toml or uv.lock changes.
#SBATCH --job-name=uap_setup_env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/job_logs/%x.%j.out
#SBATCH --error=slurm/job_logs/%x.%j.out

source slurm/env.sh

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --frozen --all-groups
source .venv/bin/activate

python - <<'EOF'
import lightning
import pytorch_forecasting
import torch

print("torch", torch.__version__, "| lightning", lightning.__version__,
      "| pytorch-forecasting", pytorch_forecasting.__version__)
assert torch.cuda.is_available(), "CUDA is not available on this node"
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"GPU: {name}, compute capability {capability}")
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
EOF
