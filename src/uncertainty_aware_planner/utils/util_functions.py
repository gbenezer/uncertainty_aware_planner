# derived from https://github.com/Dingdong-Inc/frn-50k-baseline/blob/main/latent_demand_recovery/exp/app.py
# retrieved 30Oct2026

import random
import numpy as np
import torch
from pathlib import Path
import subprocess

def set_seed(seed_value=1024):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    return

def get_git_state(repo_dir: Path) -> str:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return sha + ("-dirty" if status else "")