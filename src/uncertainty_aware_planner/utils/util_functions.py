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

# helper function to raise appropriate KeyErrors if the required columns
# are not present
def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    """helper function to check if the input DataFrame has the required
    columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

# helper function to filter hours from the input dataframe 
# that are not from valid imputation days
def _valid_day_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows from days that have at least one masked hour and no true stockout."""
    # if the input dataframe has appropriate columns, then use them
    # otherwise fall back to assumptions
    masked = df["is_masked"] if "is_masked" in df.columns else df["actual_demand"].notna()

    # assumption is the df includes only stockout and masked hours, and those statuses
    # are mutually exclusive
    stockout = df["is_stockout"] if "is_stockout" in df.columns else ~masked
    
    # create a summary df that counts stockout hours and masked hours on a daily basis
    counts = (
        pd.DataFrame({**{k: df[k] for k in DAY_KEYS}, "masked": masked, "stockout": stockout})
        .groupby(DAY_KEYS)[["masked", "stockout"]]
        .sum()
    )
    # get the day indices for valid days
    valid_days = counts.index[(counts["masked"] > 0) & (counts["stockout"] == 0)]

    # filter based on the valid multi-index data (store-product-days)
    keys = pd.MultiIndex.from_frame(df[DAY_KEYS])
    return df[keys.isin(valid_days)]