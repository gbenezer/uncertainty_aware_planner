import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uncertainty_aware_planner.data.config import load_config, load_yaml
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import (
    to_imputed_hourly,
    to_imputed_hourly_interval,
    to_imputed_hourly_interval_median,
    get_series_hourly,
    add_series_metadata,
    filter_by_mean_demand,
)
from uncertainty_aware_planner.utils.util_functions import set_seed, get_git_state
from uncertainty_aware_planner.models.imputation import (
    DLinear_Imputation_Quantile,
    DLinear_Imputation_Baseline,
    TimesNet_Imputation_Quantile,
)
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_errors_df,
    scaled_absolute_errors,
    summarize_scaled_errors,
)
from uncertainty_aware_planner.evaluation.uncertainty_metrics import (
    prediction_interval_coverage_fraction,
    mean_prediction_interval_width,
)

# ---------------------------------------------------------------- constants
CONFIG_DIR = Path("configs/imputation_models")
CONFIG_FILES = [
    "dlinear_imputation_quantile.yaml",
    "dlinear_imputation_quantile_median.yaml",
    "timesnet_imputation_quantile_median.yaml",
    "dlinear_imputation_baseline.yaml",
]
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")

DATA_DIR = Path("experiments/gil/data")
MODEL_DIRECTORY = Path("experiments/gil/models")
RESULTS_FILE = Path(
    "results/imputation/"
    "full_train_data_calbration_data_eval_imputation_no_conformalization.csv"
)
REPO_DIR = Path(__file__).resolve().parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [86400, 4, 42]
MASE_LAGS = (1, 24, 168)
VOLUME_GROUPS = ("low", "high")

# inference latency measurement
LATENCY_WARMUP_RUNS = 1  # untimed; its output is the one used for evaluation
LATENCY_TIMED_RUNS = 5


# ------------------------------------------------- model / conversion lookup
def _baseline_to_df(calib_set, calib_index, output):
    df = to_imputed_hourly(calib_set, calib_index, output[..., 0])
    return df.assign(q025=np.nan, q975=np.nan)


# (model name, num quantiles) -> (model class, checkpoint subdirectory suffix)
MODELS = {
    ("DLinear_Imputation_Baseline", None): (
        DLinear_Imputation_Baseline,
        "dlinear_baseline",
    ),
    ("DLinear_Imputation_Quantile", 2): (
        DLinear_Imputation_Quantile,
        "dlinear_quantile_interval_only",
    ),
    ("DLinear_Imputation_Quantile", 3): (
        DLinear_Imputation_Quantile,
        "dlinear_quantile",
    ),
    ("TimesNet_Imputation_Quantile", 3): (TimesNet_Imputation_Quantile, "timesnet"),
}

# num quantiles -> function turning raw model output into an hourly dataframe
TO_DATAFRAME = {
    None: _baseline_to_df,
    2: to_imputed_hourly_interval,
    3: to_imputed_hourly_interval_median,
}


# ------------------------------------------------------------------ latency
def _sync():
    """Wait for queued GPU work so wall-clock timings are accurate."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def impute_with_latency(model, dataset):
    """Run imputation, returning (output, flat dict of latency metrics).

    Warmup runs absorb one-time costs (CUDA init, kernel autotuning, caches).
    The output of the first warmup run is returned for evaluation, so the
    timed repeats never affect the metrics.
    """
    output = None
    for _ in range(LATENCY_WARMUP_RUNS):
        out = model.impute(dataset)
        output = out if output is None else output

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(LATENCY_TIMED_RUNS):
        _sync()
        start = time.perf_counter()
        model.impute(dataset)
        _sync()
        times.append(time.perf_counter() - start)

    times = np.asarray(times)
    n_samples = dataset["X"].shape[0]  # the PyPOTS set is a dict, not a sequence
    latency = {
        "latency_n_samples": n_samples,
        "latency_n_timed_runs": LATENCY_TIMED_RUNS,
        "latency_total_s_mean": times.mean(),
        "latency_total_s_std": times.std(ddof=1) if len(times) > 1 else np.nan,
        "latency_total_s_min": times.min(),
        "latency_total_s_median": np.median(times),
        "latency_ms_per_sample_mean": 1000 * times.mean() / n_samples,
        "latency_samples_per_s": n_samples / times.mean(),
        "latency_peak_gpu_mem_mb": (
            torch.cuda.max_memory_allocated() / 2**20
            if DEVICE.type == "cuda"
            else np.nan
        ),
    }
    return output, latency


# ---------------------------------------------------------------- evaluation
def _flat(prefix, metrics):
    """{'value': v, 'n_x': n} -> {prefix: v, 'prefix_n_x': n}."""
    return {
        (prefix if k == "value" else f"{prefix}_{k}"): v for k, v in metrics.items()
    }


def evaluate_subset(df, errors):
    """All metrics for one slice of the imputation dataframe, as a flat dict.

    `errors` is the per-row output of `scaled_absolute_errors` for the full
    frame; the slice is picked out by index, so every slice shares one MASE
    scale per series.

    The baseline needs no special case: its q025/q975 columns are NaN, so the
    interval metrics return NaN with a count of 0. This relies on every metric
    returning the same keys when the slice is empty (see the library patch).
    """
    out = dict(calculate_imputation_errors_df(imputed_hourly=df))  # wape, wpe, n_pe

    sub_errors = errors.loc[df.index]
    for lag in MASE_LAGS:
        out.update(_flat(f"mase_{lag}hr", summarize_scaled_errors(sub_errors, lag)))

    out.update(
        _flat("picp", prediction_interval_coverage_fraction(actual_and_interval_df=df))
    )
    out.update(_flat("mpiw", mean_prediction_interval_width(interval_df=df)))
    return out


def make_subsets(df, series_hourly):
    """overall / weekday / weekend, each further split by low / high volume."""
    is_weekday = df["weekday"] & ~df["holiday_flag"]
    day_masks = {
        "overall": np.ones(len(df), bool),
        "weekday": is_weekday,
        "weekend": ~is_weekday,
    }

    # per-series mean demand does not depend on the day slice, and each call
    # groups the whole hourly table, so do it once per group, not once per slice
    # (df has a fresh RangeIndex after add_series_metadata, so index membership
    # is a safe row mask)
    volume_masks = {
        group: df.index.isin(
            filter_by_mean_demand(
                imputed_df=df,
                series_hourly=series_hourly,
                group=group,
                mean_source="full",
            ).index
        )
        for group in VOLUME_GROUPS
    }

    subsets = {}
    for day_name, day_mask in day_masks.items():
        subsets[day_name] = df[day_mask]
        for group, vol_mask in volume_masks.items():
            subsets[f"{day_name}_{group}_volume"] = df[day_mask & vol_mask]
    return subsets


# ------------------------------------------------------------------- helpers
def run_key(seed, model_type, num_quantiles):
    """Normalise so NaN (read from CSV) and None (baseline) compare equal."""
    n = None if pd.isna(num_quantiles) else int(num_quantiles)
    return seed, model_type, n


def load_completed_runs():
    if not RESULTS_FILE.exists():
        return set()
    done = pd.read_csv(
        RESULTS_FILE, usecols=["seed", "model_type", "num_quantiles"]
    ).drop_duplicates()
    return {run_key(*row) for row in done.itertuples(index=False)}


def load_series_hourly():
    cache = DATA_DIR / "series_hourly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = get_series_hourly()
    df.to_parquet(cache)
    return df


# ---------------------------------------------------------------------- main
def main():
    series_hourly = load_series_hourly()
    completed = load_completed_runs()
    configs = [load_yaml(CONFIG_DIR / f) for f in CONFIG_FILES]

    cfg = load_config()
    cube = load_cube(cfg)
    masks = pd.read_parquet("data/processed/eval_masks.parquet")

    # independent of seed and model, so built once
    calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)

    experiment_dict = {
        "repo_dir_commit": get_git_state(REPO_DIR),
        "public_data_config_file": PUBLIC_DATA_CONFIG_FILE,
        "split_config_file": SPLIT_CONFIG_FILE,
        "device": DEVICE,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for config in configs:
            model_type = config["name"]
            quantiles = config.get("quantiles")
            num_quantiles = len(quantiles) if quantiles is not None else None

            if run_key(seed, model_type, num_quantiles) in completed:
                print(f"Skipping seed={seed}, {model_type}, q={num_quantiles} (done)")
                continue

            set_seed(seed)

            # load + impute
            model_cls, subdir = MODELS[(model_type, num_quantiles)]
            model = model_cls(config)
            model.load(MODEL_DIRECTORY / f"seed_{seed}_{subdir}")
            output, latency = impute_with_latency(model, calib_set)

            # convert to dataframe and attach metadata
            imputation_df = TO_DATAFRAME[num_quantiles](calib_set, calib_index, output)
            imputation_df = add_series_metadata(imputation_df, series_hourly)

            # evaluate every subset
            row = {
                **experiment_dict,
                "execution_datetime": datetime.datetime.now(),
                "seed": seed,
                "eval_data": "calibration",
                "model_type": model_type,
                "quantiles": quantiles,
                "num_quantiles": num_quantiles,
                **latency,
            }
            # scaled errors once per run (one pass over the hourly table per
            # lag), then every subset just slices them
            errors = scaled_absolute_errors(imputation_df, MASE_LAGS, series_hourly)
            for subset_name, subset_df in make_subsets(
                imputation_df, series_hourly
            ).items():
                metrics = evaluate_subset(subset_df, errors)
                row.update({f"{subset_name}_{k}": v for k, v in metrics.items()})

            # one append per run, so a run is either fully recorded or not at all
            pd.DataFrame([row]).to_csv(
                RESULTS_FILE,
                mode="a",
                header=not RESULTS_FILE.exists(),
                index=False,
            )


if __name__ == "__main__":
    main()
