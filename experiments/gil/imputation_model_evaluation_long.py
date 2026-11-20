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
RESULTS_DIR = Path("results/imputation")
RESULTS_STEM = "full_train_data_calbration_data_eval_imputation_no_conformalization"
RUNS_FILE = RESULTS_DIR / f"{RESULTS_STEM}_runs.csv"  # one row per run
METRICS_FILE = (
    RESULTS_DIR / f"{RESULTS_STEM}_metrics.csv"
)  # long: run x subset x metric
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
def evaluate_subset(df, errors):
    """Long-format metrics for one slice: columns metric, value, n, n_dropped.

    `n` is the number of rows the metric was computed on; `n_dropped` is only
    set for MASE (rows without a valid scale).

    `errors` is the per-row output of `scaled_absolute_errors` for the full
    frame; the slice is picked out by index, so every slice shares one MASE
    scale per series.

    The baseline needs no special case: its q025/q975 columns are NaN, so the
    interval metrics return NaN with n = 0.
    """
    rows = []

    pe = calculate_imputation_errors_df(imputed_hourly=df)
    rows += [
        ("wape", pe["wape"], pe["n_pe"], np.nan),
        ("wpe", pe["wpe"], pe["n_pe"], np.nan),
    ]

    sub_errors = errors.loc[df.index]
    for lag in MASE_LAGS:
        m = summarize_scaled_errors(sub_errors, lag)
        rows.append(
            (f"mase_{lag}hr", m["value"], m["n_mase"], m["n_mase_dropped_no_scale"])
        )

    picp = prediction_interval_coverage_fraction(actual_and_interval_df=df)
    rows.append(("picp", picp["value"], picp["n_picp"], np.nan))
    mpiw = mean_prediction_interval_width(interval_df=df)
    rows.append(("mpiw", mpiw["value"], mpiw["n_mpiw"], np.nan))

    return pd.DataFrame(rows, columns=["metric", "value", "n", "n_dropped"])


def make_subsets(df, series_hourly):
    """{(day_type, volume): slice}: overall / weekday / weekend, each for
    all / low / high volume series."""
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
        subsets[(day_name, "all")] = df[day_mask]
        for group, vol_mask in volume_masks.items():
            subsets[(day_name, group)] = df[day_mask & vol_mask]
    return subsets


# ------------------------------------------------------------------- helpers
def make_run_id(seed, model_type, num_quantiles):
    q = "none" if num_quantiles is None else num_quantiles
    return f"{model_type}__q{q}__seed{seed}"


def load_completed_runs():
    """run_ids in the runs table.

    A run's runs row is written after its metric rows, so metric rows without
    one come from an interrupted run; they are dropped so the run can be redone.
    """
    done = set()
    if RUNS_FILE.exists():
        done = set(pd.read_csv(RUNS_FILE, usecols=["run_id"])["run_id"])
    if METRICS_FILE.exists():
        metrics = pd.read_csv(METRICS_FILE)
        keep = metrics["run_id"].isin(done)
        if not keep.all():
            metrics[keep].to_csv(METRICS_FILE, index=False)
    return done


def append_csv(df, path):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for config in configs:
            model_type = config["name"]
            quantiles = config.get("quantiles")
            num_quantiles = len(quantiles) if quantiles is not None else None

            run_id = make_run_id(seed, model_type, num_quantiles)
            if run_id in completed:
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

            # scaled errors once per run (one pass over the hourly table per
            # lag), then every subset just slices them
            errors = scaled_absolute_errors(imputation_df, MASE_LAGS, series_hourly)
            frames = [
                evaluate_subset(subset_df, errors).assign(
                    day_type=day_type, volume=volume
                )
                for (day_type, volume), subset_df in make_subsets(
                    imputation_df, series_hourly
                ).items()
            ]
            metrics = pd.concat(frames, ignore_index=True)[
                ["day_type", "volume", "metric", "value", "n", "n_dropped"]
            ]
            metrics.insert(0, "run_id", run_id)

            run_row = {
                "run_id": run_id,
                **experiment_dict,
                "execution_datetime": datetime.datetime.now(),
                "seed": seed,
                "eval_data": "calibration",
                "model_type": model_type,
                "quantiles": quantiles,
                "num_quantiles": num_quantiles,
                **latency,
            }

            # metrics first, runs row last: a run counts as done only once its
            # runs row exists (see load_completed_runs)
            append_csv(metrics, METRICS_FILE)
            append_csv(pd.DataFrame([run_row]), RUNS_FILE)


if __name__ == "__main__":
    main()
