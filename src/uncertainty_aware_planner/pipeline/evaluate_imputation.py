"""Train and evaluate imputation models on cross-validation folds, within job time limits.

One *unit* is one (seed, fold, model config). For each unit the script trains on
the fold's training split, then evaluates on its validation split with the same
metrics, subsets and output tables as `pretrained_model_eval.py` (the shared
pipeline is in `imputation_evaluation_helpers`).

Job time limits
---------------
A unit can take longer than one job (TimesNet: 15+ epochs, ~6 h or more), so the
script is built to be stopped and restarted at any point:

* Training runs one epoch per `fit()` call. After each epoch it checks the time
  left and stops cleanly if another epoch (as long as the slowest so far) would
  not fit before the deadline.
* The model and a small state file are checkpointed together, atomically, at
  most every CHECKPOINT_MIN_INTERVAL_S seconds and always before stopping. A
  later job resumes from the last checkpoint.
* Evaluation only starts if there is time for it. Otherwise the trained model
  stays checkpointed and a later job just evaluates it, so training is never
  repeated because evaluation did not fit.
* Each finished unit writes its own result files, so jobs working on different
  units (a job array, say) never touch the same file. The combined tables are
  rebuilt from them at the end of every invocation.
* The exit code is 0 when every selected unit is done and 75 when work remains,
  so a submit script can resubmit until it sees 0.

Model API (the only coupling to the model classes). Written against the
wrappers in models/imputation: the constructor takes the config dict, whose
"epochs" is the total number of epochs; `fit(set)` fits every sub-model (one per
quantile); `save(dir)` / `load(dir)` write and read one .pypots file per
sub-model. It also relies on PyPOTS's `fit` continuing from the model's current
weights when called again, which is not visible in the wrappers.

Differences from one uninterrupted `fit`: because training is chunked into
one-epoch fits, early stopping (`patience`) and PyPOTS's restore of the best
epoch by training loss do not apply, so the final epoch is kept; and the AdamW
moments are not part of a checkpoint, so they restart whenever a unit resumes
in a new job.
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.config import load_config, load_yaml
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.utils.util_functions import set_seed
from uncertainty_aware_planner.evaluation import imputation_evaluation_helpers as ie

# ---------------------------------------------------------------- constants
CONFIG_DIR = Path("configs/imputation_models")
CONFIG_FILES = [
    "dlinear_imputation_quantile.yaml",
    "dlinear_imputation_quantile_median.yaml",
    "timesnet_imputation_quantile.yaml",
    "timesnet_imputation_quantile_median.yaml",
    "dlinear_imputation_baseline.yaml",
]
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")
DATA_DIR = Path("data/hf_cache")
REPO_DIR = Path(__file__).resolve().parent

SEEDS = [86400, 4, 42]
FOLDS = range(1, 7)

RESULTS_DIR = Path("results/imputation/cv_folds")
UNITS_DIR = RESULTS_DIR / "units"  # one directory of result files per finished unit
RUNS_FILE = RESULTS_DIR / "cv_folds_runs.csv"  # combined: one row per unit
METRICS_FILE = RESULTS_DIR / "cv_folds_metrics.csv"  # combined: long format
CHECKPOINT_DIR = Path("models/cv_folds")

CHECKPOINT_MIN_INTERVAL_S = 600
EXIT_MORE_WORK = 75  # EX_TEMPFAIL: work remains, run again

NEW_STATE = {
    "epochs_done": 0,
    "train_seconds": 0.0,  # summed over all sessions
    "epoch_seconds": 0.0,  # slowest epoch seen, 0 = unknown
    "sessions": 0,  # number of jobs that trained this unit
    "train_complete": False,
}


def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------- units
@dataclass
class Unit:
    seed: int
    fold: int
    config: dict

    @property
    def model_type(self):
        return self.config["name"]

    @property
    def num_quantiles(self):
        q = self.config.get("quantiles")
        return len(q) if q is not None else None

    @property
    def run_id(self):
        base = ie.make_run_id(self.seed, self.model_type, self.num_quantiles)
        return f"{base}__fold{self.fold}"

    @property
    def result_dir(self):
        return UNITS_DIR / self.run_id

    @property
    def ckpt_dir(self):
        return CHECKPOINT_DIR / self.run_id

    @property
    def is_done(self):
        # runs.csv is written last, so its existence means the unit is complete
        return (self.result_dir / "runs.csv").exists()


def build_model(unit, epochs=None):
    model_cls = ie.MODEL_CLASSES[unit.model_type]
    config = unit.config
    if epochs is not None:
        # patience=None: with one-epoch fits early stopping could never trigger
        # anyway, and it keeps `patience > epochs` from being an issue
        config = {**config, "epochs": epochs, "patience": None}
    return model_cls(config)


# ------------------------------------------------------------- file helpers
def write_csv_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _recover(ckpt_dir):
    """Undo a save that was interrupted between its two renames."""
    old = ckpt_dir.with_name(ckpt_dir.name + ".old")
    if not ckpt_dir.exists() and old.exists():
        old.rename(ckpt_dir)


def load_state(unit):
    _recover(unit.ckpt_dir)
    state_file = unit.ckpt_dir / "state.json"
    if state_file.exists():
        return {**NEW_STATE, **json.loads(state_file.read_text())}
    return dict(NEW_STATE)


def save_checkpoint(model, state, ckpt_dir):
    """Write model + state into a temp dir, then swap it in, so a kill at any
    moment leaves either the old or the new checkpoint intact."""
    tmp = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    old = ckpt_dir.with_name(ckpt_dir.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save(tmp / "model")
    (tmp / "state.json").write_text(json.dumps(state))
    if ckpt_dir.exists():
        shutil.rmtree(old, ignore_errors=True)
        ckpt_dir.rename(old)
    tmp.rename(ckpt_dir)
    shutil.rmtree(old, ignore_errors=True)


def collect():
    """Rebuild the combined tables from the finished units' result files."""
    done = sorted(p.parent for p in UNITS_DIR.glob("*/runs.csv"))
    if not done:
        return
    for name, out in (("runs.csv", RUNS_FILE), ("metrics.csv", METRICS_FILE)):
        write_csv_atomic(
            pd.concat([pd.read_csv(d / name) for d in done], ignore_index=True), out
        )
    log(f"Collected {len(done)} finished units into {RESULTS_DIR}")


# ------------------------------------------------------------------ training
def train(unit, state, env):
    """Train until done or out of time; returns (model, finished)."""
    total_epochs = unit.config["epochs"]

    set_seed(unit.seed)
    model = build_model(unit, epochs=1)  # each fit() call is one epoch
    if state["epochs_done"] > 0:
        model.load(unit.ckpt_dir / "model")
        log(f"{unit.run_id}: resumed at epoch {state['epochs_done']}/{total_epochs}")
    state["sessions"] += 1

    train_set = env.train_set(unit.fold)
    durations, dirty, last_save = [], False, time.time()

    while state["epochs_done"] < total_epochs:
        # leave room for the slowest epoch seen so far (10% slack)
        est = 1.1 * max(durations, default=state["epoch_seconds"])
        if time.time() + est > env.deadline:
            log(f"{unit.run_id}: out of time at epoch {state['epochs_done']}/{total_epochs}")
            break

        # reseed per epoch so a resumed run draws the same random numbers an
        # uninterrupted one would
        seed = np.random.SeedSequence([unit.seed, unit.fold, state["epochs_done"]])
        set_seed(int(seed.generate_state(1)[0]))

        start = time.time()
        model.fit(train_set)
        durations.append(time.time() - start)

        state["epochs_done"] += 1
        state["train_seconds"] += durations[-1]
        state["epoch_seconds"] = max(durations)
        state["train_complete"] = state["epochs_done"] >= total_epochs
        dirty = True

        if state["train_complete"] or time.time() - last_save >= CHECKPOINT_MIN_INTERVAL_S:
            save_checkpoint(model, state, unit.ckpt_dir)
            dirty, last_save = False, time.time()

    if dirty:
        save_checkpoint(model, state, unit.ckpt_dir)
    return model, state["train_complete"]


# ---------------------------------------------------------------- evaluation
def evaluate_and_write(unit, model, state, env):
    start = time.time()
    val_set, val_index = env.val_set(unit.fold)
    metrics, latency = ie.evaluate_run(
        model,
        val_set,
        val_index,
        unit.num_quantiles,
        env.series_hourly,
        latency_timed_runs=env.latency_runs,
    )
    metrics.insert(0, "run_id", unit.run_id)

    run_row = {
        "run_id": unit.run_id,
        **env.experiment,
        "execution_datetime": datetime.datetime.now(),
        "seed": unit.seed,
        "fold": unit.fold,
        "eval_data": "validation",
        "model_type": unit.model_type,
        "quantiles": unit.config.get("quantiles"),
        "num_quantiles": unit.num_quantiles,
        "train_epochs": state["epochs_done"],
        "train_seconds": state["train_seconds"],
        "train_sessions": state["sessions"],
        "eval_seconds": time.time() - start,
        **latency,
    }

    # metrics first, runs row last (see Unit.is_done)
    write_csv_atomic(metrics, unit.result_dir / "metrics.csv")
    write_csv_atomic(pd.DataFrame([run_row]), unit.result_dir / "runs.csv")


# --------------------------------------------------------------------- units
def run_unit(unit, env):
    """Advance one unit as far as the time budget allows; True if it finished."""
    if unit.is_done:
        return True

    state = load_state(unit)

    if not state["train_complete"]:
        estimate = (
            state["epoch_seconds"]
            or env.epoch_estimates.get(unit.model_type)
            or env.min_start_s
        )
        if env.deadline - time.time() < estimate:
            log(f"{unit.run_id}: not enough time left to start, skipping")
            return False

        model, finished = train(unit, state, env)
        if state["epoch_seconds"]:
            env.epoch_estimates[unit.model_type] = state["epoch_seconds"]
        if not finished:
            return False
    else:
        model = build_model(unit)
        model.load(unit.ckpt_dir / "model")

    if env.deadline - time.time() < env.eval_reserve_s:
        log(f"{unit.run_id}: trained, but not enough time left to evaluate")
        return False

    evaluate_and_write(unit, model, state, env)
    log(f"{unit.run_id}: done")
    return True


def run_all(units, env):
    """Process the units in priority order; returns the number left undone."""
    todo = [u for u in units if not u.is_done]
    # finish units that already have a checkpoint before starting new ones
    todo.sort(key=lambda u: not (u.ckpt_dir / "state.json").exists())

    remaining = 0
    for unit in todo:
        if not run_unit(unit, env):
            remaining += 1
    log(f"{len(units) - remaining}/{len(units)} selected units done")
    return remaining


# ---------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--folds", type=int, nargs="+", default=list(FOLDS))
    p.add_argument(
        "--models", nargs="+", default=None,
        help="config file stems to run (default: all), e.g. dlinear_imputation_baseline",
    )
    p.add_argument("--time-limit-hours", type=float, default=8.0,
                   help="the job's wall-clock cap, counted from script start")
    p.add_argument("--margin-minutes", type=float, default=25.0,
                   help="stop this long before the cap (checkpoint writes, slack)")
    p.add_argument("--eval-reserve-minutes", type=float, default=30.0,
                   help="only start evaluating a trained unit if this much time is left")
    p.add_argument("--min-start-minutes", type=float, default=45.0,
                   help="time needed to start a unit whose epoch time is not known yet")
    p.add_argument("--latency-runs", type=int, default=None,
                   help=f"timed inference repeats per unit (default: {ie.LATENCY_TIMED_RUNS})")
    return p.parse_args(argv)


def main(argv=None):
    started = time.time()
    args = parse_args(argv)
    configs = {Path(f).stem: load_yaml(CONFIG_DIR / f) for f in CONFIG_FILES}
    selected = args.models or list(configs)
    unknown = set(selected) - set(configs)
    if unknown:
        sys.exit(f"unknown --models {sorted(unknown)}; choose from {sorted(configs)}")

    units = [
        Unit(seed, fold, configs[name])
        for seed in args.seeds
        for fold in args.folds
        for name in selected
    ]

    # fail now, not after loading data and training the first few units
    for name in selected:
        quantiles = configs[name].get("quantiles")
        ie.check_supported(
            configs[name]["name"], len(quantiles) if quantiles is not None else None
        )
    ids = [u.run_id for u in units]
    clashes = sorted({i for i in ids if ids.count(i) > 1})
    if clashes:
        sys.exit(
            "two configs share a model name and number of quantiles, so their "
            f"runs would overwrite each other: {clashes[:2]}"
        )

    cfg = load_config()
    cube = load_cube(cfg)
    masks = pd.read_parquet("data/processed/eval_masks.parquet")

    # only the current fold's data is kept in memory
    @lru_cache(maxsize=1)
    def train_set(fold):
        return build_pypots_set(cube, cfg, "train", fold=fold)[0]

    @lru_cache(maxsize=1)
    def val_set(fold):
        return build_pypots_set(cube, cfg, "val", fold=fold, masks=masks)

    env = SimpleNamespace(
        series_hourly=ie.load_series_hourly(DATA_DIR),
        train_set=train_set,
        val_set=val_set,
        experiment=ie.experiment_metadata(
            REPO_DIR, PUBLIC_DATA_CONFIG_FILE, SPLIT_CONFIG_FILE
        ),
        latency_runs=args.latency_runs or ie.LATENCY_TIMED_RUNS,
        deadline=started + 3600 * args.time_limit_hours - 60 * args.margin_minutes,
        eval_reserve_s=60 * args.eval_reserve_minutes,
        min_start_s=60 * args.min_start_minutes,
        epoch_estimates={},  # model name -> slowest epoch seen in this process
    )

    remaining = run_all(units, env)
    collect()
    return EXIT_MORE_WORK if remaining else 0


if __name__ == "__main__":
    sys.exit(main())