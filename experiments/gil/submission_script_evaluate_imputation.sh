#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=evaluate_imputation
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --output=experiments/gil/job_logs/evaluate_imputation.%A_%a.out
#SBATCH --error=experiments/gil/job_logs/evaluate_imputation.%A_%a.err

# Train and evaluate the imputation models on the cross-validation folds.
#
# Submit from the project root, after `mkdir -p experiments/gil/job_logs`:
#     sbatch experiments/gil/submission_script_evaluate_imputation.sh
#
# The gpu partition allows 4 running and 4 pending jobs, so the work is split
# across 4 workers (the 4 array tasks) instead of one task per fold. Worker k
# takes every 4th unit (`--shard k/4`) of the 90 units (3 seeds x 6 folds x 5
# models), so workers never touch the same unit.
#
# A worker's share is far more than one 8 h job (about 9 TimesNet units), so
# the Python script checkpoints and stops before the time limit, exiting with
# code 75 if work remains. This script then resubmits the same worker, which
# resumes from the checkpoints, until the script exits 0. At any moment each
# worker has one running job, plus one pending job while it resubmits, which
# keeps the total within 4 running + 4 pending.
#
# Before training, every worker needs data/hf_cache/series_hourly.parquet. If it
# is missing, the first worker to get there builds it and the others wait for it
# (see build_series_hourly below). data/processed/eval_masks.parquet is never
# built here: if it is missing, the worker exits with code 2 and does not resubmit.

NUM_WORKERS=4    # must equal the size of --array above

# keep in sync with --time above; the Python script stops MARGIN_MINUTES early
TIME_LIMIT_HOURS=8
MARGIN_MINUTES=25

SUBMIT_SCRIPT=experiments/gil/submission_script_evaluate_imputation.sh
MAX_ATTEMPTS=25  # stops a worker that keeps asking for more time without progress
ATTEMPT=${ATTEMPT:-1}

# series_hourly build coordination
SERIES_CACHE_DIR=data/hf_cache   # keep equal to DATA_DIR in evaluate_imputation.py
SERIES_FILE="${SERIES_CACHE_DIR}/series_hourly.parquet"
LOCK_DIR="${SERIES_CACHE_DIR}/.series_hourly.build.lock"
EVAL_MASKS=data/processed/eval_masks.parquet
WAIT_MAX_MINUTES=${WAIT_MAX_MINUTES:-90}   # how long a waiting worker gives the builder
POLL_SECONDS=${POLL_SECONDS:-30}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-30}
STALE_MINUTES=${STALE_MINUTES:-5}          # a lock untouched this long has no live builder

echo "worker ${SLURM_ARRAY_TASK_ID}/${NUM_WORKERS} (attempt ${ATTEMPT}/${MAX_ATTEMPTS})"

source .venv/bin/activate || exit 2

# The masks come from an earlier pipeline step: fail now, before building or waiting.
if [ ! -f "${EVAL_MASKS}" ]; then
    echo "worker ${SLURM_ARRAY_TASK_ID}: missing ${EVAL_MASKS} (submit from the project root, after the masks are built)"
    exit 2
fi

# Make sure series_hourly.parquet exists. The first worker to create the lock
# directory (mkdir is atomic) builds the table, touching a heartbeat file while it
# works; the others poll. If the builder dies without releasing the lock (node
# failure, OOM kill), its heartbeat stops and a waiter takes the lock over. In the
# rare case two workers take over together, both build, which is safe because the
# table is written to a temporary file and renamed into place.
# load_series_hourly writes the file atomically, so its existence means it is complete.
build_series_hourly() {
    mkdir -p "${SERIES_CACHE_DIR}"
    local waited=0 build_pid build_status
    while [ ! -f "${SERIES_FILE}" ]; do
        if mkdir "${LOCK_DIR}" 2>/dev/null; then
            trap 'rm -rf "${LOCK_DIR}"' EXIT
            echo "${SLURM_JOB_ID}" > "${LOCK_DIR}/owner"
            touch "${LOCK_DIR}/heartbeat"
            echo "worker ${SLURM_ARRAY_TASK_ID}: ${SERIES_FILE} not found, building it"
            python -u -c "
from uncertainty_aware_planner.evaluation import imputation_evaluation_helpers as ie
ie.load_series_hourly('${SERIES_CACHE_DIR}')
" &
            build_pid=$!
            while kill -0 "${build_pid}" 2>/dev/null; do
                touch "${LOCK_DIR}/heartbeat"
                sleep "${HEARTBEAT_SECONDS}"
            done
            wait "${build_pid}"
            build_status=$?
            rm -rf "${LOCK_DIR}"
            trap - EXIT
            if [ "${build_status}" -ne 0 ] || [ ! -f "${SERIES_FILE}" ]; then
                echo "worker ${SLURM_ARRAY_TASK_ID}: building ${SERIES_FILE} failed (status ${build_status})"
                exit 2
            fi
            echo "worker ${SLURM_ARRAY_TASK_ID}: built ${SERIES_FILE}"
            return 0
        fi

        # another worker holds the lock: wait, unless nothing there is alive
        if [ -d "${LOCK_DIR}" ] && [ -z "$(find "${LOCK_DIR}" -mmin "-${STALE_MINUTES}" 2>/dev/null)" ]; then
            echo "worker ${SLURM_ARRAY_TASK_ID}: build lock untouched for ${STALE_MINUTES} min, taking it over"
            rm -rf "${LOCK_DIR}"
            continue
        fi
        if [ "${waited}" -ge $(( WAIT_MAX_MINUTES * 60 )) ]; then
            echo "worker ${SLURM_ARRAY_TASK_ID}: ${SERIES_FILE} still missing after ${WAIT_MAX_MINUTES} min"
            exit 2
        fi
        echo "worker ${SLURM_ARRAY_TASK_ID}: waiting for another worker to build ${SERIES_FILE}"
        sleep "${POLL_SECONDS}"
        waited=$(( waited + POLL_SECONDS ))
    done
}
build_series_hourly

# Waiting and building count against this job's wall clock, but the Python script
# measures its limit from its own start, so give it what is actually left.
# ($SECONDS is the time since this script started.)
REMAINING_HOURS=$(awk -v used="${SECONDS}" -v limit="${TIME_LIMIT_HOURS}" \
    'BEGIN { printf "%.4f", limit - used / 3600 }')

python -u src/uncertainty_aware_planner/pipeline/evaluate_imputation.py \
    --shard "${SLURM_ARRAY_TASK_ID}/${NUM_WORKERS}" \
    --time-limit-hours "${REMAINING_HOURS}" --margin-minutes "${MARGIN_MINUTES}"
status=$?

# exit code 75 means "ran out of time, work remains": resubmit this worker
if [ "${status}" -eq 75 ]; then
    if [ "${ATTEMPT}" -ge "${MAX_ATTEMPTS}" ]; then
        echo "worker ${SLURM_ARRAY_TASK_ID}: still unfinished after ${ATTEMPT} attempts, not resubmitting"
        exit 1
    fi
    echo "worker ${SLURM_ARRAY_TASK_ID}: work remains, resubmitting"
    # retry for a while in case the pending-job limit is briefly full
    for try in 1 2 3 4 5 6 7 8 9 10; do
        if sbatch --array="${SLURM_ARRAY_TASK_ID}" \
                  --export=ALL,ATTEMPT=$(( ATTEMPT + 1 )) "${SUBMIT_SCRIPT}"; then
            exit 0
        fi
        echo "worker ${SLURM_ARRAY_TASK_ID}: sbatch failed (try ${try}/10), retrying"
        sleep "${RETRY_SLEEP:-60}"
    done
    echo "worker ${SLURM_ARRAY_TASK_ID}: resubmission failed"
    exit 75
fi

exit "${status}"