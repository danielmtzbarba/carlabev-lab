#!/bin/bash
#SBATCH --job-name=wm_bench
#SBATCH --output=/data/horse/ws/dama898h-carlabev/results/logs/world_model_benchmark_%j.out
#SBATCH --error=/data/horse/ws/dama898h-carlabev/results/logs/world_model_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -L horse

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

source "${SLURM_SUBMIT_DIR}/infra/slurm/common_storage.sh"

RUN_NAME="${RUN_NAME:-lewm-bench}"
DATASET_PATH="${DATASET_PATH:-datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_1/train/seed_2}"
BATCH_SIZES=(${BATCH_SIZES:-16 32 64})
CHUNK_LENGTHS=(${CHUNK_LENGTHS:-8 16})
WARMUP_BATCHES="${WARMUP_BATCHES:-2}"
MEASURE_BATCHES="${MEASURE_BATCHES:-10}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGE_TO_TMP="${STAGE_TO_TMP:-true}"
SYNC_RESULTS_BACK="${SYNC_RESULTS_BACK:-true}"

REPO_ROOT="${CARLABEV_REPO_ROOT}"
JOB_TMP_ROOT="${TMPDIR:-/tmp}"
LOCAL_JOB_ROOT="${JOB_TMP_ROOT%/}/${SLURM_JOB_ID:-manual}_${RUN_NAME}"
LOCAL_DATASET_ROOT="${LOCAL_JOB_ROOT}/dataset"
FINAL_RUN_DIR="${CARLABEV_RUNS_ROOT}/world_model/${RUN_NAME}"

echo "World-model benchmark (horse workspace-aware)"
echo "  host=$(hostname)"
echo "  repo_root=${REPO_ROOT}"
echo "  artifact_root=${CARLABEV_ARTIFACT_ROOT}"
echo "  dataset_path=${DATASET_PATH}"
echo "  run_name=${RUN_NAME}"
echo "  batch_sizes=${BATCH_SIZES[*]}"
echo "  chunk_lengths=${CHUNK_LENGTHS[*]}"
echo "  warmup_batches=${WARMUP_BATCHES}"
echo "  measure_batches=${MEASURE_BATCHES}"
echo "  device=${DEVICE}"
echo "  num_workers=${NUM_WORKERS}"
echo "  stage_to_tmp=${STAGE_TO_TMP}"
echo "  sync_results_back=${SYNC_RESULTS_BACK}"
echo "  local_job_root=${LOCAL_JOB_ROOT}"

if [[ "${STAGE_TO_TMP}" == "true" ]]; then
    echo "Staging dataset to node-local storage..."
    mkdir -p "${LOCAL_DATASET_ROOT}"
    rsync -a --delete "${DATASET_PATH%/}/" "${LOCAL_DATASET_ROOT}/"
    BENCHMARK_DATASET_PATH="${LOCAL_DATASET_ROOT}"
else
    BENCHMARK_DATASET_PATH="${DATASET_PATH}"
fi

echo "Launching benchmark..."
set -x
uv run drl world-model benchmark \
    --run-name "${RUN_NAME}" \
    --data.dataset-paths "${BENCHMARK_DATASET_PATH}" \
    --data.num-workers "${NUM_WORKERS}" \
    --batch-sizes "${BATCH_SIZES[@]}" \
    --chunk-lengths "${CHUNK_LENGTHS[@]}" \
    --warmup-batches "${WARMUP_BATCHES}" \
    --measure-batches "${MEASURE_BATCHES}" \
    --training.device "${DEVICE}"
set +x

if [[ "${SYNC_RESULTS_BACK}" == "true" ]]; then
    echo "Results stay on horse under ${FINAL_RUN_DIR}"
    echo "Benchmark log: ${FINAL_RUN_DIR}/benchmark.log"
    echo "Slurm logs: ${CARLABEV_RESULTS_ROOT}/logs"
fi

echo "Done."
