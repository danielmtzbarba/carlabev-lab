#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

source "${SLURM_SUBMIT_DIR:-$(pwd)}/infra/slurm/common_storage.sh"

DATASET_PATH="${DATASET_PATH:-/data/horse/ws/dama898h-carlabev/datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_1/train/seed_2}"
DEST_NAME="${DEST_NAME:-carlabev-world-model/seed_2}"
RUN_PREFIX="${RUN_PREFIX:-wm-storage-ablation}"
BATCH_SIZES=(${BATCH_SIZES:-32})
CHUNK_LENGTHS=(${CHUNK_LENGTHS:-4})
NUM_WORKERS="${NUM_WORKERS:-0}"
PIN_MEMORY="${PIN_MEMORY:-True}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-False}"
PREFETCH_FACTORS=(${PREFETCH_FACTORS:-2})
WARMUP_BATCHES="${WARMUP_BATCHES:-1}"
MEASURE_BATCHES="${MEASURE_BATCHES:-3}"
DEVICE="${DEVICE:-cuda}"

TMP_ROOT="${TMPDIR:-/tmp}/${SLURM_JOB_ID:-manual}"
TMP_DATASET_PATH="${TMP_ROOT%/}/${DEST_NAME}"

echo "World-model storage ablation"
echo "  dataset_path=${DATASET_PATH}"
echo "  tmp_dataset_path=${TMP_DATASET_PATH}"
echo "  run_prefix=${RUN_PREFIX}"
echo "  batch_sizes=${BATCH_SIZES[*]}"
echo "  chunk_lengths=${CHUNK_LENGTHS[*]}"
echo "  num_workers=${NUM_WORKERS}"
echo "  pin_memory=${PIN_MEMORY}"
echo "  persistent_workers=${PERSISTENT_WORKERS}"
echo "  prefetch_factors=${PREFETCH_FACTORS[*]}"
echo "  warmup_batches=${WARMUP_BATCHES}"
echo "  measure_batches=${MEASURE_BATCHES}"
echo "  device=${DEVICE}"

echo
echo "[1/3] Preparing shard cache on horse dataset"
uv run drl world-model prepare-cache \
  --path "${DATASET_PATH}"

echo
echo "[2/3] Staging dataset to tmp with prepared shards"
uv run drl world-model stage \
  --path "${DATASET_PATH}" \
  --tmp-root "${TMP_ROOT}" \
  --dest-name "${DEST_NAME}"

echo
echo "[3/3] Running loader probe on horse + prepared shards"
uv run drl world-model probe-loader \
  --run-name "${RUN_PREFIX}-horse-prepared" \
  --data.dataset-paths "${DATASET_PATH}" \
  --batch-sizes "${BATCH_SIZES[@]}" \
  --chunk-lengths "${CHUNK_LENGTHS[@]}" \
  --num-workers-options "${NUM_WORKERS}" \
  --pin-memory-options "${PIN_MEMORY}" \
  --persistent-workers-options "${PERSISTENT_WORKERS}" \
  --prefetch-factors "${PREFETCH_FACTORS[@]}" \
  --warmup-batches "${WARMUP_BATCHES}" \
  --measure-batches "${MEASURE_BATCHES}" \
  --move-to-device \
  --device "${DEVICE}"

echo
echo "[4/4] Running loader probe on tmp + prepared shards"
uv run drl world-model probe-loader \
  --run-name "${RUN_PREFIX}-tmp-prepared" \
  --data.dataset-paths "${TMP_DATASET_PATH}" \
  --batch-sizes "${BATCH_SIZES[@]}" \
  --chunk-lengths "${CHUNK_LENGTHS[@]}" \
  --num-workers-options "${NUM_WORKERS}" \
  --pin-memory-options "${PIN_MEMORY}" \
  --persistent-workers-options "${PERSISTENT_WORKERS}" \
  --prefetch-factors "${PREFETCH_FACTORS[@]}" \
  --warmup-batches "${WARMUP_BATCHES}" \
  --measure-batches "${MEASURE_BATCHES}" \
  --move-to-device \
  --device "${DEVICE}"

echo
echo "Compare these result files:"
echo "  ${CARLABEV_RUNS_ROOT}/world_model/${RUN_PREFIX}-horse-prepared/artifacts/loader_probe_results.json"
echo "  ${CARLABEV_RUNS_ROOT}/world_model/${RUN_PREFIX}-tmp-prepared/artifacts/loader_probe_results.json"
