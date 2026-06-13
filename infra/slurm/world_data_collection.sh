#!/bin/bash

set -euo pipefail

STUDY_ID="${STUDY_ID:-PPO_NAVIGATION_DIFFICULTY}"
DATASET_NAME="${DATASET_NAME:-lewm-ppo-difficulty-hpc}"
SPLIT="${SPLIT:-train}"
POLICY="${POLICY:-ppo}"
DEVICE="${DEVICE:-cuda}"
NUM_ENVS="${NUM_ENVS:-14}"
TOTAL_TRANSITIONS="${TOTAL_TRANSITIONS:-100000}"
STEPS_PER_SHARD="${STEPS_PER_SHARD:-16384}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"

# Winning backbone from the difficulty study:
# - exp 1: no traffic
# - exp 3: medium
EXP_IDS=(${EXP_IDS:-1 3})
SEEDS=(${SEEDS:-2 3 5 7 11})

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

echo "World-model dataset collection"
echo "  study_id=${STUDY_ID}"
echo "  dataset_name=${DATASET_NAME}"
echo "  split=${SPLIT}"
echo "  policy=${POLICY}"
echo "  device=${DEVICE}"
echo "  num_envs=${NUM_ENVS}"
echo "  total_transitions=${TOTAL_TRANSITIONS}"
echo "  steps_per_shard=${STEPS_PER_SHARD}"
echo "  exp_ids=${EXP_IDS[*]}"
echo "  seeds=${SEEDS[*]}"

for SEED in "${SEEDS[@]}"; do
    for EXP_ID in "${EXP_IDS[@]}"; do
        echo
        echo "Collecting dataset for exp_id=${EXP_ID} seed=${SEED}"
        uv run drl world-model collect exp \
            --study-id "${STUDY_ID}" \
            --exp-id "${EXP_ID}" \
            --seed "${SEED}" \
            --num-envs "${NUM_ENVS}" \
            --total-transitions "${TOTAL_TRANSITIONS}" \
            --steps-per-shard "${STEPS_PER_SHARD}" \
            --policy "${POLICY}" \
            --dataset-name "${DATASET_NAME}" \
            --split "${SPLIT}" \
            --device "${DEVICE}" \
            --show-progress "${SHOW_PROGRESS}"
    done
done
