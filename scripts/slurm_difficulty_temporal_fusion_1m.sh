#!/bin/bash
#SBATCH --job-name=carlabev_diff_tf_1m
#SBATCH --output=results/logs/difficulty_temporal_fusion_1m_%A_%a.out
#SBATCH --error=results/logs/difficulty_temporal_fusion_1m_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-24

# First-pass 1M-step sweep for:
#   PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION
#
# Factors:
#   difficulty_id       = rt_no_traffic_v1, rt_easy_v1, rt_medium_v1, rt_hard_v1
#   temporal_fusion     = stack, vehicle_temporal, vehicle_weighted
#   fov_anchor          = center, lookahead_75
#
# This launcher runs one seed per experiment for the initial screening pass.

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_DIFFICULTY_TEMPORAL_FUSION"

TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=30
FINAL_EVAL_EPISODES=100
SEED=0

EXP_IDS=(
    1 2 3 4 5 6
    7 8 9 10 11 12
    13 14 15 16 17 18
    19 20 21 22 23 24
)

module purge

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR}"

mkdir -p results/logs

TOTAL_TASKS=${#EXP_IDS[@]}
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= TOTAL_TASKS )); then
    echo "Array index ${ARRAY_INDEX} is out of bounds for ${TOTAL_TASKS} tasks."
    exit 1
fi

EXP_ID="${EXP_IDS[$ARRAY_INDEX]}"

echo "Starting difficulty/temporal-fusion sweep on node: $(hostname)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Resolved study_id: ${STUDY_ID}"
echo "Resolved exp_id: ${EXP_ID}"
echo "Resolved seed: ${SEED}"
echo "Resolved total_timesteps: ${TOTAL_TIMESTEPS}"
echo "Resolved eval_episodes: ${EVAL_EPISODES}"
echo "Resolved eval_final_episodes: ${FINAL_EVAL_EPISODES}"

sleep_time=$((SLURM_ARRAY_TASK_ID * 5))
echo "Sleeping ${sleep_time}s before launch..."
sleep "${sleep_time}"

srun uv run python train.py exp \
    --study-id "${STUDY_ID}" \
    --exp-id "${EXP_ID}" \
    --seed "${SEED}" \
    --run-mode headless \
    --ppo.total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-final-episodes "${FINAL_EVAL_EPISODES}" \
    --no-capture-video
