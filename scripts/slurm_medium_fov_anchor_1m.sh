#!/bin/bash
#SBATCH --job-name=carlabev_med_fov_1m
#SBATCH --output=results/logs/medium_fov_anchor_1m_%A_%a.out
#SBATCH --error=results/logs/medium_fov_anchor_1m_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=1-6

# 1M-step medium-difficulty FOV-anchor ablation:
#   PPO_NAVIGATION_MEDIUM_FOV_ANCHOR
#
# Variations:
#   exp 1 = center
#   exp 2 = lookahead_75
#
# Seeds:
#   0, 555, 9999
#
# Run artifacts now land under:
#   runs/PPO_NAVIGATION_MEDIUM_FOV_ANCHOR/exp_<exp_id>/trial_<trial>/seed_<seed>/
# with checkpoints/, eval/, and videos/ subdirectories per run.

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR"

TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=100
FINAL_EVAL_EPISODES=1000

EXP_IDS=(
    1 1 1
    2 2 2
)

SEEDS=(
    0 555 9999
    0 555 9999
)

module purge

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export OPTUNA_SQLITE_BUSY_TIMEOUT_SECONDS=120
export OPTUNA_STUDY_ACQUIRE_MAX_WAIT_SECONDS=900
export OPTUNA_STUDY_ACQUIRE_RETRY_MIN_SECONDS=5
export OPTUNA_STUDY_ACQUIRE_RETRY_MAX_SECONDS=15

cd "${SLURM_SUBMIT_DIR}"

mkdir -p results/logs

TOTAL_TASKS=${#EXP_IDS[@]}
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= TOTAL_TASKS )); then
    echo "Array index ${ARRAY_INDEX} is out of bounds for ${TOTAL_TASKS} tasks."
    exit 1
fi

EXP_ID="${EXP_IDS[$ARRAY_INDEX]}"
SEED="${SEEDS[$ARRAY_INDEX]}"

echo "Starting medium FOV-anchor run on node: $(hostname)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Resolved study_id: ${STUDY_ID}"
echo "Resolved exp_id: ${EXP_ID}"
echo "Resolved seed: ${SEED}"
echo "Resolved total_timesteps: ${TOTAL_TIMESTEPS}"
echo "Resolved eval_episodes: ${EVAL_EPISODES}"
echo "Resolved eval_final_episodes: ${FINAL_EVAL_EPISODES}"
echo "Artifacts will be recorded under runs/${STUDY_ID}/exp_${EXP_ID}/..."

sleep_time=$(((SLURM_ARRAY_TASK_ID - 1) * 20))
echo "Sleeping ${sleep_time}s before launch..."
sleep "${sleep_time}"

srun env PYTHONUNBUFFERED=1 uv run python train.py exp \
    --study-id "${STUDY_ID}" \
    --exp-id "${EXP_ID}" \
    --seed "${SEED}" \
    --run-mode headless \
    --ppo.total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-final-episodes "${FINAL_EVAL_EPISODES}"
