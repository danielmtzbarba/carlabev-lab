#!/bin/bash
#SBATCH --job-name=carlabev_semlook_confirm3m
#SBATCH --output=results/logs/semantic_lookahead_confirm3m_%A_%a.out
#SBATCH --error=results/logs/semantic_lookahead_confirm3m_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --array=1-12

# 3M-step confirmation pass for the top semantic/lookahead configurations:
#   exp 3 = 2-class, center
#   exp 4 = 2-class, lookahead_75
#   exp 5 = 4-class, center
#   exp 6 = 4-class, lookahead_75
#
# Runs 3 seeds per experiment under the same study:
#   PPO_NAVIGATION_SEMANTIC_LOOKAHEAD

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_SEMANTIC_LOOKAHEAD"

# Confirmation budget
TOTAL_TIMESTEPS=3000000
EVAL_EPISODES=30
FINAL_EVAL_EPISODES=100

EXP_IDS=(
    3 3 3
    4 4 4
    5 5 5
    6 6 6
)

SEEDS=(
    0 1 2
    0 1 2
    0 1 2
    0 1 2
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
SEED="${SEEDS[$ARRAY_INDEX]}"

echo "Starting 3M confirmation run on node: $(hostname)"
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
