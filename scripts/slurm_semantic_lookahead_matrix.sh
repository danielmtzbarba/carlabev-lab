#!/bin/bash
#SBATCH --job-name=carlabev_semlook_matrix
#SBATCH --output=results/logs/semantic_lookahead_%A_%a.out
#SBATCH --error=results/logs/semantic_lookahead_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=1-6

# Launch the full semantic-mask x FOV-anchor matrix for
# PPO_NAVIGATION_SEMANTIC_LOOKAHEAD across a configurable seed grid.
# This pilot launcher samples six (exp_id, seed) pairs across the matrix
# to estimate runtime before expanding to the full 36-task grid.

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_SEMANTIC_LOOKAHEAD"

# First-pass budget
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=30
FINAL_EVAL_EPISODES=100

# Pilot sample across the matrix.
PILOT_EXP_IDS=(1 2 5 6 9 10)
PILOT_SEEDS=(0 0 0 0 0 0)

module purge

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR}"

mkdir -p results/logs

TOTAL_TASKS=${#PILOT_EXP_IDS[@]}
ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= TOTAL_TASKS )); then
    echo "Array index ${ARRAY_INDEX} is out of bounds for ${TOTAL_TASKS} tasks."
    exit 1
fi

EXP_ID="${PILOT_EXP_IDS[$ARRAY_INDEX]}"
SEED="${PILOT_SEEDS[$ARRAY_INDEX]}"

echo "Starting semantic/lookahead matrix run on node: $(hostname)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Resolved study_id: ${STUDY_ID}"
echo "Resolved exp_id: ${EXP_ID}"
echo "Resolved seed: ${SEED}"
echo "Resolved total_timesteps: ${TOTAL_TIMESTEPS}"
echo "Resolved eval_episodes: ${EVAL_EPISODES}"
echo "Resolved eval_final_episodes: ${FINAL_EVAL_EPISODES}"

# Stagger starts a bit to reduce simultaneous SQLite initialization pressure.
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
