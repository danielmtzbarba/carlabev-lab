#!/bin/bash
#SBATCH --job-name=carlabev_semlook_remaining
#SBATCH --output=results/logs/semantic_lookahead_remaining_%A_%a.out
#SBATCH --error=results/logs/semantic_lookahead_remaining_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --array=1-32

# Launch only the missing first-pass semantic-mask x FOV-anchor runs for
# PPO_NAVIGATION_SEMANTIC_LOOKAHEAD. This excludes the four completed pilot
# runs:
#   (exp_id=5, seed=0)
#   (exp_id=6, seed=0)
#   (exp_id=9, seed=0)
#   (exp_id=10, seed=0)

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_SEMANTIC_LOOKAHEAD"

# First-pass budget
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=30
FINAL_EVAL_EPISODES=100

# Missing (exp_id, seed) pairs after the 6-task pilot.
EXP_IDS=(
    1 1 1
    2 2 2
    3 3 3
    4 4 4
    5 5
    6 6
    7 7 7
    8 8 8
    9 9
    10 10
    11 11 11
    12 12 12
)

SEEDS=(
    0 1 2
    0 1 2
    0 1 2
    0 1 2
    1 2
    1 2
    0 1 2
    0 1 2
    1 2
    1 2
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

echo "Starting remaining semantic/lookahead matrix run on node: $(hostname)"
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
