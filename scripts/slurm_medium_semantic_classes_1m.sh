#!/bin/bash
#SBATCH --job-name=carlabev_med_sem_1m
#SBATCH --output=results/logs/medium_semantic_classes_1m_%A_%a.out
#SBATCH --error=results/logs/medium_semantic_classes_1m_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-18

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES"

TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=100
FINAL_EVAL_EPISODES=1000

EXP_IDS=(
    1 1 1
    2 2 2
    3 3 3
    4 4 4
    5 5 5
    6 6 6
)

SEEDS=(
    0 555 9999
    0 555 9999
    0 555 9999
    0 555 9999
    0 555 9999
    0 555 9999
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

echo "Starting medium semantic-classes ablation on node: $(hostname)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Resolved study_id: ${STUDY_ID}"
echo "Resolved exp_id: ${EXP_ID}"
echo "Resolved seed: ${SEED}"

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
