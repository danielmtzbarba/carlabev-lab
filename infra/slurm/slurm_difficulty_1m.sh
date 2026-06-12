#!/bin/bash
#SBATCH --job-name=carlabev_difficulty_1m
#SBATCH --output=results/logs/difficulty_1m_%A_%a.out
#SBATCH --error=results/logs/difficulty_1m_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=1-40

# Run artifacts now land under:
#   runs/PPO_NAVIGATION_DIFFICULTY/exp_<exp_id>/trial_<trial>/seed_<seed>/
# with checkpoints/, eval/, and videos/ subdirectories per run.
#
# Video plan for 1M runs:
#   - 20 training probe videos
#   - 5 intermediate-eval videos per scheduled eval
#   - 10 final-eval videos across the 1000 final episodes

set -euo pipefail

STUDY_ID="PPO_NAVIGATION_DIFFICULTY"

TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=100
FINAL_EVAL_EPISODES=1000

PRIME_SEEDS=(2 3 5 7 11 13 17 19 23 29)
EXP_VARIANTS=(1 2 3 4)
EXP_IDS=()
SEEDS=()
for exp_id in "${EXP_VARIANTS[@]}"; do
    for seed in "${PRIME_SEEDS[@]}"; do
        EXP_IDS+=("${exp_id}")
        SEEDS+=("${seed}")
    done
done

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

echo "Starting difficulty ablation on node: $(hostname)"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Resolved study_id: ${STUDY_ID}"
echo "Resolved exp_id: ${EXP_ID}"
echo "Resolved seed: ${SEED}"
echo "Artifacts will be recorded under runs/${STUDY_ID}/exp_${EXP_ID}/..."

sleep_time=$(((SLURM_ARRAY_TASK_ID - 1) * 20))
echo "Sleeping ${sleep_time}s before launch..."
sleep "${sleep_time}"

srun env PYTHONUNBUFFERED=1 uv run carlabev-lab train exp \
    --study-id "${STUDY_ID}" \
    --exp-id "${EXP_ID}" \
    --seed "${SEED}" \
    --run-mode headless \
    --ppo.total-timesteps "${TOTAL_TIMESTEPS}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --eval-final-episodes "${FINAL_EVAL_EPISODES}"
