#!/bin/bash
#SBATCH --job-name=carlabev_optuna_phase1
#SBATCH --output=results/logs/phase1_%A_%a.out
#SBATCH --error=results/logs/phase1_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=1-10

# Note: Based on TU Dresden HPC Compendium
# We use a Job Array (--array=1-10) with 1 node per array task
# to robustly launch 10 independent Optuna workers across 10 nodes.

# Set up environment
module purge

# Export OMP_NUM_THREADS to match requested CPUs as per TU Dresden best practices
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the directory where the job was submitted (should be the project root)
cd $SLURM_SUBMIT_DIR
# Alternatively, you can hardcode the absolute path to your project on the HPC:
# cd /absolute/path/to/carlabev-lab

# Ensure the results/logs directory exists
mkdir -p results/logs

echo "Starting Optuna Phase 1 worker on node: $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Run the Optuna Tuner Phase 1 using uv
srun uv run python -m src.tuning.optuna_tuner \
    --exp-id 26 \
    --phase 1 \
    --n-trials-phase-1 100 \
    --timesteps-phase-1 1000000 \
    --save-every-phase-1 25 \
    --eval-episodes 30 \
    --eval-final-episodes 100 \
    --top-k-phase-1 10
