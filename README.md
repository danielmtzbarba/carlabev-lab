# CarlaBEV Optuna Tuning Guide

This repository utilizes [Optuna](https://optuna.org/) to run parallel hyperparameter optimization searches directly on the TU Dresden HPC Slurm cluster. The search leverages a robust SQLite backend with concurrency support (WAL journaling & connection timeouts) and `constant_liar` sampling to ensure parallel nodes efficiently explore different parameters without duplicating work.

## 1. Running on TU Dresden HPC (Slurm)

To run a massive 10-node search for 8 hours on the GPU partition, use the provided Slurm Launcher script. 

The script defaults to 10 GPU nodes, 14 CPUs per task, and executes **Phase 1** (Continuous Parameters) with the exact default values found in `optuna_tuner.py`.

```bash
# Submit the Job Array for 10 concurrent nodes
sbatch scripts/slurm_phase1_launcher.sh
```

*(Note: You can view logs for each unique node in `results/logs/phase1_%A_%a.out`)*

## 2. Running Locally (Interactive)

You can manually execute tuning phases sequentially from your terminal using `uv`:

### Phase 1: Tune Continuous Hyperparameters
```bash
uv run python -m src.tuning.optuna_tuner \
    --exp-id 26 \
    --phase 1 \
    --n-trials-phase-1 100 \
    --timesteps-phase-1 1000000 \
    --save-every-phase-1 25 \
    --eval-episodes 30 \
    --eval-final-episodes 100
```
*Note: Video and model saving is automatically disabled during Phase 1 to reduce IO overhead.*

### Phase 2: Tune Categorical Hyperparameters
```bash
uv run python -m src.tuning.optuna_tuner \
    --exp-id 26 \
    --phase 2 \
    --n-trials-phase-2 50 \
    --timesteps-phase-2 2000000 \
    --save-every-phase-2 25 \
    --eval-episodes 30 \
    --eval-final-episodes 100
```

## 3. Analysis and Visualization

Once some trials have finished (or while they are running), you can instantly generate HTML visualizations (Parameter Importance, Parallel Coordinates, Optimization History) and print comprehensive SQL metrics.

```bash
uv run python -m src.tuning.optuna_analysis --exp-id 26 --top-k 5
```
Interactive plots will be saved to `results/carlabev_optuna_26_plots/`.
