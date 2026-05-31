<div align="center">

# CarlaBEV-Lab 🧠🚀

**The Deep Reinforcement Learning Training Suite for CarlaBEV**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-API-blueviolet)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Tuning-FF1493)](https://optuna.org/)

</div>

---

## 📌 Overview

**CarlaBEV-Lab** is the centralized experimental playground and training suite for the [CarlaBEV API](https://github.com/yourusername/carlabev-env). It is engineered for mass-scaling Deep Reinforcement Learning experiments for autonomous driving R&D. 

By natively combining **Stable Baselines3** with high-performance hyperparameter tuning by **Optuna**, CarlaBEV-Lab simplifies launching everything from single-node local debugging to massively parallelized multi-node searches on HPC resources.

<!-- Hero Banner Image Placeholder -->
<div align="center">
  <img src="assets/images/carlabev_lab_hero_banner.png" alt="CarlaBEV-Lab Hero Banner" width="800">
</div>

---

## ✨ Key Features

- 🧠 **Ready-to-run RL Models:** Instantly train models using Stable Baselines3 (`PPO`, `SAC`, etc.) directly on the CarlaBEV simulations.
- 🎛️ **Massive Parallel Tuning:** Integrates robust Optuna searches with an SQLite backend, utilizing `constant_liar` sampling and connection timeouts for conflict-free concurrent parameter optimization.
- 🖥️ **HPC Grid Search Ready:** Out-of-the-box support for generating Slurm Job Arrays to run immense GPU searches efficiently across massive computing clusters.
- 📊 **Rich Analysis & Visualization:** Instantly generate HTML visualizations (Parameter Importance, Parallel Coordinates, Optimization History) of any search.

---

## 🚀 Getting Started

### Prerequisites
CarlaBEV-Lab depends directly on `CarlaBEV` being locally accessible. Use [`uv`](https://github.com/astral-sh/uv) to securely resolve the `pyproject.toml` pointing to your local `carlabev-env` project directory.

### Installation

1. Ensure the `carlabev-env` repository exists locally alongside this project.
2. Initialize and sync the training environment:
    ```bash
    uv sync
    ```

### Base Training & Evaluation

To verify everything is working, you can manually execute an evaluation, debugging loop, or train a base agent.

### Git Hooks

This repo uses `pre-commit` to run local quality checks.

Install the hooks once per clone:

```bash
./scripts/setup-hooks.sh
```

Hook behavior:

- `pre-commit`: `ruff --fix` and `ruff-format`
- `pre-push`: `ruff --fix` and `ruff-format`
- `pre-push`: `uv run pytest`


```bash
uv run python train.py exp --study-id PPO_NAVIGATION --exp-id 26
uv run python eval.py
uv run python test.py
```

### Configuration Contract

CarlaBEV-Lab now uses CarlaBEV's public config contract internally. The canonical environment fields are:

- `map_name`
- `obs_mode` (`bev_rgb`, `bev_semantic`, `vector`)
- `action_mode` (`discrete`, `continuous`)
- `reward_mode` (`shaping`, `carl`)

Legacy aliases such as `obs_space`, `action_space`, and `reward_type` are still accepted for compatibility, but they emit deprecation warnings and are only retained at the boundary of older configs or experiments.

### Study Registry

Experiments are now organized under named studies instead of one global mutable experiment table.

- A study contains metadata, its Optuna study name, its SQLite database path, train/eval protocol registries, and a dictionary of `exp_id -> ExperimentSpec`.
- Study definitions live in separate modules under `src/config/studies/`.
- Registry and lookup helpers live in `src/config/studies/registry.py`.
- `PPO_NAVIGATION` is the default migrated navigation study containing the original 29 experiment variants.
- `EDGE_CASE_SCENARIOS` is a second study for curated hazardous scenarios such as `jaywalk`, `lead_brake`, and `red_light_runner`.
- Authored edge-case scenes are copied locally under `assets/scenes/` and grouped by family through `src/config/authored_scenarios.py`.
- Authored-scene variation behavior is declared in the study protocol specs, not hardcoded in trainers:
  - edge-case train protocols use randomized authored variants
  - edge-case eval protocols use the same authored scenes with variation disabled

Each experiment now references:

- one `train_protocol_id`
- one or more `eval_protocol_ids`

That lets a study train on one distribution and evaluate on another.

Current examples:

- `PPO_NAVIGATION`: trains on random generated navigation scenes and evaluates on random generated navigation scenes
- `EDGE_CASE_SCENARIOS`:
  - `exp-id 1`: train on all authored `jaywalk-*` scenes, evaluate on all authored edge-case scenes
  - `exp-id 2`: train on all authored `leadbrake-*` scenes, evaluate on all authored edge-case scenes
  - `exp-id 3`: train on all authored `redlightrunner-*` scenes, evaluate on all authored edge-case scenes
  - `exp-id 4`: train on all authored edge-case scenes, evaluate on all authored edge-case scenes

For authored-scene studies, the reset protocol can also declare:

- `variation_enabled`
- `variation_seed_mode`
- `variation_seed`
- `variation_seed_min`
- `variation_seed_max`

This allows train/eval variation policy to remain fully declarative.

### Manual Runs vs Optuna

A normal training command such as:

```bash
uv run python train.py exp --study-id EDGE_CASE_SCENARIOS --exp-id 1
```

is still recorded through Optuna. It is treated as a single fixed trial using the current config values for that experiment.

Use this when you want:

- one baseline run
- one reproduced run with fixed parameters
- one study/experiment run logged into the same Optuna database

Use `src.tuning.optuna_tuner` when you want actual hyperparameter search across many trials.

Typical workflow:

```bash
uv run python train.py exp --study-id PPO_NAVIGATION --exp-id 26
uv run python eval.py exp --study-id PPO_NAVIGATION --exp-id 26
uv run python train.py exp --study-id EDGE_CASE_SCENARIOS --exp-id 1
uv run python train.py exp --study-id EDGE_CASE_SCENARIOS --exp-id 4
```

---

## 🔬 Optuna Hyperparameter Tuning

CarlaBEV-Lab is structured into continuous and categorical search phases. You can run tuning locally, or scale it up across several nodes.

### 1. Running Locally (Interactive)

Execute tuning phases sequentially from your terminal:

**Phase 1: Tune Continuous Hyperparameters**
```bash
uv run python -m src.tuning.optuna_tuner \
    --study-id PPO_NAVIGATION \
    --exp-id 26 \
    --phase 1 \
    --n-trials-phase-1 100 \
    --timesteps-phase-1 1000000 \
    --eval-episodes 30 \
    --eval-final-episodes 100
```
*(Video and model saving is automatically disabled during Phase 1 to reduce IO overhead.)*

**Phase 2: Tune Categorical Hyperparameters**
```bash
uv run python -m src.tuning.optuna_tuner \
    --study-id PPO_NAVIGATION \
    --exp-id 26 \
    --phase 2a \
    --n-trials-phase-2a 50 \
    --timesteps-phase-2a 2000000
```

### 2. Large Scale HPC Grids

To deploy robust searches (for example, across multiple GPU nodes), utilize the included Slurm launcher scripts.
```bash
# Example: Submit the Job Array for 10 concurrent nodes to solve Phase 1
sbatch scripts/slurm_phase1_launcher.sh
```
*(Logs for each concurrent node scale gracefully into `results/logs/phaseX_%A_%a.out`)*

### 3. Analysis and Visualization

Review tuning runs instantly:
```bash
uv run python -m src.tuning.optuna_analysis --study-id PPO_NAVIGATION --exp-id 26 --top-k 5
```
This loads the Optuna study configured for `PPO_NAVIGATION`, filters to experiment `26`, and generates parameter curves, importance breakdowns, and history charts under `results/`.

---

## 🗺️ System Architecture

The project splits the Deep RL process cleanly from tuning logic:
- `src/agents/`: Definitions, policy constructors, and hyperparameter ingestion.
- `src/config/`: Study registry, typed experiment definitions, and configuration loaders that bridge CLI arguments to the SB3 training loops.
- `src/config/reset_protocol.py`: Shared train/eval reset samplers built from study protocol definitions.
- `src/trainers/`: Main logic for initializing `CarlaBEV` environments, applying Gym Wrappers, and training via model checkpoints.
- `src/tuning/`: Contains the distributed Optuna optimizer logic (`optuna_tuner.py`), and visualization toolings (`optuna_analysis.py`).
- `scripts/`: Collection of Bash scripts for HPC massive slurm-launch grid scaling.

---

## 📄 License
Created for robust R&D internal purposes.
