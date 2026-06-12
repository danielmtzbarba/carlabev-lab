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

- 🧠 **Study-driven PPO training:** Train and evaluate the maintained PPO navigation and scenario studies directly on CarlaBEV simulations.
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
uv run python eval.py exp --study-id PPO_NAVIGATION --exp-id 26
uv run pytest
```

### Testing

The repo ships with a `pytest` suite focused on the maintained PPO path.

Run the full suite:

```bash
uv run pytest
```

Run a focused subset:

```bash
uv run pytest tests/config
uv run pytest tests/factories tests/agents
uv run pytest -m integration
```

Current test coverage is organized as:

- `tests/config/`: study schema, experiment loading, reset protocol sampling, and run-path generation
- `tests/factories/`: PPO-only factory dispatch and unsupported-algorithm rejection
- `tests/agents/`: CNN PPO backbone, discrete head, and continuous head behavior
- `tests/eval/`: evaluation aggregation and study scoring
- `tests/trainers/`: PPO smoke coverage with fake vector environments
- `tests/tuning/`: Optuna phase mutation and orchestration behavior

Markers declared in `pyproject.toml`:

- `unit`: fast isolated tests
- `integration`: multi-module local integration tests
- `slow`: heavier smoke coverage
- `envdep`: tests that rely on the CarlaBEV runtime contract

### Configuration Contract

CarlaBEV-Lab now uses CarlaBEV's public config contract internally. The canonical environment fields are:

- `map_name`
- `obs_mode` (`bev_rgb`, `bev_semantic`, `vector`)
- `action_mode` (`discrete`, `continuous`)
- `reward_mode` (`shaping`, `carl`)

For the current study path, the preferred declarative selectors are the profile IDs exported by `carlabev-env`:

- `difficulty_id`
- `action_profile_id`
- `reward_profile_id`

Legacy aliases such as `obs_space`, `action_space`, and `reward_type` are still accepted for compatibility, but they emit deprecation warnings and are intended only as migration shims at older config boundaries.

### Study Registry

Experiments are now organized under named studies instead of one global mutable experiment table.

- A study contains metadata, its Optuna study name, its SQLite database path, train/eval protocol registries, and a dictionary of `exp_id -> ExperimentSpec`.
- Study definitions live in separate modules under `src/config/studies/`.
- Registry and lookup helpers live in `src/config/studies/registry.py`.
- `PPO_NAVIGATION` is the default migrated navigation study and currently contains 31 experiment variants.
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

### Reset Seed Scheduling

The active PPO path now separates:

- run seed: experiment identity and reproducibility anchor
- reset seed schedule: episode-to-episode and env-slot diversity

For `random_navigation` protocols, the reset sampler derives deterministic
per-env, per-episode seeds from the top-level run seed and protocol id.

Current modes:

- `fixed`: each env slot reuses its own fixed reset seed across episodes
- `incremental`: each env slot advances a deterministic counter-based seed
- `hashed_episode`: each env slot gets a hashed `(run_seed, protocol_id, env_slot, reset_count)` seed

The default navigation setting is `hashed_episode`.

Practical consequences:

- parallel env slots no longer collapse onto the same reset seed
- episode diversity is maintained inside a run
- rerunning with the same top-level run seed reproduces the same reset-seed schedule

Study-level seed policy:

- Optuna phases and the main study launchers now use the shared 10-prime seed set
- `2, 3, 5, 7, 11, 13, 17, 19, 23, 29`
- this keeps study comparisons aligned across trials, studies, and manual sweeps

Current limitation:

- `scenario_catalog` protocols still build one shared reset options payload per
  vector reset call, so authored-scene entry choice is not yet independently
  diversified per env slot in the same reset wave
- this does not affect the random-navigation seed pathology fix

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

### Evaluation Metrics And Study Scoring

Evaluation payloads now include comfort-aware metrics in addition to task success:

- `success_rate`
- `collision_rate`
- `unfinished_rate`
- `mean_abs_accel_long`
- `mean_abs_accel_lat`
- `mean_abs_jerk_long`
- `mean_abs_jerk_lat`
- `mean_abs_yaw_rate`
- `mean_abs_yaw_acc`
- `comfort_violation_rate`
- `harsh_brake_rate`
- `comfort_score`
- `normalized_score`

`normalized_score` is the main bounded study-ranking score. It is designed to stay interpretable across reward changes by combining:

- task completion
- collision avoidance
- unfinished episodes
- comfort

Raw `mean_return` is still logged and saved, but it is no longer the preferred leaderboard metric for study comparisons.

### Result Inspection

Print raw completed trials for a study:

```bash
uv run python scripts/print_top_study_results.py \
  --study-id PPO_NAVIGATION_DIFFICULTY \
  --top-k 10
```

Print seed-averaged experiment summaries:

```bash
uv run python scripts/print_top_experiments_by_seed_average.py \
  --study-id PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION \
  --top-k 3
```

Print the normalized-score leaderboard:

```bash
uv run python scripts/print_normalized_study_leaderboard.py \
  --study-id PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES \
  --top-k 6
```

### Seed Scene Diagnostics

The seed-scene diagnostic tooling is now split into two stages so you do not need to rerun the simulator every time you tweak the figures.

Run the expensive data-generation pass once:

```bash
uv run python scripts/analyze_seed_scene_distribution.py analyze \
  --difficulty-ids rt_medium_v1 \
  --samples-per-seed 1000 \
  --seed-mode incremental \
  --save-frames-per-pair 6 \
  --output-dir results/seed_scene_diag_medium_incremental
```

This writes reusable artifacts for each `(difficulty, seed)` pair:

```text
results/seed_scene_diag_medium_incremental/
  summary.json
  all_samples.csv
  <difficulty_id>/
    seed_<seed>/
      samples.csv
      spawn_points.csv
      route_points.csv
      frames/
```

Re-render figures from those saved artifacts without touching the simulator:

```bash
uv run python scripts/analyze_seed_scene_distribution.py visualize \
  --output-dir results/seed_scene_diag_medium_incremental
```

Spawn clustering is visualization-only and can be tuned without rerunning analysis:

```bash
uv run python scripts/analyze_seed_scene_distribution.py visualize \
  --output-dir results/seed_scene_diag_medium_incremental \
  --spawn-cluster-radius 20 \
  --spawn-top-k 10
```

Behavior of the current plots:

- `spawn_*`: top repeated spawn zones shown as numbered cluster centroids
- `route_*`: route corridor density heatmaps
- default spawn clustering merges nearby starts within a `16`-pixel radius and shows the top `10` clusters

If you still want the original one-command workflow, the script defaults to `full` mode when no subcommand is provided.

Interpretation after the seed fix:

- `fixed` is the old collapse mode and is useful as a control
- `incremental` and the lab runtime `hashed_episode` mode are the diversity-preserving settings

### Run Artifact Layout

Study runs now use a short, structured scaffold instead of long descriptive folder names:

```text
runs/
  <study_id>/
    exp_<exp_id>/
      trial_<trial_id>|trial_manual/
        seed_<seed>/
          config.yaml
          train.log
          status.json
          checkpoints/
          eval/
          videos/
```

Two identifiers are tracked in config and status files:

- `run_label`: `{study_id}_e{exp_id}`
- `run_id`: `{study_id}_e{exp_id}_t{trial_id}_s{seed}`

The latest resolved run for each `(study_id, exp_id)` is also written to:

```text
runs/<study_id>/exp_<exp_id>/LATEST_RUN.json
```

### Video Capture Policy

The study launchers now use explicit milestone-based capture:

- training:
  - `20` probe videos across a `1_000_000`-step run
- intermediate evaluation:
  - `5` videos sampled across the `100` eval episodes
- final evaluation:
  - `10` videos sampled across the `1000` final episodes

All video outputs live under the run-local `videos/` tree:

```text
videos/
  train/
  eval/
    intermediate/
    final/
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

The project splits the PPO training flow cleanly from tuning logic:
- `src/agents/`: PPO policy and backbone constructors used by the maintained study path.
- `src/config/`: Study registry, typed experiment definitions, and configuration loaders that bridge CLI arguments to CarlaBEV run configs.
- `src/config/reset_protocol.py`: Shared train/eval reset samplers built from study protocol definitions.
- `src/trainers/`: PPO training loop and checkpoint/evaluation orchestration.
- `src/eval/`: PPO evaluation, protocol aggregation, and study scoring.
- `src/tuning/`: Distributed Optuna orchestration (`optuna_tuner.py`) and analysis tooling (`optuna_analysis.py`).
- `scripts/`: Slurm launchers and study/result inspection utilities.

---

## 📄 License
Created for robust R&D internal purposes.
