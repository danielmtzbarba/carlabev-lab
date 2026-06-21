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

By natively combining **Stable Baselines3** with high-performance hyperparameter tuning by **Optuna**, CarlaBEV-Lab supports a study-driven workflow for local training, evaluation, and iterative search.

<!-- Hero Banner Image Placeholder -->
<div align="center">
  <img src="assets/images/carlabev_lab_hero_banner.png" alt="CarlaBEV-Lab Hero Banner" width="800">
</div>

---

## ✨ Key Features

- 🧠 **Study-driven PPO training:** Train and evaluate the maintained PPO navigation and scenario studies directly on CarlaBEV simulations.
- 🎛️ **Massive Parallel Tuning:** Integrates robust Optuna searches with an SQLite backend, utilizing `constant_liar` sampling and connection timeouts for conflict-free concurrent parameter optimization.
- 🧪 **Study-owned tuning stages:** Optuna search is declared per study with explicit tuning stages such as policy dynamics, rollout geometry, and loss regularization.
- 📊 **Rich Analysis & Visualization:** Instantly generate HTML visualizations (Parameter Importance, Parallel Coordinates, Optimization History) of any search.

---

## 🚀 Getting Started

### Prerequisites
CarlaBEV-Lab now resolves `CarlaBEV` from the local sibling checkout declared in
`pyproject.toml`:

- `../../driverless/carlabev-env`

That keeps the lab pinned to the current env workspace during active
co-development. After pulling env changes, rerun:

```bash
uv sync
```

### Installation

1. Initialize and sync the training environment:
    ```bash
    uv sync
    ```

### HPC Storage Layout

For the TU Dresden setup discussed in this repo, the recommended split is:

- code and `.venv`: `/home/h6/dama898h/carlabev-lab`
- active artifacts: `/data/horse/ws/dama898h-carlabev`
- optional long-term archive later: a separate `walrus` workspace

The Python runtime now resolves heavy artifact roots centrally:

- `runs/` -> `${CARLABEV_RUNS_ROOT}` or `${CARLABEV_ARTIFACT_ROOT}/runs`
- `results/` -> `${CARLABEV_RESULTS_ROOT}` or `${CARLABEV_ARTIFACT_ROOT}/results`
- `datasets/` -> `${CARLABEV_DATASETS_ROOT}` or `${CARLABEV_ARTIFACT_ROOT}/datasets`

If none of those environment variables are set, the repo auto-detects
`/data/horse/ws/$USER-carlabev` when it exists; otherwise it falls back to local
relative paths for non-HPC development.

For your current cluster setup, exporting this once is the clearest option:

```bash
export CARLABEV_ARTIFACT_ROOT=/data/horse/ws/dama898h-carlabev
```

Suggested workspace tree:

```text
/data/horse/ws/dama898h-carlabev/
  datasets/
    world_model/
  runs/
    PPO_NAVIGATION/
    PPO_NAVIGATION_DIFFICULTY/
    world_model/
  results/
    logs/
    diagnostics/
    optuna/
```

Separation of responsibilities:

- `runs/`: per-run artifacts such as checkpoints, eval outputs, run-local logs, and all PPO videos
- `results/`: cross-run artifacts such as Slurm logs, Optuna databases/exports, diagnostics, and reports
- `datasets/`: reusable collected data such as world-model shards

The Slurm scripts now mirror that split by writing:

- Slurm stdout/stderr to `/data/horse/ws/dama898h-carlabev/results/logs/`
- Python run artifacts to `/data/horse/ws/dama898h-carlabev/runs/`
- collected datasets to `/data/horse/ws/dama898h-carlabev/datasets/`

Right now the lab tracks the `main` branch of `carlabev-env`. Once you create a release tag such as `v0.1.0`, switch `[tool.uv.sources].CarlaBEV` from `branch = "main"` to `tag = "v0.1.0"` and run `uv sync` again for a reproducible pinned release.

### Base Training & Evaluation

To verify everything is working, you can manually execute an evaluation, debugging loop, or train a base agent.

### CLI Entry Point

The packaged CLI is now the primary way to interact with the lab:

```bash
uv run drl --help
```

After `uv sync`, the installed entry point also lives at `.venv/bin/drl`, so you
can run it directly once the virtualenv is activated or `.venv/bin` is on your
`PATH`.

The main command groups are:

- `train`: run one configured study experiment
- `eval`: evaluate the latest run for a configured study experiment
- `tune`: launch or analyze Optuna tuning stages
- `results`: inspect leaderboards, plots, and report assets
- `scene-library`: prebuild study-owned CarlaBEV scene databases
- `db`: inspect or clean Optuna state
- `diagnostics`: run seed-scene and pruning diagnostics
- `world-model`: collect, inspect, validate, benchmark, and train offline world models for the LeWM proof of concept

`drl` is the primary installed command.

### Git Hooks

This repo uses `pre-commit` to run local quality checks.

Install the hooks once per clone:

```bash
./tools/setup-hooks.sh
```

Hook behavior:

- `pre-commit`: `ruff --fix` and `ruff-format`
- `pre-push`: `ruff --fix` and `ruff-format`
- `pre-push`: `uv run pytest`


```bash
uv run drl train exp --study-id PPO_NAVIGATION --exp-id 26
uv run drl eval exp --study-id PPO_NAVIGATION --exp-id 26
uv run pytest
```

## Docs

Project documentation now lives under [`docs/`](docs/README.md) so the root
README stays focused on setup and entry points.

If you are working on the LeWM proof of concept, start with:

- [`docs/world-model.md`](docs/world-model.md): dataset format, collector,
  validation, training, checkpoint evaluation, and study-aware CLI usage
- [`docs/world-model-hpc.md`](docs/world-model-hpc.md): staging, prepared shard
  caches, loader probing, benchmarking, and current H100 findings

The current quick-start flow for the world-model PoC is:

1. collect or stage a dataset
2. validate it with `uv run drl world-model validate`
3. train with `uv run drl world-model train exp ...`
4. evaluate the checkpoint with `uv run drl world-model eval-checkpoint ...`

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

For the current study path, the preferred declarative selectors are:

- `action_profile_id`
- `reward_profile_id`
- study-owned random-navigation backbones declared through `RandomNavigationProtocol`
- scene-library policy declared per protocol

The maintained study path no longer uses experiment-level `difficulty_id`,
`traffic`, or `curriculum` switches. Those ideas now live in protocol-owned
scene generation:

- `ExperimentSpec` owns agent-facing choices such as action mode, input type,
  semantic layout, temporal fusion, reward mode, FOV settings, and
  train/eval protocol ids
- `RandomNavigationProtocol` owns reset seeding, curriculum behavior,
  scene-library policy, and one explicit `SceneGenerationBackbone`
- `SceneGenerationBackbone` owns random-scene controls such as
  `route_extent`, `route_dist_range`, `speed_profile`, `num_vehicles`,
  `num_vehicles_near_ego`, `traffic_role_profile`, and
  `guaranteed_candidate_role`

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
- `PPO_NAVIGATION_DIFFICULTY`: now acts as a traffic-density-only study with
  explicit `no_traffic`, `easy`, `medium`, and `hard` near-ego scene backbones
- `PPO_NAVIGATION_MEDIUM_FOV_ANCHOR`,
  `PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES`, and
  `PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION`: use fixed medium scene backbones and
  vary only the targeted ablation axis
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

### Scene Library Workflow

The maintained PPO studies now assume a study-specific CarlaBEV scene library.
The intended workflow is:

1. Prebuild the scene corpus for the study with `carlabev-env`.
2. Point the study protocols at the resulting `.db` path.
3. Train and evaluate with the scene library enabled in read-only mode.

Each migrated random-navigation study now declares its own scene-library path in
the protocol spec, for example:

- `assets/scene_libraries/ppo_navigation.db`
- `assets/scene_libraries/ppo_navigation_difficulty.db`
- `assets/scene_libraries/ppo_navigation_medium_fov_anchor.db`

This keeps scene generation explicit, repeatable, and decoupled from training.

The preferred way to populate those databases from the lab repo is now:

```bash
uv run drl scene-library build --study-id PPO_NAVIGATION --dry-run
uv run drl scene-library build --study-id PPO_NAVIGATION
```

Default behavior:

- dedupe identical random-navigation backbones across train/eval protocols
- use the shared 10 prime study seeds: `2 3 5 7 11 13 17 19 23 29`
- request `1000` scenes per study seed per unique backbone

Useful overrides:

```bash
uv run drl scene-library build --study-id PPO_NAVIGATION --episodes-per-seed 500
uv run drl scene-library build --study-id PPO_NAVIGATION_DIFFICULTY --protocol-ids easy_train easy_eval
uv run drl scene-library build --study-id PPO_NAVIGATION --include-eval --dry-run --json
```

The lab command delegates to CarlaBEV's public scene-library builder using the
structured seed interface:

- `study_id`
- `backbone_id`
- `study_seed`
- `episode_index`

That means the corpus is deterministic per study seed without relying on opaque
manual seed offsets.

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
uv run drl train exp --study-id EDGE_CASE_SCENARIOS --exp-id 1
```

is still recorded through Optuna. It is treated as a single fixed trial using the current config values for that experiment.

Use this when you want:

- one baseline run
- one reproduced run with fixed parameters
- one study/experiment run logged into the same Optuna database

Use `uv run drl tune run` when you want actual hyperparameter search across many trials.

Typical workflow:

```bash
uv run drl train exp --study-id PPO_NAVIGATION --exp-id 26
uv run drl eval exp --study-id PPO_NAVIGATION --exp-id 26
uv run drl train exp --study-id EDGE_CASE_SCENARIOS --exp-id 1
uv run drl train exp --study-id EDGE_CASE_SCENARIOS --exp-id 4
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
uv run drl results top-trials \
  --study-id PPO_NAVIGATION_DIFFICULTY \
  --top-k 10
```

Print seed-averaged experiment summaries:

```bash
uv run drl results top-experiments \
  --study-id PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION \
  --top-k 3
```

Print the normalized-score leaderboard:

```bash
uv run drl results leaderboard \
  --study-id PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES \
  --top-k 6
```

### Seed Scene Diagnostics

The seed-scene diagnostic tooling is now split into two stages so you do not need to rerun the simulator every time you tweak the figures.

Run the expensive data-generation pass once:

```bash
uv run drl diagnostics seed-scenes analyze \
  --scene-profile-ids medium \
  --samples-per-seed 1000 \
  --seed-mode incremental \
  --save-frames-per-pair 6 \
  --output-dir results/seed_scene_diag_medium_incremental
```

This writes reusable artifacts for each `(scene_profile, seed)` pair:

```text
results/seed_scene_diag_medium_incremental/
  summary.json
  all_samples.csv
  <scene_profile_id>/
    seed_<seed>/
      samples.csv
      spawn_points.csv
      route_points.csv
      frames/
```

Re-render figures from those saved artifacts without touching the simulator:

```bash
uv run drl diagnostics seed-scenes visualize \
  --output-dir results/seed_scene_diag_medium_incremental
```

Spawn clustering is visualization-only and can be tuned without rerunning analysis:

```bash
uv run drl diagnostics seed-scenes visualize \
  --output-dir results/seed_scene_diag_medium_incremental \
  --spawn-cluster-radius 20 \
  --spawn-top-k 10
```

Behavior of the current plots:

- `spawn_*`: top repeated spawn zones shown as numbered cluster centroids
- `route_*`: route corridor density heatmaps

Built-in scene-profile ids currently include:

- `no_traffic`
- `easy`
- `medium`
- `hard`
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

### Repository Layout

The repo root is intentionally kept narrow:

- `src/`: maintained Python package code and CLI modules
- `infra/slurm/`: cluster launchers for study sweeps
- `tools/`: local repo helpers such as hook installation
- `docs/`: generated figures and authored notes
- `assets/`: curated static scene inputs and images

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

CarlaBEV-Lab now declares tuning through the study system. Each study can own an explicit sequence of tuning stages instead of opaque phase numbers.

### 1. Running Locally (Interactive)

Execute tuning stages sequentially from your terminal:

**Policy Dynamics: tune learning rate, GAE lambda, and discount factor**
```bash
uv run drl tune run \
    --study-id PPO_NAVIGATION \
    --exp-id 26 \
    --stage policy_dynamics \
    --n-trials 100 \
    --total-timesteps 1000000 \
    --eval-episodes 30 \
    --eval-final-episodes 100
```
*(Video and model saving are disabled during tuning stages by the study-owned tuning config.)*

**Rollout Geometry: tune rollout horizon and minibatch/update geometry**
```bash
uv run drl tune run \
    --study-id PPO_NAVIGATION \
    --exp-id 26 \
    --stage rollout_geometry \
    --n-trials 50 \
    --total-timesteps 2000000
```

The current `PPO_NAVIGATION` study declares these stages:

- `policy_dynamics`
- `rollout_geometry`
- `loss_regularization`
- `network_capacity`

### 2. Analysis and Visualization

Review tuning runs instantly:
```bash
uv run drl tune analyze --study-id PPO_NAVIGATION --exp-id 26 --top-k 5
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
- `src/carlabev_lab/`: package-backed CLI commands for reporting, diagnostics, and DB maintenance.
- `infra/slurm/`: cluster launchers for study sweeps.
- `tools/`: local repository helpers such as hook installation.

---

## 📄 License
Created for robust R&D internal purposes.
