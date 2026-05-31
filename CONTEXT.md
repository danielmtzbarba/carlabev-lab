# CONTEXT.md

## Project Summary

`carlabev-lab` is a reinforcement learning experiment harness built around an external `CarlaBEV` environment package. Its core design is to separate:

- experiment declaration
- environment reset distribution selection
- algorithm training/evaluation
- hyperparameter optimization

The repository is optimized for study-based PPO experimentation, with additional older algorithm implementations still present in the tree.

## External Dependency Boundary

The environment implementation does not live in this repository.

- `pyproject.toml` declares `CarlaBEV` as a local editable dependency.
- The active source path is `/home/danielmtz/Projects/carlabev-env`.
- `run_experiment()` and `evaluate_ppo()` import `CarlaBEV.envs.make_env`.

This repo therefore owns experiment orchestration, not the simulator/environment internals.

## Architecture

### 1. Entry Layer

- `train.py` loads a configured experiment and runs it.
- `eval.py` loads the same experiment config and evaluates the final PPO checkpoint.

Both entrypoints rely on `src/config/experiment_loader.py`.

### 2. Configuration Layer

The repository uses dataclasses plus Pydantic models:

- dataclasses in `src/config/base_config.py` hold mutable runtime config
- Pydantic models in `src/config/studies/models.py` define validated, declarative study specifications

The flow is:

1. CLI creates `ArgsCarlaBEV`
2. study registry resolves a `StudyConfig`
3. an `ExperimentSpec` mutates `ArgsCarlaBEV`
4. runtime code consumes the mutated dataclass

This gives a declarative experiment surface while keeping trainer code simple.

### 3. Study Registry Layer

`src/config/studies/registry.py` is the catalog entrypoint.

Current studies:

- `PPO_NAVIGATION`
- `EDGE_CASE_SCENARIOS`

Each study owns:

- `study_id`
- description and metadata
- Optuna study name
- SQLite db path
- train protocols
- eval protocols
- experiment table

This design removes the need for one global mutable experiment table.

### 4. Reset Protocol Layer

`src/config/reset_protocol.py` converts declarative protocol specs into concrete `env.reset(..., options=...)` payloads.

Two protocol modes exist:

- `random_navigation`
- `scenario_catalog`

Design intent:

- trainers should not hardcode train/eval reset distributions
- study definitions choose distributions
- samplers produce deterministic-per-study/per-seed protocol randomness

Notable detail:

- protocol RNG seeds are derived from `cfg.seed`, `cfg.study_id`, and `protocol_id` through SHA-256

That makes authored-scenario selection reproducible across runs without tying it to Python’s global RNG state.

### 5. Training Layer

`src/trainers/__init__.py` dispatches algorithms.

The most complete path is PPO:

- `src/trainers/ppo.py`
- `src/agents/cnn_ppo.py`
- `src/eval/eval_ppo.py`

PPO trainer responsibilities:

- compute rollout tensor sizes from PPO config
- build agent and optimizer
- sample reset options through `ResetProtocolSampler`
- run vectorized rollouts
- normalize rewards
- compute GAE
- optimize actor/critic
- trigger periodic evaluation
- report metrics to TensorBoard, console, SQLite, and Optuna

The trainer performs both training and lifecycle orchestration. It is not a pure optimization loop.

### 6. Evaluation Layer

`src/eval/eval_ppo.py` is protocol-aware.

Important behavior:

- builds eval samplers from `cfg.eval_protocol_ids`
- evaluates each protocol independently
- aggregates weighted metrics across protocols
- saves a `.npy` payload under `runs/<exp_name>/`

Evaluation metrics:

- `mean_return`
- `std_return`
- `mean_length`
- `success_rate`
- `collision_rate`
- `unfinished_rate`

This means evaluation is intentionally distribution-aware rather than tied to a single fixed reset regime.

### 7. Logging Layer

`src/utils/logger.py` is the observability hub.

Outputs:

- TensorBoard under `runs/<exp_name>/`
- console summaries through Rich
- file logging through `drlog.log`
- optional SQLite rows into the same Optuna database file

It records:

- per-episode training stats
- learning losses and coefficients
- evaluation summaries
- threshold-crossing benchmark times

Design intent:

- Optuna trials should be inspectable beyond final scalar objective values
- training/eval curves should be recoverable from SQLite and TensorBoard

### 8. Tuning Layer

`src/tuning/` implements multi-phase Optuna search for PPO.

Phases:

- phase 1: continuous dynamics parameters
  - `learning_rate`
  - `gae_lambda`
  - `gamma`
- phase 2a: rollout/update structure
  - `num_steps`
  - `update_epochs`
  - `num_minibatches`
- phase 2b: coefficients and regularization schedules
  - `clip_coef_start`
  - `ent_coef_start`
  - `vf_coef_start`
  - `max_grad_norm`
  - decay factors
- phase 3: architecture shape
  - convolution channel multiplier
  - FC size

`optuna_tuner.py` uses:

- SQLite-backed `RDBStorage`
- `TPESampler(constant_liar=True)`
- `MedianPruner`

It is explicitly designed for concurrent Slurm workers sharing one SQLite database.

## Main Modules

### `src/config/base_config.py`

Mutable runtime state.

Important nested configs:

- `EnvConfig`
- `PPOConfig`
- `LoggerConfig`
- `ArgsCarlaBEV`

### `src/config/experiment_loader.py`

The effective application service of the repo.

Key responsibilities:

- load CLI args with `tyro`
- apply experiment spec to runtime config
- save the resolved run config as YAML
- wrap standalone runs inside Optuna
- attach trial metadata and launch the trainer

### `src/config/studies/ppo_navigation.py`

Defines the main navigation study and a large experiment matrix over:

- action space
- traffic
- input type
- reward type
- curriculum mode
- FOV masking

### `src/config/studies/edge_case_scenarios.py`

Defines scenario-focused experiments over curated authored scene families:

- `jaywalk`
- `lead_brake`
- `red_light_runner`

Training protocols randomize variations; eval protocols cycle deterministically through authored scenarios.

### `src/trainers/ppo.py`

The main optimization implementation.

Notable design choices:

- reward normalization before PPO storage
- dynamic minibatch correction if batch is too small
- annealed LR
- decayed entropy/value/clip coefficients
- periodic evaluation during training
- Optuna pruning based on a custom safety score

The objective returned to Optuna is not raw return. It is a weighted safety-oriented composite:

- positive success rate
- penalty for collisions
- penalty for unfinished episodes
- small contribution from mean return

### `src/agents/cnn_ppo.py`

The canonical policy/value network definitions.

Two active variants:

- discrete convolutional PPO
- continuous convolutional PPO

Important decision:

- the continuous policy stores and scores raw Gaussian actions, then squashes to environment bounds with Jacobian correction

### `src/tuning/optuna_analysis.py`

Generates analysis artifacts:

- optimization history
- param importance
- parallel coordinates
- contour/slice plots
- EDF
- timeline
- custom learning curves
- threshold-time plots
- tabbed HTML dashboard

## Design Decisions

### Declarative Studies Over Hardcoded Experiments

The central decision is to move experiment identity into study modules. That makes experiment tables explicit, versionable, and easier to extend by domain.

### Reset Distribution As First-Class Configuration

Train/eval scenario selection is not buried inside trainers. Protocol specs define reset behavior, and samplers materialize it at runtime.

This is especially important for edge-case evaluation, where training and evaluation distributions differ intentionally.

### Manual Runs Still Use Optuna Storage

Standalone runs are enqueued into the same Optuna study database. This unifies:

- baselines
- reproductions
- tuning trials

Tradeoff:

- simpler experiment accounting
- but manual runs are no longer conceptually separate from hyperparameter search

### Multi-Seed Trial Averaging

Each Optuna objective phase runs several seeds and averages the resulting score. This favors robust configurations over lucky single-seed outcomes.

### HPC-Conscious SQLite Usage

The project is explicitly built for many workers contending on one SQLite database, using:

- connection timeout
- WAL mode in logger SQLite writes
- staggered Slurm startup
- `constant_liar` sampler

This is pragmatic rather than elegant, but consistent with the repository’s stated HPC use case.

## Runtime Artifacts

### `runs/`

Per-run output directory, typically:

- `config.yaml`
- TensorBoard event files
- saved PPO checkpoints
- evaluation `.npy` payloads
- benchmark CSVs

### `results/`

Cross-run artifacts:

- Optuna SQLite databases
- plot directories
- dashboards
- Slurm logs
- exported trial CSVs

## Supported Vs Legacy Areas

### Most Supported

- study registry
- PPO training
- PPO evaluation
- Optuna tuning
- scenario-catalog protocols

### Present But Likely Legacy / Incomplete

- DQN
- SAC
- MuZero
- shell scripts that assume fixed experiment IDs

Evidence:

- string-based dispatch through `exp_name`
- missing package references in DQN code
- separate APIs not reflected in `ArgsCarlaBEV`
- README drift

## Ambiguities / Open Questions

These are the main points where the code does not fully answer intent:

1. Is PPO the only maintained algorithm, or are DQN/SAC/MuZero expected to be revived?
2. Should agent/trainer dispatch be based on `args.algorithm` only, instead of string matching against `exp_name`?
3. Are logger benchmark threshold keys meant to use dots or underscores consistently?
4. Is `README.md` intentionally keeping historical command examples, or should it be treated as out of date?
5. Should authored scenarios be immutable assets, or part of normal experiment iteration?

## Recommended Mental Model

Treat this repo as three systems sharing one config object:

1. A study registry that declares what an experiment is.
2. A PPO training/evaluation engine that executes that declaration against `CarlaBEV`.
3. An Optuna infrastructure layer that scales the same experiment family locally or on HPC.

That model matches the actual code better than thinking of this repository as a generic multi-algorithm RL zoo.
