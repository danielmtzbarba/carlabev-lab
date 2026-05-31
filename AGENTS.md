# AGENTS.md

## Purpose

This repository is a Python training lab for `CarlaBEV`, centered on:

- study-driven experiment selection
- PPO training and evaluation
- Optuna-based multi-phase tuning
- authored scenario evaluation for edge cases

The codebase is small enough to traverse fully, but not all modules are equally current. Treat the PPO + study registry path as the canonical implementation.

## First Read Order

When entering the project, read files in this order:

1. `README.md`
2. `pyproject.toml`
3. `train.py`
4. `src/config/experiment_loader.py`
5. `src/config/studies/models.py`
6. `src/config/studies/registry.py`
7. `src/config/studies/ppo_navigation.py`
8. `src/config/studies/edge_case_scenarios.py`
9. `src/config/reset_protocol.py`
10. `src/trainers/ppo.py`
11. `src/eval/eval_ppo.py`
12. `src/agents/__init__.py`
13. `src/agents/cnn_ppo.py`
14. `src/utils/logger.py`
15. `src/tuning/optuna_tuner.py`

That sequence reconstructs the real execution path from CLI to environment creation, training, evaluation, and tuning.

## Canonical Runtime Path

For current work, assume the supported path is:

`train.py` -> `load_experiment()` -> study registry lookup -> mutable `ArgsCarlaBEV` -> `CarlaBEV.envs.make_env()` -> trainer dispatch -> PPO trainer -> evaluation -> TensorBoard/SQLite/results

The canonical evaluation path is:

`eval.py` -> `load_experiment()` -> `src/eval/eval_ppo.py`

The canonical tuning path is:

`src/tuning/optuna_tuner.py` -> `phase1/2a/2b/3` -> `run_experiment()`

## High-Value Directories

### `src/config/`

This is the control plane.

- `base_config.py`: runtime dataclasses
- `experiment_loader.py`: CLI loading, study application, Optuna/manual run wrapper
- `reset_protocol.py`: train/eval reset option samplers
- `studies/`: declarative study definitions and registry
- `authored_scenarios.py`: maps scenario families to JSON scene files

If behavior differs by experiment, study, reset mode, or scenario family, start here.

### `src/trainers/`

This is the algorithm execution layer.

- `ppo.py` is the main implementation in active use
- `utils.py` contains curriculum schedules
- `dqn.py`, `sac.py`, and `muzero.py` exist, but are not aligned with the current config surface as cleanly as PPO

If the task is about training loops, rollout storage, loss terms, or evaluation cadence, start here.

### `src/agents/`

This is the model factory layer.

- `cnn_ppo.py` contains the active PPO backbones and actor/critic heads
- `__init__.py` chooses the agent implementation
- other algorithms are present but look older and less integrated

### `src/eval/`

- `eval_ppo.py` is the real evaluation entrypoint for the current architecture
- DQN evaluation code references old package paths and should not be treated as canonical

### `src/tuning/`

This is the Optuna orchestration layer.

- `optuna_tuner.py` owns study creation/loading and phase execution
- `phase1.py`, `phase2a.py`, `phase2b.py`, `phase3.py` mutate PPO sub-configs
- `optuna_analysis.py` generates HTML dashboards and plots from Optuna + SQLite logs

### `assets/scenes/`

Curated authored scenarios used by `EDGE_CASE_SCENARIOS`.

### `scripts/`

Primarily Slurm launchers for distributed Optuna workers.

## Current State: What To Trust

Trust these first:

- study registry and experiment spec model
- reset protocol sampling
- PPO trainer
- PPO evaluator
- Optuna phase orchestration
- TensorBoard + SQLite logging integration

Be cautious with these:

- `README.md` examples that mention `test.py`
- `run.sh` and `test.sh` because they hardcode older workflows
- DQN/SAC/MuZero paths, which compile but reference older assumptions and package names
- any logic that keys behavior off `args.exp_name` string matching instead of `args.algorithm`

## Known Drift / Ambiguities

Agents working in this repo should keep these in mind:

1. The repo depends on a local editable `CarlaBEV` package path from `pyproject.toml`. The environment implementation is intentionally outside this repo.
2. `README.md` says `uv run python test.py`, but the repo only contains `test.sh`.
3. `build_trainer()` dispatches by substring on algorithm name, while `build_agent()` dispatches mostly by substring on `exp_name`.
4. `dqn.py` and `src/eval/dqn_eval.py` reference `src.evals` / `src.envs`, which are not present here.
5. `phase2b.py` does not set `args.logging.db_path` explicitly, although `run_experiment()` later fills it in for active trials.
6. `src/utils/logger.py` writes benchmark keys like `time_to_reach_0.1`, but the SQLite insert reads `time_to_reach_0_1`; those names do not match.
7. `optuna_analysis.py` labels its theme helpers as “dark” while using a light palette; that is cosmetic drift, not architectural drift.

Do not silently “normalize” these unless the task is explicitly to clean them up.

## Safe Working Rules

1. Prefer editing the study-driven PPO path unless the task explicitly targets legacy algorithms.
2. Preserve the declarative split:
   - studies decide experiment combinations
   - reset protocols decide reset distributions
   - trainers decide rollout/optimization behavior
3. Keep environment-specific behavior out of trainers when it can live in protocol definitions.
4. If adding a new experiment family, extend `src/config/studies/` first, not ad hoc conditionals in trainers.
5. If changing evaluation semantics, update both `src/eval/eval_ppo.py` and the logging expectations in `src/utils/logger.py`.
6. If changing Optuna phases, keep parameter ownership by phase explicit.

## How To Traverse By Task

If asked about experiment identity:

- read `src/config/studies/*.py`
- then `src/config/experiment_loader.py`

If asked about how resets/scenarios are chosen:

- read `src/config/reset_protocol.py`
- then `src/config/authored_scenarios.py`
- then `assets/scenes/*.json`

If asked about PPO behavior:

- read `src/trainers/ppo.py`
- then `src/agents/cnn_ppo.py`
- then `src/eval/eval_ppo.py`

If asked about tuning:

- read `src/tuning/optuna_tuner.py`
- then phase files
- then `src/tuning/optuna_analysis.py`

If asked about logging/results:

- read `src/utils/logger.py`
- inspect `runs/` and `results/`

## Typical Commands

Setup:

```bash
uv sync
```

Train one study experiment:

```bash
uv run python train.py exp --study-id PPO_NAVIGATION --exp-id 26
```

Evaluate one study experiment:

```bash
uv run python eval.py exp --study-id PPO_NAVIGATION --exp-id 26
```

Run Optuna phase 1:

```bash
uv run python -m src.tuning.optuna_tuner --study-id PPO_NAVIGATION --exp-id 26 --phase 1
```

Generate Optuna analysis:

```bash
uv run python -m src.tuning.optuna_analysis --study-id PPO_NAVIGATION --exp-id 26 --top-k 5
```

## What Not To Assume

- Do not assume all algorithms are equally production-ready.
- Do not assume this repo contains the CarlaBEV environment internals.
- Do not assume shell helpers are authoritative over Python entrypoints.
- Do not assume README examples are fully current.

## If Clarification Is Needed

Ask the user these questions before large structural changes:

1. Is PPO the only maintained training path, or should DQN/SAC/MuZero be treated as supported too?
2. Should this repo continue to depend on a sibling local `carlabev-env` checkout, or is packaging/install behavior changing?
3. Are authored scenario JSONs considered stable public inputs, or should agents feel free to edit them during feature work?
