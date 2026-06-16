# World Model / LeWM PoC

This repo includes a world-model pipeline for a fast LeWorld Model style proof
of concept against the maintained PPO baseline.

The Phase 1 loop is:

1. collect offline transition datasets from study-owned PPO environments
2. validate dataset quality and training readiness
3. train a latent dynamics model on fixed-length sequence windows
4. evaluate checkpoints offline before integrating a controller loop

## Dataset layout

Collected datasets are stored under:

```text
datasets/world_model/<dataset_name>/<study_id>/exp_<exp_id>/<split>/seed_<seed>/
```

Each dataset directory contains shard files plus a `summary.json` manifest.

Transition shards currently include:

- `obs`
- `actions`
- `rewards`
- `dones`
- `terminated`
- `truncated`
- `next_obs`
- `env_index`
- `episode_id`
- `step_in_episode`
- `protocol_id`
- `reset_seed`
- `route_signature`
- `scene_signature`
- `straight_fraction`
- `left_turn_fraction`
- `right_turn_fraction`

The manifest records shard counts, transition totals, collection policy, and
PPO checkpoint provenance when applicable.

## Collection

Collect a random-policy dataset:

```bash
uv run drl world-model collect exp \
  --study-id PPO_NAVIGATION \
  --exp-id 26 \
  --seed 0 \
  --num-envs 14 \
  --total-transitions 100000 \
  --steps-per-shard 4096 \
  --policy random \
  --dataset-name lewm-random-100k \
  --split train \
  --device cpu
```

Collect a PPO-policy dataset from the latest compatible run:

```bash
uv run drl world-model collect exp \
  --study-id PPO_NAVIGATION_DIFFICULTY \
  --exp-id 1 \
  --seed 2 \
  --num-envs 14 \
  --total-transitions 100000 \
  --steps-per-shard 4096 \
  --policy ppo \
  --dataset-name lewm-ppo-difficulty-hpc \
  --split train \
  --device cuda
```

For `--policy ppo`, the collector resolves the latest matching checkpoint from
the experiment run metadata when `--checkpoint-path` is omitted. It also fails
fast if the checkpoint contract is incompatible with the requested environment
configuration.

Compatibility checks include:

- `study_id`
- `exp_id`
- algorithm family
- observation shape and semantic layout
- temporal fusion mode and stack depth
- action mode and action profile
- FOV configuration
- live action-space contract

## Inspect, summarize, validate

Inspect one shard:

```bash
uv run drl world-model inspect \
  --path datasets/world_model/lewm-random-100k/PPO_NAVIGATION/exp_26/train/seed_0
```

Summarize a whole dataset:

```bash
uv run drl world-model summary \
  --path datasets/world_model/lewm-random-100k/PPO_NAVIGATION/exp_26/train/seed_0
```

Validate training readiness:

```bash
uv run drl world-model validate \
  --paths datasets/world_model/lewm-random-100k/PPO_NAVIGATION/exp_26/train/seed_0
```

Validation focuses on:

- missing arrays or dtype mismatches
- empty or malformed shards
- episode-length statistics
- action balance
- route and scene diversity
- valid chunk-window counts for sequence training

Uniqueness reporting distinguishes episode-level diversity from transition-level
concentration, so long episodes do not trigger misleading route-diversity
warnings.

## Training contract

Phase 1 training uses fixed-length episode-aware windows and predicts the next
latent state from current observations plus action conditioning.

The trainer contract is:

- data input: sequence windows derived from collected shard directories
- action space: discrete action ids from the PPO navigation studies
- split expectation: train roots and validation windows are built from the same
  dataset root unless multiple roots are mixed later
- rollout chunk length: configurable, with `4` currently the most practical H100
  baseline

The trainer surface is structured as:

- `data.*`: dataset roots, chunk length, batch size, loader settings, cache control
- `model.*`: encoder backend, ViT size, latent dimensions
- `optimizer.*`: optimizer and gradient clipping
- `training.*`: epochs, device, AMP, compile, logging cadence, checkpointing

`training.device` defaults to `cuda`.

## Encoder baseline

The default encoder backend is `stable_pretraining_vit_hf`, which uses the
[`stable-pretraining`](https://github.com/galilai-group/stable-pretraining)
`vit_hf` construction path as the closest open-source baseline to the LeWM
encoder setup.

Current baseline encoder settings:

- backend: `stable_pretraining_vit_hf`
- size: `small`
- image size: `96`
- patch size: `8`
- hidden size: `384`
- layers: `12`
- attention heads: `6`
- MLP dim: `1536`

For BEV stacks with more than three channels, the adapter projects inputs to RGB
before the ViT and projects token features back to the configured latent size.

## Training commands

Raw config-style training:

```bash
uv run drl world-model train \
  --run-name lewm-random-phase1 \
  --data.dataset-paths datasets/world_model/lewm-random-100k/PPO_NAVIGATION/exp_26/train/seed_0 \
  --data.batch-size 32 \
  --data.chunk-length 4 \
  --data.num-workers 0 \
  --data.pin-memory True \
  --training.device cuda \
  --training.amp \
  --training.amp-dtype bfloat16 \
  --training.epochs 5
```

Study-aware training, aligned with the repo’s PPO study system:

```bash
uv run drl world-model train exp \
  --study-id PPO_NAVIGATION_DIFFICULTY \
  --exp-id 1 \
  --seed 2
```

The first built-in study preset targets `PPO_NAVIGATION_DIFFICULTY` and currently
uses the same parameters we validated on H100:

- dataset name: `lewm-ppo-difficulty-hpc`
- split: `train`
- batch size: `32`
- chunk length: `4`
- num workers: `0`
- pin memory: `True`
- epochs: `5`
- device: `cuda`
- AMP: `bfloat16`

When the dataset is already staged to local node storage, override the source:

```bash
uv run drl world-model train exp \
  --study-id PPO_NAVIGATION_DIFFICULTY \
  --exp-id 1 \
  --seed 2 \
  --dataset-path /tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2
```

## Checkpoint evaluation

Offline checkpoint evaluation computes train or validation losses from a saved
world-model checkpoint without resuming training.

```bash
uv run drl world-model eval-checkpoint \
  --checkpoint-path /data/horse/ws/dama898h-carlabev/runs/world_model/lewm-train-ppo-5epochs/checkpoints/world_model_best.pt \
  --dataset-paths /tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2 \
  --device cuda \
  --batch-size 32
```

Include the train split too when you want a direct train/val comparison:

```bash
uv run drl world-model eval-checkpoint \
  --checkpoint-path /data/horse/ws/dama898h-carlabev/runs/world_model/lewm-train-ppo-5epochs/checkpoints/world_model_best.pt \
  --dataset-paths /tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2 \
  --device cuda \
  --batch-size 32 \
  --include-train-split
```

Artifacts are written under:

```text
runs/world_model/<run_name>/
```

including:

- `checkpoints/world_model_best.pt`
- `checkpoints/world_model_final.pt`
- `artifacts/history.json`
- `artifacts/validation_report.json`
- `artifacts/eval_checkpoint.json`
- `config.json`

## Current baseline status

The current PPO-only smoke baseline on the H100 reached:

- `epoch=5/5`
- `train_loss=0.363`
- `val_loss=0.354`

That is enough to say the current trainer is learning and gives us a stable
Phase 1 baseline for the next LeWM control-loop work.
