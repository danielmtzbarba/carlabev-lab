# World Model HPC Workflow

This page captures the practical workflow for running the LeWM proof of concept
on the TU Dresden `horse` nodes.

## Recommended storage split

Use:

- code and `.venv`: `/home/h6/dama898h/carlabev-lab`
- persistent artifacts: `/data/horse/ws/dama898h-carlabev`
- node-local temporary data: `/tmp/$SLURM_JOB_ID`

The repo resolves heavy artifact roots through:

- `CARLABEV_ARTIFACT_ROOT`
- `CARLABEV_RUNS_ROOT`
- `CARLABEV_RESULTS_ROOT`
- `CARLABEV_DATASETS_ROOT`

Typical setup:

```bash
export CARLABEV_ARTIFACT_ROOT=/data/horse/ws/dama898h-carlabev
```

## Staging datasets

Stage a dataset into node-local storage before probing, benchmarking, or
training:

```bash
uv run drl world-model stage \
  --path /data/horse/ws/dama898h-carlabev/datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_1/train/seed_2 \
  --tmp-root /tmp/$SLURM_JOB_ID \
  --dest-name carlabev-world-model/seed_2
```

You do not need to clean `/tmp` first if you are reusing the same destination
and want it refreshed. The staging command rewrites the staged dataset.

Staging logs:

- copy start and destination
- shard preparation start
- per-shard load timing
- prepared shard save events
- final shard count and byte count

## Prepared shard cache

By default, staging also prepares an uncompressed shard cache under:

```text
/tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2/.wm_cache/prepared_shards/
```

This cache stores arrays as `.npy` files plus per-shard manifests. The world-model
loader prefers these prepared shards over the original `.npz` files when the
cache is complete.

Why this matters:

- raw `.npz` shard loading was taking about `13s` per shard on the shared path
- repeated decompression became the main bottleneck
- prepared shards trade more disk space for much faster repeated access

If you want to prepare the cache in place on the `horse` workspace without
staging first, use:

```bash
uv run drl world-model prepare-cache \
  --path /data/horse/ws/dama898h-carlabev/datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_1/train/seed_2
```

## Sequence-window cache

Sequence-window indices are cached separately under:

```text
<dataset_root>/.wm_cache/
```

These cache files store valid training windows only, not observation payloads.
They are reused by:

- `world-model validate`
- `world-model benchmark`
- `world-model probe-loader`
- `world-model train`

## Loader probe

Use the loader probe to measure data-path performance before paying for model
forward/backward passes:

```bash
uv run drl world-model probe-loader \
  --run-name lewm-loader-probe-prepared-sweep \
  --data.dataset-paths /tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2 \
  --batch-sizes 16 32 64 128 \
  --chunk-lengths 4 8 \
  --num-workers-options 0 \
  --pin-memory-options True \
  --persistent-workers-options False \
  --prefetch-factors 2 \
  --warmup-batches 1 \
  --measure-batches 3 \
  --move-to-device \
  --device cuda
```

Per candidate, the probe logs and times:

- dataset/index reuse
- loader construction
- warmup batches
- measured batches
- host-to-device transfer time

The probe also writes machine-readable results under:

```text
runs/world_model/<run_name>/artifacts/loader_probe_results.{json,csv}
```

## Storage ablation

We measured the practical question that matters most for the loader path:

1. `horse + prepared shards`
2. `/tmp + prepared shards`

Setup:

- dataset:
  `PPO_NAVIGATION_DIFFICULTY / exp_1 / train / seed_2`
- batch size: `32`
- chunk length: `4`
- workers: `0`
- pin memory: `True`
- prefetch factor: `2`
- move to device: enabled
- device: `cuda`

Measured results:

- `horse + prepared shards`
  - `samples/s=18.63`
  - `first_batch=2.063s`
  - `measure_fetch=1714.361ms`
  - `measure_transfer=2.813ms`
  - `measure_total=1717.174ms`
  - `index_build=1690.460ms`
  - `chunk_build=341.920ms`
- `/tmp + prepared shards`
  - `samples/s=82.71`
  - `first_batch=0.665s`
  - `measure_fetch=383.510ms`
  - `measure_transfer=2.788ms`
  - `measure_total=386.297ms`
  - `index_build=532.832ms`
  - `chunk_build=22.525ms`

Takeaways:

- `/tmp + prepared shards` was about `4.4x` faster in steady-state throughput
- fetch time improved by about `4.5x`
- first-batch latency improved by about `3.1x`
- host-to-device transfer stayed basically unchanged, so the bottleneck is the
  filesystem path, not GPU transfer

Conclusion:

- prepared shards help, but they are not enough to make `horse` competitive for
  repeated random-access training
- staging to `/tmp/$SLURM_JOB_ID` remains the recommended workflow even after
  the shard cache exists on `horse`

To reproduce this comparison on a node, use:

```bash
DATASET_PATH=/data/horse/ws/dama898h-carlabev/datasets/world_model/lewm-ppo-difficulty-hpc/PPO_NAVIGATION_DIFFICULTY/exp_1/train/seed_2 \
DEST_NAME=carlabev-world-model/seed_2 \
RUN_PREFIX=wm-storage-exp1-seed2 \
BATCH_SIZES="32" \
CHUNK_LENGTHS="4" \
NUM_WORKERS=0 \
PIN_MEMORY=True \
PERSISTENT_WORKERS=False \
PREFETCH_FACTORS="2" \
WARMUP_BATCHES=1 \
MEASURE_BATCHES=3 \
DEVICE=cuda \
sh infra/slurm/world_model_storage_ablation.sh
```

## Benchmark

Use the full benchmark once loader settings are safe:

```bash
uv run drl world-model benchmark \
  --run-name lewm-bench-h100-opt \
  --data.dataset-paths /tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2 \
  --batch-sizes 32 64 128 256 \
  --chunk-lengths 4 8 \
  --data.num-workers 0 \
  --data.pin-memory True \
  --training.device cuda \
  --training.amp \
  --training.amp-dtype bfloat16 \
  --measure-batches 3
```

The benchmark runs real train steps and reports:

- status
- samples per second
- tokens per second
- first-batch latency
- peak CUDA memory

It also writes partial artifacts after each completed candidate so interrupted
runs still preserve useful results.

## H100 findings

The most useful H100 loader probe so far used:

- GPU: `NVIDIA H100 96GB`
- dataset: `PPO_NAVIGATION_DIFFICULTY / exp_1 / train / seed_2`
- observation shape: `(16, 96, 96)`
- encoder backend: `stable_pretraining_vit_hf`
- encoder size: `small`
- AMP: `bfloat16`

Best prepared-shard loader candidates observed:

- `chunk=4, batch=32, workers=0, pin_memory=True`
  - `samples/s=82.98`
  - `first_batch=0.524s`
  - `peak_mb=144.0`
- `chunk=4, batch=16, workers=0, pin_memory=True`
  - `samples/s=80.23`
  - `first_batch=0.427s`
  - `peak_mb=72.0`
- `chunk=8, batch=16, workers=0, pin_memory=True`
  - `samples/s=46.34`
  - `first_batch=0.336s`
  - `peak_mb=144.0`

Main conclusions:

- `chunk_length=4` is currently the best practical training baseline
- `batch_size=32` is the strongest loader-only configuration tested so far
- `num_workers > 0` was unstable on the tested node and not needed
- the biggest bottleneck was storage fetch time, not GPU memory
- staging to `/tmp` remains materially better than reading prepared shards
  directly from `horse`

## Recommended smoke-training command

```bash
uv run drl world-model train exp \
  --study-id PPO_NAVIGATION_DIFFICULTY \
  --exp-id 1 \
  --seed 2 \
  --dataset-path /tmp/$SLURM_JOB_ID/carlabev-world-model/seed_2
```

This uses the current study preset:

- batch size `32`
- chunk length `4`
- num workers `0`
- pin memory `True`
- AMP `bfloat16`
- device `cuda`
- epochs `5`

## Slurm helper

The repo also includes a collection helper:

```bash
sh infra/slurm/world_data_collection.sh
```

and the benchmark launcher:

```bash
sbatch infra/slurm/world_model_benchmark.sh
```

For a direct `horse` vs `/tmp` prepared-shard comparison, use:

```bash
sh infra/slurm/world_model_storage_ablation.sh
```

The practical loop on `horse` is:

1. collect on workspace storage
2. optionally prepare the shard cache in place on `horse`
3. stage to `/tmp/$SLURM_JOB_ID`
4. run `probe-loader`
5. run `benchmark`
6. launch training with the validated preset
