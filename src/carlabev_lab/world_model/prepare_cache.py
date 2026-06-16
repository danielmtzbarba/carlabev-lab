from __future__ import annotations

from dataclasses import dataclass

import tyro

from src.utils.common_logging import configure_logging
from src.world_model.staging import prepare_dataset_shard_cache


@dataclass
class PrepareCacheArgs:
    path: str


def main() -> None:
    configure_logging()
    args = tyro.cli(PrepareCacheArgs)
    result = prepare_dataset_shard_cache(args.path)
    print(f"Prepared shard cache in {result.dataset_dir}", flush=True)
    print(f"Shards: {result.shard_count}", flush=True)
    print(f"Prepared shards: {result.prepared_shards}", flush=True)


if __name__ == "__main__":
    main()
