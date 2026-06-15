from __future__ import annotations

from dataclasses import dataclass

import tyro

from src.utils.common_logging import configure_logging
from src.world_model.staging import stage_dataset_to_tmp


@dataclass
class StageDatasetArgs:
    path: str
    tmp_root: str | None = None
    dest_name: str | None = None
    overwrite: bool = True
    prepare_shards: bool = True


def main() -> None:
    configure_logging()
    args = tyro.cli(StageDatasetArgs)
    result = stage_dataset_to_tmp(
        args.path,
        tmp_root=args.tmp_root,
        dest_name=args.dest_name,
        overwrite=args.overwrite,
        prepare_shards=args.prepare_shards,
    )
    print(f"Staged dataset to {result.staged_dir}", flush=True)
    print(f"Source dataset: {result.source_dir}", flush=True)
    print(f"Shards: {result.shard_count}", flush=True)
    print(f"Prepared shards: {result.prepared_shards}", flush=True)
    print(f"Bytes: {result.total_bytes}", flush=True)
    return None


if __name__ == "__main__":
    main()
