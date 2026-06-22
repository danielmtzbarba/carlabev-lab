from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

from src.carlabev_lab.scene_library.build import resolve_benchmark_scene_library_plan
from src.config.scene_benchmarks.registry import get_scene_benchmark_config
from src.utils.storage_paths import resolve_artifact_path


@dataclass(frozen=True)
class BuildShardTask:
    backbone_id: str
    scene_profile_id: str
    scene_split: str
    study_seed: int
    episodes_per_seed: int
    shard_path: str

    @property
    def label(self) -> str:
        return f"{self.scene_profile_id}:{self.scene_split}:seed_{self.study_seed}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a benchmark scene-library on one node by running one shard per prime seed, "
            "merging the shard databases, and running final library analysis."
        )
    )
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--prime-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes-per-seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--output-db", default=None)
    parser.add_argument("--include-train", action="store_true", default=False)
    parser.add_argument("--include-eval", action="store_true", default=False)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _effective_splits(*, include_train: bool, include_eval: bool) -> tuple[bool, bool]:
    return (
        include_train or not include_eval,
        include_eval or not include_train,
    )


def _default_workers(task_count: int) -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            cpu_budget = int(slurm_cpus)
        except ValueError:
            cpu_budget = os.cpu_count() or 1
    else:
        cpu_budget = os.cpu_count() or 1
    return max(1, min(task_count, cpu_budget))


def _default_shard_dir(benchmark_id: str) -> str:
    return f"assets/scene_libraries/shards/{benchmark_id}"


def _sanitize_backbone_id(backbone_id: str) -> str:
    return backbone_id.replace(":", ".")


def _build_tasks(
    *,
    benchmark_id: str,
    profiles: list[str] | None,
    prime_seeds: list[int] | None,
    episodes_per_seed: int | None,
    include_train: bool,
    include_eval: bool,
    shard_dir: str,
) -> tuple[list[BuildShardTask], str]:
    benchmark = get_scene_benchmark_config(benchmark_id)
    plans, default_prime_seeds, default_episodes = resolve_benchmark_scene_library_plan(
        benchmark_id,
        profiles=profiles,
        include_train=include_train,
        include_eval=include_eval,
    )
    selected_seeds = list(default_prime_seeds if prime_seeds is None else prime_seeds)
    selected_episodes = int(default_episodes if episodes_per_seed is None else episodes_per_seed)
    tasks: list[BuildShardTask] = []
    for plan in plans:
        profile_id = str(plan.request_kwargs["scene_profile_id"])
        scene_split = str(plan.request_kwargs["scene_split"])
        for study_seed in selected_seeds:
            shard_name = (
                f"{benchmark_id}.{_sanitize_backbone_id(plan.backbone_id)}.seed_{study_seed}.db"
            )
            tasks.append(
                BuildShardTask(
                    backbone_id=plan.backbone_id,
                    scene_profile_id=profile_id,
                    scene_split=scene_split,
                    study_seed=study_seed,
                    episodes_per_seed=selected_episodes,
                    shard_path=str(Path(shard_dir) / shard_name),
                )
            )
    return tasks, str(benchmark.scene_library.path)


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _build_command(benchmark_id: str, task: BuildShardTask) -> list[str]:
    command = [
        "uv",
        "run",
        "drl",
        "scene-library",
        "build",
        "--benchmark-id",
        benchmark_id,
        "--profiles",
        task.scene_profile_id,
        "--prime-seeds",
        str(task.study_seed),
        "--episodes-per-seed",
        str(task.episodes_per_seed),
        "--scene-library-path",
        task.shard_path,
    ]
    if task.scene_split == "train":
        command.append("--include-train")
    else:
        command.append("--include-eval")
    return command


def _merge_command(output_db: str, shard_dir: str) -> list[str]:
    return [
        "uv",
        "run",
        "drl",
        "scene-library",
        "merge",
        "--output",
        output_db,
        "--inputs",
        str(Path(shard_dir) / "*.db"),
    ]


def _analysis_commands(output_db: str) -> list[list[str]]:
    base = [
        "uv",
        "run",
        "carlabev",
        "scene",
        "analyze-library",
    ]
    return [
        [*base, "summary", "--scene-library-path", output_db],
        [*base, "quality", "--scene-library-path", output_db],
        [
            *base,
            "breakdown",
            "--scene-library-path",
            output_db,
            "--by",
            "scene-profile",
            "--by",
            "scene-split",
            "--by",
            "main-actor-role",
            "--by",
            "traffic-role-profile",
        ],
    ]


def _remove_existing(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _run_logged(command: Sequence[str], *, env: dict[str, str], cwd: Path):
    logger.info("Exec | {}", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _run_parallel_builds(
    *,
    benchmark_id: str,
    tasks: list[BuildShardTask],
    workers: int,
    cwd: Path,
    env: dict[str, str],
):
    logger.info(
        "Starting shard builds | benchmark {} | tasks {} | workers {}",
        benchmark_id,
        len(tasks),
        workers,
    )
    pending = list(tasks)
    running: dict[Future[None], BuildShardTask] = {}
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while pending or running:
            while pending and len(running) < workers:
                task = pending.pop(0)
                logger.info(
                    "Launch | {} | shard {}",
                    task.label,
                    resolve_artifact_path(task.shard_path),
                )
                future = executor.submit(
                    _run_logged,
                    _build_command(benchmark_id, task),
                    env=env,
                    cwd=cwd,
                )
                running[future] = task
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                future.result()
                logger.info("Done | {}", task.label)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    include_train, include_eval = _effective_splits(
        include_train=args.include_train,
        include_eval=args.include_eval,
    )
    shard_dir = args.shard_dir or _default_shard_dir(args.benchmark_id)
    tasks, default_output_db = _build_tasks(
        benchmark_id=args.benchmark_id,
        profiles=args.profiles,
        prime_seeds=args.prime_seeds,
        episodes_per_seed=args.episodes_per_seed,
        include_train=include_train,
        include_eval=include_eval,
        shard_dir=shard_dir,
    )
    output_db = args.output_db or default_output_db
    workers = args.workers or _default_workers(len(tasks))
    cwd = Path(__file__).resolve().parents[1]
    env = _command_env()

    shard_dir_path = Path(resolve_artifact_path(shard_dir))
    output_db_path = Path(resolve_artifact_path(output_db))

    if args.dry_run:
        logger.info(
            "Dry run | benchmark {} | tasks {} | workers {} | shard_dir {} | output {}",
            args.benchmark_id,
            len(tasks),
            workers,
            shard_dir_path,
            output_db_path,
        )
        for task in tasks:
            logger.info("Task | {} | {}", task.label, " ".join(_build_command(args.benchmark_id, task)))
        logger.info("Merge | {}", " ".join(_merge_command(output_db, shard_dir)))
        for command in _analysis_commands(output_db):
            logger.info("Analyze | {}", " ".join(command))
        return 0

    if args.fresh:
        logger.info("Fresh start | removing {}", shard_dir_path)
        _remove_existing(shard_dir_path)
        if not args.skip_merge:
            logger.info("Fresh start | removing {}", output_db_path)
            _remove_existing(output_db_path)

    shard_dir_path.mkdir(parents=True, exist_ok=True)
    output_db_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        _run_parallel_builds(
            benchmark_id=args.benchmark_id,
            tasks=tasks,
            workers=workers,
            cwd=cwd,
            env=env,
        )

    if not args.skip_merge:
        _run_logged(_merge_command(output_db, shard_dir), env=env, cwd=cwd)
        if not args.keep_shards:
            logger.info("Removing shard directory {}", shard_dir_path)
            _remove_existing(shard_dir_path)

    if not args.skip_analysis:
        for command in _analysis_commands(output_db):
            _run_logged(command, env=env, cwd=cwd)

    logger.info(
        "Complete | benchmark {} | output {}",
        args.benchmark_id,
        output_db_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
