from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path
from typing import Any

from CarlaBEV.src.managers.scene_library import SceneLibrary
from loguru import logger

from src.utils.storage_paths import resolve_artifact_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge scene-library shard databases into one canonical SQLite database."
    )
    parser.add_argument("--output", required=True, help="Merged output .db path.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input shard database paths or glob patterns.",
    )
    parser.add_argument("--generator-version", default="role_traffic_v1")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _expand_input_paths(patterns: list[str], *, output_path: Path) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(str(resolve_artifact_path(pattern)))
        if not matches:
            candidate = Path(resolve_artifact_path(pattern))
            matches = [str(candidate)]
        for match in matches:
            path = Path(match).resolve()
            if path == output_path.resolve():
                continue
            if path in seen:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Shard database not found: {path}")
            seen.add(path)
            expanded.append(path)
    return sorted(expanded)


def _scene_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("PRAGMA table_info(scenes)").fetchall()
    return [str(row[1]) for row in rows if str(row[1]) != "scene_id"]


def _attached_scene_columns(conn: sqlite3.Connection, alias: str) -> list[str]:
    rows = conn.execute(f"PRAGMA {alias}.table_info(scenes)").fetchall()
    return [str(row[1]) for row in rows if str(row[1]) != "scene_id"]


def merge_scene_library_shards(
    *,
    output_path: str,
    input_patterns: list[str],
    generator_version: str = "role_traffic_v1",
) -> dict[str, Any]:
    output_db = Path(resolve_artifact_path(output_path)).resolve()
    output_db.parent.mkdir(parents=True, exist_ok=True)
    SceneLibrary(str(output_db), generator_version=generator_version)
    inputs = _expand_input_paths(input_patterns, output_path=output_db)
    if not inputs:
        raise ValueError("No shard databases matched the provided --inputs.")

    rows_scanned = 0
    rows_inserted = 0
    merged_from: list[str] = []
    with sqlite3.connect(output_db) as output_conn:
        output_columns = _scene_columns(output_conn)
        for index, input_db in enumerate(inputs):
            alias = f"shard_{index}"
            output_conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(input_db),))
            try:
                shard_count = int(
                    output_conn.execute(f"SELECT COUNT(*) FROM {alias}.scenes").fetchone()[0]
                )
                shard_columns = _attached_scene_columns(output_conn, alias)
                merge_columns = [column for column in output_columns if column in shard_columns]
                before_changes = output_conn.total_changes
                quoted_columns = ", ".join(merge_columns)
                output_conn.execute(
                    f"""
                    INSERT OR IGNORE INTO scenes ({quoted_columns})
                    SELECT {quoted_columns}
                    FROM {alias}.scenes
                    """
                )
                output_conn.commit()
                inserted = output_conn.total_changes - before_changes
                rows_scanned += shard_count
                rows_inserted += inserted
                merged_from.append(str(input_db))
                logger.info(
                    "Merge {} | rows {} | inserted {} | duplicates {}",
                    input_db,
                    shard_count,
                    inserted,
                    shard_count - inserted,
                )
            finally:
                output_conn.execute(f"DETACH DATABASE {alias}")

        total_rows = int(output_conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])

    return {
        "output_path": str(output_db),
        "input_paths": merged_from,
        "input_databases": len(merged_from),
        "rows_scanned": rows_scanned,
        "rows_inserted": rows_inserted,
        "rows_skipped_as_duplicates": rows_scanned - rows_inserted,
        "total_rows": total_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = merge_scene_library_shards(
        output_path=args.output,
        input_patterns=list(args.inputs),
        generator_version=args.generator_version,
    )
    if args.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        logger.info(
            "Merged {} shards into {} | scanned {} | inserted {} | duplicates {} | total {}",
            summary["input_databases"],
            summary["output_path"],
            summary["rows_scanned"],
            summary["rows_inserted"],
            summary["rows_skipped_as_duplicates"],
            summary["total_rows"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
