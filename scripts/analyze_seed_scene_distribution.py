from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seed_scene_diag.analysis import generate_dataset
from seed_scene_diag.common import DEFAULT_MAP_ASSET_SIZE
from seed_scene_diag.visualization import render_from_artifacts


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 555, 9999], help="Top-level run seeds to analyze.")
    parser.add_argument(
        "--difficulty-ids",
        nargs="+",
        default=["rt_no_traffic_v1", "rt_easy_v1", "rt_medium_v1", "rt_hard_v1"],
        help="Difficulty preset IDs to analyze.",
    )
    parser.add_argument("--samples-per-seed", type=int, default=1000, help="Number of reset scenes to sample for each seed+difficulty pair.")
    parser.add_argument(
        "--seed-mode",
        choices=["fixed", "incremental"],
        default="fixed",
        help="'fixed' reuses the same seed on every reset. 'incremental' uses seed + sample_index.",
    )
    parser.add_argument("--save-frames-per-pair", type=int, default=6, help="Number of representative spawn frames to save for each seed+difficulty pair.")
    parser.add_argument("--frame-size", type=int, default=128, help="Rendered frame size for spawn-frame captures.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/seed_scene_diagnostics"), help="Directory where summary artifacts will be written.")
    parser.add_argument("--map-asset-size", type=int, default=DEFAULT_MAP_ASSET_SIZE, help="Town01 raster size used for coverage rendering.")


def _add_visualization_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spawn-cluster-radius", type=float, default=16.0, help="Radius in map pixels used to merge nearby spawn locations into one cluster.")
    parser.add_argument("--spawn-top-k", type=int, default=10, help="Number of top spawn clusters to render and report.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize random-navigation scene distributions across seeds and difficulty presets."
    )
    subparsers = parser.add_subparsers(dest="mode")

    analyze_parser = subparsers.add_parser("analyze", help="Run the simulator pass and write reusable artifacts.")
    _add_shared_args(analyze_parser)

    visualize_parser = subparsers.add_parser("visualize", help="Render figures from saved CSV/JSON artifacts without rerunning the simulator.")
    visualize_parser.add_argument("--output-dir", type=Path, default=Path("results/seed_scene_diagnostics"), help="Directory containing summary.json and per-pair point CSVs.")
    visualize_parser.add_argument("--map-asset-size", type=int, default=DEFAULT_MAP_ASSET_SIZE, help="Town01 raster size used for coverage rendering.")
    _add_visualization_args(visualize_parser)

    full_parser = subparsers.add_parser("full", help="Run analysis and visualization in one pass.")
    _add_shared_args(full_parser)
    _add_visualization_args(full_parser)

    return parser


def _coerce_legacy_full_mode(argv: list[str]) -> list[str]:
    if not argv:
        return ["full"]
    if argv[0] in {"analyze", "visualize", "full", "-h", "--help"}:
        return argv
    return ["full", *argv]


def main(argv: list[str] | None = None) -> None:
    cli_argv = _coerce_legacy_full_mode(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(cli_argv)

    if args.mode == "analyze":
        summary = generate_dataset(
            seeds=args.seeds,
            difficulty_ids=args.difficulty_ids,
            samples_per_seed=args.samples_per_seed,
            seed_mode=args.seed_mode,
            save_frames_per_pair=args.save_frames_per_pair,
            frame_size=args.frame_size,
            output_dir=args.output_dir,
            map_asset_size=args.map_asset_size,
        )
        print(f"Saved analysis artifacts to {args.output_dir}")
        print(f"Pairs analyzed: {len(summary['pairs'])}")
        return

    if args.mode == "visualize":
        result = render_from_artifacts(
            output_dir=args.output_dir,
            map_asset_size=args.map_asset_size,
            spawn_cluster_radius=args.spawn_cluster_radius,
            spawn_top_k=args.spawn_top_k,
        )
        print(f"Rendered figures in {args.output_dir}")
        print(f"Pairs rendered: {result['pairs']}")
        print(f"Difficulties rendered: {result['difficulties']}")
        return

    if args.mode == "full":
        summary = generate_dataset(
            seeds=args.seeds,
            difficulty_ids=args.difficulty_ids,
            samples_per_seed=args.samples_per_seed,
            seed_mode=args.seed_mode,
            save_frames_per_pair=args.save_frames_per_pair,
            frame_size=args.frame_size,
            output_dir=args.output_dir,
            map_asset_size=args.map_asset_size,
        )
        result = render_from_artifacts(
            output_dir=args.output_dir,
            map_asset_size=args.map_asset_size,
            spawn_cluster_radius=args.spawn_cluster_radius,
            spawn_top_k=args.spawn_top_k,
        )
        print(f"Saved diagnostics to {args.output_dir}")
        print(f"Pairs analyzed: {len(summary['pairs'])}")
        print(f"Pairs rendered: {result['pairs']}")
        return

    parser.error("Unknown mode")


if __name__ == "__main__":
    main()
