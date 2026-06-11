from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from seed_scene_diag.analysis import generate_dataset
from seed_scene_diag.common import DEFAULT_MAP_ASSET_SIZE
from seed_scene_diag.visualization import render_from_artifacts
from rich.console import Console
from rich.table import Table


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Top-level run seeds to analyze.")
    parser.add_argument(
        "--prime-seeds-count",
        type=int,
        default=None,
        help="Generate the first N prime-number seeds. Cannot be combined with --seeds.",
    )
    parser.add_argument(
        "--prime-seeds-start",
        type=int,
        default=2,
        help="Lower bound used when generating prime-number seeds with --prime-seeds-count.",
    )
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
    parser.add_argument(
        "--ego-route-graph",
        choices=["full_vehicle", "right_lane", "left_lane"],
        default="full_vehicle",
        help="Planner graph used for ego-route sampling.",
    )
    parser.add_argument(
        "--route-profile",
        choices=["any", "mostly_straight", "single_left", "single_right", "multi_turn", "mixed"],
        default=None,
        help="Optional route-profile constraint passed to RandomNavigationReset.",
    )
    parser.add_argument(
        "--route-profile-mix",
        nargs="+",
        default=None,
        help="Weighted route-profile mix entries like 'mostly_straight=0.4 single_left=0.3 single_right=0.3'.",
    )
    parser.add_argument("--min-turns", type=int, default=None, help="Optional minimum accepted turn-count for ego routes.")
    parser.add_argument("--max-turns", type=int, default=None, help="Optional maximum accepted turn-count for ego routes.")
    parser.add_argument(
        "--max-route-attempts",
        type=int,
        default=None,
        help="Maximum route candidates to try per reset before failing constrained route generation.",
    )
    parser.add_argument(
        "--intersection-required",
        action="store_true",
        help="Require routes classified as intersection-like.",
    )
    parser.add_argument(
        "--intersection-forbidden",
        action="store_true",
        help="Reject routes classified as intersection-like.",
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


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    limit = int(value**0.5) + 1
    for factor in range(3, limit, 2):
        if value % factor == 0:
            return False
    return True


def _first_n_primes(count: int, start: int = 2) -> list[int]:
    if count <= 0:
        raise ValueError("--prime-seeds-count must be positive.")
    candidate = max(2, start)
    primes: list[int] = []
    while len(primes) < count:
        if _is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None and args.prime_seeds_count is not None:
        raise ValueError("Use either --seeds or --prime-seeds-count, not both.")
    if args.prime_seeds_count is not None:
        return _first_n_primes(args.prime_seeds_count, start=args.prime_seeds_start)
    if args.seeds is not None:
        return args.seeds
    return [0, 555, 9999]


def _resolve_route_profile_mix(args: argparse.Namespace) -> dict[str, float] | None:
    if args.route_profile_mix is None:
        return None
    mix: dict[str, float] = {}
    for item in args.route_profile_mix:
        if "=" not in item:
            raise ValueError(f"Invalid --route-profile-mix entry {item!r}. Use profile=weight.")
        profile, weight_text = item.split("=", 1)
        profile = profile.strip()
        if profile not in {"any", "mostly_straight", "single_left", "single_right", "multi_turn", "mixed"}:
            raise ValueError(f"Unknown route profile {profile!r} in --route-profile-mix.")
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"Invalid weight {weight_text!r} in --route-profile-mix.") from exc
        mix[profile] = weight
    return mix


def _resolve_intersection_required(args: argparse.Namespace) -> bool | None:
    if args.intersection_required and args.intersection_forbidden:
        raise ValueError("Use either --intersection-required or --intersection-forbidden, not both.")
    if args.intersection_required:
        return True
    if args.intersection_forbidden:
        return False
    return None


def _print_pair_summary(summary: dict) -> None:
    console = Console()
    table = Table(
        title="Seed Scene Diagnostics Summary",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Difficulty", justify="left")
    table.add_column("Seed", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Unique Scenes", justify="right")
    table.add_column("Unique Routes", justify="right")
    table.add_column("Scene Repeat %", justify="right")
    table.add_column("Route Repeat %", justify="right")
    table.add_column("Spawn Valid %", justify="right")
    table.add_column("Profile", justify="left")
    table.add_column("Profile Share %", justify="right")
    table.add_column("Turns", justify="right")
    table.add_column("Intersections %", justify="right")
    table.add_column("Straight %", justify="right")
    table.add_column("Left %", justify="right")
    table.add_column("Right %", justify="right")

    for pair in sorted(summary["pairs"].values(), key=lambda item: (item["difficulty_id"], item["seed"])):
        route_profile_counts = pair.get("route_profile_counts", {})
        dominant_profile = max(route_profile_counts, key=route_profile_counts.get) if route_profile_counts else "-"
        dominant_profile_share = (
            (route_profile_counts.get(dominant_profile, 0) / pair["num_samples"]) * 100.0
            if route_profile_counts and pair["num_samples"] > 0
            else 0.0
        )
        table.add_row(
            pair["difficulty_label"],
            str(pair["seed"]),
            str(pair["num_samples"]),
            str(pair["num_unique_scenes"]),
            str(pair.get("num_unique_routes", 0)),
            f"{pair['scene_repeat_ratio'] * 100:.1f}",
            f"{pair.get('route_repeat_ratio', 0.0) * 100:.1f}",
            f"{(pair['spawn_valid_rate'] or 0.0) * 100:.1f}",
            dominant_profile,
            f"{dominant_profile_share:.1f}",
            f"{pair.get('mean_turn_count', 0.0):.2f}",
            f"{(pair.get('intersection_like_rate') or 0.0) * 100:.1f}",
            f"{(pair['mean_straight_fraction'] or 0.0) * 100:.1f}",
            f"{(pair['mean_left_turn_fraction'] or 0.0) * 100:.1f}",
            f"{(pair['mean_right_turn_fraction'] or 0.0) * 100:.1f}",
        )
    console.print(table)

    if summary["pairs"]:
        aggregate = Table(
            title="Aggregate Seed Summary",
            show_header=True,
            header_style="bold cyan",
        )
        aggregate.add_column("Difficulty", justify="left")
        aggregate.add_column("Seeds", justify="right")
        aggregate.add_column("Mean Unique Routes", justify="right")
        aggregate.add_column("Mean Route Repeat %", justify="right")
        aggregate.add_column("Mean Turns", justify="right")
        aggregate.add_column("Mean Intersections %", justify="right")
        aggregate.add_column("Mean Straight %", justify="right")

        grouped: dict[str, list[dict]] = {}
        for pair in summary["pairs"].values():
            grouped.setdefault(pair["difficulty_label"], []).append(pair)

        for difficulty_label, items in sorted(grouped.items()):
            seed_count = len(items)
            mean_unique_routes = sum(item.get("num_unique_routes", 0) for item in items) / seed_count
            mean_route_repeat = sum(item.get("route_repeat_ratio", 0.0) for item in items) / seed_count
            mean_turns = sum(item.get("mean_turn_count", 0.0) or 0.0 for item in items) / seed_count
            mean_intersections = sum(item.get("intersection_like_rate", 0.0) or 0.0 for item in items) / seed_count
            mean_straight = sum(item.get("mean_straight_fraction", 0.0) or 0.0 for item in items) / seed_count
            aggregate.add_row(
                difficulty_label,
                str(seed_count),
                f"{mean_unique_routes:.1f}",
                f"{mean_route_repeat * 100:.1f}",
                f"{mean_turns:.2f}",
                f"{mean_intersections * 100:.1f}",
                f"{mean_straight * 100:.1f}",
            )
        console.print(aggregate)


def main(argv: list[str] | None = None) -> None:
    cli_argv = _coerce_legacy_full_mode(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(cli_argv)
    seeds = _resolve_seeds(args)
    route_profile_mix = _resolve_route_profile_mix(args)
    intersection_required = _resolve_intersection_required(args)

    if args.mode == "analyze":
        summary = generate_dataset(
            seeds=seeds,
            difficulty_ids=args.difficulty_ids,
            samples_per_seed=args.samples_per_seed,
            seed_mode=args.seed_mode,
            route_profile=args.route_profile,
            route_profile_mix=route_profile_mix,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            intersection_required=intersection_required,
            max_route_attempts=args.max_route_attempts,
            ego_route_graph=args.ego_route_graph,
            save_frames_per_pair=args.save_frames_per_pair,
            frame_size=args.frame_size,
            output_dir=args.output_dir,
            map_asset_size=args.map_asset_size,
        )
        _print_pair_summary(summary)
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
            seeds=seeds,
            difficulty_ids=args.difficulty_ids,
            samples_per_seed=args.samples_per_seed,
            seed_mode=args.seed_mode,
            route_profile=args.route_profile,
            route_profile_mix=route_profile_mix,
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            intersection_required=intersection_required,
            max_route_attempts=args.max_route_attempts,
            ego_route_graph=args.ego_route_graph,
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
        _print_pair_summary(summary)
        print(f"Saved diagnostics to {args.output_dir}")
        print(f"Pairs analyzed: {len(summary['pairs'])}")
        print(f"Pairs rendered: {result['pairs']}")
        return

    parser.error("Unknown mode")


if __name__ == "__main__":
    main()
