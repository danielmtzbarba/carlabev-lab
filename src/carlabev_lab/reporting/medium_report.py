from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import tyro

from CarlaBEV.config import EnvConfig, RandomNavigationReset, build_random_navigation_options
from CarlaBEV.envs.carlabev import CarlaBEV
from CarlaBEV.envs.fov import FovRenderSpec, FovRenderer
from CarlaBEV.tools.visualize_semantic_channels import _semantic_rows
from CarlaBEV.tools.visualize_vehicle_temporal_fusion import _build_variants, _draw_variant_grid
from CarlaBEV.wrappers.rgb_to_semantic import rgb_to_semantic_mask, vehicle_channel_index


@dataclass
class Args:
    output_dir: str = "docs"
    map_name: str = "Town01"
    size: int = 128
    route_extent: str = "medium"
    route_dist_range: tuple[int, int] = (50, 130)
    speed_profile: str = "medium"
    num_vehicles: int = 4
    num_vehicles_near_ego: int = 4
    traffic_role_profile: str = "mix"
    guaranteed_candidate_role: str = "mix"
    semantic_mask_ch: str = "4-class"
    history_frames: int = 3
    warmup_steps: int = 6
    max_scene_attempts: int = 64
    min_vehicle_pixels: int = 40
    seed_start: int = 0
    figure_dpi: int = 180
    interpolation: str = "nearest"
    weights: tuple[float, float, float] = (1.0, 0.5, 0.25)
    max_cols: int = 4


@dataclass
class CapturedScene:
    env: CarlaBEV
    rgb_history: np.ndarray
    center_frame: np.ndarray
    vehicle_pixels: int
    seed: int
    num_vehicles: int
    route_length: int


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("@", "")
        .replace("/", "_")
    )


def _save_panel_image(plt, path: Path, image: np.ndarray, *, is_rgb: bool, interpolation: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=180, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if is_rgb:
        ax.imshow(image, interpolation=interpolation)
    else:
        ax.imshow(image, cmap="magma", vmin=0.0, vmax=1.0, interpolation=interpolation)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _build_env(args: Args) -> CarlaBEV:
    cfg = EnvConfig(
        size=args.size,
        map_name=args.map_name,
        obs_mode="bev_rgb",
        render_mode="rgb_array",
        action_mode="discrete",
        ego_anchor_x_frac=0.5,
        ego_anchor_y_frac=0.5,
    )
    return CarlaBEV(cfg)


def _medium_reset_options(args: Args) -> dict[str, object]:
    return build_random_navigation_options(
        RandomNavigationReset(
            route_extent=args.route_extent,
            route_dist_range=args.route_dist_range,
            speed_profile=args.speed_profile,
            num_vehicles=args.num_vehicles,
            num_vehicles_near_ego=args.num_vehicles_near_ego,
            traffic_role_profile=args.traffic_role_profile,
            guaranteed_candidate_role=args.guaranteed_candidate_role,
        )
    )


def _capture_scene(args: Args) -> CapturedScene:
    vehicle_idx = vehicle_channel_index(args.semantic_mask_ch)
    options = _medium_reset_options(args)
    best_scene: CapturedScene | None = None

    for seed in range(args.seed_start, args.seed_start + args.max_scene_attempts):
        env = _build_env(args)
        try:
            obs, _ = env.reset(seed=seed, options=options)
            for _ in range(args.warmup_steps):
                obs, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    obs, _ = env.reset(seed=seed, options=options)
                    break

            history: list[np.ndarray] = []
            history.append(np.asarray(obs).copy())
            while len(history) < args.history_frames:
                obs, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    obs, _ = env.reset(seed=seed, options=options)
                history.append(np.asarray(obs).copy())

            rgb_history = np.stack(history, axis=0)
            center_frame = rgb_history[-1].copy()
            semantic = rgb_to_semantic_mask(center_frame, mode=args.semantic_mask_ch)
            vehicle_pixels = int(semantic[vehicle_idx].sum())
            if vehicle_pixels >= args.min_vehicle_pixels:
                return CapturedScene(
                    env=env,
                    rgb_history=rgb_history,
                    center_frame=center_frame,
                    vehicle_pixels=vehicle_pixels,
                    seed=seed,
                    num_vehicles=int(getattr(env, "num_vehicles", 0)),
                    route_length=int(getattr(env, "len_ego_route", 0)),
                )
            candidate = CapturedScene(
                env=env,
                rgb_history=rgb_history,
                center_frame=center_frame,
                vehicle_pixels=vehicle_pixels,
                seed=seed,
                num_vehicles=int(getattr(env, "num_vehicles", 0)),
                route_length=int(getattr(env, "len_ego_route", 0)),
            )
            if best_scene is None or candidate.vehicle_pixels > best_scene.vehicle_pixels:
                if best_scene is not None:
                    best_scene.env.close()
                best_scene = candidate
                continue
        except Exception:
            env.close()
            raise
        env.close()

    if best_scene is not None:
        print(
            "Falling back to the best available medium scene "
            f"(seed={best_scene.seed}, vehicle_pixels={best_scene.vehicle_pixels})."
        )
        return best_scene

    raise RuntimeError("Failed to capture any medium scene.")


def _set_anchor_and_render(env: CarlaBEV, *, y_frac: float) -> tuple[np.ndarray, tuple[int, int]]:
    spec = FovRenderSpec(
        output_size=env.cfg.size,
        ego_anchor_x_frac=0.5,
        ego_anchor_y_frac=y_frac,
        mask_fov=env.map.mask_fov,
    )
    env.map.fov_renderer = FovRenderer(spec)
    env.map._fov_surface = pygame.Surface(env.map.fov_renderer.output_resolution)
    env.map.crop_resolution = env.map.fov_renderer.crop_resolution
    env.map.camera.CROP_W, env.map.camera.CROP_H = env.map.crop_resolution
    env.map.camera.CONST = pygame.math.Vector2(
        -env.map.camera.CROP_W / 2,
        -env.map.camera.CROP_H / 2,
    )
    env.map.camera.scroll()
    env.map.draw_fov()
    frame = np.asarray(env.render()).copy()
    return frame, env.map.fov_renderer.anchor_px


def _save_rgb_frame(plt, output_dir: Path, scene: CapturedScene, args: Args) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=args.figure_dpi, constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.imshow(scene.center_frame, interpolation=args.interpolation)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        (
            "Representative medium BEV frame\n"
            f"seed={scene.seed} | vehicles={scene.num_vehicles} | "
            f"route_len={scene.route_length} | vehicle_pixels={scene.vehicle_pixels}"
        ),
        fontsize=11,
    )
    path = output_dir / "medium_scene_representative_rgb.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    rgb_dir = output_dir / "representative_rgb"
    _save_panel_image(
        plt,
        rgb_dir / "current_rgb.png",
        scene.center_frame,
        is_rgb=True,
        interpolation=args.interpolation,
    )


def _save_semantic_modes(plt, output_dir: Path, scene: CapturedScene, args: Args) -> None:
    rows = _semantic_rows(scene.center_frame)
    max_cols = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(
        len(rows),
        max_cols,
        figsize=(max_cols * 3.2, len(rows) * 3.0),
        dpi=args.figure_dpi,
        squeeze=False,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle("Semantic observation modes on the same medium-traffic scene", fontsize=13)

    for row_idx, (row_name, images) in enumerate(rows):
        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if col_idx >= len(images):
                ax.axis("off")
                continue

            image, label = images[col_idx]
            if row_name == "raw_rgb":
                ax.imshow(image, interpolation="bilinear")
            else:
                ax.imshow(image, cmap="magma", vmin=0.0, vmax=1.0, interpolation=args.interpolation)
            ax.set_title(label, fontsize=10, pad=8)
        axes[row_idx][0].set_ylabel(row_name, rotation=0, labelpad=36, va="center", fontsize=11)

    path = output_dir / "medium_scene_semantic_modes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    semantic_dir = output_dir / "semantic_modes"
    for row_name, images in rows:
        row_dir = semantic_dir / _slug(row_name)
        for image, label in images:
            _save_panel_image(
                plt,
                row_dir / f"{_slug(label)}.png",
                image,
                is_rgb=(row_name == "raw_rgb"),
                interpolation="bilinear" if row_name == "raw_rgb" else args.interpolation,
            )


def _save_anchor_modes(plt, output_dir: Path, scene: CapturedScene, args: Args) -> None:
    center_frame, center_anchor = _set_anchor_and_render(scene.env, y_frac=0.5)
    lookahead_frame, lookahead_anchor = _set_anchor_and_render(scene.env, y_frac=0.75)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=args.figure_dpi, constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("FOV anchor comparison on the same medium-traffic world state", fontsize=13)
    panels = (
        ("center", center_frame, center_anchor),
        ("lookahead_75", lookahead_frame, lookahead_anchor),
    )
    for ax, (title, frame, anchor) in zip(axes, panels, strict=True):
        ax.imshow(frame, interpolation=args.interpolation)
        ax.scatter([anchor[0]], [anchor[1]], s=90, c="#ffcc00", edgecolors="black", linewidths=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} | ego anchor={anchor}", fontsize=11)

    path = output_dir / "medium_scene_anchor_modes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    anchor_dir = output_dir / "anchor_modes"
    _save_panel_image(
        plt,
        anchor_dir / "center" / "frame.png",
        center_frame,
        is_rgb=True,
        interpolation=args.interpolation,
    )
    _save_panel_image(
        plt,
        anchor_dir / "lookahead_75" / "frame.png",
        lookahead_frame,
        is_rgb=True,
        interpolation=args.interpolation,
    )


def _save_temporal_fusion_modes(plt, output_dir: Path, scene: CapturedScene, args: Args) -> None:
    variants = _build_variants(scene.rgb_history, args)
    fig = plt.figure(
        figsize=(13.5, 13.0),
        dpi=args.figure_dpi,
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Temporal fusion post-processing on a medium-traffic scene\n"
        "Rows: RGB history, stacked semantic channels, explicit vehicle history, weighted vehicle history",
        fontsize=13,
    )
    subfigs = fig.subfigures(len(variants), 1)
    if not isinstance(subfigs, np.ndarray):
        subfigs = np.array([subfigs])

    for subfig, (row_name, images, labels) in zip(subfigs, variants, strict=True):
        _draw_variant_grid(subfig, row_name, images, labels, args)

    path = output_dir / "medium_scene_temporal_fusion_modes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    fusion_dir = output_dir / "temporal_fusion"
    for row_name, images, labels in variants:
        row_dir = fusion_dir / _slug(row_name)
        is_rgb = row_name == "rgb_history"
        for image, label in zip(images, labels, strict=True):
            _save_panel_image(
                plt,
                row_dir / f"{_slug(label)}.png",
                image,
                is_rgb=is_rgb,
                interpolation=args.interpolation,
            )


def main(args: Args) -> None:
    if args.history_frames != 3:
        raise ValueError("This report figure script expects history_frames=3.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate the report figures.") from exc

    scene = _capture_scene(args)
    try:
        _save_rgb_frame(plt, output_dir, scene, args)
        _save_semantic_modes(plt, output_dir, scene, args)
        _save_anchor_modes(plt, output_dir, scene, args)
        _save_temporal_fusion_modes(plt, output_dir, scene, args)
    finally:
        scene.env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
