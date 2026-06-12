from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .common import (
    COMPARISON_HEADER,
    COMPARISON_MARGIN,
    DEFAULT_MAP_ASSET_SIZE,
    DIFFICULTY_LABELS,
    HEATMAP_ALPHA,
    LABEL_COLOR,
    LEGEND_HEIGHT_ROUTE,
    LEGEND_HEIGHT_SPAWN,
    LEGEND_TEXT_COLOR,
    LEGEND_WIDTH,
    PANEL_BACKGROUND,
    ROUTE_HEATMAP_SIGMA,
    SPAWN_CLUSTER_COLORS,
    SUBTITLE_COLOR,
    TITLE_COLOR,
    SpawnCluster,
    load_font,
    load_points_csv,
    town_map_asset_path,
)


def _base_map(map_name: str, map_asset_size: int) -> Image.Image:
    return Image.open(town_map_asset_path(map_name, map_asset_size)).convert("RGBA")


def _heat_color(normalized: float) -> tuple[int, int, int, int]:
    t = float(np.clip(normalized, 0.0, 1.0))
    if t < 0.35:
        local = t / 0.35
        color = (
            int(12 + 18 * local),
            int(108 + 88 * local),
            int(156 + 48 * local),
            int(35 + HEATMAP_ALPHA * (0.35 + 0.25 * local)),
        )
    elif t < 0.7:
        local = (t - 0.35) / 0.35
        color = (
            int(30 + 210 * local),
            int(196 - 28 * local),
            int(204 - 150 * local),
            int(110 + HEATMAP_ALPHA * (0.30 + 0.25 * local)),
        )
    else:
        local = (t - 0.7) / 0.3
        color = (
            int(240 + 15 * local),
            int(168 - 90 * local),
            int(54 - 34 * local),
            int(170 + HEATMAP_ALPHA * (0.18 + 0.17 * local)),
        )
    return tuple(int(channel) for channel in color)


def _make_density_map(points: list[tuple[float, float]], *, width: int, height: int) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float32)
    for x_raw, y_raw in points:
        x = int(np.clip(round(x_raw), 0, width - 1))
        y = int(np.clip(round(y_raw), 0, height - 1))
        density[y, x] += 1.0
    return density


def _route_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    transformed = np.log1p(values)
    vmax = float(np.percentile(transformed[transformed > 0], 99.5))
    if vmax <= 0.0:
        vmax = float(transformed.max())
    if vmax <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(transformed / vmax, 0.0, 1.0).astype(np.float32)


def top_spawn_clusters(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    merge_radius: float = 16.0,
    top_k: int = 10,
) -> list[SpawnCluster]:
    if not points:
        return []

    counts: dict[tuple[int, int], int] = defaultdict(int)
    for x_raw, y_raw in points:
        x = int(np.clip(round(x_raw), 0, width - 1))
        y = int(np.clip(round(y_raw), 0, height - 1))
        counts[(x, y)] += 1

    candidates = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    clusters: list[dict[str, float]] = []
    radius_sq = merge_radius * merge_radius

    for (x, y), count in candidates:
        assigned = False
        for cluster in clusters:
            dx = cluster["x"] - x
            dy = cluster["y"] - y
            if dx * dx + dy * dy <= radius_sq:
                total = cluster["count"] + count
                cluster["x"] = (cluster["x"] * cluster["count"] + x * count) / total
                cluster["y"] = (cluster["y"] * cluster["count"] + y * count) / total
                cluster["count"] = total
                assigned = True
                break
        if not assigned:
            clusters.append({"x": float(x), "y": float(y), "count": float(count)})

    clusters = sorted(clusters, key=lambda cluster: cluster["count"], reverse=True)[:top_k]
    total_points = max(len(points), 1)
    return [
        SpawnCluster(
            rank=index + 1,
            center_x=cluster["x"],
            center_y=cluster["y"],
            count=int(cluster["count"]),
            share=float(cluster["count"]) / total_points,
        )
        for index, cluster in enumerate(clusters)
    ]


def render_spawn_cluster_map(
    points: list[tuple[float, float]],
    *,
    map_name: str,
    map_asset_size: int,
    merge_radius: float,
    top_k: int,
) -> tuple[Image.Image, list[SpawnCluster]]:
    base_map = _base_map(map_name, map_asset_size)
    width, height = base_map.size
    clusters = top_spawn_clusters(points, width=width, height=height, merge_radius=merge_radius, top_k=top_k)
    draw = ImageDraw.Draw(base_map)
    badge_font = load_font(16)
    for cluster in clusters:
        color = SPAWN_CLUSTER_COLORS[(cluster.rank - 1) % len(SPAWN_CLUSTER_COLORS)]
        x = int(round(cluster.center_x))
        y = int(round(cluster.center_y))
        radius = 10 + min(cluster.count // 8, 10)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(24, 28, 32, 255), width=2)
        label = str(cluster.rank)
        bbox = draw.textbbox((0, 0), label, font=badge_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x - text_w / 2, y - text_h / 2 - 1), label, font=badge_font, fill=(20, 24, 28, 255))
    return base_map, clusters


def render_route_heatmap(points: list[tuple[float, float]], *, map_name: str, map_asset_size: int) -> Image.Image:
    base_map = _base_map(map_name, map_asset_size)
    width, height = base_map.size
    density = _make_density_map(points, width=width, height=height)
    if float(density.max()) <= 0.0:
        return base_map
    density_img = Image.fromarray(np.clip(density / max(float(density.max()), 1.0) * 255.0, 0, 255).astype(np.uint8))
    density_img = density_img.filter(ImageFilter.GaussianBlur(radius=ROUTE_HEATMAP_SIGMA))
    smoothed = _route_normalize(np.asarray(density_img, dtype=np.float32))
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    active = smoothed > 0.03
    for y, x in np.argwhere(active):
        overlay[y, x] = _heat_color(smoothed[y, x])
    return Image.alpha_composite(base_map, Image.fromarray(overlay))


def _annotate_image(image: Image.Image, *, title: str, subtitle: str, labels: list[str], map_width: int) -> None:
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    subtitle_font = load_font(16)
    label_font = load_font(18)
    draw.text((COMPARISON_MARGIN, 14), title, font=title_font, fill=TITLE_COLOR)
    draw.text((COMPARISON_MARGIN, 52), subtitle, font=subtitle_font, fill=SUBTITLE_COLOR)
    for index, label in enumerate(labels):
        x0 = COMPARISON_MARGIN + index * (map_width + COMPARISON_MARGIN)
        draw.text((x0, 88), label, font=label_font, fill=LABEL_COLOR)


def _draw_legend(
    image: Image.Image,
    *,
    anchor_x: int,
    anchor_y: int,
    coverage_label: str,
    spawn_clusters: list[SpawnCluster] | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    title_font = load_font(17)
    body_font = load_font(13)
    legend_height = LEGEND_HEIGHT_SPAWN if coverage_label == "spawn" else LEGEND_HEIGHT_ROUTE
    box = (anchor_x, anchor_y, anchor_x + LEGEND_WIDTH, anchor_y + legend_height)
    draw.rounded_rectangle(box, radius=12, fill=(250, 246, 238, 230), outline=(205, 192, 175, 255), width=1)
    draw.text((anchor_x + 14, anchor_y + 10), "Density legend", font=title_font, fill=LEGEND_TEXT_COLOR)
    if coverage_label == "spawn":
        draw.text((anchor_x + 14, anchor_y + 30), "Top repeated start zones", font=body_font, fill=SUBTITLE_COLOR)
        line_y = anchor_y + 52
        for cluster in (spawn_clusters or [])[:10]:
            color = SPAWN_CLUSTER_COLORS[(cluster.rank - 1) % len(SPAWN_CLUSTER_COLORS)]
            draw.rounded_rectangle((anchor_x + 14, line_y, anchor_x + 28, line_y + 14), radius=4, fill=color)
            text = f"{cluster.rank}. n={cluster.count} ({cluster.share * 100:.1f}%)"
            draw.text((anchor_x + 36, line_y - 1), text, font=body_font, fill=LEGEND_TEXT_COLOR)
            line_y += 16
    else:
        draw.text((anchor_x + 14, anchor_y + 30), "Low activity          High activity", font=body_font, fill=SUBTITLE_COLOR)
        bar_x0 = anchor_x + 14
        bar_y0 = anchor_y + 48
        bar_width = LEGEND_WIDTH - 28
        for dx in range(bar_width):
            t = dx / max(bar_width - 1, 1)
            draw.line(((bar_x0 + dx, bar_y0), (bar_x0 + dx, bar_y0 + 10)), fill=_heat_color(t), width=1)
        draw.text((anchor_x + 14, bar_y0 + 16), "Route corridor use", font=body_font, fill=LEGEND_TEXT_COLOR)


def save_single_panel(
    *,
    output_path: Path,
    map_image: Image.Image,
    difficulty_id: str,
    seed: int,
    samples_per_seed: int,
    coverage_label: str,
    spawn_clusters: list[SpawnCluster] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = Image.new(
        "RGBA",
        (map_image.size[0] + COMPARISON_MARGIN * 2, map_image.size[1] + COMPARISON_HEADER + COMPARISON_MARGIN),
        PANEL_BACKGROUND,
    )
    panel.alpha_composite(map_image, (COMPARISON_MARGIN, COMPARISON_HEADER))
    title = f"{DIFFICULTY_LABELS.get(difficulty_id, difficulty_id)} · Seed {seed} · {coverage_label.title()} Coverage"
    if coverage_label == "spawn":
        subtitle = f"{samples_per_seed} sampled scenes. Numbered markers show the most repeated spawn zones over Town01."
    else:
        subtitle = f"{samples_per_seed} sampled scenes. Colors encode repeated occupancy density over Town01."
    _annotate_image(panel, title=title, subtitle=subtitle, labels=[], map_width=map_image.size[0])
    _draw_legend(
        panel,
        anchor_x=panel.size[0] - LEGEND_WIDTH - COMPARISON_MARGIN,
        anchor_y=18,
        coverage_label=coverage_label,
        spawn_clusters=spawn_clusters,
    )
    panel.save(output_path)


def save_comparison_panel(
    *,
    output_path: Path,
    difficulty_id: str,
    coverage_label: str,
    rendered: list[tuple[int, Image.Image]],
    samples_per_seed: int,
    legend_clusters: list[SpawnCluster] | None = None,
) -> None:
    if not rendered:
        return
    map_width, map_height = rendered[0][1].size
    panel_width = COMPARISON_MARGIN * (len(rendered) + 1) + map_width * len(rendered)
    panel_height = COMPARISON_HEADER + COMPARISON_MARGIN * 2 + map_height
    panel = Image.new("RGBA", (panel_width, panel_height), PANEL_BACKGROUND)
    title = f"{DIFFICULTY_LABELS.get(difficulty_id, difficulty_id)} · {coverage_label.title()} Coverage by Seed"
    if coverage_label == "spawn":
        subtitle = (
            f"Town01 top spawn clusters from {samples_per_seed} sampled scenes per seed. "
            f"Compare repeated start zones, not score."
        )
    else:
        subtitle = (
            f"Town01 occupancy density from {samples_per_seed} sampled scenes per seed. "
            f"Compare geographic concentration, not score."
        )
    _annotate_image(panel, title=title, subtitle=subtitle, labels=[f"Seed {seed}" for seed, _ in rendered], map_width=map_width)
    for index, (_, image) in enumerate(rendered):
        x0 = COMPARISON_MARGIN + index * (map_width + COMPARISON_MARGIN)
        panel.alpha_composite(image, (x0, COMPARISON_HEADER))
    _draw_legend(
        panel,
        anchor_x=panel.size[0] - LEGEND_WIDTH - COMPARISON_MARGIN,
        anchor_y=18,
        coverage_label=coverage_label,
        spawn_clusters=legend_clusters,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def render_from_artifacts(
    *,
    output_dir: Path,
    map_name: str = "Town01",
    map_asset_size: int = DEFAULT_MAP_ASSET_SIZE,
    spawn_cluster_radius: float = 16.0,
    spawn_top_k: int = 10,
) -> dict[str, int]:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    rendered_pairs = 0
    rendered_difficulties = 0
    for difficulty_id in summary["difficulty_ids"]:
        spawn_rendered: list[tuple[int, Image.Image]] = []
        route_rendered: list[tuple[int, Image.Image]] = []
        legend_clusters: list[SpawnCluster] | None = None
        for seed in summary["seeds"]:
            pair_dir = output_dir / difficulty_id / f"seed_{seed}"
            spawn_points = load_points_csv(pair_dir / "spawn_points.csv")
            route_points = load_points_csv(pair_dir / "route_points.csv")
            spawn_image, spawn_clusters = render_spawn_cluster_map(
                spawn_points,
                map_name=map_name,
                map_asset_size=map_asset_size,
                merge_radius=spawn_cluster_radius,
                top_k=spawn_top_k,
            )
            route_image = render_route_heatmap(route_points, map_name=map_name, map_asset_size=map_asset_size)
            save_single_panel(
                output_path=pair_dir / "spawn_heatmap.png",
                map_image=spawn_image,
                difficulty_id=difficulty_id,
                seed=seed,
                samples_per_seed=summary["samples_per_seed"],
                coverage_label="spawn",
                spawn_clusters=spawn_clusters,
            )
            save_single_panel(
                output_path=pair_dir / "route_heatmap.png",
                map_image=route_image,
                difficulty_id=difficulty_id,
                seed=seed,
                samples_per_seed=summary["samples_per_seed"],
                coverage_label="route",
            )
            spawn_rendered.append((seed, spawn_image))
            route_rendered.append((seed, route_image))
            if legend_clusters is None:
                legend_clusters = spawn_clusters
            rendered_pairs += 1

        difficulty_dir = output_dir / difficulty_id
        save_comparison_panel(
            output_path=difficulty_dir / "spawn_comparison.png",
            difficulty_id=difficulty_id,
            coverage_label="spawn",
            rendered=spawn_rendered,
            samples_per_seed=summary["samples_per_seed"],
            legend_clusters=legend_clusters,
        )
        save_comparison_panel(
            output_path=difficulty_dir / "route_comparison.png",
            difficulty_id=difficulty_id,
            coverage_label="route",
            rendered=route_rendered,
            samples_per_seed=summary["samples_per_seed"],
        )
        rendered_difficulties += 1

    return {"pairs": rendered_pairs, "difficulties": rendered_difficulties}
