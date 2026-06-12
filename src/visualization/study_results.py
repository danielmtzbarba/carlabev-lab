from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SeedStyle:
    color: str
    marker: str
    label: str


@dataclass(frozen=True)
class StudySpec:
    study_id: str
    title: str
    factor_name: str
    factor_labels: dict[int, str]
    factor_colors: dict[int, str]


@dataclass(frozen=True)
class CompletedRun:
    study_id: str
    exp_id: int
    condition_label: str
    seed: int
    trial_number: int
    run_dir: Path
    started_at: str
    finished_at: str
    duration_minutes: float | None
    normalized_score: float
    success_rate: float
    collision_rate: float
    unfinished_rate: float
    comfort_score: float
    comfort_violation_rate: float
    harsh_brake_rate: float
    mean_return: float
    threshold_minutes: dict[str, float]


SEED_STYLES: dict[int, SeedStyle] = {
    0: SeedStyle(color="#0f4c81", marker="o", label="Seed 0"),
    555: SeedStyle(color="#d95f02", marker="s", label="Seed 555"),
    9999: SeedStyle(color="#1b9e77", marker="^", label="Seed 9999"),
}


STUDY_SPECS: dict[str, StudySpec] = {
    "PPO_NAVIGATION_DIFFICULTY": StudySpec(
        study_id="PPO_NAVIGATION_DIFFICULTY",
        title="Difficulty Ablation",
        factor_name="Difficulty preset",
        factor_labels={
            1: "No traffic",
            2: "Easy traffic",
            3: "Medium traffic",
            4: "Hard traffic",
        },
        factor_colors={
            1: "#0f766e",
            2: "#65a30d",
            3: "#ca8a04",
            4: "#b91c1c",
        },
    ),
    "PPO_NAVIGATION_MEDIUM_FOV_ANCHOR": StudySpec(
        study_id="PPO_NAVIGATION_MEDIUM_FOV_ANCHOR",
        title="Medium FOV Anchor",
        factor_name="FOV anchor",
        factor_labels={
            1: "Center",
            2: "Lookahead_75",
        },
        factor_colors={
            1: "#1d4ed8",
            2: "#ea580c",
        },
    ),
    "PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES": StudySpec(
        study_id="PPO_NAVIGATION_MEDIUM_SEMANTIC_CLASSES",
        title="Medium Semantic Classes",
        factor_name="Semantic class layout",
        factor_labels={
            1: "Binary",
            2: "2-class",
            3: "4-class",
            4: "5-class",
            5: "6-class",
            6: "7-class",
        },
        factor_colors={
            1: "#475569",
            2: "#0f766e",
            3: "#2563eb",
            4: "#ca8a04",
            5: "#ea580c",
            6: "#b91c1c",
        },
    ),
    "PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION": StudySpec(
        study_id="PPO_NAVIGATION_MEDIUM_TEMPORAL_FUSION",
        title="Medium Temporal Fusion",
        factor_name="Temporal fusion mode",
        factor_labels={
            1: "Stack",
            2: "Vehicle temporal",
            3: "Vehicle weighted",
        },
        factor_colors={
            1: "#2563eb",
            2: "#0f766e",
            3: "#c2410c",
        },
    ),
}

THRESHOLDS = ["0.1", "0.3", "0.5", "0.6", "0.7"]
METRIC_LABELS = {
    "normalized_score": "Normalized score",
    "success_rate": "Success rate",
    "collision_rate": "Collision rate",
    "unfinished_rate": "Unfinished rate",
    "comfort_score": "Comfort score",
}


def _parse_iso8601(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else float("nan")


def _safe_std(values: Iterable[float]) -> float:
    values = list(values)
    return stdev(values) if len(values) > 1 else 0.0 if values else float("nan")


def _read_threshold_minutes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    result: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            threshold = row.get("threshold")
            elapsed_time = row.get("elapsed_time")
            if threshold is None or elapsed_time is None:
                continue
            try:
                result[threshold] = float(elapsed_time) / 60.0
            except ValueError:
                continue
    return result


def load_completed_runs(
    runs_dir: str | Path = "runs",
    study_ids: Iterable[str] | None = None,
) -> list[CompletedRun]:
    runs_root = Path(runs_dir)
    selected_studies = list(study_ids or STUDY_SPECS.keys())
    rows: list[CompletedRun] = []

    for study_id in selected_studies:
        spec = STUDY_SPECS[study_id]
        for status_path in runs_root.glob(f"{study_id}/exp_*/trial_*/seed_*/status.json"):
            run_dir = status_path.parent
            final_eval_path = run_dir / "eval" / "final" / "summary.npy"
            if not ((run_dir / "COMPLETED").exists() and final_eval_path.exists()):
                continue

            status = json.loads(status_path.read_text(encoding="utf-8"))
            payload = np.load(final_eval_path, allow_pickle=True).item()["aggregate"]
            start = _parse_iso8601(status.get("started_at"))
            finish = _parse_iso8601(status.get("finished_at"))
            duration_minutes = (
                (finish - start).total_seconds() / 60.0 if start is not None and finish is not None else None
            )
            trial_token = run_dir.parts[-2]
            trial_number = int(trial_token.removeprefix("trial_"))

            rows.append(
                CompletedRun(
                    study_id=study_id,
                    exp_id=int(status["exp_id"]),
                    condition_label=spec.factor_labels[int(status["exp_id"])],
                    seed=int(status["seed"]),
                    trial_number=trial_number,
                    run_dir=run_dir,
                    started_at=status["started_at"],
                    finished_at=status["finished_at"],
                    duration_minutes=duration_minutes,
                    normalized_score=float(payload["normalized_score"]),
                    success_rate=float(payload["success_rate"]),
                    collision_rate=float(payload["collision_rate"]),
                    unfinished_rate=float(payload["unfinished_rate"]),
                    comfort_score=float(payload["comfort_score"]),
                    comfort_violation_rate=float(payload["comfort_violation_rate"]),
                    harsh_brake_rate=float(payload["harsh_brake_rate"]),
                    mean_return=float(payload["mean_return"]),
                    threshold_minutes=_read_threshold_minutes(run_dir / "benchmark_results.csv"),
                )
            )

    return sorted(rows, key=lambda row: (row.study_id, row.exp_id, row.seed, row.trial_number))


def _group_by_experiment(runs: Iterable[CompletedRun]) -> dict[int, list[CompletedRun]]:
    grouped: dict[int, list[CompletedRun]] = {}
    for run in runs:
        grouped.setdefault(run.exp_id, []).append(run)
    return dict(sorted(grouped.items()))


def _study_runs(rows: list[CompletedRun], study_id: str) -> list[CompletedRun]:
    return [row for row in rows if row.study_id == study_id]


def _apply_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")


def _save(fig: plt.Figure, path: Path, dpi: int, formats: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_seed_explicit_score(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    grouped = _group_by_experiment(runs)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    xticks = []
    labels = []
    for idx, (exp_id, exp_runs) in enumerate(grouped.items(), start=1):
        xticks.append(idx)
        labels.append(spec.factor_labels[exp_id])
        factor_color = spec.factor_colors[exp_id]

        seed_points = sorted(exp_runs, key=lambda row: row.seed)
        for offset_idx, run in enumerate(seed_points):
            jitter = (-0.18, 0.0, 0.18)[offset_idx] if len(seed_points) == 3 else 0.0
            seed_style = SEED_STYLES.get(run.seed, SeedStyle("#475569", "o", f"Seed {run.seed}"))
            ax.scatter(
                idx + jitter,
                run.normalized_score,
                s=90,
                color=seed_style.color,
                marker=seed_style.marker,
                edgecolors=factor_color,
                linewidths=1.6,
                zorder=3,
            )

        y_values = [run.normalized_score for run in exp_runs]
        ax.plot([idx - 0.2, idx + 0.2], [mean(y_values), mean(y_values)], color=factor_color, linewidth=3.5, solid_capstyle="round")
        ax.scatter(idx, mean(y_values), s=120, color=factor_color, marker="D", edgecolors="black", linewidths=0.9, zorder=4)

    _apply_axes_style(ax)
    ax.set_xticks(xticks, labels, rotation=18, ha="right")
    ax.set_ylabel("Normalized score")
    ax.set_title(f"{spec.title}: normalized score by condition and seed", fontsize=14, color="#0f172a")

    seed_handles = [
        plt.Line2D([], [], linestyle="", marker=style.marker, markersize=8, markerfacecolor=style.color, markeredgecolor="#0f172a", label=style.label)
        for _, style in SEED_STYLES.items()
    ]
    mean_handle = plt.Line2D([], [], linestyle="", marker="D", markersize=8, markerfacecolor="#f8fafc", markeredgecolor="#0f172a", label="Seed mean")
    ax.legend(handles=seed_handles + [mean_handle], frameon=False, ncols=4, loc="upper right")

    _save(fig, out_dir / "score_seed_explicit", dpi, formats)


def plot_seed_averaged_bars(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    grouped = _group_by_experiment(runs)
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    xs = np.arange(len(grouped))
    means = [_safe_mean(run.normalized_score for run in exp_runs) for exp_runs in grouped.values()]
    stds = [_safe_std(run.normalized_score for run in exp_runs) for exp_runs in grouped.values()]
    colors = [spec.factor_colors[exp_id] for exp_id in grouped]

    ax.bar(xs, means, color=colors, edgecolor="#0f172a", linewidth=1.0, alpha=0.9)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="#0f172a", elinewidth=1.5, capsize=5)

    _apply_axes_style(ax)
    ax.set_xticks(xs, [spec.factor_labels[exp_id] for exp_id in grouped], rotation=18, ha="right")
    ax.set_ylabel("Normalized score")
    ax.set_title(f"{spec.title}: seed-averaged normalized score", fontsize=14, color="#0f172a")

    _save(fig, out_dir / "score_seed_average", dpi, formats)


def plot_top10_lollipop(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    top = sorted(runs, key=lambda row: row.normalized_score, reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(top) * 0.55)), constrained_layout=True)
    fig.patch.set_facecolor("white")

    y_positions = np.arange(len(top))[::-1]
    for y, run in zip(y_positions, top):
        factor_color = spec.factor_colors[run.exp_id]
        seed_style = SEED_STYLES.get(run.seed, SeedStyle("#475569", "o", f"Seed {run.seed}"))
        ax.hlines(y, 0, run.normalized_score, color="#cbd5e1", linewidth=2.0)
        ax.scatter(run.normalized_score, y, s=110, color=seed_style.color, marker=seed_style.marker, edgecolors=factor_color, linewidths=1.6, zorder=3)

    _apply_axes_style(ax)
    ax.set_yticks(
        y_positions,
        [
            f"{spec.factor_labels[run.exp_id]} | seed {run.seed} | trial {run.trial_number}"
            for run in top
        ],
    )
    ax.set_xlabel("Normalized score")
    ax.set_title(f"{spec.title}: top 10 completed trials", fontsize=14, color="#0f172a")

    _save(fig, out_dir / "top10_lollipop", dpi, formats)


def plot_outcome_composition(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    grouped = _group_by_experiment(runs)
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    xs = np.arange(len(grouped))
    success = [_safe_mean(run.success_rate for run in exp_runs) for exp_runs in grouped.values()]
    collision = [_safe_mean(run.collision_rate for run in exp_runs) for exp_runs in grouped.values()]
    unfinished = [_safe_mean(run.unfinished_rate for run in exp_runs) for exp_runs in grouped.values()]

    ax.bar(xs, success, color="#0f766e", label="Success")
    ax.bar(xs, collision, bottom=success, color="#b91c1c", label="Collision")
    ax.bar(xs, unfinished, bottom=np.array(success) + np.array(collision), color="#ca8a04", label="Unfinished")

    _apply_axes_style(ax)
    ax.set_xticks(xs, [spec.factor_labels[exp_id] for exp_id in grouped], rotation=18, ha="right")
    ax.set_ylabel("Outcome fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{spec.title}: outcome composition", fontsize=14, color="#0f172a")
    ax.legend(frameon=False, ncols=3, loc="upper right")

    _save(fig, out_dir / "outcome_composition", dpi, formats)


def plot_success_collision_scatter(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for run in runs:
        seed_style = SEED_STYLES.get(run.seed, SeedStyle("#475569", "o", f"Seed {run.seed}"))
        ax.scatter(
            run.collision_rate,
            run.success_rate,
            s=90,
            marker=seed_style.marker,
            color=seed_style.color,
            edgecolors=spec.factor_colors[run.exp_id],
            linewidths=1.5,
            alpha=0.95,
        )

    _apply_axes_style(ax)
    ax.set_xlabel("Collision rate")
    ax.set_ylabel("Success rate")
    ax.set_title(f"{spec.title}: success vs collision", fontsize=14, color="#0f172a")

    seed_handles = [
        plt.Line2D([], [], linestyle="", marker=style.marker, markersize=8, markerfacecolor=style.color, markeredgecolor="#0f172a", label=style.label)
        for _, style in SEED_STYLES.items()
    ]
    factor_handles = [
        plt.Line2D([], [], linestyle="", marker="o", markersize=8, markerfacecolor="white", markeredgewidth=2, markeredgecolor=spec.factor_colors[exp_id], label=label)
        for exp_id, label in spec.factor_labels.items()
    ]
    ax.legend(handles=seed_handles + factor_handles, frameon=False, loc="lower left", ncols=2)

    _save(fig, out_dir / "success_vs_collision", dpi, formats)


def plot_score_comfort_scatter(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for run in runs:
        seed_style = SEED_STYLES.get(run.seed, SeedStyle("#475569", "o", f"Seed {run.seed}"))
        ax.scatter(
            run.comfort_score,
            run.normalized_score,
            s=90,
            marker=seed_style.marker,
            color=seed_style.color,
            edgecolors=spec.factor_colors[run.exp_id],
            linewidths=1.5,
            alpha=0.95,
        )

    _apply_axes_style(ax)
    ax.set_xlabel("Comfort score")
    ax.set_ylabel("Normalized score")
    ax.set_title(f"{spec.title}: normalized score vs comfort", fontsize=14, color="#0f172a")
    _save(fig, out_dir / "score_vs_comfort", dpi, formats)


def plot_finish_time(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    grouped = _group_by_experiment(runs)
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    xs = np.arange(1, len(grouped) + 1)
    data = [[run.duration_minutes for run in exp_runs if run.duration_minutes is not None] for exp_runs in grouped.values()]
    box = ax.boxplot(data, positions=xs, widths=0.45, patch_artist=True, showmeans=True, meanline=False)
    for patch, exp_id in zip(box["boxes"], grouped):
        patch.set_facecolor(spec.factor_colors[exp_id])
        patch.set_alpha(0.35)
        patch.set_edgecolor("#0f172a")
    for median in box["medians"]:
        median.set_color("#0f172a")
        median.set_linewidth(1.8)
    for mean_marker in box["means"]:
        mean_marker.set_marker("D")
        mean_marker.set_markerfacecolor("#0f172a")
        mean_marker.set_markeredgecolor("white")
        mean_marker.set_markersize(7)
    for whisker in box["whiskers"]:
        whisker.set_color("#475569")

    for x, exp_runs in zip(xs, grouped.values()):
        for run in exp_runs:
            if run.duration_minutes is None:
                continue
            seed_style = SEED_STYLES.get(run.seed, SeedStyle("#475569", "o", f"Seed {run.seed}"))
            ax.scatter(x, run.duration_minutes, s=75, color=seed_style.color, marker=seed_style.marker, edgecolors="#0f172a", linewidths=0.9, zorder=3)

    _apply_axes_style(ax)
    ax.set_xticks(xs, [spec.factor_labels[exp_id] for exp_id in grouped], rotation=18, ha="right")
    ax.set_ylabel("Finished runtime (minutes)")
    ax.set_title(f"{spec.title}: wall-clock finish time", fontsize=14, color="#0f172a")
    _save(fig, out_dir / "finish_time", dpi, formats)


def plot_threshold_heatmap(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    grouped = _group_by_experiment(runs)
    matrix = np.full((len(grouped), len(THRESHOLDS)), np.nan, dtype=float)

    for row_idx, exp_id in enumerate(grouped):
        exp_runs = grouped[exp_id]
        for col_idx, threshold in enumerate(THRESHOLDS):
            values = [run.threshold_minutes[threshold] for run in exp_runs if threshold in run.threshold_minutes]
            if values:
                matrix[row_idx, col_idx] = _safe_mean(values)

    fig, ax = plt.subplots(figsize=(8, max(3.8, len(grouped) * 0.8)), constrained_layout=True)
    fig.patch.set_facecolor("white")

    cmap = plt.get_cmap("YlOrBr").copy()
    cmap.set_bad(color="#f8fafc")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(THRESHOLDS)), [f"Success ≥ {threshold}" for threshold in THRESHOLDS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(grouped)), [spec.factor_labels[exp_id] for exp_id in grouped])
    ax.set_title(f"{spec.title}: threshold crossing time (minutes)", fontsize=14, color="#0f172a")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = "-" if np.isnan(value) else f"{value:.1f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="#111827", fontsize=10)

    cbar = fig.colorbar(image, ax=ax, shrink=0.92)
    cbar.set_label("Minutes to first threshold crossing")
    _save(fig, out_dir / "threshold_heatmap", dpi, formats)


def plot_seed_profile(runs: list[CompletedRun], spec: StudySpec, out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    grouped = _group_by_experiment(runs)
    seed_order = sorted(SEED_STYLES)
    x_positions = np.arange(len(seed_order))

    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for exp_id, exp_runs in grouped.items():
        score_by_seed = {run.seed: run.normalized_score for run in exp_runs}
        y_values = [score_by_seed.get(seed, np.nan) for seed in seed_order]
        ax.plot(x_positions, y_values, color=spec.factor_colors[exp_id], linewidth=2.5, marker="o", markersize=7, label=spec.factor_labels[exp_id])
        for x, seed in zip(x_positions, seed_order):
            if seed not in score_by_seed:
                continue
            seed_style = SEED_STYLES[seed]
            ax.scatter(x, score_by_seed[seed], s=85, marker=seed_style.marker, color=seed_style.color, edgecolors=spec.factor_colors[exp_id], linewidths=1.2, zorder=3)

    _apply_axes_style(ax)
    ax.set_xticks(x_positions, [str(seed) for seed in seed_order])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Normalized score")
    ax.set_title(f"{spec.title}: seed sensitivity profile", fontsize=14, color="#0f172a")
    ax.legend(frameon=False, loc="upper right", ncols=2)

    _save(fig, out_dir / "seed_profile", dpi, formats)


def plot_cross_study_seed_distribution(rows: list[CompletedRun], out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    study_ids = list(STUDY_SPECS)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for idx, study_id in enumerate(study_ids, start=1):
        study_rows = [row for row in rows if row.study_id == study_id]
        for offset_idx, seed in enumerate(sorted(SEED_STYLES)):
            seed_rows = [row.normalized_score for row in study_rows if row.seed == seed]
            if not seed_rows:
                continue
            jitter = (-0.2, 0.0, 0.2)[offset_idx]
            style = SEED_STYLES[seed]
            ax.scatter(
                np.full(len(seed_rows), idx + jitter),
                seed_rows,
                s=80,
                color=style.color,
                marker=style.marker,
                edgecolors="#0f172a",
                linewidths=0.8,
                alpha=0.9,
            )
            ax.plot([idx + jitter - 0.07, idx + jitter + 0.07], [mean(seed_rows), mean(seed_rows)], color=style.color, linewidth=3)

    _apply_axes_style(ax)
    ax.set_xticks(np.arange(1, len(study_ids) + 1), [STUDY_SPECS[study_id].title for study_id in study_ids], rotation=15, ha="right")
    ax.set_ylabel("Normalized score")
    ax.set_title("Cross-study normalized score distribution by seed", fontsize=14, color="#0f172a")
    handles = [
        plt.Line2D([], [], linestyle="", marker=style.marker, markersize=8, markerfacecolor=style.color, markeredgecolor="#0f172a", label=style.label)
        for style in SEED_STYLES.values()
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncols=3)
    _save(fig, out_dir / "cross_study_seed_distribution", dpi, formats)


def plot_cross_study_success_collision(rows: list[CompletedRun], out_dir: Path, dpi: int, formats: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for run in rows:
        style = SEED_STYLES.get(run.seed, SeedStyle("#475569", "o", f"Seed {run.seed}"))
        factor_color = STUDY_SPECS[run.study_id].factor_colors[run.exp_id]
        ax.scatter(
            run.collision_rate,
            run.success_rate,
            s=85,
            marker=style.marker,
            color=style.color,
            edgecolors=factor_color,
            linewidths=1.3,
            alpha=0.95,
        )

    _apply_axes_style(ax)
    ax.set_xlabel("Collision rate")
    ax.set_ylabel("Success rate")
    ax.set_title("All completed runs: success vs collision", fontsize=14, color="#0f172a")
    _save(fig, out_dir / "cross_study_success_vs_collision", dpi, formats)


def generate_all_plots(
    runs_dir: str | Path = "runs",
    output_dir: str | Path = "docs/study_plots",
    dpi: int = 180,
    formats: Iterable[str] = ("png",),
) -> dict[str, list[Path]]:
    rows = load_completed_runs(runs_dir=runs_dir)
    out_root = Path(output_dir)
    generated: dict[str, list[Path]] = {}

    for study_id, spec in STUDY_SPECS.items():
        study_rows = _study_runs(rows, study_id)
        if not study_rows:
            continue
        study_dir = out_root / study_id.lower()
        plot_seed_explicit_score(study_rows, spec, study_dir, dpi, formats)
        plot_seed_averaged_bars(study_rows, spec, study_dir, dpi, formats)
        plot_top10_lollipop(study_rows, spec, study_dir, dpi, formats)
        plot_outcome_composition(study_rows, spec, study_dir, dpi, formats)
        plot_success_collision_scatter(study_rows, spec, study_dir, dpi, formats)
        plot_score_comfort_scatter(study_rows, spec, study_dir, dpi, formats)
        plot_finish_time(study_rows, spec, study_dir, dpi, formats)
        plot_threshold_heatmap(study_rows, spec, study_dir, dpi, formats)
        plot_seed_profile(study_rows, spec, study_dir, dpi, formats)
        generated[study_id] = sorted(study_dir.glob("*"))

    overview_dir = out_root / "cross_study"
    plot_cross_study_seed_distribution(rows, overview_dir, dpi, formats)
    plot_cross_study_success_collision(rows, overview_dir, dpi, formats)
    generated["cross_study"] = sorted(overview_dir.glob("*"))
    return generated

