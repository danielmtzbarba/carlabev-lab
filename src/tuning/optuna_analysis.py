import os
import tyro
import optuna
import optuna.visualization as vis
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Optional

from src.config.experiment_loader import get_study_db_path, get_study_name
from src.tuning.engine import stage_sort_key, stage_title, trial_stage_name

# ─── Global Theme ─────────────────────────────────────────────────────────────
# A dark, scientific dark-mode theme for all plots
DARK_BG        = "#ffffff"
PANEL_BG       = "#ffffff"
GRID_COLOR     = "#e5e7eb"
TEXT_COLOR     = "#111827"
ACCENT_COLOR   = "#1e3a8a"   # deep navy-blue accent
BEST_COLOR     = "#059669"   # professional emerald-green for "best so far"
COLORSCALE     = "Cividis"   # professional/academic/colorblind-friendly scale

PHASE_PALETTE = [
    "#1e3a8a",  # navy    – phase 1
    "#6b21a8",  # violet  – phase 2a
    "#065f46",  # teal    – phase 2b
    "#9a3412",  # rusted orange
    "#9f1239",  # rose
]

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif"

GLOBAL_LAYOUT = dict(
    font=dict(family=FONT_FAMILY, size=14, color=TEXT_COLOR),
    title_font=dict(size=20, family=FONT_FAMILY, color=TEXT_COLOR),
    title_x=0.5,
    plot_bgcolor=PANEL_BG,
    paper_bgcolor=DARK_BG,
    margin=dict(l=90, r=50, t=90, b=80),
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=GRID_COLOR,
        borderwidth=1,
        font=dict(color=TEXT_COLOR, size=12),
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
    ),
)

AXIS_STYLE = dict(
    showgrid=True,
    gridwidth=1,
    gridcolor=GRID_COLOR,
    linecolor=GRID_COLOR,
    tickcolor=TEXT_COLOR,
    color=TEXT_COLOR,
    zerolinecolor=GRID_COLOR,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def apply_dark_theme(fig: go.Figure, title: str) -> go.Figure:
    """Apply the unified dark scientific theme to any Plotly figure.

    Only touches layout/axes — never mutates trace data so Optuna's
    internal figure structures (parcoords, contour, heatmap…) are preserved.
    """
    fig.update_layout(**GLOBAL_LAYOUT)
    fig.update_layout(title=dict(text=title, font=dict(size=20, color=TEXT_COLOR)))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


def save(fig: go.Figure, path: str, show: bool = False) -> None:
    """Write self-contained HTML (Plotly.js embedded — no internet needed)."""
    fig.write_html(path, include_plotlyjs=True)
    if show:
        fig.show()


def phase_color(i: int) -> str:
    return PHASE_PALETTE[i % len(PHASE_PALETTE)]


# ─── DB helpers ───────────────────────────────────────────────────────────────

def print_compact_trial_info(trial, db_conn, rank=None):
    if db_conn is None:
        print(
            f"Rank {rank} | Trial {trial.number:<3} | "
            f"Score: {trial.value:.4f} | Stage: {trial_stage_name(trial)}"
        )
        return

    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT success_rate, collision_rate FROM trial_eval_logs "
        "WHERE trial_number = ? ORDER BY global_step DESC LIMIT 1",
        (trial.number,),
    )
    eval_row = cursor.fetchone()
    succ = eval_row[0] if eval_row and eval_row[0] is not None else 0.0
    coll = eval_row[1] if eval_row and eval_row[1] is not None else 0.0

    cursor.execute(
        "SELECT MAX(global_step), AVG(mean_return) FROM trial_train_logs WHERE trial_number = ?",
        (trial.number,),
    )
    train_row = cursor.fetchone()
    steps = train_row[0] if train_row and train_row[0] is not None else 0
    ret   = train_row[1] if train_row and train_row[1] is not None else 0.0

    params_str = ", ".join(
        [
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in trial.params.items()
        ]
    )

    print(
        f"Rank {rank} | Trial {trial.number:<3} | Score: {trial.value:7.4f} | "
        f"Succ: {succ:5.2f}  Coll: {coll:5.2f} | Ret: {ret:6.2f} "
        f"({steps:<8} steps) | Stage: {trial_stage_name(trial):<20} | {params_str}"
    )


# ─── Individual Plot Functions ─────────────────────────────────────────────────

def plot_optimization_history(phase_study: optuna.Study, phase: str, plot_dir: str, show: bool) -> None:
    """
    1. Optimization History
    Proves convergence; multi-phase jumps in performance are clearly visible.
    """
    fig = vis.plot_optimization_history(phase_study)

    # Build a lookup: sequential index inside this phase study → original trial number
    phase_trial_numbers = [t.number for t in phase_study.trials]

    scatter_idx = 0  # counts scatter (individual trial) traces
    for trace in fig.data:
        name = (trace.name or "").lower()
        if "best" in name:
            trace.line.color = BEST_COLOR
            trace.line.width = 3
        else:
            # Individual trial dots — inject trial numbers into hover
            n_points = len(trace.x) if trace.x is not None else 0
            trial_nums = phase_trial_numbers[:n_points]
            trace.customdata = [[tn] for tn in trial_nums]
            trace.hovertemplate = (
                "<b>Trial #%{customdata[0]}</b><br>"
                "Objective: %{y:.4f}<br>"
                "<extra></extra>"
            )
            trace.marker.color = ACCENT_COLOR
            trace.marker.opacity = 0.65
            trace.marker.size = 8
            scatter_idx += 1

    fig = apply_dark_theme(fig, f"{stage_title(phase)} — Optimization History")
    fig.update_yaxes(title_text="Objective Value (Reward)")
    fig.update_xaxes(title_text="Trial Number")
    save(fig, os.path.join(plot_dir, "optimization_history.html"), show)


def plot_param_importances(phase_study: optuna.Study, phase: str, plot_dir: str, show: bool) -> None:
    """
    2. Hyperparameter Importance (FAnova / MDI)
    Identifies which PPO knobs genuinely moved the needle.
    """
    fig = vis.plot_param_importances(phase_study)
    # Colour bars with a gradient based on importance rank
    for trace in fig.data:
        if hasattr(trace, "marker"):
            n = len(trace.x) if hasattr(trace, "x") and trace.x is not None else 1
            trace.marker.color = list(range(n))
            trace.marker.colorscale = "Blues"  # Professional monochromatic scale for importance
            trace.marker.showscale = False
    fig = apply_dark_theme(fig, f"{stage_title(phase)} — Hyperparameter Importance (FAnova)")
    fig.update_xaxes(title_text="Relative Importance")
    save(fig, os.path.join(plot_dir, "param_importances.html"), show)


def plot_parallel_coordinate(phase_study: optuna.Study, phase: str, plot_dir: str, show: bool) -> None:
    """
    3. Parallel Coordinate Plot
    Reveals clusters of high-performing configurations ("the winning corridor").
    """
    fig = vis.plot_parallel_coordinate(phase_study)
    # Style the colour axis label
    fig = apply_dark_theme(fig, f"{stage_title(phase)} — Parallel Coordinates (Best Configurations)")
    # Force colorscale on the parcoords trace
    for trace in fig.data:
        if trace.type == "parcoords":
            trace.line.colorscale = COLORSCALE
            trace.line.showscale = True
            trace.line.colorbar = dict(
                title=dict(text="Objective", font=dict(color=TEXT_COLOR, size=12)),
                tickfont=dict(color=TEXT_COLOR, size=11),
                bgcolor=PANEL_BG,
                bordercolor=GRID_COLOR,
            )
            # Style each dimension label
            if trace.dimensions:
                for dim in trace.dimensions:
                    dim.label = dim.label  # keep Optuna's label
                    if hasattr(dim, "tickfont"):
                        dim.tickfont = dict(color=TEXT_COLOR, size=10)
                    if hasattr(dim, "labelfont"):
                        dim.labelfont = dict(color=TEXT_COLOR, size=11)
    save(fig, os.path.join(plot_dir, "parallel_coordinate.html"), show)


def plot_contour(phase_study: optuna.Study, phase: str, plot_dir: str, show: bool,
                 top_trial_numbers: set | None = None) -> None:
    """
    4. Contour Plot (Interdependency)
    Exposes non-linear synergies between PPO hyperparameters.
    Top-ranked trials are highlighted as red stars.
    """
    TOP_COLOR    = "#b91c1c"   # professional deep red for top-ranked points
    OTHER_COLOR  = "#1e40af"   # professional steel blue for regular points
    CONTOUR_SCALE = "Cividis"   # professional/academic/colorblind-friendly scale

    top_trial_numbers = top_trial_numbers or set()

    fig = vis.plot_contour(phase_study)

    # Build ordered trial list for hover annotation
    phase_trials_ordered = [
        t for t in phase_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Use trial numbers directly to avoid hashing FrozenTrial objects (which may contain lists)
    top_nums = top_trial_numbers

    for trace in fig.data:
        if trace.type == "contour":
            trace.colorscale = CONTOUR_SCALE
            trace.contours = dict(coloring="heatmap")
            trace.line = dict(smoothing=0.85)
            if hasattr(trace, "colorbar") and trace.colorbar is not None:
                trace.colorbar.tickfont = dict(color=TEXT_COLOR)
                trace.colorbar.title = dict(font=dict(color=TEXT_COLOR))
        if trace.type == "scatter":
            # Per-point colour: red for top, blue for rest
            n_pts = len(trace.x) if trace.x is not None else 0
            matched = phase_trials_ordered[:n_pts]
            colors  = [TOP_COLOR if t.number in top_nums else OTHER_COLOR for t in matched]
            sizes   = [12 if t.number in top_nums else 7 for t in matched]
            symbols = ["star" if t.number in top_nums else "circle" for t in matched]
            ranks   = [
                next((i + 1 for i, tr in enumerate(sorted(phase_trials_ordered, key=lambda x: x.value, reverse=True))
                      if tr.number == t.number), "-")
                for t in matched
            ]
            trace.update(
                customdata=[[t.number, t.value, r] for t, r in zip(matched, ranks)],
                hovertemplate=(
                    "<b>Trial #%{customdata[0]}</b><br>"
                    "Objective: %{customdata[1]:.4f}<br>"
                    "Rank: #%{customdata[2]}<br>"
                    "x: %{x}<br>y: %{y}<br>"
                    "<extra></extra>"
                ),
                marker=dict(
                    color=colors,
                    size=sizes,
                    symbol=symbols,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
            )

    fig = apply_dark_theme(fig, f"{stage_title(phase)} — Search Space Contour (Parameter Interactions)")
    # Add legend annotation explaining markers
    fig.add_annotation(
        text="★ Top-ranked trial",
        xref="paper", yref="paper",
        x=1.0, y=1.04, showarrow=False,
        font=dict(color=TOP_COLOR, size=12),
        xanchor="right",
    )
    save(fig, os.path.join(plot_dir, "contour_plot.html"), show)

    # Also save the slice plot alongside — inject trial numbers into hover
    fig_slice = vis.plot_slice(phase_study)
    for trace in fig_slice.data:
        if trace.type == "scatter" and trace.mode is not None and "markers" in str(trace.mode):
            n_pts = len(trace.x) if trace.x is not None else 0
            matched = phase_trials_ordered[:n_pts]
            colors  = [TOP_COLOR if t.number in top_nums else OTHER_COLOR for t in matched]
            sizes   = [12 if t.number in top_nums else 7 for t in matched]
            symbols = ["star" if t.number in top_nums else "circle" for t in matched]
            ranks   = [
                next((i + 1 for i, tr in enumerate(sorted(phase_trials_ordered, key=lambda x: x.value, reverse=True))
                      if tr.number == t.number), "-")
                for t in matched
            ]
            trace.update(
                customdata=[[t.number, t.value, r] for t, r in zip(matched, ranks)],
                hovertemplate=(
                    "<b>Trial #%{customdata[0]}</b><br>"
                    "Objective: %{customdata[1]:.4f}<br>"
                    "Rank: #%{customdata[2]}<br>"
                    "Param value: %{x}<br>"
                    "<extra></extra>"
                ),
                marker=dict(
                    color=colors,
                    size=sizes,
                    symbol=symbols,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                    showscale=False,  # Remove the trial/color bar
                ),
            )
    fig_slice = apply_dark_theme(fig_slice, f"{stage_title(phase)} — Parameter Slice Analysis")
    save(fig_slice, os.path.join(plot_dir, "slice_plot.html"), show)


def plot_edf(phase_study: optuna.Study, phase: str, plot_dir: str, show: bool) -> None:
    """
    5. Empirical Distribution Function (EDF)
    Measures search reliability and objective robustness across the phase.
    """
    fig = vis.plot_edf(phase_study)
    for trace in fig.data:
        if hasattr(trace, "line"):
            trace.line.color = ACCENT_COLOR
            trace.line.width = 2.5
        if hasattr(trace, "fill"):
            trace.fill = "tozeroy"
            trace.fillcolor = "rgba(0,212,255,0.12)"
    fig = apply_dark_theme(fig, f"{stage_title(phase)} — EDF (Search Reliability)")
    fig.update_xaxes(title_text="Objective Value (Reward)")
    fig.update_yaxes(title_text="Cumulative Probability")
    save(fig, os.path.join(plot_dir, "edf_plot.html"), show)


def plot_timeline(phase_study: optuna.Study, phase: str, plot_dir: str, show: bool) -> None:
    """
    6. Trial Timeline
    Shows wall-clock duration per trial — invaluable for multi-phase parallel runs.
    """
    try:
        fig = vis.plot_timeline(phase_study)
        fig = apply_dark_theme(fig, f"{stage_title(phase)} — Trial Timeline (Wall-Clock)")
        fig.update_xaxes(title_text="Wall-Clock Time")
        fig.update_yaxes(title_text="Trial")
        save(fig, os.path.join(plot_dir, "timeline.html"), show)
    except AttributeError:
        # plot_timeline added in Optuna 3.2; gracefully skip on older versions
        print("  ⚠ plot_timeline not available in this Optuna version — skipping.")


def plot_learning_curves(top_trials, phase: str, plot_dir: str, db_conn, show: bool) -> None:
    """
    Custom: top-k training learning curves overlaid on one figure.
    """
    fig = go.Figure()
    for rank, trial in enumerate(top_trials):
        df = pd.read_sql_query(
            "SELECT global_step, mean_return FROM trial_train_logs WHERE trial_number = ? ORDER BY global_step",
            db_conn,
            params=(trial.number,),
        )
        if not df.empty:
            color = phase_color(rank)
            fig.add_trace(
                go.Scatter(
                    x=df["global_step"],
                    y=df["mean_return"],
                    mode="lines",
                    name=f"Rank {rank + 1}  (Trial {trial.number})",
                    line=dict(color=color, width=2),
                    opacity=0.9,
                )
            )

    if len(fig.data) == 0:
        return

    fig = apply_dark_theme(fig, f"{stage_title(phase)} — Learning Curves (Top Trials)")
    fig.update_xaxes(title_text="Environment Steps")
    fig.update_yaxes(title_text="Mean Return")
    save(fig, os.path.join(plot_dir, "top_trials_learning_curves.html"), show)


def plot_time_to_reach(top_trials, phase: str, plot_dir: str, db_conn, show: bool) -> None:
    """
    Custom: wall-time required to reach each success-rate threshold.
    """
    threshold_cols = ["0_1", "0_2", "0_3", "0_4", "0_5", "0_6", "0_7", "0_8", "0_9", "0_95", "0_99"]
    fig = go.Figure()

    for rank, trial in enumerate(top_trials):
        df = pd.read_sql_query(
            "SELECT * FROM trial_eval_logs WHERE trial_number = ? ORDER BY global_step DESC LIMIT 1",
            db_conn,
            params=(trial.number,),
        )
        if df.empty:
            continue

        x_vals, y_vals = [], []
        for tc in threshold_cols:
            col_name = f"time_to_reach_{tc}"
            if col_name in df.columns and pd.notna(df.iloc[0][col_name]):
                x_vals.append(float(tc.replace("_", ".")))
                y_vals.append(df.iloc[0][col_name] / 3600.0)

        if x_vals:
            color = phase_color(rank)
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"Rank {rank + 1}  (Trial {trial.number})",
                    line=dict(color=color, width=2),
                    marker=dict(color=color, size=8, symbol="circle"),
                )
            )

    if len(fig.data) == 0:
        return

    fig = apply_dark_theme(fig, f"{stage_title(phase)} — Wall-Time to Success Threshold")
    fig.update_xaxes(title_text="Success-Rate Threshold")
    fig.update_yaxes(title_text="Training Elapsed Time (hours)")
    save(fig, os.path.join(plot_dir, "top_trials_time_to_reach.html"), show)


# ─── Dataclass & Main ─────────────────────────────────────────────────────────

@dataclass
class AnalyzeArgs:
    study_id:   str           = "PPO_NAVIGATION"
    exp_id:     Optional[int] = None
    stage:      Optional[str] = None
    top_k:      int           = 5
    show_plots: bool          = False
    save_dir:   str           = "results"


def generate_dashboard(save_dir: str, db_name: str, phases: list[str]) -> None:
    """
    Generate a single consolidated HTML dashboard with a tabbed interface.
    Each tuning stage has a section, and each section has tabs for the core plots.
    """
    dashboard_path = os.path.join(save_dir, f"{db_name}_dashboard.html")
    report_path = os.path.join(save_dir, "optuna_study_report.md")
    
    # Helper to extract parameter tables from the MD report
    phase_params = {}
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            lines = f.readlines()
        
        current_phase = None
        in_table = False
        table_lines = []
        
        for line in lines:
            if "## Policy Dynamics:" in line:
                current_phase = "policy_dynamics"
            elif "## Rollout Geometry:" in line:
                current_phase = "rollout_geometry"
            elif "## Loss Regularization:" in line:
                current_phase = "loss_regularization"
            elif "## Network Capacity:" in line:
                current_phase = "network_capacity"

            if current_phase:
                if "| Parameter |" in line:
                    in_table = True
                    table_lines = [line]
                elif in_table:
                    if "|" in line:
                        table_lines.append(line)
                    else:
                        in_table = False
                        # Convert MD table to Simple HTML table
                        html_table = '<table style="width:100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px;">'
                        for i, tl in enumerate(table_lines):
                            if "---" in tl:
                                continue
                            cells = [c.strip() for c in tl.split("|")[1:-1]]
                            tag = "th" if i == 0 else "td"
                            style = "border: 1px solid #e5e7eb; padding: 8px; text-align: left;"
                            if i == 0:
                                style += "background-color: #f9fafb; color: #1e3a8a; font-weight: 600;"
                            html_table += "<tr>" + "".join([f'<{tag} style="{style}">{c}</{tag}>' for c in cells]) + "</tr>"
                        html_table += "</table>"
                        phase_params[current_phase] = html_table
                        current_phase = None # Reset to avoid grabbing implementation details

    # CSS for a clean, modern tabbed interface
    css = """
    body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; color: #111827; margin: 0; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    h1 { text-align: center; color: #1e3a8a; margin-bottom: 40px; }
    .phase-container { margin-bottom: 60px; border-bottom: 2px solid #e5e7eb; padding-bottom: 20px; }
    .phase-title { font-size: 24px; color: #1e3a8a; border-left: 6px solid #1e3a8a; padding-left: 15px; margin-bottom: 20px; }
    .params-section { margin-bottom: 25px; }
    .params-title { font-size: 16px; font-weight: 600; color: #4b5563; margin-bottom: 10px; }
    
    /* Tabs */
    .tabs { display: flex; flex-wrap: wrap; margin-bottom: 10px; border-bottom: 1px solid #e5e7eb; }
    .tab-label { padding: 12px 24px; cursor: pointer; background: #f9fafb; border: 1px solid #e5e7eb; border-bottom: none; margin-right: 4px; border-radius: 6px 6px 0 0; font-weight: 500; transition: 0.2s; }
    .tab-label:hover { background: #f3f4f6; }
    .tab-radio { display: none; }
    .tab-content { display: none; width: 100%; padding: 20px; border: 1px solid #e5e7eb; border-top: none; border-radius: 0 0 8px 8px; background: white; box-sizing: border-box; }
    
    /* Logic: when radio is checked, show content and style label */
    """
    
    # Dynamic logic for each radio button
    for p in phases:
        for plot in ["History", "Importance", "Contour", "Slice"]:
            css += f'#tab-{p}-{plot}:checked ~ .content-{p}-{plot} {{ display: block; }}\n'
            css += f'#tab-{p}-{plot}:checked ~ .labels-{p} label[for="tab-{p}-{plot}"] {{ background: white; border-bottom: 2px solid white; color: #1e3a8a; position: relative; top: 1px; }}\n'

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>CarlaBEV: Optuna PPO hyperparameter optimization</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <h1>CarlaBEV: Optuna PPO hyperparameter optimization</h1>
    """

    for p in sorted(phases, key=stage_sort_key):
        plots = [
            ("History", "optimization_history.html"),
            ("Importance", "param_importances.html"),
            ("Contour", "contour_plot.html"),
            ("Slice", "slice_plot.html"),
        ]
        
        params_html = phase_params.get(p, '<p style="color: #6b7280; font-style: italic;">No parameter details found in report.</p>')

        html_content += f"""
        <div class="phase-container">
            <div class="phase-title">{stage_title(p)}</div>
            
            <div class="params-section">
                <div class="params-title">Target Hyperparameters</div>
                {params_html}
            </div>

            <div class="tabs-wrapper">
        """
        
        # Radios
        for i, (name, _) in enumerate(plots):
            checked = "checked" if i == 0 else ""
            html_content += f'<input type="radio" name="tabs-{p}" class="tab-radio" id="tab-{p}-{name}" {checked}>\n'
            
        # Labels
        html_content += f'<div class="tabs labels-{p}">\n'
        for name, _ in plots:
             html_content += f'<label for="tab-{p}-{name}" class="tab-label">{name}</label>\n'
        html_content += '</div>\n'
        
        # Contents
        for name, filename in plots:
            # Note: paths are relative to the dashboard.html location in results/
            rel_path = f"{db_name}_phase_{p}_plots/{filename}"
            html_content += f"""
            <div class="tab-content content-{p}-{name}">
                <iframe src="{rel_path}" width="100%" height="800px" frameborder="0"></iframe>
            </div>
            """
            
        html_content += """
            </div>
        </div>
        """

    html_content += """
    </div>
</body>
</html>
"""
    with open(dashboard_path, "w") as f:
        f.write(html_content)
    print(f"\n✨ Consolidated Dashboard generated at: {dashboard_path}")


def main():
    args = tyro.cli(AnalyzeArgs)
    study_name = get_study_name(args.study_id)
    db_path = get_study_db_path(args.study_id)
    db_name = os.path.splitext(os.path.basename(db_path))[0]
    if args.exp_id is not None:
        db_name = f"{db_name}_exp_{args.exp_id}"

    storage_name = f"sqlite:///{db_path}"

    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist!")
        return

    import sqlite3
    db_conn   = None
    real_path = storage_name.replace("sqlite:///", "")
    if os.path.exists(real_path):
        db_conn = sqlite3.connect(real_path)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Loading study: {study_name} from {storage_name}")
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    scoped_trials = [
        t
        for t in study.trials
        if args.exp_id is None or t.user_attrs.get("base_exp_id") == args.exp_id
    ]
    complete_trials = [
        t for t in scoped_trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # ── Summary Table ──────────────────────────────────────────────────────────
    print("\n--- Study Statistics by Stage ---")
    print(f"Total Trials: {len(scoped_trials)}")

    summary_data = [
        {"Stage": trial_stage_name(t), "State": t.state.name}
        for t in scoped_trials
    ]

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        pivot_df = df_summary.pivot_table(index="Stage", columns="State", aggfunc=len, fill_value=0)
        for col in ["COMPLETE", "PRUNED", "RUNNING", "FAIL"]:
            if col not in pivot_df.columns:
                pivot_df[col] = 0
        pivot_df["TOTAL"] = pivot_df[["COMPLETE", "PRUNED", "RUNNING", "FAIL"]].sum(axis=1)
        pivot_df = pivot_df[["COMPLETE", "PRUNED", "RUNNING", "FAIL", "TOTAL"]]
        print("\n" + pivot_df.to_string())

    print("-" * 40)

    if not complete_trials:
        print("No complete trials found yet.")
        return

    best_trial = max(complete_trials, key=lambda t: t.value)
    print("\n🏆 Best Trial Overall:")
    print_compact_trial_info(best_trial, db_conn=db_conn, rank="BEST")

    # ── Stage Loop ─────────────────────────────────────────────────────────────
    phases = set(trial_stage_name(t) for t in complete_trials)
    if args.stage:
        phases = {args.stage}

    best_trials_by_phase: dict = {}

    # Print cross-phase overview
    for phase in sorted(phases, key=stage_sort_key):
        phase_trials = [t for t in complete_trials if trial_stage_name(t) == phase]
        if phase_trials:
            phase_trials.sort(key=lambda t: t.value, reverse=True)
            best_trials_by_phase[phase] = phase_trials[0]

    print("\n" + "=" * 80)
    print("--- 🥇 Best Trial By Stage Overview ---")
    for phase, best_trial in best_trials_by_phase.items():
        print_compact_trial_info(best_trial, db_conn=db_conn, rank=f"🥇 BEST ({stage_title(phase)})")
    print("=" * 80 + "\n")

    for phase in sorted(phases, key=stage_sort_key):
        phase_trials = [t for t in complete_trials if trial_stage_name(t) == phase]
        if not phase_trials:
            continue

        phase_trials.sort(key=lambda t: t.value, reverse=True)
        top_curr_phase = phase_trials[: args.top_k]

        print(f"\n--- Top {min(args.top_k, len(phase_trials))} Trials ({stage_title(phase)}) ---")
        for i, trial in enumerate(top_curr_phase):
            print_compact_trial_info(trial, db_conn=db_conn, rank=i + 1)

        # Build an isolated in-memory study for this phase's visualisations
        phase_study = optuna.create_study(direction=study.direction)
        for t in scoped_trials:
            if trial_stage_name(t) == phase:
                phase_study.add_trial(t)

        plot_dir = os.path.join(args.save_dir, f"{db_name}_phase_{phase}_plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"\nGenerating {stage_title(phase)} plots → {plot_dir}")

        try:
            # 1. Optimization History
            plot_optimization_history(phase_study, phase, plot_dir, args.show_plots)

            # 2. Hyperparameter Importance
            if len(phase_trials) > 1:
                plot_param_importances(phase_study, phase, plot_dir, args.show_plots)

            # 3. Parallel Coordinate
            if len(phase_trials) > 1:
                plot_parallel_coordinate(phase_study, phase, plot_dir, args.show_plots)

            # 4. Contour + Slice
            if len(phase_trials) > 1:
                top_nums = {t.number for t in top_curr_phase}
                plot_contour(phase_study, phase, plot_dir, args.show_plots,
                             top_trial_numbers=top_nums)

            # 5. EDF
            if len(phase_trials) > 1:
                plot_edf(phase_study, phase, plot_dir, args.show_plots)

            # 6. Timeline
            plot_timeline(phase_study, phase, plot_dir, args.show_plots)

            # 7. Custom: Learning Curves
            if db_conn is not None and len(top_curr_phase) > 0:
                plot_learning_curves(top_curr_phase, phase, plot_dir, db_conn, args.show_plots)

            # 8. Custom: Time-to-Reach Success Thresholds
            if db_conn is not None and len(top_curr_phase) > 0:
                plot_time_to_reach(top_curr_phase, phase, plot_dir, db_conn, args.show_plots)

            print(f"  ✅ {stage_title(phase)} plots saved.")

        except Exception as e:
            print(f"  ⚠ {stage_title(phase)} plot error: {e}")

    # ── Final Dashboard ────────────────────────────────────────────────────────
    if phases:
        generate_dashboard(args.save_dir, db_name, list(phases))

    if db_conn is not None:
        db_conn.close()


if __name__ == "__main__":
    main()
