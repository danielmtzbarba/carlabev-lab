import os
import tyro
import optuna
import optuna.visualization as vis
import pandas as pd
from dataclasses import dataclass

@dataclass
class AnalyzeArgs:
    exp_id: int = 26
    top_k: int = 5
    show_plots: bool = False
    save_dir: str = "results"

def print_compact_trial_info(trial, db_conn, rank=None):
    if db_conn is None:
        print(f"Rank {rank} | Trial {trial.number:<3} | Score: {trial.value:.4f} | Phase: {trial.user_attrs.get('phase', '?')}")
        return

    cursor = db_conn.cursor()
    # Get final eval
    cursor.execute('SELECT success_rate, collision_rate FROM trial_eval_logs WHERE trial_number = ? ORDER BY global_step DESC LIMIT 1', (trial.number,))
    eval_row = cursor.fetchone()
    
    succ = eval_row[0] if eval_row and eval_row[0] is not None else 0.0
    coll = eval_row[1] if eval_row and eval_row[1] is not None else 0.0
    
    # Get train summary 
    cursor.execute('SELECT MAX(global_step), AVG(mean_return) FROM trial_train_logs WHERE trial_number = ?', (trial.number,))
    train_row = cursor.fetchone()
    steps = train_row[0] if train_row and train_row[0] is not None else 0
    ret = train_row[1] if train_row and train_row[1] is not None else 0.0
    
    # Format params compactly
    params_str = ", ".join([f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in trial.params.items()])
    
    print(f"Rank {rank} | Trial {trial.number:<3} | Score: {trial.value:7.4f} | Succ: {succ:5.2f} Coll: {coll:5.2f} | Ret: {ret:6.2f} ({steps:<8} steps) | Phase: {trial.user_attrs.get('phase', '?'):<2} | {params_str}")


def main():
    args = tyro.cli(AnalyzeArgs)
    study_name = f"carlabev_optuna_{args.exp_id}"
    storage_name = f"sqlite:///results/{study_name}.db"
    
    if not os.path.exists(f"results/{study_name}.db"):
        print(f"Database results/{study_name}.db does not exist!")
        return

    import sqlite3
    db_conn = None
    real_path = storage_name.replace("sqlite:///", "")
    if os.path.exists(real_path):
        db_conn = sqlite3.connect(real_path)

    print(f"Loading study: {study_name} from {storage_name}")
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    
    print("\n--- Study Statistics ---")
    print(f"Total Trials: {len(study.trials)}")
    
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"  Complete: {len(complete_trials)}")
    print(f"  Pruned:   {len(pruned_trials)}")
    print(f"  Failed:   {len(failed_trials)}")
    print("-" * 24)

    if not complete_trials:
        print("No complete trials found yet.")
        return

    print("\n🏆 Best Trial Overall:")
    print_compact_trial_info(study.best_trial, db_conn=db_conn, rank="BEST")
    
    complete_trials.sort(key=lambda t: t.value, reverse=True) # Maximize direction
    
    print(f"\n--- Top {min(args.top_k, len(complete_trials))} Trials ---")
    top_trials = complete_trials[:args.top_k]
    for i, trial in enumerate(top_trials):
        print_compact_trial_info(trial, db_conn=db_conn, rank=i+1)

    # Visualization
    plot_dir = os.path.join(args.save_dir, f"{study_name}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\nGenerating plots in {plot_dir}...")
    
    # Common layout enhancements for publication quality
    layout_enhancements = dict(
        font=dict(family="Computer Modern, Arial, sans-serif", size=16, color="black"),
        title_font=dict(size=22, color="black", family="Computer Modern, Arial, sans-serif"),
        title_x=0.5, # Center title
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=40, t=80, b=80),
    )
    
    def apply_publication_style(fig, title):
        fig.update_layout(**layout_enhancements)
        fig.update_layout(title=title)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', color='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', color='black')
        return fig
    
    try:
        # 1. Optimization History
        fig_history = vis.plot_optimization_history(study)
        fig_history = apply_publication_style(fig_history, "Optimization History")
        fig_history.write_html(os.path.join(plot_dir, "optimization_history.html"))
        
        # 2. Parameter Importances
        if len(complete_trials) > 1:
            fig_importance = vis.plot_param_importances(study)
            fig_importance = apply_publication_style(fig_importance, "Hyperparameter Importances")
            fig_importance.write_html(os.path.join(plot_dir, "param_importances.html"))
            
        # 3. Contour (Topography / Heatmap)
        if len(complete_trials) > 1:
            fig_contour = vis.plot_contour(study)
            fig_contour = apply_publication_style(fig_contour, "Search Space Topography")
            fig_contour.write_html(os.path.join(plot_dir, "contour_plot.html"))
            
            fig_slice = vis.plot_slice(study)
            fig_slice = apply_publication_style(fig_slice, "Parameter Slice Analysis")
            fig_slice.write_html(os.path.join(plot_dir, "slice_plot.html"))

        # 5. Top Trials Learning Curves
        if db_conn is not None and len(top_trials) > 0:
            import plotly.graph_objects as go
            fig_learning = go.Figure()
            
            for rank, trial in enumerate(top_trials):
                # Fetch mean return curve
                df = pd.read_sql_query('SELECT global_step, mean_return FROM trial_train_logs WHERE trial_number = ? ORDER BY global_step', db_conn, params=(trial.number,))
                if not df.empty:
                    fig_learning.add_trace(go.Scatter(x=df['global_step'], y=df['mean_return'], mode='lines', name=f"Rank {rank+1} (Trial {trial.number})"))
                    
            if len(fig_learning.data) > 0:
                fig_learning = apply_publication_style(fig_learning, "Learning Curves (Top Trials)")
                fig_learning.write_html(os.path.join(plot_dir, "top_trials_learning_curves.html"))
                
        # 6. Time-to-Reach Success Thresholds
        if db_conn is not None and len(top_trials) > 0:
            fig_thresholds = go.Figure()
            threshold_cols = [
                '0_1','0_2','0_3','0_4','0_5','0_6','0_7','0_8','0_9','0_95','0_99'
            ]
            
            for rank, trial in enumerate(top_trials):
                # Fetch threshold array (latest row for this trial)
                df = pd.read_sql_query('SELECT * FROM trial_eval_logs WHERE trial_number = ? ORDER BY global_step DESC LIMIT 1', db_conn, params=(trial.number,))
                if not df.empty:
                    x_vals = []
                    y_vals = []
                    for tc in threshold_cols:
                        col_name = f'time_to_reach_{tc}'
                        if col_name in df.columns and pd.notna(df.iloc[0][col_name]):
                            hr_threshold = float(tc.replace('_', '.'))
                            x_vals.append(hr_threshold)
                            y_vals.append(df.iloc[0][col_name] / 3600.0) # Convert seconds to hours
                            
                    if len(x_vals) > 0:
                        fig_thresholds.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=f"Rank {rank+1} (Trial {trial.number})"))
            
            if len(fig_thresholds.data) > 0:
                fig_thresholds = apply_publication_style(fig_thresholds, "Walltime to Success Threshold (Hours)")
                fig_thresholds.update_xaxes(title_text="Success Rate Threshold")
                fig_thresholds.update_yaxes(title_text="Training Elapsed Time (Hours)")
                fig_thresholds.write_html(os.path.join(plot_dir, "top_trials_time_to_reach.html"))
            
        print("Plots successfully saved!")
        
        if args.show_plots:
            fig_history.show()
            if len(complete_trials) > 1:
                fig_importance.show()
                fig_contour.show()
                fig_slice.show()
                
    except Exception as e:
        print(f"Warning: Could not save some plots (perhaps parsing error or only 1 trial?). Error: {e}")
        
    finally:
        if db_conn is not None:
            db_conn.close()

if __name__ == "__main__":
    main()
