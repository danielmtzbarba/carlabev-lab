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

def print_trial_info(trial, db_conn, rank=None):
    prefix = f"Rank {rank}: " if rank else ""
    print(f"\n{prefix}Trial {trial.number}")
    print(f"  Value: {trial.value}")
    print("  Parameters:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")
    
    phase = trial.user_attrs.get("phase", "Unknown")
    print(f"  Phase: {phase}")

    if db_conn is not None:
        cursor = db_conn.cursor()
        
        # Fetch the latest evaluation logs for this trial
        cursor.execute('''
            SELECT * FROM trial_eval_logs 
            WHERE trial_number = ? 
            ORDER BY global_step DESC LIMIT 1
        ''', (trial.number,))
        
        eval_row = cursor.fetchone()
        if eval_row:
            # Reconstruct column names to values mapping
            col_names = [description[0] for description in cursor.description]
            eval_data = dict(zip(col_names, eval_row))
            
            print("  Final Evaluation Metrics (from DB):")
            metrics_keys = ['mean_return', 'std_return', 'mean_length', 'success_rate', 'collision_rate', 'unfinished_rate']
            for k in metrics_keys:
                if k in eval_data and eval_data[k] is not None:
                    print(f"    {k}: {eval_data[k]:.4f}")
            
            print("  Success Thresholds Reached (from DB):")
            threshold_keys = [k for k in col_names if k.startswith('time_to_reach_')]
            found_thresholds = False
            for k in threshold_keys:
                if eval_data[k] is not None:
                    found_thresholds = True
                    print(f"    {k}: {eval_data[k]}")
            if not found_thresholds:
                print("    None")
                
        # Give a quick summary of training performance
        cursor.execute('''
            SELECT MAX(global_step), AVG(mean_return), AVG(pg_loss), AVG(v_loss)
            FROM trial_train_logs
            WHERE trial_number = ?
        ''', (trial.number,))
        train_row = cursor.fetchone()
        if train_row and train_row[0] is not None:
            print(f"  Training Summary (from DB):")
            print(f"    Total Steps: {train_row[0]}")
            print(f"    Avg Return:  {train_row[1]:.4f}")
            print(f"    Avg PG Loss: {train_row[2]:.4f}")
            print(f"    Avg V Loss:  {train_row[3]:.4f}")
            
    else:
        # Fallback to user attributes if DB is not available
        thresholds = {k: v for k, v in trial.user_attrs.items() if "time_to_reach" in k or "step_to_reach" in k}
        metrics = {k: v for k, v in trial.user_attrs.items() if k.startswith("final_")}
        
        if metrics:
            print("  Final Evaluation Metrics (Attributes Backup):")
            for k, v in metrics.items():
                print(f"    {k.replace('final_', '')}: {v:.4f}" if isinstance(v, float) else f"    {k.replace('final_', '')}: {v}")
                
        if thresholds:
            print("  Success Thresholds Reached (Attributes Backup):")
            for k, v in thresholds.items():
                print(f"    {k}: {v}")

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
    print_trial_info(study.best_trial, db_conn=db_conn)
    
    complete_trials.sort(key=lambda t: t.value, reverse=True) # Maximize direction
    
    print(f"\n--- Top {min(args.top_k, len(complete_trials))} Trials ---")
    for i, trial in enumerate(complete_trials[:args.top_k]):
        print_trial_info(trial, db_conn=db_conn, rank=i+1)

    if db_conn is not None:
        db_conn.close()

    # Visualization
    plot_dir = os.path.join(args.save_dir, f"{study_name}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\nGenerating plots in {plot_dir}...")
    
    try:
        # 1. Optimization History
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html(os.path.join(plot_dir, "optimization_history.html"))
        
        # 2. Parameter Importances
        if len(complete_trials) > 1:
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html(os.path.join(plot_dir, "param_importances.html"))
            
        # 3. Parallel Coordinate
        if len(complete_trials) > 1:
            fig_parallel = vis.plot_parallel_coordinate(study)
            fig_parallel.write_html(os.path.join(plot_dir, "parallel_coordinate.html"))
            
        # 4. Slice Plot
        if len(complete_trials) > 1:
            fig_slice = vis.plot_slice(study)
            fig_slice.write_html(os.path.join(plot_dir, "slice_plot.html"))
            
        print("Plots successfully saved!")
        
        if args.show_plots:
            fig_history.show()
            if len(complete_trials) > 1:
                fig_importance.show()
                fig_parallel.show()
                
    except Exception as e:
        print(f"Warning: Could not save some plots (perhaps parsing error or only 1 trial?). Error: {e}")

if __name__ == "__main__":
    main()
