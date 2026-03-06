import os
import tyro
import optuna
import pandas as pd
from datetime import datetime
import time
import random

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import apply_experiment_config, get_global_db_path, get_global_study_name
from src.tuning.optuna_utils import OptunaArgs
from src.tuning.phase1 import phase_1_objective
from src.tuning.phase2a import phase_2a_objective
from src.tuning.phase2b import phase_2b_objective
from src.tuning.phase3 import phase_3_objective


def main():
    cli_args = tyro.cli(OptunaArgs)
    
    # Create base ArgsCarlaBEV to get environment configuration
    base_args = ArgsCarlaBEV(exp_id=cli_args.exp_id)
    base_args = apply_experiment_config(base_args, cli_args.exp_id)
    print(f"⚙️ Running Optuna Hyperparameter tuning for Base Experiment ID = {cli_args.exp_id}")
    
    # Reverted to MedianPruner per user request
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,  # Number of initial trials before pruning starts
        n_warmup_steps=5,    # Number of evaluations to wait before pruning a given trial
        interval_steps=1
    )
    # Implement SQLite storage with concurrency support
    os.makedirs("results", exist_ok=True)
    db_path = get_global_db_path()
    storage_name = f"sqlite:///{db_path}"
    
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"connect_args": {"timeout": 60}}
    )

    sampler = optuna.samplers.TPESampler(constant_liar=True)
    
    study = None
    
    # Attempt to create or load the study robustly to avoid race conditions when many Slurm nodes start simultaneously
    for _ in range(20):
        try:
            study = optuna.create_study(
                storage=storage,
                load_if_exists=True,
                direction="maximize", 
                pruner=pruner,
                sampler=sampler,
                study_name=get_global_study_name()
            )
            break
        except Exception as e:
            print(f"Study creation collision or lock detected: {e}. Retrying in a few seconds...")
            time.sleep(random.uniform(2, 6))
            
    if study is None:
        raise RuntimeError("Failed to create or load the Optuna study after multiple attempts due to database locking.")

    try:
        if str(cli_args.phase) == "1":
            print(f"--- Starting Phase 1: Continuous Params (Budget: {cli_args.timesteps_phase_1}) ---")
            completed_phase_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "1"]
            trials_needed = cli_args.n_trials_phase_1 - len(completed_phase_trials)
            if trials_needed <= 0:
                print(f"Phase 1 already has {len(completed_phase_trials)} completed trials (Target: {cli_args.n_trials_phase_1}). Skipping optimization.")
            else:
                study.optimize(lambda trial: phase_1_objective(trial, base_args, cli_args), n_trials=trials_needed)
        
        elif str(cli_args.phase) == "2a":
            print(f"--- Starting Phase 2a: Categorical Params (Budget: {cli_args.timesteps_phase_2a}) ---")
            
            # Fetch completed Phase 1 trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "1"]
            if len(completed_trials) == 0:
                print("Error: Cannot run Phase 2a without completed Phase 1 trials in the database.")
                return

            completed_trials.sort(key=lambda t: t.value, reverse=True)
            top_trials = completed_trials[:cli_args.top_k_phase_1]
            print(f"Found {len(completed_trials)} Phase 1 trials. Using parameters from top {len(top_trials)} trials.")
            
            # We will use the best trial from phase 1 for each phase 2a trial
            best_phase_1_trial = top_trials[0]
            top_params = {
                "learning_rate": best_phase_1_trial.params["learning_rate"],
                "gae_lambda": best_phase_1_trial.params["gae_lambda"],
                "gamma": best_phase_1_trial.params["gamma"],
            }
            
            print("Selected continuous params for Phase 2a:", top_params)
            completed_phase_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "2a"]
            trials_needed = cli_args.n_trials_phase_2a - len(completed_phase_trials)
            if trials_needed <= 0:
                print(f"Phase 2a already has {len(completed_phase_trials)} completed trials (Target: {cli_args.n_trials_phase_2a}). Skipping optimization.")
            else:
                study.optimize(lambda trial: phase_2a_objective(trial, base_args, cli_args, top_params), n_trials=trials_needed)
            
        elif str(cli_args.phase) == "2b":
            print(f"--- Starting Phase 2b: PPO Coefficients (Budget: {cli_args.timesteps_phase_2b}) ---")
            
            # Fetch completed Phase 2a trials to get both fixed continuous and tuned categorical settings
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "2a"]
            if len(completed_trials) == 0:
                print("Error: Cannot run Phase 2b without completed Phase 2a trials in the database.")
                return

            completed_trials.sort(key=lambda t: t.value, reverse=True)
            top_trials = completed_trials[:cli_args.top_k_phase_2a]
            print(f"Found {len(completed_trials)} Phase 2a trials. Using parameters from top {len(top_trials)} trials.")
            
            best_phase_2a_trial = top_trials[0]
            
            phase_1_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "1"]
            phase_1_trials.sort(key=lambda t: t.value, reverse=True)
            best_p1_trial = phase_1_trials[0]

            top_params = {
                "learning_rate": best_p1_trial.params["learning_rate"],
                "gae_lambda": best_p1_trial.params["gae_lambda"],
                "gamma": best_p1_trial.params["gamma"],
                "num_steps": best_phase_2a_trial.params["num_steps"],
                "update_epochs": best_phase_2a_trial.params["update_epochs"],
                "num_minibatches": best_phase_2a_trial.params["num_minibatches"],
            }
            
            print("Selected fixed params for Phase 2b Tuning:", top_params)
            completed_phase_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "2b"]
            trials_needed = cli_args.n_trials_phase_2b - len(completed_phase_trials)
            if trials_needed <= 0:
                print(f"Phase 2b already has {len(completed_phase_trials)} completed trials (Target: {cli_args.n_trials_phase_2b}). Skipping optimization.")
            else:
                study.optimize(lambda trial: phase_2b_objective(trial, base_args, cli_args, top_params), n_trials=trials_needed)
            
        elif str(cli_args.phase) == "3":
            print(f"--- Starting Phase 3: Architecture Params (Budget: {cli_args.timesteps_phase_3}) ---")
            
            # Fetch completed Phase 2b trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "2b"]
            if len(completed_trials) == 0:
                print("Error: Cannot run Phase 3 without completed Phase 2b trials in the database (or run with phase 2 baselines).")
                # Fallback to phase 2 baselines if no 2b exist could be implemented, but strict is safer
                return

            completed_trials.sort(key=lambda t: t.value, reverse=True)
            top_trials = completed_trials[:cli_args.top_k_phase_2b]
            print(f"Found {len(completed_trials)} Phase 2b trials. Using parameters from top {len(top_trials)} trials.")
            
            best_phase_2b_trial = top_trials[0]
            
            phase_2a_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "2a"]
            phase_2a_trials.sort(key=lambda t: t.value, reverse=True)
            best_p2a_trial = phase_2a_trials[0]
            
            phase_1_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "1"]
            phase_1_trials.sort(key=lambda t: t.value, reverse=True)
            best_p1_trial = phase_1_trials[0]

            top_params = {
                "learning_rate": best_p1_trial.params["learning_rate"],
                "gae_lambda": best_p1_trial.params["gae_lambda"],
                "gamma": best_p1_trial.params["gamma"],
                "num_steps": best_p2a_trial.params["num_steps"],
                "update_epochs": best_p2a_trial.params["update_epochs"],
                "num_minibatches": best_p2a_trial.params["num_minibatches"],
                "clip_coef_start": best_phase_2b_trial.params["clip_coef_start"],
                "ent_coef_start": best_phase_2b_trial.params["ent_coef_start"],
                "vf_coef_start": best_phase_2b_trial.params["vf_coef_start"],
                "max_grad_norm": best_phase_2b_trial.params["max_grad_norm"],
                "ent_decay_factor": best_phase_2b_trial.params["ent_decay_factor"],
                "vf_decay_factor": best_phase_2b_trial.params["vf_decay_factor"],
                "clip_decay_factor": best_phase_2b_trial.params["clip_decay_factor"],
            }
            
            print("Selected fixed params for Phase 3 Network Tuning:", top_params)
            completed_phase_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "3"]
            trials_needed = cli_args.n_trials_phase_3 - len(completed_phase_trials)
            if trials_needed <= 0:
                print(f"Phase 3 already has {len(completed_phase_trials)} completed trials (Target: {cli_args.n_trials_phase_3}). Skipping optimization.")
            else:
                study.optimize(lambda trial: phase_3_objective(trial, base_args, cli_args, top_params), n_trials=trials_needed)
            
    except KeyboardInterrupt:
        print("\nInterrupted early! Saving study results so far...")
        
    print(f"Number of finished trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print("Best trial overall:")
        trial = study.best_trial
        
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        print("  Phase: ", trial.user_attrs.get("phase", "Unknown"))
    else:
        print("No trials completed. Skipping best trial extraction.")
        
    # Save study statistics to CSV
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/optuna_study_{base_args.exp_id}_{timestamp}.csv"
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved Optuna statistics to: {csv_path}")


if __name__ == "__main__":
    main()
