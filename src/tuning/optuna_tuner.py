import os
import tyro
import optuna
import torch
import pandas as pd
from datetime import datetime

import copy
from dataclasses import dataclass

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import apply_experiment_config, save_run_config
from CarlaBEV.envs import make_env
from src.utils.logger import DRLogger
from src.trainers.ppo import train_ppo


@dataclass
class OptunaArgs:
    exp_id: int = 26
    phase: int = 1 # Choose 1 or 2
    n_trials_phase_1: int = 100
    n_trials_phase_2: int = 50
    timesteps_phase_1: int = 1_000_000
    timesteps_phase_2: int = 2_000_000
    save_every_phase_1: int = 25 
    save_every_phase_2: int = 25
    eval_episodes: int = 30
    eval_final_episodes: int = 100
    
    top_k_phase_1: int = 10 # Number of best trials to consider for Phase 2

def phase_1_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs) -> float:
    import copy
    args = copy.deepcopy(base_args)

    # Sample ONLY continuous hyperparameters
    args.ppo.learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    args.ppo.ent_coef_start = trial.suggest_float("ent_coef_start", 5e-4, 2e-2, log=True)
    args.ppo.vf_coef_start = trial.suggest_float("vf_coef_start", 0.3, 0.9)
    args.ppo.clip_coef_start = trial.suggest_float("clip_coef_start", 0.1, 0.25)
    args.ppo.gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    args.ppo.gamma = trial.suggest_float("gamma", 0.95, 0.999)

    # Fixed values for phase 1 (defaults or specifically chosen)
    args.ppo.num_steps = 256
    args.ppo.update_epochs = 4
    args.ppo.num_minibatches = 4

    # Calculate decay 
    args.ppo.ent_coef_end = args.ppo.ent_coef_start * 0.2
    args.ppo.clip_coef_end = max(0.08, args.ppo.clip_coef_start * 0.7)
    args.ppo.vf_coef_end = args.ppo.vf_coef_start
    
    # 3. Budget settings
    args.ppo.total_timesteps = opt_args.timesteps_phase_1
    args.save_every = opt_args.save_every_phase_1
    args.eval_episodes = opt_args.eval_episodes
    args.eval_final_episodes = opt_args.eval_final_episodes
    
    # Enable SQLite auxiliary logging
    args.logging.db_path = f"results/carlabev_optuna_{opt_args.exp_id}.db"
    args.logging.trial_number = trial.number
    
    # Disable heavy IO tracking during Phase 1
    args.save_model = False
    args.capture_video = False
    
    trial.set_user_attr("phase", 1)

    return _run_trial_training(trial, args)

def phase_2_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs, top_continuous_params: dict) -> float:
    import copy
    args = copy.deepcopy(base_args)
    
    # Apply the best continuous hyperparameters found in Phase 1
    args.ppo.learning_rate = top_continuous_params["learning_rate"]
    args.ppo.ent_coef_start = top_continuous_params["ent_coef_start"]
    args.ppo.vf_coef_start = top_continuous_params["vf_coef_start"]
    args.ppo.clip_coef_start = top_continuous_params["clip_coef_start"]
    args.ppo.gae_lambda = top_continuous_params["gae_lambda"]
    args.ppo.gamma = top_continuous_params["gamma"]
    
    # Calculate decay based on the Phase 1 top params
    args.ppo.ent_coef_end = args.ppo.ent_coef_start * 0.2
    args.ppo.clip_coef_end = max(0.08, args.ppo.clip_coef_start * 0.7)
    args.ppo.vf_coef_end = args.ppo.vf_coef_start

    # Sample categorical hyperparameters (num_steps, update_epochs, num_minibatches)
    args.ppo.num_steps = trial.suggest_categorical("num_steps", [64, 128, 256, 512])
    args.ppo.update_epochs = trial.suggest_int("update_epochs", 3, 10)
    args.ppo.num_minibatches = trial.suggest_categorical("num_minibatches", [2, 4, 8])

    # 3. Budget settings
    args.ppo.total_timesteps = opt_args.timesteps_phase_2
    args.save_every = opt_args.save_every_phase_2
    args.eval_episodes = opt_args.eval_episodes
    args.eval_final_episodes = opt_args.eval_final_episodes
    
    # Enable SQLite auxiliary logging
    args.logging.db_path = f"results/carlabev_optuna_{opt_args.exp_id}.db"
    args.logging.trial_number = trial.number
    
    trial.set_user_attr("phase", 2)
    
    return _run_trial_training(trial, args)

def _run_trial_training(trial: optuna.Trial, args: ArgsCarlaBEV) -> float:

    # Modify exp_name to be trial-specific
    args.exp_name = f"{args.exp_name}_optuna_trial_{trial.number}"
    
    # Save config for this trial
    save_run_config(args)

    # Save trial attributes for filtering
    trial.set_user_attr("action_space", args.env.action_space)
    trial.set_user_attr("traffic_enabled", args.env.traffic_enabled)
    trial.set_user_attr("input_type", "masks" if args.env.masked else "rgb")
    trial.set_user_attr("fov_masked", args.env.fov_masked)
    trial.set_user_attr("reward_type", args.env.reward_type)
    trial.set_user_attr("curriculum", args.env.curriculum_mode if args.env.curriculum_enabled else "off")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 4. Initialize environments and logger
    envs = make_env(args)
    logger = DRLogger(config=args, stats_interval=100)

    # 5. Run training
    # Optuna may raise TrialPruned inside train_ppo
    final_score = train_ppo(args, envs, logger, device, trial=trial)
    
    return final_score


def main():
    cli_args = tyro.cli(OptunaArgs)
    
    # Create base ArgsCarlaBEV to get environment configuration
    base_args = ArgsCarlaBEV(exp_id=cli_args.exp_id)
    base_args = apply_experiment_config(base_args, cli_args.exp_id)
    print(f"⚙️ Running Optuna Hyperparameter tuning for Base Experiment ID = {cli_args.exp_id}")
    
    # We will use MedianPruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100, interval_steps=50)
    
    # Implement SQLite storage with concurrency support
    os.makedirs("results", exist_ok=True)
    storage_name = f"sqlite:///results/carlabev_optuna_{cli_args.exp_id}.db"
    
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"connect_args": {"timeout": 60}}
    )

    sampler = optuna.samplers.TPESampler(constant_liar=True)
    
    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,
        direction="maximize", 
        pruner=pruner,
        sampler=sampler,
        study_name=f"carlabev_optuna_{cli_args.exp_id}"
    )

    try:
        if cli_args.phase == 1:
            print(f"--- Starting Phase 1: Continuous Params (Budget: {cli_args.timesteps_phase_1}) ---")
            study.optimize(lambda trial: phase_1_objective(trial, base_args, cli_args), n_trials=cli_args.n_trials_phase_1)
        
        elif cli_args.phase == 2:
            print(f"--- Starting Phase 2: Categorical Params (Budget: {cli_args.timesteps_phase_2}) ---")
            
            # Fetch completed Phase 1 trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("phase") == 1]
            if len(completed_trials) == 0:
                print("Error: Cannot run Phase 2 without completed Phase 1 trials in the database.")
                return

            completed_trials.sort(key=lambda t: t.value, reverse=True)
            top_trials = completed_trials[:cli_args.top_k_phase_1]
            print(f"Found {len(completed_trials)} Phase 1 trials. Using parameters from top {len(top_trials)} trials.")
            
            # We will use the best trial from phase 1 for each phase 2 trial
            # Or sample from the top config list for a bit more variation
            # Here we just take the absolute best trial's continuous params
            best_phase_1_trial = top_trials[0]
            top_params = {
                "learning_rate": best_phase_1_trial.params["learning_rate"],
                "ent_coef_start": best_phase_1_trial.params["ent_coef_start"],
                "vf_coef_start": best_phase_1_trial.params["vf_coef_start"],
                "clip_coef_start": best_phase_1_trial.params["clip_coef_start"],
                "gae_lambda": best_phase_1_trial.params["gae_lambda"],
                "gamma": best_phase_1_trial.params["gamma"],
            }
            
            print("Selected continuous params for Phase 2:", top_params)
            study.optimize(lambda trial: phase_2_objective(trial, base_args, cli_args, top_params), n_trials=cli_args.n_trials_phase_2)
            
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
