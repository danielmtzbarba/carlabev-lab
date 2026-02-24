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
    exp_id: int = 1
    n_trials: int = 100
    timesteps_budget: int = 2_000_000
    save_every: int = 100

def objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs) -> float:
    # 1. Start from base configurations
    # We deepcopy to avoid mutating across trials, though tyro/dataclass might need a clean re-init
    # Better to just re-apply the base configs to a fresh ArgsCarlaBEV
    import copy
    args = copy.deepcopy(base_args)

    # 2. Sample hyperparameters
    args.ppo.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    args.ppo.ent_coef_start = trial.suggest_float("ent_coef_start", 1e-4, 0.1, log=True)
    args.ppo.vf_coef_start = trial.suggest_float("vf_coef_start", 0.1, 1.0)
    args.ppo.clip_coef_start = trial.suggest_float("clip_coef_start", 0.1, 0.4)
    args.ppo.gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    args.ppo.gamma = trial.suggest_float("gamma", 0.9, 0.999)
    args.ppo.num_steps = trial.suggest_categorical("num_steps", [64, 128, 256, 512])
    args.ppo.update_epochs = trial.suggest_int("update_epochs", 3, 10)
    args.ppo.num_minibatches = trial.suggest_categorical("num_minibatches", [2, 4, 8])

    # Let ends equal starts for simplicity, or we can tune decays later
    args.ppo.ent_coef_end = args.ppo.ent_coef_start / 10.0
    args.ppo.vf_coef_end = args.ppo.vf_coef_start / 2.0
    args.ppo.clip_coef_end = args.ppo.clip_coef_start / 2.0

    # 3. Reduce Timesteps Budget for faster trials
    args.ppo.total_timesteps = opt_args.timesteps_budget
    args.save_every = opt_args.save_every

    # Modify exp_name to be trial-specific
    args.exp_name = f"{args.exp_name}_optuna_trial_{trial.number}"
    
    # Save config for this trial
    save_run_config(args)

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
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name=f"carlabev_optuna_{cli_args.exp_id}")
    
    try:
        study.optimize(lambda trial: objective(trial, base_args, cli_args), n_trials=cli_args.n_trials)
    except KeyboardInterrupt:
        print("\nInterrupted early! Saving study results so far...")
        
    print(f"Number of finished trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print("Best trial:")
        trial = study.best_trial
        
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
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
