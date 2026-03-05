import copy
import numpy as np
import optuna

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import run_experiment
from src.tuning.optuna_utils import OptunaArgs

def phase_1_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs) -> float:
    """Tunes Continuous Learning Dynamics (LR, Lambda, Gamma)"""
    # Sample ONLY continuous hyperparameters approved for Phase 1
    sampled_lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    sampled_gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    sampled_gamma = trial.suggest_float("gamma", 0.98, 0.9995)

    scores = []
    base_seed = base_args.seed
    
    # Robust seeding: ensures independence across trials but strict reproducibility
    trial_offset = trial.number * 10000

    for i in range(opt_args.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + trial_offset + i
        
        args.ppo.learning_rate = sampled_lr
        args.ppo.gae_lambda = sampled_gae_lambda
        args.ppo.gamma = sampled_gamma

        # Budget settings
        args.ppo.total_timesteps = opt_args.timesteps_phase_1
        args.eval_episodes = opt_args.eval_episodes
        args.eval_final_episodes = opt_args.eval_final_episodes
        
        # Enable SQLite auxiliary logging
        args.logging.db_path = f"results/carlabev_optuna_{opt_args.exp_id}.db"
        args.logging.trial_number = trial.number
        
        # Disable heavy IO tracking during Phase 1
        args.save_model = False
        args.capture_video = False
        
        trial.set_user_attr("phase", 1)

        score = run_experiment(args, trial=trial, seed_idx=i)
        scores.append(score)

    return float(np.mean(scores))
