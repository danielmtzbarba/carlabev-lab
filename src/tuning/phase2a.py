import copy
import numpy as np
import optuna

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import run_experiment
from src.tuning.optuna_utils import OptunaArgs

def phase_2a_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs, top_continuous_params: dict) -> float:
    """Tunes Categorical Environment Rollout constraints using top continuous params"""
    # Apply the best continuous hyperparameters found in Phase 1
    sampled_lr = top_continuous_params["learning_rate"]
    sampled_gae_lambda = top_continuous_params["gae_lambda"]
    sampled_gamma = top_continuous_params["gamma"]

    # Sample categorical hyperparameters (num_steps, update_epochs, num_minibatches)
    sampled_num_steps = trial.suggest_categorical("num_steps", [128, 256, 512])
    sampled_update_epochs = trial.suggest_int("update_epochs", 3, 8)
    sampled_num_minibatches = trial.suggest_categorical("num_minibatches", [2, 4, 8])
    
    scores = []
    base_seed = base_args.seed

    for i in range(opt_args.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + i
        
        args.ppo.learning_rate = sampled_lr
        args.ppo.gae_lambda = sampled_gae_lambda
        args.ppo.gamma = sampled_gamma
        
        args.ppo.num_steps = sampled_num_steps
        args.ppo.update_epochs = sampled_update_epochs
        args.ppo.num_minibatches = sampled_num_minibatches

        # Calculate decay based on the Phase 1 top params
        args.ppo.ent_coef_end = args.ppo.ent_coef_start * 0.2
        args.ppo.clip_coef_end = max(0.08, args.ppo.clip_coef_start * 0.7)
        args.ppo.vf_coef_end = args.ppo.vf_coef_start

        # Budget settings
        args.ppo.total_timesteps = opt_args.timesteps_phase_2a
        args.save_every = opt_args.save_every_phase_2a
        args.eval_episodes = opt_args.eval_episodes
        args.eval_final_episodes = opt_args.eval_final_episodes
        
        # Enable SQLite auxiliary logging
        args.logging.db_path = f"results/carlabev_optuna_{opt_args.exp_id}.db"
        args.logging.trial_number = trial.number
        trial.set_user_attr("phase", "2a")
        
        score = run_experiment(args, trial=trial, seed_idx=i)
        scores.append(score)
        
    return float(np.mean(scores))
