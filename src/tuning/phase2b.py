import copy
import numpy as np
import optuna

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import run_experiment
from src.tuning.optuna_utils import OptunaArgs

def phase_2b_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs, top_params: dict) -> float:
    """Tunes PPO Coefficients and Regularizations using top categorical and continuous params"""
    
    # Sample PPO coefficients
    sampled_clip_coef_start = trial.suggest_float("clip_coef_start", 0.1, 0.25)
    sampled_ent_coef_start = trial.suggest_float("ent_coef_start", 5e-4, 3e-2, log=True)
    sampled_vf_coef_start = trial.suggest_float("vf_coef_start", 0.4, 0.9)
    sampled_max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    # Sample Decay factors
    sampled_ent_decay_factor = trial.suggest_float("ent_decay_factor", 0.1, 0.5)
    sampled_vf_decay_factor = trial.suggest_float("vf_decay_factor", 0.5, 1.0)
    sampled_clip_decay_factor = trial.suggest_float("clip_decay_factor", 0.4, 0.9)

    scores = []
    base_seed = base_args.seed

    # Robust seeding: ensures independence across trials but strict reproducibility
    trial_offset = trial.number * 10000
    
    for i in range(opt_args.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + trial_offset + i
        
        # Apply the best fixed parameters from Phase 1 and 2a
        args.ppo.learning_rate = top_params["learning_rate"]
        args.ppo.gae_lambda = top_params["gae_lambda"]
        args.ppo.gamma = top_params["gamma"]
        args.ppo.num_steps = top_params["num_steps"]
        args.ppo.update_epochs = top_params["update_epochs"]
        args.ppo.num_minibatches = top_params["num_minibatches"]

        # Apply the explicitly sampled Phase 2b parameters
        args.ppo.clip_coef_start = sampled_clip_coef_start
        args.ppo.ent_coef_start = sampled_ent_coef_start
        args.ppo.vf_coef_start = sampled_vf_coef_start
        args.ppo.max_grad_norm = sampled_max_grad_norm

        # Apply decay factors directly
        args.ppo.ent_decay_factor = sampled_ent_decay_factor
        args.ppo.vf_decay_factor = sampled_vf_decay_factor
        args.ppo.clip_decay_factor = sampled_clip_decay_factor

        # Budget settings
        args.ppo.total_timesteps = opt_args.timesteps_phase_2b
        args.eval_episodes = opt_args.eval_episodes
        args.eval_final_episodes = opt_args.eval_final_episodes
        
        # Enable SQLite auxiliary logging
        trial.set_user_attr("phase", "2b")
        
        score = run_experiment(args, trial=trial, seed_idx=i)
        scores.append(score)
        
    return float(np.mean(scores))
