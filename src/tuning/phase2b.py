import copy
import numpy as np
import optuna

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import run_experiment
from src.tuning.optuna_utils import OptunaArgs

def phase_2b_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs, top_params: dict) -> float:
    """Tunes PPO Coefficients and Regularizations using top categorical and continuous params"""
    
    # Sample PPO coefficients
    sampled_clip_coef_start = trial.suggest_float("clip_coef_start", 0.05, 0.4)
    sampled_ent_coef_start = trial.suggest_float("ent_coef_start", 1e-4, 0.1, log=True)
    sampled_vf_coef_start = trial.suggest_float("vf_coef_start", 0.1, 1.0)
    sampled_max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 2.0)
    
    # Sample Decay factors
    sampled_ent_decay_factor = trial.suggest_float("ent_decay_factor", 0.01, 1.0)
    sampled_vf_decay_factor = trial.suggest_float("vf_decay_factor", 0.5, 1.0)

    scores = []
    base_seed = base_args.seed

    for i in range(opt_args.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + i
        
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

        # Compute endpoints using decay factors
        args.ppo.ent_coef_end = args.ppo.ent_coef_start * sampled_ent_decay_factor
        args.ppo.vf_coef_end = args.ppo.vf_coef_start * sampled_vf_decay_factor
        
        # Keep clip decay fixed to a reasonable heuristic or match what was there
        args.ppo.clip_coef_end = max(0.01, args.ppo.clip_coef_start * 0.5)

        # Budget settings
        args.ppo.total_timesteps = opt_args.timesteps_phase_2b
        args.save_every = opt_args.save_every_phase_2b
        args.eval_episodes = opt_args.eval_episodes
        args.eval_final_episodes = opt_args.eval_final_episodes
        
        # Enable SQLite auxiliary logging
        trial.set_user_attr("phase", "2b")
        
        score = run_experiment(args, trial=trial, seed_idx=i)
        scores.append(score)
        
    return float(np.mean(scores))
