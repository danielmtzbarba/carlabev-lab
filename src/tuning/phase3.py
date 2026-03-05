import copy
import numpy as np
import optuna

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import run_experiment
from src.tuning.optuna_utils import OptunaArgs

def phase_3_objective(trial: optuna.Trial, base_args: ArgsCarlaBEV, opt_args: OptunaArgs, top_params: dict) -> float:
    """Tunes Network Architecture shapes using optimal Phase 1/2 baselines"""
    # Base Network policy configuration
    base_channels = [32, 64, 64]
    
    # Sample Architecture multipliers
    channel_multiplier = trial.suggest_categorical("channel_multiplier", [0.5, 1.0, 1.5, 2.0])
    sampled_fc_size = trial.suggest_categorical("fc_size", [256, 512, 1024])
    
    sampled_channels = [int(c * channel_multiplier) for c in base_channels]

    scores = []
    base_seed = base_args.seed

    # Robust seeding: ensures independence across trials but strict reproducibility
    trial_offset = trial.number * 10000
    
    for i in range(opt_args.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + trial_offset + i
        
        # Apply the best Phase 1/2 fixed parameters
        args.ppo.learning_rate = top_params["learning_rate"]
        args.ppo.gae_lambda = top_params["gae_lambda"]
        args.ppo.gamma = top_params["gamma"]
        args.ppo.num_steps = top_params["num_steps"]
        args.ppo.update_epochs = top_params["update_epochs"]
        args.ppo.num_minibatches = top_params["num_minibatches"]

        # Apply Phase 2b parameters
        args.ppo.clip_coef_start = top_params["clip_coef_start"]
        args.ppo.ent_coef_start = top_params["ent_coef_start"]
        args.ppo.vf_coef_start = top_params["vf_coef_start"]
        args.ppo.max_grad_norm = top_params["max_grad_norm"]

        args.ppo.ent_decay_factor = top_params["ent_decay_factor"]
        args.ppo.vf_decay_factor = top_params["vf_decay_factor"]
        args.ppo.clip_decay_factor = top_params["clip_decay_factor"]

        # Apply the sampled Phase 3 parameters
        args.ppo.channels = sampled_channels
        args.ppo.fc_size = sampled_fc_size

        # Budget settings
        args.ppo.total_timesteps = opt_args.timesteps_phase_3
        args.eval_episodes = opt_args.eval_episodes
        args.eval_final_episodes = opt_args.eval_final_episodes
        
        # Enable SQLite auxiliary logging
        args.logging.db_path = f"results/carlabev_optuna_{opt_args.exp_id}.db"
        args.logging.trial_number = trial.number
        
        trial.set_user_attr("phase", 3)
        
        score = run_experiment(args, trial=trial, seed_idx=i)
        scores.append(score)
        
    return float(np.mean(scores))
