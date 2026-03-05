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

    for i in range(opt_args.num_seeds):
        args = copy.deepcopy(base_args)
        args.seed = base_seed + i
        
        args.ppo.learning_rate = sampled_lr
        args.ppo.gae_lambda = sampled_gae_lambda
        args.ppo.gamma = sampled_gamma

        # Fixed values for phase 1 (defaults or specifically chosen)
        args.ppo.num_steps = 256
        args.ppo.update_epochs = 4
        args.ppo.num_minibatches = 6

        # Calculate decay using existing base params
        args.ppo.ent_coef_end = args.ppo.ent_coef_start * 0.2
        args.ppo.clip_coef_end = max(0.08, args.ppo.clip_coef_start * 0.7)
        args.ppo.vf_coef_end = args.ppo.vf_coef_start
        
        # Budget settings
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

        score = run_experiment(args, trial=trial, seed_idx=i)
        scores.append(score)

    return float(np.mean(scores))
