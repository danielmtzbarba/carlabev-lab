# src/config/experiment_loader.py

import os
import tyro
import yaml

from src.config.base_config import ArgsCarlaBEV, EnvConfig, PPOConfig

def get_global_study_name():
    return "carlabev"

def get_global_db_path():
    return "results/carlabev_optuna.db"


# ============================================================
# 1. Tabla de 24 configuraciones experimentales
# (Action Space, Traffic, Input, Reward, curriculum_mode, fov_mask)
# ============================================================

EXPERIMENT_TABLE = {
    # Reward and Input type - Traffic OFF:
    1:  ("discrete", "off", "rgb",   "shaping", "off", "off"),
    2:  ("discrete", "off", "rgb",   "shaping", "on",  "off"),
    3:  ("discrete", "off", "rgb",   "carl",    "off", "off"),
    4:  ("discrete", "off", "rgb",   "carl",    "on",  "off"),
    5:  ("discrete", "off", "masks", "shaping", "off", "off"),
    6:  ("discrete", "off", "masks", "shaping", "on",  "off"),
    7:  ("discrete", "off", "masks", "carl",    "off", "off"),
    8:  ("discrete", "off", "masks", "carl",    "on",  "off"),

    # Reward and Input type - Traffic ON:
    9:  ("discrete", "on",  "rgb",   "shaping", "off", "off"),
    10: ("discrete", "on",  "rgb",   "shaping", "on",  "off"),
    11: ("discrete", "on",  "rgb",   "carl",    "off", "off"),
    12: ("discrete", "on",  "rgb",   "carl",    "on",  "off"),
    13: ("discrete", "on",  "masks", "shaping", "off", "off"),
    14: ("discrete", "on",  "masks", "shaping", "on",  "off"),
    15: ("discrete", "on",  "masks", "carl",    "off", "off"),
    16: ("discrete", "on",  "masks", "carl",    "on",  "off"),

    # Curriculum learning variants
    17: ("discrete", "on", "rgb",   "carl", "vehicles_only", "off"),
    18: ("discrete", "on", "rgb",   "carl", "route_only",    "off"),
    19: ("discrete", "on", "rgb",   "carl", "both",          "off"),
    20: ("discrete", "on", "masks", "carl", "vehicles_only", "off"),
    21: ("discrete", "on", "masks", "carl", "route_only",    "off"),
    22: ("discrete", "on", "masks", "carl", "both",          "off"),

    # Baseline extra runs
    23: ("discrete", "off", "rgb",   "carl", "route_only", "off"),
    24: ("discrete", "off", "masks", "carl", "route_only", "off"),

    # Only Edge scenarios
    25: ("discrete", "on", "masks", "carl", "off", "off"),

    # FOV masked 
    26: ("discrete", "on", "masks", "carl", "route_only", "off"),
    27: ("discrete", "on", "masks", "carl", "route_only", "on"),

    # Continuous experiments
    28: ("continuous", "on", "masks", "carl", "route_only", "off"),
    29: ("continuous", "on", "masks", "carl", "route_only", "on"),
}


# ============================================================
# 2. Aplica parámetros según ID de experimento
# ============================================================


def apply_experiment_config(args: ArgsCarlaBEV, exp_id: int):
    action_space, traffic, input_type, reward_type, curriculum, fov_mask = EXPERIMENT_TABLE[exp_id]

    env: EnvConfig() = args.env
    ppo: PPOConfig() = args.ppo

    # ===== Action Space
    env.action_space = action_space

    # ===== Fov Mask (corners) 
    env.fov_masked = fov_mask == "on"

    # ===== Traffic
    env.traffic_enabled = traffic == "on"

    # ===== Input type
    if input_type == "rgb":
        env.masked = False
        env.obs_space = "bev"
    elif input_type == "masks":
        env.masked = True
        env.obs_space = "bev"

    # ===== Reward
    env.reward_type = reward_type  # You switch rewards in env builder

    # ===== Curriculum
    if curriculum == "off":
        env.curriculum_enabled = False
    else:
        env.curriculum_enabled = True
        if curriculum == "vehicles_only":
            env.curriculum_mode = "vehicles"
        elif curriculum == "route_only":
            env.curriculum_mode = "route"
        else:
            env.curriculum_mode = "both"

    # ===== Base name
    args.exp_name = f"exp-{exp_id}_{args.algorithm}_act-{action_space}_traffic-{traffic}_input-{input_type}_rwd-{reward_type}_curr-{curriculum}_fovmask-{fov_mask}"

    return args


# ============================================================
# 3. Save config to runs/<name>/config.yaml
# ============================================================


def save_run_config(args):
    out_dir = os.path.join("runs", args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(args, f)
    print(f"Saved: runs/{args.exp_name}/config.yaml")


# ============================================================
# 4. CLI Entry
# ============================================================


def load_experiment():
    parser = tyro.extras.subcommand_type_from_defaults(
        {
            "exp": ArgsCarlaBEV(),
        }
    )

    args = tyro.cli(parser)

    print(f"⚙ Selecting experiment ID = {args.exp_id}")

    args = apply_experiment_config(args, args.exp_id)
    save_run_config(args)

    return args

# ============================================================
# 5. Universal Execution Wrapper (Manual + Optuna)
# ============================================================

def run_experiment(args: ArgsCarlaBEV, trial=None, seed_idx: int = None) -> float:
    """Universal execution wrapper for both single standalone trains and Optuna searches."""
    import torch
    import optuna
    from CarlaBEV.envs import make_env
    from src.utils.logger import DRLogger
    from src.trainers import build_trainer

    # If running normally (not orchestrated by Optuna Tuner directly), we enqueue it into the global study
    if trial is None:
        print(f"🔗 Integrating standalone run into Optuna SQLite Database (Exp ID: {args.exp_id})")
        db_path = get_global_db_path()
        storage_name = f"sqlite:///{db_path}"
        
        study = optuna.create_study(
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            study_name=get_global_study_name()
        )
        
        # Enqueue this exact config so Optuna logs it as the next trial
        enqueued_params = {
            "learning_rate": args.ppo.learning_rate,
            "gae_lambda": args.ppo.gae_lambda,
            "gamma": args.ppo.gamma,
            "num_steps": args.ppo.num_steps,
            "update_epochs": args.ppo.update_epochs,
            "num_minibatches": args.ppo.num_minibatches,
        }
        study.enqueue_trial(enqueued_params)
        
        # Execute exactly 1 trial (the one we just enqueued)
        # That trial block will recursively call run_experiment AGAIN, but this time with a valid `trial` object!
        final_scores = []
        def _manual_objective(t):
            return run_experiment(args, trial=t, seed_idx=seed_idx)
            
        study.optimize(_manual_objective, n_trials=1)
        return
        
    # --- IF REACHING HERE, WE HAVE AN ACTIVE OPTUNA TRIAL ---

    # Modify exp_name to be trial-specific and optionally seed-specific
    if seed_idx is not None:
        args.exp_name = f"{args.exp_name}_optuna_trial_{trial.number}_seed_{args.seed}"
    else:
        args.exp_name = f"{args.exp_name}_optuna_trial_{trial.number}"
    
    # Save config for this trial execution
    save_run_config(args)

    # Universally format database path
    args.logging.db_path = get_global_db_path()
    args.logging.trial_number = trial.number

    # Save trial attributes for filtering in DB
    trial.set_user_attr("base_exp_id", args.exp_id)
    trial.set_user_attr("action_space", args.env.action_space)
    trial.set_user_attr("traffic_enabled", args.env.traffic_enabled)
    trial.set_user_attr("input_type", "masks" if args.env.masked else "rgb")
    trial.set_user_attr("fov_masked", args.env.fov_masked)
    trial.set_user_attr("reward_type", args.env.reward_type)
    trial.set_user_attr("curriculum", args.env.curriculum_mode if args.env.curriculum_enabled else "off")

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Initialize environments and logger
    envs = make_env(args)
    logger = DRLogger(config=args, stats_interval=100)
    logger.msg(f"Environments - {args.env.env_id}:{args.num_envs} built.")

    # Dynamically build trainer based on the algorithm specified in config
    trainer = build_trainer(args.algorithm)
    logger.msg(f"Trainer built for algorithm: {args.algorithm}")

    # Run training
    # Optuna may raise TrialPruned inside the trainer loop
    final_score = trainer(args, envs, logger, device, trial=trial)
    
    return final_score


if __name__ == "__main__":
    load_experiment()
