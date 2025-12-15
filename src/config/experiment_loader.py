# src/config/experiment_loader.py

import os
import tyro
import yaml

from src.config.base_config import ArgsCarlaBEV, EnvConfig, PPOConfig


# ============================================================
# 1. Tabla de 24 configuraciones experimentales
# ============================================================

EXPERIMENT_TABLE = {
    # Reward and Input type - Traffic OFF:
    1:  ("off", "rgb",   "shaping", "off", "off"),
    2:  ("off", "rgb",   "shaping", "on",  "off"),
    3:  ("off", "rgb",   "carl",    "off", "off"),
    4:  ("off", "rgb",   "carl",    "on",  "off"),
    5:  ("off", "masks", "shaping", "off", "off"),
    6:  ("off", "masks", "shaping", "on",  "off"),
    7:  ("off", "masks", "carl",    "off", "off"),
    8:  ("off", "masks", "carl",    "on",  "off"),

    # Reward and Input type - Traffic ON:
    9:  ("on",  "rgb",   "shaping", "off", "off"),
    10: ("on",  "rgb",   "shaping", "on",  "off"),
    11: ("on",  "rgb",   "carl",    "off", "off"),
    12: ("on",  "rgb",   "carl",    "on",  "off"),
    13: ("on",  "masks", "shaping", "off", "off"),
    14: ("on",  "masks", "shaping", "on",  "off"),
    15: ("on",  "masks", "carl",    "off", "off"),
    16: ("on",  "masks", "carl",    "on",  "off"),

    # Curriculum learning variants
    17: ("on", "rgb",   "carl", "vehicles_only", "off"),
    18: ("on", "rgb",   "carl", "route_only",    "off"),
    19: ("on", "rgb",   "carl", "both",          "off"),
    20: ("on", "masks", "carl", "vehicles_only", "off"),
    21: ("on", "masks", "carl", "route_only",    "off"),
    22: ("on", "masks", "carl", "both",          "off"),

    # Baseline extra runs
    23: ("off", "rgb",   "carl", "route_only", "off"),
    24: ("off", "masks", "carl", "route_only", "off"),

    # Only Edge scenarios
    25: ("on", "masks", "carl", "off", "off"),

    # FOV masked 
    26: ("on", "masks", "carl", "route_only", "on"),
}


# ============================================================
# 2. Aplica parámetros según ID de experimento
# ============================================================


def apply_experiment_config(args: ArgsCarlaBEV, exp_id: int):
    traffic, input_type, reward_type, curriculum, fov_mask = EXPERIMENT_TABLE[exp_id]

    env: EnvConfig() = args.env
    ppo: PPOConfig() = args.ppo

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
    args.exp_name = f"exp-{exp_id}_cnn-ppo_traffic-{traffic}_input-{input_type}_rwd-{reward_type}_curr-{curriculum}_fovmask-{fov_mask}"

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
