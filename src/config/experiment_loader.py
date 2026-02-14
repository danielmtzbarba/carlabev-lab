# src/config/experiment_loader.py

import os
import tyro
import yaml

from src.config.base_config import ArgsCarlaBEV, EnvConfig, PPOConfig


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
    args.exp_name = f"exp-{exp_id}_cnn-ppo_act-{action_space}_traffic-{traffic}_input-{input_type}_rwd-{reward_type}_curr-{curriculum}_fovmask-{fov_mask}"

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


if __name__ == "__main__":
    load_experiment()
