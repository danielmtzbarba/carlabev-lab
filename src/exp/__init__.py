import tyro
import yaml
import os
import numpy as np
import random
import torch

from src.utils.logger import DRLogger


def save_run_config_yaml(config):
    """
    Saves a dictionary of hyperparameters to runs/<run_name>/params.yaml

    Args:
        run_name (str): Name of the experiment/run.
        config_dict (dict): Dictionary of hyperparameters.
    """
    run_dir = os.path.join("runs", config.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"✅ Hyperparameters saved to {config_path}")


def get_experiment(experiment):
    from .cnn_ppo_discrete_traffic import ArgsCarlaBEV
    args = tyro.cli(ArgsCarlaBEV)
    save_run_config_yaml(args)
    log = DRLogger(args)

    return args, log
