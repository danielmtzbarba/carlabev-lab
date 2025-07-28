import tyro

import numpy as np
import random
import torch


from src.utils.logger import DRLogger


def get_experiment(experiment):
    if "DQN" in experiment:
        from .dqn_carlabev import ArgsCarlaBEV
    elif "vector-ppo-discrete" in experiment:
        from .vector_ppo_discrete_carlabev import ArgsCarlaBEV
    elif "cnn-ppo-discrete" in experiment:
        from .cnn_ppo_discrete_carlabev import ArgsCarlaBEV
    elif "SAC" in experiment:
        from .sac_carlabev import ArgsCarlaBEV
    else:
        exit("Unregisted experiment...")

    args = tyro.cli(ArgsCarlaBEV)
    log = DRLogger(args)

    return args, log
