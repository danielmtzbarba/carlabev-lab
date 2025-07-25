import tyro

import numpy as np
import random
import torch


from src.utils.logger import DRLogger


def get_experiment(experiment):
    if "DQN" in experiment:
        from .dqn_carlabev import ArgsCarlaBEV
    elif "PPO" in experiment:
        from .ppo_carlabev import ArgsCarlaBEV
    elif "SAC" in experiment:
        from .sac_carlabev import ArgsCarlaBEV
    else:
        exit("Unregisted experiment...")

    args = tyro.cli(ArgsCarlaBEV)
    log = DRLogger(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    return args, log
