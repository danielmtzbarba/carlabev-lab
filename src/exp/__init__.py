import tyro

import numpy as np
import random
import torch

from torch.utils.tensorboard import SummaryWriter
from .carlabev_demo import ArgsCarlaBEVDemo


def get_experiment():
    args = tyro.cli(ArgsCarlaBEVDemo)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    writer = SummaryWriter(f"runs/{args.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    return args, writer
