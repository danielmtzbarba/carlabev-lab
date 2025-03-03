import tyro

import numpy as np
import random
import torch

from torch.utils.tensorboard import SummaryWriter
from .dqn_racing import Args
from .dqn_gridworld import ArgsGridWorld
from .dqn_carlabev import ArgsCarlaBEV
from .dqn_carlabev_test import ArgsCarlaBEVTest
from .carlabev_demo import ArgsCarlaBEVDemo


def get_experiment(exp_name: str):
    if "dqn_carlabev" in exp_name:
        args = tyro.cli(ArgsCarlaBEVDemo)
        assert args.num_envs == 1, "vectorized envs are not supported at the moment"
        # run_name = f"{args.exp_name}-seed_{args.seed}-bs_{args.buffer_size}"
        run_name = args.exp_name
    elif "dqn_racing_discrete" in exp_name:
        args = tyro.cli(Args)
        assert args.num_envs == 1, "vectorized envs are not supported at the moment"
        run_name = f"{args.exp_name}-seed_{args.seed}-bs_{args.buffer_size}"
    elif "dqn_gridworld" in exp_name:
        args = tyro.cli(ArgsGridWorld)
        assert args.num_envs == 1, "vectorized envs are not supported at the moment"
        run_name = f"{args.exp_name}-seed_{args.seed}-bs_{args.buffer_size}"

    else:
        exit()

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
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

    return args, run_name, writer
