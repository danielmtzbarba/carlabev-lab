import torch
import time
from torch import nn
import numpy as np

from src.agents import build_agent

# TODO: PLACE SELF-PLAY THEN TRAIN


def train_muzero(args, envs, logger, device):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    agent, optimizer = build_agent(args, envs, device)

    envs.close()
    logger.writer.close()
