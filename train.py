import torch

import warnings

from src.exp import get_experiment
from src.envs import make_env
from src.trainers import build_trainer


warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXPERIMENT = "DQN"


def main():
    args, logger = get_experiment(EXPERIMENT)
    envs = make_env(args)
    trainer = build_trainer(EXPERIMENT)
    trainer(args, envs, logger, device)


if __name__ == "__main__":
    main()
