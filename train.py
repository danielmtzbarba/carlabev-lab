import torch

import warnings

from CarlaBEV.envs import make_env
from src.trainers import build_trainer

from src.config.experiment_loader import load_experiment
from src.utils.logger import DRLogger

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EXPERIMENT = "cnn-ppo-discrete"


def main():
    # cfg, logger = get_experiment(EXPERIMENT)
    cfg = load_experiment()
    envs = make_env(cfg, eval=False)
    logger = DRLogger(cfg)
    logger.msg(f"Environments - {cfg.env.env_id}:{cfg.num_envs} built.")
    trainer = build_trainer(EXPERIMENT)
    logger.msg("Trainer built")
    trainer(cfg, envs, logger, device)


if __name__ == "__main__":
    main()
