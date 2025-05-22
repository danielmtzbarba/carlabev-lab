import os
from torch.utils.tensorboard import SummaryWriter

import sys
import logging

file_handler = logging.FileHandler(filename="drlog.log")  # , mode="w")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    handlers=handlers,
    format="[%(asctime)s] %(levelname)s ==> %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    encoding="utf-8",
    level=logging.INFO,
)
logger = logging.getLogger("drlab")


class DRLogger:
    def __init__(self, config):
        # Create tensorboard summary
        self.writer = SummaryWriter(f"runs/{config.exp_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )
        self._logger = logger
        self._config = config

    def log_loss(self, global_step, loss, sps, q):
        self.writer.add_scalar("losses/td_loss", loss, global_step)
        self.writer.add_scalar("losses/q_values", q, global_step)
        self.writer.add_scalar("stats/SPS", sps, global_step)

    def log_episode(self, infos):
        try:
            num_ep = infos["termination"]["episode"][0]
            ret = infos["termination"]["return"][0]
            cause = infos["termination"]["termination"][0]
            success_rate = infos["termination"]["success_rate"][0]
            collision_rate = infos["termination"]["collision_rate"][0]
            reward = infos["episode"]["r"][0]
            length = infos["episode"]["l"][0]
            mean_reward = infos["termination"]["mean_reward"][0]

        except Exception as e:
            num_ep = infos["termination"]["episode"]
            ret = infos["termination"]["return"]
            cause = infos["termination"]["termination"]
            success_rate = infos["termination"]["success_rate"]
            collision_rate = infos["termination"]["collision_rate"]
            reward = infos["episode"]["r"]
            length = infos["episode"]["l"]
            mean_reward = infos["termination"]["mean_reward"]

        #
        logger.info(f"episode-{num_ep}: {ret}-{cause}")
        #
        self.writer.add_scalar("stats/episodic_return", reward, num_ep)
        self.writer.add_scalar("stats/episodic_length", length, num_ep)
        self.writer.add_scalar("stats/mean_reward", mean_reward, num_ep)
        self.writer.add_scalar("stats/collision_rate", collision_rate, num_ep)
        self.writer.add_scalar("stats/success_rate", success_rate, num_ep)

        return num_ep
