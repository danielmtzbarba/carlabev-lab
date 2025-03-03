from collections import deque
import numpy as np
import cv2
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
        self._summary = SummaryWriter(os.path.join(config.logdir))
        self._logger = logger
        self._config = config
        self._setup_buffers()

    def _setup_buffers(self):
        # History metrics
        self._total_rewards = deque(maxlen=self._config.max_episodes)
        self._mean_rewards = deque(maxlen=self._config.max_episodes)
        self._epsilon = deque(maxlen=self._config.max_episodes)
        self._losses = deque()
        self._qs = deque()

    def new_episode(self, num_episode):
        self._logger.info(f"{'-' * 70}")
        self._logger.info(
            f"{' ' * 25}Episode {num_episode}/{self._config.max_episodes}"
        )
        self._logger.info(f"{'-' * 70}")

    def log_loss(self, q, loss, iteration):
        self._qs.append(q)
        self._losses.append(loss)
        self._summary.add_scalar("train/loss", float(loss), iteration)
        self._summary.add_scalar("train/q", float(q), iteration)

    def _log_state(self, episode):
        if episode.id % self._config.save_every == 0:
            for step, state in enumerate(episode.states):
                resized = cv2.resize(
                    np.transpose(np.squeeze(state, 0), (1, 2, 0)),
                    (512, 512),
                    interpolation=cv2.INTER_AREA,
                )
                self._summary.add_image(
                    f"train/episode_state_resized/{episode.id}",
                    resized,
                    step,
                    dataformats=self._config.log_img_type,
                )

    def log_episode(self, episode, epsilon):
        "LOG EPISODE METRICS"
        self._log_state(episode)
        self._total_rewards.append(episode.total_reward)
        self._mean_rewards.append(episode.mean_reward)
        self._epsilon.append(epsilon)

        # Update tensorboard
        self._summary.add_scalar(
            "train/total_reward", float(episode.total_reward), episode.id
        )
        self._summary.add_scalar(
            "train/mean_reward", float(episode.mean_reward), episode.id
        )

        self._summary.add_scalar("train/epsilon", float(epsilon), episode.id)
        #
        self._logger.info(f"Epsilon {epsilon}")
        self._logger.info(
            f"Total reward: {episode.total_reward}, Mean reward: {episode.mean_reward}"
        )
        self._logger.info(f"{'-' * 70}")
        #
