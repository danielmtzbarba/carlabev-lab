import os
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
import time

file_handler = logging.FileHandler("drlog.log")
stdout_handler = logging.StreamHandler(sys.stdout)
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
        run_dir = f"runs/{config.exp_name}"
        self.writer = SummaryWriter(run_dir)
        self._logger = logger
        self._config = config
        self._start_time = time.time()

        # Log hyperparameters
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )

    def log_loss(self, global_step, loss, sps, q):
        self.writer.add_scalar("train/loss_td", loss, global_step)
        self.writer.add_scalar("train/q_value_mean", q, global_step)
        self.writer.add_scalar("train/steps_per_second", sps, global_step)

    def log_episode(self, infos, global_step=None):
        # Support dicts or dict-of-lists
        try:
            num_ep = infos["termination"]["episode"][0]
            ret = infos["termination"]["return"][0]
            cause = infos["termination"]["termination"][0]
            success_rate = infos["termination"]["success_rate"][0]
            collision_rate = infos["termination"]["collision_rate"][0]
            reward = infos["episode"]["r"][0]
            length = infos["episode"]["l"][0]
            mean_reward = infos["termination"]["mean_reward"][0]
        except Exception:
            num_ep = infos["termination"]["episode"]
            ret = infos["termination"]["return"]
            cause = infos["termination"]["termination"]
            success_rate = infos["termination"]["success_rate"]
            collision_rate = infos["termination"]["collision_rate"]
            reward = infos["episode"]["r"]
            length = infos["episode"]["l"]
            mean_reward = infos["termination"]["mean_reward"]

        elapsed = time.time() - self._start_time

        # --- Console/File Logging ---
        self._logger.info(
            f"Ep {num_ep:04d} | Ret: {ret:.2f} | Len: {length} | "
            f"MeanR: {mean_reward:.2f} | Succ: {success_rate:.2f} | Coll: {collision_rate:.2f} | "
            f"Cause: {cause} | Step: {global_step}"
        )

        # --- TensorBoard ---
        self.writer.add_scalar("eval/episodic_return", reward, num_ep)
        self.writer.add_scalar("eval/episodic_length", length, num_ep)
        self.writer.add_scalar("eval/mean_reward", mean_reward, num_ep)
        self.writer.add_scalar("safety/collision_rate", collision_rate, num_ep)
        self.writer.add_scalar("safety/success_rate", success_rate, num_ep)
        self.writer.add_scalar("time/elapsed_sec", elapsed, num_ep)

        if global_step is not None:
            self.writer.add_scalar("train/episodic_return", reward, global_step)

        # Flush periodically
        if num_ep % 20 == 0:
            self.writer.flush()

        return num_ep
