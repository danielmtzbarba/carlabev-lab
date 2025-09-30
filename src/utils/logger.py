import os
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table

# --- Logging setup ---
file_handler = logging.FileHandler(filename="drlog.log")
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
console = Console()


class DRLogger:
    def __init__(self, config, stats_interval: int = 10):
        self.writer = SummaryWriter(f"runs/{config.exp_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(config).items()])),
        )
        self._logger = logger
        self._config = config
        self._stats_interval = stats_interval

    def log_loss(self, global_step, loss, sps, q):
        self.writer.add_scalar("losses/td_loss", loss, global_step)
        self.writer.add_scalar("losses/q_values", q, global_step)
        self.writer.add_scalar("stats/SPS", sps, global_step)

    def log_episode(self, infos, iteration=None):
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

        # TensorBoard logging
        self.writer.add_scalar("stats/episodic_return", reward, num_ep)
        self.writer.add_scalar("stats/episodic_length", length, num_ep)
        self.writer.add_scalar("stats/mean_reward", mean_reward, num_ep)
        self.writer.add_scalar("stats/collision_rate", collision_rate, num_ep)
        self.writer.add_scalar("stats/success_rate", success_rate, num_ep)

        # --- Console output ---
        if iteration is not None:
            console.print(f"[cyan]Iter {iteration}[/cyan] | Ep {num_ep} → "
                          f"Return: [green]{ret:.2f}[/green] | Cause: {cause} | Len: {length}")
        else:
            console.print(f"Ep {num_ep} → Return: [green]{ret:.2f}[/green] | Cause: {cause} | Len: {length}")

        # Print aggregate stats every n episodes
        if num_ep % 10  == 0:
            table = Table(title=f"Episode Stats (up to Ep {num_ep})", show_header=True, header_style="bold magenta")
            table.add_column("Mean Reward", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Collision Rate", justify="right")

            table.add_row(
                f"{mean_reward:.2f}",
                f"{success_rate:.2%}",
                f"{collision_rate:.2%}",
            )
            console.print(table)
        return num_ep

    def msg(self, text):
        self._logger.info(text)
