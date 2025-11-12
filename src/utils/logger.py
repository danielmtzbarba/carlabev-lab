import os
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
import numpy as np

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


class DRLogger(object):
    """
    Logger for RL training.
    Only logs finished episodes passed from the trainer.
    Tracks global episode counter, handles success/collision/unfinished based on cause.
    Supports TensorBoard, Rich console, and arbitrary learning variable logging.
    """

    def __init__(self, config, stats_interval=100):
        self.writer = SummaryWriter(f"runs/{config.exp_name}")
        self._console = Console()
        self._logger = logger
        self._stats_interval = stats_interval

        # Global episode counter
        self.global_episode = 0

        # History for aggregate stats
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_unfinished = []

        # Log hyperparameters
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{k}|{v}|" for k, v in vars(config).items()])),
        )

    def log_episode(self, infos, mean_return, idx, global_step=0):
        """
        infos: dict containing 'termination' with keys:
               'cause', 'return', 'length', 'mean_reward', 'success_rate', 'collision_rate', 'unfinished_rate'
        """
        self.global_episode += 1
        
        data = infos
        # Console output
        self._console.print(
            f"Step {global_step} | Ep {self.global_episode} | Return: [green]{data["return"][idx]:.2f}[/green] | "
                f"Len: {data["length"][idx]} | Num Vehicles: {data["num_vehicles"][idx]} | Return MA50: {mean_return:.2f} | "
            f"Cause: {data["termination"][idx]} |"
        )

        cause = data["termination"][idx]
        ep_return = data["return"][idx]
        ep_length= data["length"][idx]
        mean_reward = float(data["mean_reward"][idx])

        # Determine success / collision / unfinished
        success = 1.0 if cause == "success" else 0.0
        collision = 1.0 if cause == "collision" else 0.0
        unfinished = 1.0 if cause in ["out_of_bounds", "off_road", "max_actions"] else 0.0

        # Store history for aggregate stats
        self.episode_returns.append(ep_return)
        self.episode_lengths.append(ep_length)
        self.episode_successes.append(success)
        self.episode_collisions.append(collision)
        self.episode_unfinished.append(unfinished)

        # TensorBoard logging
        self.writer.add_scalar("stats/episodic_return", ep_return, self.global_episode)
        self.writer.add_scalar("stats/episodic_length", ep_length, self.global_episode)
        self.writer.add_scalar("stats/mean_reward", mean_reward, self.global_episode)
        self.writer.add_scalar("stats/success_rate", success, self.global_episode)
        self.writer.add_scalar("stats/collision_rate", collision, self.global_episode)
        self.writer.add_scalar("stats/unfinished_rate", unfinished, self.global_episode)


        # Aggregate table every stats_interval episodes
        if self.global_episode % self._stats_interval == 0:
            mean_ret = np.mean(self.episode_returns[-self._stats_interval :])
            mean_len = np.mean(self.episode_lengths[-self._stats_interval :])
            mean_succ = np.mean(self.episode_successes[-self._stats_interval :])
            mean_col = np.mean(self.episode_collisions[-self._stats_interval :])
            mean_unfin = np.mean(self.episode_unfinished[-self._stats_interval :])

            table = Table(
                title=f"Episode Stats (last {self._stats_interval} eps)",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Mean Return", justify="right")
            table.add_column("Mean Length", justify="right")
            table.add_column("Success Rate", justify="right")
            table.add_column("Collision Rate", justify="right")
            table.add_column("Unfinished Rate", justify="right")
            table.add_row(
                f"{mean_ret:.2f}",
                f"{mean_len:.1f}",
                f"{mean_succ:.2%}",
                f"{mean_col:.2%}",
                f"{mean_unfin:.2%}",
            )
            self._console.print(table)

        return self.global_episode

    def log_learning(
        self,
        global_step,
        pg_loss=None,
        v_loss=None,
        entropy=None,
        approx_kl=None,
        clip_frac=None,
    ):
        """Log learning-related variables to TensorBoard."""
        if pg_loss is not None:
            self.writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        if v_loss is not None:
            self.writer.add_scalar("losses/value_loss", v_loss, global_step)
        if entropy is not None:
            self.writer.add_scalar("stats/entropy", entropy, global_step)
        if approx_kl is not None:
            self.writer.add_scalar("stats/approx_kl", approx_kl, global_step)
        if clip_frac is not None:
            self.writer.add_scalar("stats/clip_fraction", clip_frac, global_step)

    # === Add this method inside DRLogger ===
    def log_evaluation(self, results_dict: dict, global_step: int = None):
        """
        Logs evaluation results from evaluate_ppo() to TensorBoard and console.

        Parameters
        ----------
        results_dict : dict
            Dictionary containing evaluation statistics.
            Example: {
                "mean_return": 150.2,
                "success_rate": 0.82,
                "collision_rate": 0.05,
                "avg_length": 423,
                "avg_speed": 2.8,
            }
        global_step : int, optional
            Global training step to align logs in TensorBoard.
        """
        # Log to TensorBoard
        for key, value in results_dict.items():
            try:
                self.writer.add_scalar(f"eval/{key}", value, global_step)
            except Exception as e:
                self._logger.warning(f"⚠️ Could not log {key}: {e}")

        # Rich console table
        from rich.table import Table

        table = Table(
            title=f"Evaluation Results @ step {global_step or '-'}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", justify="right")
        table.add_column("Value", justify="center")

        for key, value in results_dict.items():
            if isinstance(value, float):
                value_str = f"{value:.3f}"
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        self._console.print(table)

        # Also log a simple summary line to file
        msg = " | ".join(
            [
                f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in results_dict.items()
            ]
        )
        self._logger.info(f"[EVAL] {msg}")

    def msg(self, text):
        self._logger.info(text)
