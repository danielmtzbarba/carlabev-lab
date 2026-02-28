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

def abbreviate_number(n):
    """
    Convert an integer into an abbreviated string:
    1234 -> '1.2K'
    1234567 -> '1.2M'
    9876543210 -> '9.9B'
    """
    if n < 1_000:
        return str(n)
    elif n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    elif n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    else:
        return f"{n / 1_000_000_000:.1f}B"

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

        # Benchmark Logging
        self.success_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.reached_thresholds = set()
        self.threshold_stats = {}

        # SQL Auxiliary Logging
        self.db_conn = None
        self.trial_number = getattr(config.logging, "trial_number", None)
        self.db_path = getattr(config.logging, "db_path", None)
        
        if self.db_path is not None and self.trial_number is not None:
            import sqlite3
            real_path = self.db_path.replace("sqlite:///", "")
            # Enable high-concurrency mode with extended timeout and WAL journal
            self.db_conn = sqlite3.connect(real_path, timeout=60)
            self.db_conn.execute("PRAGMA journal_mode=WAL;")
            self._init_db_tables()

    def _init_db_tables(self):
        query_train = """
        CREATE TABLE IF NOT EXISTS trial_train_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_number INTEGER,
            global_step INTEGER,
            walltime REAL,
            mean_return REAL,
            pg_loss REAL,
            v_loss REAL,
            entropy REAL,
            approx_kl REAL,
            clip_frac REAL,
            ent_coef REAL,
            train_success_rate REAL,
            train_collision_rate REAL,
            train_unfinished_rate REAL
        )
        """
        query_eval = """
        CREATE TABLE IF NOT EXISTS trial_eval_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_number INTEGER,
            global_step INTEGER,
            walltime REAL,
            mean_return REAL,
            std_return REAL,
            mean_length REAL,
            success_rate REAL,
            collision_rate REAL,
            unfinished_rate REAL,
            time_to_reach_0_1 REAL,
            time_to_reach_0_2 REAL,
            time_to_reach_0_3 REAL,
            time_to_reach_0_4 REAL,
            time_to_reach_0_5 REAL,
            time_to_reach_0_6 REAL,
            time_to_reach_0_7 REAL,
            time_to_reach_0_8 REAL,
            time_to_reach_0_9 REAL,
            time_to_reach_0_95 REAL,
            time_to_reach_0_99 REAL
        )
        """
        cursor = self.db_conn.cursor()
        cursor.execute(query_train)
        cursor.execute(query_eval)
        self.db_conn.commit()

    def log_episode(self, infos, mean_return, idx, global_step=0):
        """
        infos: dict containing 'termination' with keys:
               'cause', 'return', 'length', 'mean_reward', 'success_rate', 'collision_rate', 'unfinished_rate'
        """
        self.global_episode += 1
        
        data = infos
        # Console output
        self._console.print(
            f"Step {abbreviate_number(global_step)} | Ep {self.global_episode} | Ret: [green]{data["return"][idx]:.2f}[/green] | "
            f"len_route: {int(data["len_ego_route"][idx])} | veh: {data["num_vehicles"][idx]} | "
            f"len_ep: {data["length"][idx]} | cause: {data["termination"][idx]} | MA50: {mean_return:.2f} | "
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
        ent_coef=None,
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

        if self.db_conn is not None:
            import time
            mean_ret = np.mean(self.episode_returns[-self._stats_interval :]) if len(self.episode_returns) > 0 else 0.0
            mean_succ = np.mean(self.episode_successes[-self._stats_interval :]) if len(self.episode_successes) > 0 else 0.0
            mean_col = np.mean(self.episode_collisions[-self._stats_interval :]) if len(self.episode_collisions) > 0 else 0.0
            mean_unfin = np.mean(self.episode_unfinished[-self._stats_interval :]) if len(self.episode_unfinished) > 0 else 0.0
            
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO trial_train_logs 
                (trial_number, global_step, walltime, mean_return, pg_loss, v_loss, entropy, approx_kl, clip_frac, ent_coef, train_success_rate, train_collision_rate, train_unfinished_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.trial_number, global_step, time.time(), mean_ret, 
                pg_loss if pg_loss is not None else 0.0, 
                v_loss if v_loss is not None else 0.0, 
                entropy if entropy is not None else 0.0, 
                approx_kl if approx_kl is not None else 0.0, 
                clip_frac if clip_frac is not None else 0.0, 
                ent_coef if ent_coef is not None else 0.0, 
                mean_succ, mean_col, mean_unfin
            ))
            self.db_conn.commit()

    # === Add this method inside DRLogger ===
    def log_evaluation(
        self,
        results_dict: dict,
        global_step: int = None,
        iteration: int = None,
        elapsed_time: float = None,
    ):
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
        iteration : int, optional
            Current training iteration.
        elapsed_time : float, optional
            Elapsed wall-clock time in seconds.
        """
        # --- Benchmark Logging ---
        if "success_rate" in results_dict:
            success_rate = results_dict["success_rate"]
            collision_rate = results_dict.get("collision_rate", 0.0)

            for threshold in self.success_thresholds:
                if success_rate >= threshold and threshold not in self.reached_thresholds:
                    self.reached_thresholds.add(threshold)
                    
                    # Add to results_dict for TB/Console
                    key = f"time_to_reach_{threshold}"
                    results_dict[key] = elapsed_time
                    self.threshold_stats[key] = elapsed_time
                    self.threshold_stats[f"step_to_reach_{threshold}"] = global_step
                    
                    # Save to CSV
                    import csv
                    log_path = os.path.join(self.writer.log_dir, "benchmark_results.csv")
                    file_exists = os.path.isfile(log_path)
                    
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["threshold", "success_rate", "collision_rate", "global_step", "iteration", "elapsed_time"])
                        
                        writer.writerow([
                            threshold,
                            success_rate,
                            collision_rate,
                            global_step,
                            iteration,
                            elapsed_time
                        ])
                    
                    self._logger.info(
                        f"🏆 Reached success threshold {threshold} at step {global_step} "
                        f"(Iter: {iteration}, Time: {elapsed_time:.2f}s)"
                    )
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

        if self.db_conn is not None:
            import time
            cursor = self.db_conn.cursor()
            
            cursor.execute('''
                INSERT INTO trial_eval_logs 
                (trial_number, global_step, walltime, 
                 mean_return, std_return, mean_length, 
                 success_rate, collision_rate, unfinished_rate,
                 time_to_reach_0_1, time_to_reach_0_2, time_to_reach_0_3, time_to_reach_0_4, time_to_reach_0_5, 
                 time_to_reach_0_6, time_to_reach_0_7, time_to_reach_0_8, time_to_reach_0_9, time_to_reach_0_95, time_to_reach_0_99)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.trial_number, global_step if global_step is not None else 0, time.time(),
                float(results_dict.get("mean_return", 0.0)),
                float(results_dict.get("std_return", 0.0)),
                float(results_dict.get("mean_length", 0.0)),
                float(results_dict.get("success_rate", 0.0)),
                float(results_dict.get("collision_rate", 0.0)),
                float(results_dict.get("unfinished_rate", 0.0)),
                results_dict.get("time_to_reach_0.1", None),
                results_dict.get("time_to_reach_0.2", None),
                results_dict.get("time_to_reach_0.3", None),
                results_dict.get("time_to_reach_0.4", None),
                results_dict.get("time_to_reach_0.5", None),
                results_dict.get("time_to_reach_0.6", None),
                results_dict.get("time_to_reach_0.7", None),
                results_dict.get("time_to_reach_0.8", None),
                results_dict.get("time_to_reach_0.9", None),
                results_dict.get("time_to_reach_0.95", None),
                results_dict.get("time_to_reach_0.99", None)
            ))
            self.db_conn.commit()

    def msg(self, text):
        self._logger.info(text)
