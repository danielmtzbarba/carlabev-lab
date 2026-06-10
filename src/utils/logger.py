import csv
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
from rich.console import Console
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def abbreviate_number(n):
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.1f}B"


class DRLogger:
    def __init__(self, config, stats_interval=100):
        self.run_dir = getattr(config, "run_dir", os.path.join("runs", config.exp_name))
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = SummaryWriter(self.run_dir)
        self._console = Console()
        self._interactive = getattr(config, "run_mode", "interactive") == "interactive"
        self._stats_interval = stats_interval
        self._log_path = os.path.join(self.run_dir, "train.log")
        self._status_path = os.path.join(self.run_dir, "status.json")
        self._completed_path = os.path.join(self.run_dir, "COMPLETED")
        self._failed_path = os.path.join(self.run_dir, "FAILED")
        self._traceback_path = os.path.join(self.run_dir, "failure_traceback.log")
        self._logger = self._build_logger(config.exp_name)

        self.global_episode = 0
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_unfinished = []

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{k}|{v}|" for k, v in vars(config).items()])),
        )

        self.success_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        self.reached_thresholds = set()
        self.threshold_stats = {}

        self.db_conn = None
        self.trial_number = getattr(config.logging, "trial_number", None)
        self.db_path = getattr(config.logging, "db_path", None)
        self.seed = getattr(config, "seed", None)

        self._status = {
            "study_id": getattr(config, "study_id", None),
            "exp_id": getattr(config, "exp_id", None),
            "exp_name": getattr(config, "exp_name", None),
            "algorithm": getattr(config, "algorithm", None),
            "run_mode": getattr(config, "run_mode", "interactive"),
            "seed": self.seed,
            "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "state": "initialized",
            "started_at": _utcnow_iso(),
            "last_updated_at": _utcnow_iso(),
            "last_eval_step": None,
            "last_eval_metrics": None,
            "finished_at": None,
            "error": None,
        }
        self._write_status()

        if self.db_path is not None and self.trial_number is not None:
            real_path = self.db_path.replace("sqlite:///", "")
            self.db_conn = sqlite3.connect(real_path, timeout=60)
            self.db_conn.execute("PRAGMA journal_mode=WAL;")
            self._init_db_tables()

    def _build_logger(self, exp_name: str) -> logging.Logger:
        logger_name = f"drlab.{exp_name}"
        run_logger = logging.getLogger(logger_name)
        run_logger.setLevel(logging.INFO)
        run_logger.propagate = False
        run_logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s ==> %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

        file_handler = logging.FileHandler(self._log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        run_logger.addHandler(file_handler)

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        run_logger.addHandler(stdout_handler)
        return run_logger

    def _write_status(self):
        self._status["last_updated_at"] = _utcnow_iso()
        with open(self._status_path, "w", encoding="utf-8") as handle:
            json.dump(self._status, handle, indent=2, sort_keys=True)

    def set_running(self):
        self._status["state"] = "running"
        self._write_status()

    def update_status(
        self,
        *,
        state: str | None = None,
        last_eval_step: int | None = None,
        last_eval_metrics: dict | None = None,
        error: str | None = None,
        finished: bool = False,
    ):
        if state is not None:
            self._status["state"] = state
        if last_eval_step is not None:
            self._status["last_eval_step"] = int(last_eval_step)
        if last_eval_metrics is not None:
            self._status["last_eval_metrics"] = last_eval_metrics
        if error is not None:
            self._status["error"] = error
        if finished:
            self._status["finished_at"] = _utcnow_iso()
        self._write_status()

    def mark_completed(self, final_metrics: dict | None = None):
        if os.path.exists(self._failed_path):
            os.remove(self._failed_path)
        self.update_status(state="completed", last_eval_metrics=final_metrics, finished=True)
        with open(self._completed_path, "w", encoding="utf-8") as handle:
            handle.write(f"completed_at={self._status['finished_at']}\n")

    def mark_failed(self, exc: BaseException, traceback_text: str):
        if os.path.exists(self._completed_path):
            os.remove(self._completed_path)
        self.update_status(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished=True,
        )
        with open(self._failed_path, "w", encoding="utf-8") as handle:
            handle.write(f"failed_at={self._status['finished_at']}\n")
            handle.write(f"error={type(exc).__name__}: {exc}\n")
        with open(self._traceback_path, "w", encoding="utf-8") as handle:
            handle.write(traceback_text)

    def _init_db_tables(self):
        query_train = """
        CREATE TABLE IF NOT EXISTS trial_train_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trial_number INTEGER,
            seed INTEGER,
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
            seed INTEGER,
            global_step INTEGER,
            walltime REAL,
            mean_return REAL,
            std_return REAL,
            mean_length REAL,
            success_rate REAL,
            collision_rate REAL,
            unfinished_rate REAL,
            comfort_score REAL,
            normalized_score REAL,
            mean_abs_accel_long REAL,
            mean_abs_accel_lat REAL,
            mean_abs_jerk_long REAL,
            mean_abs_jerk_lat REAL,
            mean_abs_yaw_rate REAL,
            mean_abs_yaw_acc REAL,
            comfort_violation_rate REAL,
            harsh_brake_rate REAL,
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
        try:
            cursor.execute("ALTER TABLE trial_train_logs ADD COLUMN seed INTEGER;")
        except Exception:
            pass
        try:
            cursor.execute("ALTER TABLE trial_eval_logs ADD COLUMN seed INTEGER;")
        except Exception:
            pass
        for column in (
            "comfort_score",
            "normalized_score",
            "mean_abs_accel_long",
            "mean_abs_accel_lat",
            "mean_abs_jerk_long",
            "mean_abs_jerk_lat",
            "mean_abs_yaw_rate",
            "mean_abs_yaw_acc",
            "comfort_violation_rate",
            "harsh_brake_rate",
        ):
            try:
                cursor.execute(f"ALTER TABLE trial_eval_logs ADD COLUMN {column} REAL;")
            except Exception:
                pass
        self.db_conn.commit()

    def log_episode(self, infos, mean_return, idx, global_step=0):
        self.global_episode += 1

        data = infos
        if self._interactive:
            self._console.print(
                f"Step {abbreviate_number(global_step)} | Ep {self.global_episode} | Ret: [green]{data['return'][idx]:.2f}[/green] | "
                f"len_route: {int(data['len_ego_route'][idx])} | veh: {data['num_vehicles'][idx]} | "
                f"len_ep: {data['length'][idx]} | cause: {data['termination'][idx]} | MA50: {mean_return:.2f} | "
            )

        cause = data["termination"][idx]
        ep_return = data["return"][idx]
        ep_length = data["length"][idx]
        mean_reward = float(data["mean_reward"][idx])

        success = 1.0 if cause == "success" else 0.0
        collision = 1.0 if cause == "collision" else 0.0
        unfinished = 1.0 if cause in ["out_of_bounds", "off_road", "max_actions"] else 0.0

        self.episode_returns.append(ep_return)
        self.episode_lengths.append(ep_length)
        self.episode_successes.append(success)
        self.episode_collisions.append(collision)
        self.episode_unfinished.append(unfinished)

        self.writer.add_scalar("stats/episodic_return", ep_return, self.global_episode)
        self.writer.add_scalar("stats/episodic_length", ep_length, self.global_episode)
        self.writer.add_scalar("stats/mean_reward", mean_reward, self.global_episode)
        self.writer.add_scalar("stats/success_rate", success, self.global_episode)
        self.writer.add_scalar("stats/collision_rate", collision, self.global_episode)
        self.writer.add_scalar("stats/unfinished_rate", unfinished, self.global_episode)

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
            if self._interactive:
                self._console.print(table)
            self._logger.info(
                "[EP_STATS] step=%s episodes=%s mean_return=%.2f mean_length=%.1f success_rate=%.3f collision_rate=%.3f unfinished_rate=%.3f",
                abbreviate_number(global_step),
                self.global_episode,
                mean_ret,
                mean_len,
                mean_succ,
                mean_col,
                mean_unfin,
            )

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

            mean_ret = np.mean(self.episode_returns[-self._stats_interval :]) if self.episode_returns else 0.0
            mean_succ = np.mean(self.episode_successes[-self._stats_interval :]) if self.episode_successes else 0.0
            mean_col = np.mean(self.episode_collisions[-self._stats_interval :]) if self.episode_collisions else 0.0
            mean_unfin = np.mean(self.episode_unfinished[-self._stats_interval :]) if self.episode_unfinished else 0.0

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO trial_train_logs
                (trial_number, seed, global_step, walltime, mean_return, pg_loss, v_loss, entropy, approx_kl, clip_frac, ent_coef, train_success_rate, train_collision_rate, train_unfinished_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.trial_number,
                    self.seed,
                    global_step,
                    time.time(),
                    mean_ret,
                    pg_loss if pg_loss is not None else 0.0,
                    v_loss if v_loss is not None else 0.0,
                    entropy if entropy is not None else 0.0,
                    approx_kl if approx_kl is not None else 0.0,
                    clip_frac if clip_frac is not None else 0.0,
                    ent_coef if ent_coef is not None else 0.0,
                    mean_succ,
                    mean_col,
                    mean_unfin,
                ),
            )
            self.db_conn.commit()

    def log_evaluation(
        self,
        results_dict: dict,
        global_step: int = None,
        iteration: int = None,
        elapsed_time: float = None,
    ):
        if "success_rate" in results_dict:
            success_rate = results_dict["success_rate"]
            collision_rate = results_dict.get("collision_rate", 0.0)

            for threshold in self.success_thresholds:
                if success_rate >= threshold and threshold not in self.reached_thresholds:
                    self.reached_thresholds.add(threshold)
                    key = f"time_to_reach_{threshold}"
                    results_dict[key] = elapsed_time
                    self.threshold_stats[key] = elapsed_time
                    self.threshold_stats[f"step_to_reach_{threshold}"] = global_step

                    log_path = os.path.join(self.writer.log_dir, "benchmark_results.csv")
                    file_exists = os.path.isfile(log_path)
                    with open(log_path, "a", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        if not file_exists:
                            writer.writerow(["threshold", "success_rate", "collision_rate", "global_step", "iteration", "elapsed_time"])
                        writer.writerow([
                            threshold,
                            success_rate,
                            collision_rate,
                            global_step,
                            iteration,
                            elapsed_time,
                        ])

                    self._logger.info(
                        "Reached success threshold %s at step %s (iter=%s time=%.2fs)",
                        threshold,
                        global_step,
                        iteration,
                        elapsed_time if elapsed_time is not None else -1.0,
                    )

        for key, value in results_dict.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                self.writer.add_scalar(f"eval/{key}", value, global_step)

        table = Table(
            title=f"Evaluation Results @ step {global_step or '-'}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Metric", justify="right")
        table.add_column("Value", justify="center")

        for key, value in results_dict.items():
            value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
            table.add_row(key, value_str)

        if self._interactive:
            self._console.print(table)

        msg = " | ".join(
            [
                f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in results_dict.items()
            ]
        )
        self._logger.info(f"[EVAL] {msg}")
        self.update_status(
            state="running",
            last_eval_step=global_step,
            last_eval_metrics={
                key: float(value) if isinstance(value, (float, np.floating, int, np.integer)) else value
                for key, value in results_dict.items()
                if key in {
                    "mean_return",
                    "std_return",
                    "mean_length",
                    "success_rate",
                    "collision_rate",
                    "unfinished_rate",
                    "comfort_score",
                    "normalized_score",
                    "mean_abs_accel_long",
                    "mean_abs_accel_lat",
                    "mean_abs_jerk_long",
                    "mean_abs_jerk_lat",
                    "mean_abs_yaw_rate",
                    "mean_abs_yaw_acc",
                    "comfort_violation_rate",
                    "harsh_brake_rate",
                }
            },
        )

        if self.db_conn is not None:
            import time

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO trial_eval_logs
                (trial_number, seed, global_step, walltime,
                 mean_return, std_return, mean_length,
                 success_rate, collision_rate, unfinished_rate,
                 comfort_score, normalized_score,
                 mean_abs_accel_long, mean_abs_accel_lat, mean_abs_jerk_long, mean_abs_jerk_lat, mean_abs_yaw_rate, mean_abs_yaw_acc,
                 comfort_violation_rate, harsh_brake_rate,
                 time_to_reach_0_1, time_to_reach_0_2, time_to_reach_0_3, time_to_reach_0_4, time_to_reach_0_5,
                 time_to_reach_0_6, time_to_reach_0_7, time_to_reach_0_8, time_to_reach_0_9, time_to_reach_0_95, time_to_reach_0_99)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.trial_number,
                    self.seed,
                    global_step if global_step is not None else 0,
                    time.time(),
                    float(results_dict.get("mean_return", 0.0)),
                    float(results_dict.get("std_return", 0.0)),
                    float(results_dict.get("mean_length", 0.0)),
                    float(results_dict.get("success_rate", 0.0)),
                    float(results_dict.get("collision_rate", 0.0)),
                    float(results_dict.get("unfinished_rate", 0.0)),
                    float(results_dict.get("comfort_score", 0.0)),
                    float(results_dict.get("normalized_score", 0.0)),
                    float(results_dict.get("mean_abs_accel_long", 0.0)),
                    float(results_dict.get("mean_abs_accel_lat", 0.0)),
                    float(results_dict.get("mean_abs_jerk_long", 0.0)),
                    float(results_dict.get("mean_abs_jerk_lat", 0.0)),
                    float(results_dict.get("mean_abs_yaw_rate", 0.0)),
                    float(results_dict.get("mean_abs_yaw_acc", 0.0)),
                    float(results_dict.get("comfort_violation_rate", 0.0)),
                    float(results_dict.get("harsh_brake_rate", 0.0)),
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
                    results_dict.get("time_to_reach_0.99", None),
                ),
            )
            self.db_conn.commit()

    def msg(self, text):
        self._logger.info(text)

    def close(self):
        if self.db_conn is not None:
            self.db_conn.close()
            self.db_conn = None
        self.writer.close()
        for handler in list(self._logger.handlers):
            handler.flush()
            handler.close()
            self._logger.removeHandler(handler)
