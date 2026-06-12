from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.config.base_config import ArgsCarlaBEV


class DummyWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int | None]] = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), step))

    def add_text(self, *_args, **_kwargs):
        return None

    @property
    def log_dir(self) -> str:
        return "dummy-log-dir"

    def close(self):
        return None


class DummyLogger:
    def __init__(self):
        self.writer = DummyWriter()
        self.messages: list[str] = []
        self.episode_logs: list[dict] = []
        self.learning_logs: list[dict] = []
        self.evaluation_logs: list[dict] = []
        self.threshold_stats: dict[str, float] = {}

    def msg(self, text):
        self.messages.append(str(text))

    def log_episode(self, infos, mean_return, idx, global_step=0):
        self.episode_logs.append(
            {
                "infos": infos,
                "mean_return": mean_return,
                "idx": idx,
                "global_step": global_step,
            }
        )
        return len(self.episode_logs)

    def log_learning(self, **kwargs):
        self.learning_logs.append(kwargs)

    def log_evaluation(self, results_dict, global_step=None, iteration=None, elapsed_time=None):
        self.evaluation_logs.append(
            {
                "results_dict": results_dict,
                "global_step": global_step,
                "iteration": iteration,
                "elapsed_time": elapsed_time,
            }
        )

    def set_running(self):
        return None

    def mark_completed(self, final_metrics=None):
        self.final_metrics = final_metrics

    def update_status(self, **_kwargs):
        return None

    def close(self):
        self.writer.close()


class DummyTrial:
    def __init__(self, number: int = 7):
        self.number = number
        self.user_attrs: dict[str, object] = {}
        self.reports: list[tuple[float, int]] = []

    def suggest_float(self, name, low, high, log=False):
        del low, high, log
        values = {
            "learning_rate": 2e-4,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "clip_coef_start": 0.2,
            "ent_coef_start": 0.01,
            "vf_coef_start": 0.6,
            "max_grad_norm": 0.5,
            "ent_decay_factor": 0.3,
            "vf_decay_factor": 0.8,
            "clip_decay_factor": 0.7,
        }
        return values[name]

    def suggest_categorical(self, name, choices):
        return {
            "num_steps": 128,
            "num_minibatches": 2,
        }.get(name, choices[0])

    def suggest_int(self, name, low, high):
        del low, high
        values = {"update_epochs": 4}
        return values[name]

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def report(self, value, step):
        self.reports.append((float(value), int(step)))

    def should_prune(self):
        return False


@dataclass
class FakeStepResult:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


class FakeVectorEnv:
    def __init__(
        self,
        *,
        num_envs: int = 2,
        obs_shape: tuple[int, ...] = (3, 96, 96),
        action_space: gym.Space | None = None,
        terminate_every: int = 1,
    ):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(
            low=0.0,
            high=255.0,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space
        self.single_action_space = action_space or gym.spaces.Discrete(5)
        self._terminate_every = terminate_every
        self._step_count = 0

    def _obs(self) -> np.ndarray:
        fill = float(self._step_count % 255)
        return np.full(
            (self.num_envs,) + self.single_observation_space.shape,
            fill_value=fill,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        del seed, options
        self._step_count = 0
        return self._obs(), {}

    def step(self, actions):
        del actions
        self._step_count += 1
        done = self._step_count % self._terminate_every == 0
        terminated = np.full((self.num_envs,), done, dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        reward = np.full((self.num_envs,), 1.0, dtype=np.float32)
        info = {
            "episode_info": {
                "return": np.full((self.num_envs,), float(self._step_count), dtype=np.float32),
                "length": np.full((self.num_envs,), self._step_count, dtype=np.int32),
                "mean_reward": np.full((self.num_envs,), 1.0, dtype=np.float32),
                "len_ego_route": np.full((self.num_envs,), 10 + self._step_count, dtype=np.int32),
                "num_vehicles": np.full((self.num_envs,), 2, dtype=np.int32),
                "termination": np.array(
                    ["success" if done else "running"] * self.num_envs,
                    dtype=object,
                ),
                "mean_abs_accel_long": np.full((self.num_envs,), 0.1, dtype=np.float32),
                "mean_abs_accel_lat": np.full((self.num_envs,), 0.1, dtype=np.float32),
                "mean_abs_jerk_long": np.full((self.num_envs,), 0.1, dtype=np.float32),
                "mean_abs_jerk_lat": np.full((self.num_envs,), 0.1, dtype=np.float32),
                "mean_abs_yaw_rate": np.full((self.num_envs,), 0.1, dtype=np.float32),
                "mean_abs_yaw_acc": np.full((self.num_envs,), 0.1, dtype=np.float32),
                "comfort_violation_rate": np.full((self.num_envs,), 0.0, dtype=np.float32),
                "harsh_brake_rate": np.full((self.num_envs,), 0.0, dtype=np.float32),
            }
        }
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None


@pytest.fixture
def discrete_env():
    return FakeVectorEnv()


@pytest.fixture
def continuous_env():
    return FakeVectorEnv(
        action_space=gym.spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
    )


@pytest.fixture
def vector_env():
    return FakeVectorEnv(
        obs_shape=(8,),
        action_space=gym.spaces.Discrete(4),
    )


@pytest.fixture
def dummy_logger():
    return DummyLogger()


@pytest.fixture
def dummy_trial():
    return DummyTrial()


@pytest.fixture
def tiny_cfg():
    cfg = ArgsCarlaBEV()
    cfg.study_id = "PPO_NAVIGATION"
    cfg.exp_id = 26
    cfg.algorithm = "cnn-ppo"
    cfg.num_envs = 2
    cfg.seed = 123
    cfg.run_mode = "headless"
    cfg.capture_video = False
    cfg.train_video_count = 0
    cfg.eval_video_count = 0
    cfg.final_eval_video_count = 0
    cfg.save_model = False
    cfg.num_evals = 1
    cfg.eval_episodes = 1
    cfg.eval_final_episodes = 1
    cfg.ppo.total_timesteps = 8
    cfg.ppo.num_steps = 4
    cfg.ppo.num_minibatches = 1
    cfg.ppo.update_epochs = 1
    cfg.ppo.channels = [8, 16, 16]
    cfg.ppo.fc_size = 32
    cfg.env.curriculum_enabled = False
    cfg.env.traffic_enabled = True
    cfg.env.curriculum_mode = "both"
    return cfg


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fixed_eval_payload():
    aggregate = {
        "mean_return": 10.0,
        "std_return": 0.0,
        "mean_length": 3.0,
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "unfinished_rate": 0.0,
        "mean_abs_accel_long": 0.1,
        "mean_abs_accel_lat": 0.1,
        "mean_abs_jerk_long": 0.1,
        "mean_abs_jerk_lat": 0.1,
        "mean_abs_yaw_rate": 0.1,
        "mean_abs_yaw_acc": 0.1,
        "comfort_violation_rate": 0.0,
        "harsh_brake_rate": 0.0,
    }
    return {"aggregate": aggregate, "protocols": {"random_nav_eval": aggregate.copy()}}
