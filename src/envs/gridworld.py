import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation
)

from gridworld.envs import Go2TargetEnv


def make_gridworld_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = Go2TargetEnv(render_mode="rgb_array", size=8)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = Go2TargetEnv(render_mode="rgb_array", size=8)

        env = GrayscaleObservation(env)
        env = ResizeObservation(env, (96, 96))
        env = FrameStackObservation(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk
