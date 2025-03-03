import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation
)


def make_racing_env(env_id, seed, idx, capture_video, run_name, continuous=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", continuous=continuous)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, continuous=continuous)
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, (96, 96))
        env = FrameStackObservation(env, num_stack=4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk
