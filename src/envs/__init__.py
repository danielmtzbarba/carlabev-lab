import gymnasium as gym
from .carlabev import make_carlabev_env


def make_env(args):
    envs = gym.vector.SyncVectorEnv(
        [
            make_carlabev_env(
                args.env_id,
                args.seed + i,
                i,
                args.capture_video,
                args.exp_name,
                size=args.size,
            )
            for i in range(args.num_envs)
        ]
    )

    return envs
