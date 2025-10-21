import gymnasium as gym
from CarlaBEV.envs import make_carlabev_env
from CarlaBEV.envs import make_carlabev_eval

def make_env(args):
    envs = gym.vector.SyncVectorEnv(
        [
            make_carlabev_env(
                args.seed + i,
                i,
                args.capture_video,
                args.exp_name,
                obs_space=args.obs_space,
                size=args.size,
            )
            for i in range(args.num_envs)
        ]
    )

    return envs


def make_eval_env(exp_name, render=False):
    envs = gym.vector.SyncVectorEnv(
        [
            make_carlabev_eval(
                exp_name,
                obs_space="bev",
                size=128,
                render=render
            )
            for i in range(1)
        ]
    )

    return envs
