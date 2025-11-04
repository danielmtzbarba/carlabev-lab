import gymnasium as gym
from CarlaBEV.envs import make_carlabev_env
from CarlaBEV.envs import make_carlabev_eval

def make_env(cfg):
    envs = gym.vector.SyncVectorEnv(
        [
            make_carlabev_env(i, cfg) for i in range(cfg.num_envs)
        ]
    )

    return envs


def make_eval_env(cfg):
    envs = gym.vector.SyncVectorEnv(
        [
            make_carlabev_eval(cfg) for i in range(1)
        ]
    )

    return envs
