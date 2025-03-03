import gymnasium as gym

# from .racing import make_racing_env
# from .gridworld import make_gridworld_env
from .carlabev import make_carlabev_env


def make_env(args, run_name):
    if args.env_id == "CarlaBEV-v0":
        envs = gym.vector.SyncVectorEnv(
            [
                make_carlabev_env(
                    args.env_id,
                    args.seed + i,
                    i,
                    args.capture_video,
                    run_name,
                    size=args.size,
                )
                for i in range(args.num_envs)
            ]
        )
    elif args.env_id == "CarRacing-v2":
        # env setup
        envs = gym.vector.SyncVectorEnv(
            [
                """
                make_racing_env(
                    args.env_id,
                    args.seed + i,
                    i,
                    args.capture_video,
                    run_name,
                    continuous=False,
                )
                for i in range(args.num_envs)
                """
            ]
        )
    else:
        exit()

    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    return envs
