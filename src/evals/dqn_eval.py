import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from src.agents.dqn import QNetwork
from src.envs import make_carlabev_env


def eval_dqn_model(args, num_episode, q_network, run_name, writer):
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    from src.evals.dqn_eval import evaluate

    episodic_returns = evaluate(
        args,
        model_path,
        make_carlabev_env,
        args.env_id,
        num_episode=num_episode,
        eval_episodes=10,
        run_name=f"{run_name}-eval",
        Model=QNetwork,
        device="cuda:0",
        epsilon=0.05,
        writer=writer,
    )

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "DQN",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )


def evaluate(
    args,
    model_path: str,
    make_env: Callable,
    env_id: str,
    num_episode: int,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
    writer: object = None,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, 0, capture_video, run_name, size=args.size)]
    )
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, terminations, _, infos = envs.step(actions)

        if terminations[0]:
            num_ep = infos["stats_ep"]["episode"]
            ret = infos["stats_ep"]["return"]
            cause = infos["stats_ep"]["termination"]
            stats = infos["stats_ep"]["stats"]
            success_rate = infos["stats_ep"]["success_rate"]
            collision_rate = infos["stats_ep"]["collision_rate"]

            #
            writer.add_scalar(f"eval/{num_episode[0]}/episodic_return", ret, num_ep)

            episodic_returns += [ret]
        obs = next_obs

    return episodic_returns
