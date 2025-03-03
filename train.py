import sys
import random
import time

import logging
import numpy as np
import torch
import torch.nn.functional as F

import warnings

from src.evals.dqn_eval import eval_dqn_model
from src.agents import build_agent
from src.exp import get_experiment
from src.envs import make_env

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_handler = logging.FileHandler(filename="drlog.log")  # , mode="w")
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def log_episode(writer, infos):
    num_ep = infos["stats_ep"]["episode"]
    ret = infos["stats_ep"]["return"]
    cause = infos["stats_ep"]["termination"]
    stats = infos["stats_ep"]["stats"]
    success_rate = infos["stats_ep"]["success_rate"]
    collision_rate = infos["stats_ep"]["collision_rate"]

    #
    logger.info(f"episode-{num_ep}: {ret}-{cause}")
    #
    writer.add_scalar("stats/episodic_return", infos["episode"]["r"], num_ep)
    writer.add_scalar("stats/episodic_length", infos["episode"]["l"], num_ep)
    #

    writer.add_scalar(
        "stats/mean_reward",
        infos["stats_ep"]["mean_reward"],
        num_ep,
    )
    writer.add_scalar("stats/collision_rate", collision_rate, num_ep)
    writer.add_scalar("stats/success_rate", success_rate, num_ep)

    return num_ep


def train(args, envs, writer):
    q_network, optimizer, target_network, rb = build_agent(args, envs, device)
    # save blank model
    model_path = f"runs/{args.exp_name}/{args.exp_name}.cleanrl_model"
    torch.save(q_network.state_dict(), model_path)
    max_return = 0

    start_time = time.time()
    # TRY NOT TO MODIFY: start the game
    obs, infos = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(
                torch.from_numpy(obs).to(device),
                #                torch.from_numpy(poses).to(device),
            )
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if terminations[0]:
            num_ep = log_episode(writer, infos)

            if num_ep % 1000 == 0:
                eval_reward = eval_dqn_model(
                    args, num_ep, q_network, args.exp_name, writer
                )
                logger.info(f"eval-episode-{num_ep}: {eval_reward}")

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (
                        1 - data.dones.flatten()
                    )
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar(
                        "losses/q_values", old_val.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "stats/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data
                        + (1.0 - args.tau) * target_network_param.data
                    )

                if rewards[0] > max_return:
                    if args.save_model:
                        model_path = (
                            f"runs/{args.exp_name}/{args.exp_name}.cleanrl_model"
                        )
                        torch.save(q_network.state_dict(), model_path)
                        print(f"model saved to {model_path}")
                        max_return = rewards[0]

    envs.close()
    writer.close()


def main():
    args, writer = get_experiment()
    envs = make_env(args)
    train(args, envs, writer)


if __name__ == "__main__":
    main()
