import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from src.evals.dqn_eval import eval_dqn_model
from src.agents import build_agent
from src.utils.utils import linear_schedule


def train_dqn(args, envs, logger, device):
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
            )
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if terminations[0]:
            num_ep = logger.log_episode(infos)

            # TODO: IMPLEMENT VALIDATION
            if num_ep % 1000 == 0:
                pass

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = obs[idx]
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
                    sps = int(global_step / (time.time() - start_time))
                    q = old_val.mean().item()
                    logger.log_loss(global_step, loss, sps, q)

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
    logger.writer.close()
