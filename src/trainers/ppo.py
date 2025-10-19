import time
import os
import random
import torch
import numpy as np
from torch import nn

from src.agents import build_agent


class RewardNormalizer:
    def __init__(self, clip_range=(-1, 1), decay=0.99):
        self.mean = 0.0
        self.var = 1.0
        self.decay = decay
        self.clip_range = clip_range

    def update(self, batch_rewards):
        """Update statistics from a batch (not single step)."""
        batch_mean = np.mean(batch_rewards)
        batch_var = np.var(batch_rewards)
        self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
        self.var = self.decay * self.var + (1 - self.decay) * batch_var

    def normalize(self, reward):
        std = np.sqrt(self.var) + 1e-8
        normalized = (reward - self.mean) / std
        return np.clip(normalized, *self.clip_range)


def train_ppo(args, envs, logger, device):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    agent, optimizer = build_agent(args, envs, device)
    # Storage
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        dtype=torch.float32,
        device=device,
    )
    logprobs = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    rewards = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    dones = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )
    values = torch.zeros(
        (args.num_steps, args.num_envs), dtype=torch.float32, device=device
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    normalizer = RewardNormalizer(clip_range=(-1, 1), decay=0.99)
    # --- At the start of training ---
    best_return = -float("inf")  # track best episodic return

    for iteration in range(1, args.num_iterations + 1):
        # LR annealing
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step].copy_(next_obs)
            dones[step].copy_(next_done)

            # action selection
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # value shape should be (N,1) or (N,)
                values[step].copy_(value.view(-1))

            # store
            actions[step].copy_(action)
            logprobs[step].copy_(logprob.view(-1))

            # Step envs
            action_cpu = action.cpu().numpy()
            next_obs_np, reward_np, terminations, truncations, infos = envs.step(
                action_cpu
            )

            # Apply reward normalization
            norm_rewards = np.array([normalizer.normalize(r) for r in reward_np])
            rewards_tensor = torch.tensor(
                norm_rewards, dtype=torch.float32, device=device
            )

            # Store normalized rewards
            rewards[step].copy_(rewards_tensor)

            # Store next observations and done flags
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(
                np.logical_or(terminations, truncations).astype(np.float32),
                device=device,
            )

            # Episode logging for every finished env
            if "termination" in infos.keys():
                causes = infos["termination"]["termination"]
                for i, cause in enumerate(causes):
                    if cause is not None:
                        info = {
                            "length": infos["termination"]["length"][i],
                            "return": infos["termination"]["return"][i],
                            "cause": causes[i],
                        }
                        # Only pass the finished episode to the logger
                        logger.log_episode(info)

        # --- bootstrap and GAE (ensure 1D shapes everywhere) ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)  # (num_envs,)

        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            # delta is 1D: shape (num_envs,)
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
            advantages[t] = lastgaelam

        returns = advantages + values

        # flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape).to(device)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape).to(device)
        b_logprobs = logprobs.reshape(-1).to(device)
        b_advantages = advantages.reshape(-1).to(device)
        b_returns = returns.reshape(-1).to(device)
        b_values = values.reshape(-1).to(device)

        # ✅ Normalize returns (helps stabilize value function training)
        b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

        # Optimize
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl = 0.0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, mb_actions.long()
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # diagnostics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # optional logging
        if iteration % 10 == 0:
            clip_frac_mean = np.mean(clipfracs) if clipfracs else 0
            logger.log_learning(
                global_step=global_step,
                pg_loss=pg_loss.item(),
                v_loss=v_loss.item(),
                entropy=entropy_loss.item(),
                approx_kl=approx_kl if approx_kl is not None else None,
                clip_frac=clip_frac_mean,
            )

        # Save last model every iteration
        if iteration % 100 == 0:
            torch.save(
                agent.state_dict(), os.path.join(f"runs/{args.exp_name}", "ppo_last.pt")
            )
            logger.msg(f"🌟 Model saved at {iteration} iteration!")

    envs.close()
    logger.writer.close()
