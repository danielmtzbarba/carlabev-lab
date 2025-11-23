import time
import os
import random
import torch
import numpy as np
from torch import nn

from random import choice
from src.agents import build_agent
from src.trainers.utils import CurriculumState
from src.eval.eval_ppo import evaluate_ppo

from collections import deque

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


def decay_schedule(start, end, progress, mode="linear"):
    """Compute decayed value given progress in [0,1]."""
    if mode == "linear":
        return start + (end - start) * progress
    elif mode == "cosine":
        return end + (start - end) * (0.5 * (1 + np.cos(np.pi * progress)))
    elif mode == "exp":
        return end + (start - end) * np.exp(-5 * progress)
    else:
        return start


def train_ppo(cfg, envs, logger, device):
    num_envs = cfg.num_envs
    ppo_cfg = cfg.ppo

    ppo_cfg.batch_size = int(num_envs * ppo_cfg.num_steps)
    ppo_cfg.minibatch_size = int(ppo_cfg.batch_size // ppo_cfg.num_minibatches)
    ppo_cfg.num_iterations = ppo_cfg.total_timesteps // ppo_cfg.batch_size
    agent, optimizer = build_agent(cfg, envs, device)
    curr_state = CurriculumState(cfg.env)
    return_buffer = deque(maxlen=50)

    model_channels = agent.network[0].in_channels
    logger.msg(f"Observation space: {envs.observation_space}")
    logger.msg(f"Model_channels: {model_channels}")
    # Storage
    obs = torch.zeros(
        (ppo_cfg.num_steps, num_envs) + envs.single_observation_space.shape,
        dtype=torch.float32,
        device=device,
    )
    actions = torch.zeros(
        (ppo_cfg.num_steps, num_envs) + envs.single_action_space.shape,
        dtype=torch.float32,
        device=device,
    )
    logprobs = torch.zeros(
        (ppo_cfg.num_steps, num_envs), dtype=torch.float32, device=device
    )
    rewards = torch.zeros(
        (ppo_cfg.num_steps, num_envs), dtype=torch.float32, device=device
    )
    dones = torch.zeros(
        (ppo_cfg.num_steps, num_envs), dtype=torch.float32, device=device
    )
    values = torch.zeros(
        (ppo_cfg.num_steps, num_envs), dtype=torch.float32, device=device
    )

    global_step = 0
    start_time = time.time()

    options={
        #edge cases
            #    "scene": choice(["lead_brake", "jaywalk"]),
        "scene": "rdm",
        "num_vehicles": 0,
        "route_dist_range": [50, 150], 
        "reset_mask": np.full((num_envs), True)
    }

    next_obs, _ = envs.reset(seed=cfg.seed, options=options)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    normalizer = RewardNormalizer(clip_range=(-1, 1), decay=0.99)
    # --- At the start of training ---
    best_return = -float("inf")  # track best episodic return

    for iteration in range(1, ppo_cfg.num_iterations + 1):
        # LR annealing
        if ppo_cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / ppo_cfg.num_iterations
            lrnow = frac * ppo_cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --- Adaptive coefficient decays ---
        progress = (iteration - 1) / ppo_cfg.num_iterations
        ppo_cfg.ent_coef = decay_schedule(
            ppo_cfg.ent_coef_start, ppo_cfg.ent_coef_end, progress, ppo_cfg.decay_schedule
        )
        ppo_cfg.vf_coef = decay_schedule(
            ppo_cfg.vf_coef_start, ppo_cfg.vf_coef_end, progress, ppo_cfg.decay_schedule
        )
        ppo_cfg.clip_coef = decay_schedule(
            ppo_cfg.clip_coef_start, ppo_cfg.clip_coef_end, progress, ppo_cfg.decay_schedule
        )
        if iteration % (ppo_cfg.num_iterations // 6) == 0:
            ppo_cfg.ent_coef *= 1.2  # small entropy boost
            ppo_cfg.ent_coef = min(ppo_cfg.ent_coef, ppo_cfg.ent_coef_start)

        for step in range(ppo_cfg.num_steps):
            global_step += num_envs
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

            # Compute done flags
            dones_np = np.logical_or(terminations, truncations)

            # Episode logging for every finished env
            for i, ended  in enumerate(terminations):
                if ended:
                    ep_return = infos["episode_info"]["return"][i]
                    return_buffer.append(ep_return)
            
                    # Compute smoothed return
                    mean_return = sum(return_buffer) / len(return_buffer)
                    logger.log_episode(infos["episode_info"], mean_return, i,  global_step)

                    # === Reset the finished env ===
                    options={
                        #edge cases
                    #    "scene": choice(["lead_brake", "jaywalk"]),
                        "scene": "rdm",
                        "num_vehicles": curr_state.vehicle_schedule(mean_return),
                        "route_dist_range": curr_state.route_schedule(mean_return),
                        "reset_mask": dones_np
                    }
                    # reset() returns FULL batch of obs for ALL envs
                    next_obs_np, reset_info = envs.reset(seed=cfg.seed, options=options)

            # === Convert to tensors for buffer storage ===
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(dones_np.astype(np.float32), device=device)

        # --- bootstrap and GAE (ensure 1D shapes everywhere) ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).view(-1)  # (num_envs,)

        advantages = torch.zeros_like(rewards, device=device)
        lastgaelam = torch.zeros(num_envs, dtype=torch.float32, device=device)

        for t in reversed(range(ppo_cfg.num_steps)):
            if t == ppo_cfg.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            # delta is 1D: shape (num_envs,)
            delta = rewards[t] + ppo_cfg.gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = (
                delta + ppo_cfg.gamma * ppo_cfg.gae_lambda * nextnonterminal * lastgaelam
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
        b_inds = np.arange(ppo_cfg.batch_size)
        clipfracs = []
        approx_kl = 0.0
        for epoch in range(ppo_cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, ppo_cfg.batch_size, ppo_cfg.minibatch_size):
                end = start + ppo_cfg.minibatch_size
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
                        ((ratio - 1.0).abs() > ppo_cfg.clip_coef).float().mean().item()
                    )

                mb_adv = b_advantages[mb_inds]
                if ppo_cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - ppo_cfg.clip_coef, 1 + ppo_cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if ppo_cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -ppo_cfg.clip_coef, ppo_cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ppo_cfg.ent_coef * entropy_loss + ppo_cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()

            if ppo_cfg.target_kl is not None and approx_kl > ppo_cfg.target_kl:
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
            logger.writer.add_scalar("schedules/ent_coef", ppo_cfg.ent_coef, global_step)
            logger.writer.add_scalar("schedules/vf_coef", ppo_cfg.vf_coef, global_step)
            logger.writer.add_scalar("schedules/clip_coef", ppo_cfg.clip_coef, global_step)
            logger.writer.add_scalar(
                "schedules/lr", optimizer.param_groups[0]["lr"], global_step
            )

        # inject entropy boosts to enforce exploration in middle of training
            #if iteration % 5000 == 0:
        #    ppo_cfg.ent_coef = min(ppo_cfg.ent_coef * 1.3, ppo_cfg.ent_coef_start)

        # Save last model every iteration
        if iteration % cfg.save_every == 0:
            model_path = os.path.join(f"runs/{cfg.exp_name}", "ppo_last.pt")
            torch.save(agent.state_dict(), model_path)
            logger.msg(f"🌟 Model saved at {iteration} iteration!")
            eval_results = evaluate_ppo(
                cfg,
                model_path=model_path,
                num_episodes=10,
                render=False,  # turn True for visualization
                device="cuda",
            )
            logger.log_evaluation(eval_results, global_step)


    model_path = os.path.join(f"runs/{cfg.exp_name}", "ppo_final.pt")
    torch.save(agent.state_dict(), model_path)
    eval_results = evaluate_ppo(
        cfg,
        model_path=model_path,
        num_episodes=30,
        render=False,  # turn True for visualization
        device="cuda",
    )
    logger.log_evaluation(eval_results, global_step)
    logger.msg(f"🌟 Training finished at {iteration} iteration!")
    
    envs.close()
    logger.writer.close()
