import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from src.agents.muzero.muzero import MuZeroAgent
from src.agents.muzero.mcts_nav import MuZeroMCTS
from src.agents.muzero.self_play import self_play, evaluate
from src.agents.muzero.replay_buffer import ReplayBuffer, prepare_tensors

from CarlaBEV.envs import make_env_muzero

def train_network(model, optimizer, scaler, replay_buffer, config, logger, device):
    model.train()
    batch, indices, weights = replay_buffer.sample(config.batch_size, beta=0.4)

    total_losses, value_losses, policy_losses, reward_losses = [], [], [], []

    for i, episode in enumerate(batch):
        obs_tensor, action_tensor, reward_tensor, policy_tensor = prepare_tensors(episode)

        with torch.no_grad():
            hidden = model.representation(obs_tensor.to(device))
            predicted_value = model.value_head(hidden)
            observed_return = torch.tensor([[sum(step[2] for step in episode)]], device=device)
            td_error = torch.abs(predicted_value.detach() - observed_return).cpu().numpy()
            replay_buffer.update_priorities(indices[i:i+1], td_error)

        # --- Initial inference ---
        root_obs = obs_tensor[0].unsqueeze(0).to(device)
        hidden_state, _, _ = model.initial_inference(root_obs)

        total_loss = 0.0
        for step in range(config.num_unroll_steps):
            if step >= action_tensor.size(1):
                break

            action = action_tensor[:, step].to(device)
            with autocast(device_type="cuda"):
                hidden_state, policy_logits, pred_value, pred_reward = model.recurrent_inference(hidden_state, action)

                reward_loss = F.mse_loss(pred_reward, reward_tensor[step].unsqueeze(0).to(device))
                value_loss = F.mse_loss(pred_value, reward_tensor[step].unsqueeze(0).to(device))

                pred_policy = F.softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(policy_tensor * torch.log(pred_policy + 1e-8), dim=1).mean()

                step_loss = (0.01 * policy_loss) + value_loss + reward_loss
                loss = (weights[i] * step_loss).mean()
                total_loss += loss.item()

            optimizer.zero_grad()
            scaler.scale(step_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            value_losses.append(value_loss.item())
            policy_losses.append(policy_loss.item())
            reward_losses.append(reward_loss.item())

        total_losses.append(total_loss)

    # --- Logging ---
    mean_loss = np.mean(total_losses)
    logger.log_learning(
        loss=mean_loss,
        value=np.mean(value_losses),
        policy=np.mean(policy_losses),
        reward=np.mean(reward_losses),
    )


def train_muzero(config, _, logger, device):
    # --- Initialization ---
    env =  make_env_muzero()
    replay_buffer = ReplayBuffer(capacity=config.buffer_size)
    network = MuZeroAgent(hidden_dim=config.hidden_dim, action_space_size=config.action_space_size).to(device)
    mcts = MuZeroMCTS(
        network=network,
        action_space_size=config.action_space_size,
        num_simulations=config.num_simulations,
    )
    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    scaler = GradScaler()

    best_return = -float("inf")

    for it in range(1, config.iterations + 1):
        logger.msg(f"=== MuZero Iteration {it}/{config.iterations} ===")

        # --- Self-play phase ---
        self_play(env, mcts, replay_buffer, config.num_self_play_episodes, logger)

        # --- Training phase ---
        if len(replay_buffer) >= config.batch_size:
            train_network(network, optimizer, scaler, replay_buffer, config, logger, device)

        # --- Periodic evaluation ---
        if it % config.eval_interval == 0:
            avg_return = evaluate(env, mcts, config.eval_episodes, logger)
            logger.log_episode({
                "length": 0,  # placeholder if not episodic
                "return": avg_return,
                "cause": "eval"
            })

            # Save models
            torch.save(network.state_dict(), os.path.join(f"runs/{config.exp_name}", "muzero_last.pt"))
            if avg_return > best_return:
                best_return = avg_return
                torch.save(network.state_dict(), os.path.join(f"runs/{config.exp_name}", "muzero_best.pt"))
                logger.msg(f"🌟 New best model saved! Avg return: {best_return:.2f}")

    logger.msg("Training finished ✅")
    return network
