import torch
import numpy as np

from src.agents.muzero.mcts_nav import MinMaxStats


def play_episode(env, mcts, isTraining):
    next_obs, _ = env.reset()
    b, f, c, h, w = next_obs.shape
    done = False
    episode = []
    total_reward = 0

    while not done:
        min_max_stats = MinMaxStats()
        obs = torch.tensor(next_obs, dtype=torch.float32).cuda()
        obs = obs.view((b, -1, h, w))  # Combine f and c into one dim

        root = mcts.run(obs, min_max_stats, isTraining=isTraining)
        visit_counts = np.array(
            [
                root.children[a].visit_count if a in root.children else 0
                for a in range(mcts.action_space_size)
            ]
        )
        # Select action
        if isTraining:
            policy = visit_counts / visit_counts.sum()
            action = np.random.choice(mcts.action_space_size, p=policy)
        else:
            action = np.argmax(visit_counts)  # Greedy action for evaluation

        next_obs, reward, done, _, info = env.step([action])
        total_reward += reward if isinstance(reward, (int, float)) else sum(reward)

        if isTraining:
            episode.append((obs.detach().cpu(), action, reward, policy))

    return episode, info


def evaluate(env, mcts, num_episodes, logger):
    rets, lens = [], []
    for ep in range(num_episodes):
        _, info = play_episode(env, mcts, isTraining=False)
        rets.append(info["termination"]["return"][0])
        lens.append(info["termination"]["length"][0])
    logger.evaluation(rets, lens)
    return np.mean(np.array(rets))


def self_play(env, mcts, replay_buffer, num_episodes, logger):
    for ep in range(num_episodes):
        episode_data, info = play_episode(env, mcts, isTraining=True)

        td_errors = []
        for t, (obs, action, reward, policy) in enumerate(episode_data):
            with torch.no_grad():
                hiddens = mcts.muzero.representation(obs.cuda())
                predicted_value = mcts.muzero.value_head(hiddens).item()
                observed_return = sum([step[2] for step in episode_data[t:]])

                # TD error
                td_error = abs(reward + observed_return - predicted_value)
                td_errors.append(td_error)

        # Add episode
        replay_buffer.add(episode_data, td_error=max(td_errors))

        # Log average reward
        avg_rwd = replay_buffer.average_reward()
        logger.episode(info, avg_rwd)
    return
