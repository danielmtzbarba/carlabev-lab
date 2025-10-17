import torch
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1000, alpha=0.6):
        """
        Replay buffer with prioritized experience replay and sliding window sampling.

        Args:
            capacity (int): Maximum number of episodes in the buffer.
            alpha (float): Priority exponent. Higher values prioritize high-TD-error samples more.
        """
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Priority array
        self.capacity = capacity
        self.alpha = alpha
        self.position = 0  # Tracks the current position for overwriting

    def add(self, episode, td_error=1.0):
        """
        Add an episode to the buffer with an associated priority.

        Args:
            episode (list): The episode to add (list of transitions).
            td_error (float): Temporal-difference error for prioritization.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(episode)
        else:
            self.buffer[self.position] = episode

        # Assign priority
        self.priorities[self.position] = abs(td_error) + 1e-6  # Avoid zero priority
        self.position = (self.position + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size, beta=0.4, sliding_window_ratio=0.5):
        """
        Sample a batch of episodes using prioritized experience replay.

        Args:
            batch_size (int): Number of episodes to sample.
            beta (float): Importance-sampling exponent. Higher values correct for bias.
            sliding_window_ratio (float): Fraction of the buffer to sample from (e.g., 0.5 for the most recent 50%).

        Returns:
            samples (list): Sampled episodes.
            indices (list): Indices of the sampled episodes.
            weights (torch.Tensor): Importance-sampling weights for the sampled episodes.
        """
        # Determine the sliding window range
        sliding_window_start = max(0, len(self.buffer) - int(self.capacity * sliding_window_ratio))
        priorities = self.priorities[sliding_window_start:len(self.buffer)] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices based on priorities
        indices = np.random.choice(
            range(sliding_window_start, len(self.buffer)), batch_size, p=probabilities
        )
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices - sliding_window_start]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32).cuda()

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of sampled episodes.

        Args:
            indices (list): Indices of the sampled episodes.
            td_errors (list): New TD errors for the sampled episodes.
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6  # Avoid zero priority

    def average_reward(self, window=10):
        """
        Compute the average reward over the last `window` episodes.

        Args:
            window (int): Number of recent episodes to consider.

        Returns:
            float: Average reward over the last `window` episodes.
        """
        if len(self.buffer) == 0:
            return 0.0
        last_episodes = self.buffer[-window:]
        mean_rewards = 0
        for episode in last_episodes:
            mean_rewards += sum([reward for _, _, reward, _ in episode]) / len(episode)
        return mean_rewards / len(last_episodes)

    def __len__(self):
        return len(self.buffer)


def prepare_tensors(batch):
    obs_list, action_list, reward_list, policy_list = zip(*batch)
    obs_tensor = torch.stack([obs.cpu() for obs in obs_list]).squeeze(1).cuda()
    reward_tensor = torch.stack(
        [torch.tensor(reward, dtype=torch.float32) for reward in reward_list]
    ).cuda()
    policy_tensor = torch.stack(
        [torch.tensor(policy, dtype=torch.float32) for policy in policy_list]
    ).cuda()
    action_tensor = torch.stack(
        [torch.tensor(action, dtype=torch.long) for action in action_list]
    ).cuda()

    # Ensure action_tensor has a batch dimension
    if action_tensor.dim() == 1:  # Single batch element
        action_tensor = action_tensor.unsqueeze(0)  # Add batch dimension

    return obs_tensor, action_tensor, reward_tensor, policy_tensor
