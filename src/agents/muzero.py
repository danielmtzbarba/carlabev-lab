import torch
from torch import nn


class MuZeroAgent(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        # 6-channel semantic input
        self.representation = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Dynamics network: hidden state + action -> next hidden state + reward
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.reward_head = nn.Linear(hidden_dim, 1)

        # Prediction network: hidden state -> policy logits + value
        self.policy_head = nn.Linear(hidden_dim, 5)  # 5 actions
        self.value_head = nn.Linear(hidden_dim, 1)

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        hidden_state = hidden_state.mean(dim=[2, 3])  # Global pooling
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action_one_hot):
        x = torch.cat([hidden_state, action_one_hot], dim=1)
        next_hidden_state = self.dynamics(x)
        policy_logits = self.policy_head(next_hidden_state)
        value = self.value_head(next_hidden_state)
        reward = self.reward_head(next_hidden_state)
        return next_hidden_state, policy_logits, value, reward
