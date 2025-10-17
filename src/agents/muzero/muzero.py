import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class RepresentationNet(nn.Module):
    def __init__(self, input_channels=6, hidden_dim=128):
        super().__init__()
        self.initial_conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        x = self.residual_blocks(x)
        x = self.global_pool(x)  # (batch, hidden_dim, 1, 1)
        return x.view(x.size(0), -1)  # Flatten to (batch, hidden_dim)

class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim=128, action_space_size=5):
        super().__init__()
        self.action_embedding = nn.Embedding(action_space_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state, action):
        """
        hidden_state: (batch, hidden_dim)
        action: (batch,) - action indices
        """
        action_emb = self.action_embedding(action)  # (batch, hidden_dim)
        x = torch.cat([hidden_state, action_emb], dim=1)  # (batch, hidden_dim * 2)
        x = F.relu(self.fc1(x))
        next_hidden_state = F.relu(self.fc2(x))  # (batch, hidden_dim)
        reward = self.reward_head(next_hidden_state)  # (batch, 1)
        return next_hidden_state, reward

class MuZeroAgent(nn.Module):
    def __init__(self, hidden_dim=128, action_space_size=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size
        
        self.representation_net = RepresentationNet(input_channels=24, hidden_dim=hidden_dim)
        self.dynamics_net = DynamicsNet(hidden_dim=hidden_dim, action_space_size=action_space_size)

        # --- Prediction network ---
        self.policy_head = nn.Linear(hidden_dim, action_space_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def representation(self, batch):
        """
        observation: (batch, 6, H, W)
        returns: (batch, hidden_dim)
        """
        
        return self.representation_net(batch) 

    def dynamics(self, hidden_state, action_one_hot):
        """
        hidden_state: (batch, hidden_dim)
        action_one_hot: (batch, action_space_size)
        returns: next_hidden_state (batch, hidden_dim), reward (batch, 1)
        """
        return self.dynamics_net(hidden_state, action_one_hot)                        # (batch, hidden_dim)

    def prediction(self, hidden_state):
        """
        hidden_state: (batch, hidden_dim)
        returns: policy_logits (batch, action_space_size), value (batch, 1)
        """
        policy_logits = self.policy_head(hidden_state)           # (batch, action_space_size)
        value = self.value_head(hidden_state)                    # (batch, 1)
        return policy_logits, value

    def initial_inference(self, observation):
        """
        The first step: from observation to initial hidden state, policy and value
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(self, hidden_state, action_one_hot):
        """
        After the initial step: from hidden state and action to next hidden, policy, value, reward
        """
        next_hidden_state, reward = self.dynamics(hidden_state, action_one_hot)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, policy_logits, value, reward
