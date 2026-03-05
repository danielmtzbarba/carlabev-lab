import torch
from torch import nn
import numpy as np

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import gymnasium as gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNBackbone(nn.Module):
    """Shared feature extractor for both discrete and continuous agents."""
    def __init__(self, in_channels, channels=[32, 64, 64], fc_size=512):
        super().__init__()
        self.in_channels = in_channels
        
        layers = []
        current_channels = in_channels
        
        # Build Conv layers dynamically based on the channels list
        if len(channels) > 0:
            layers.append(layer_init(nn.Conv2d(current_channels, channels[0], kernel_size=8, stride=4)))
            layers.append(nn.ReLU())
            current_channels = channels[0]
            
        if len(channels) > 1:
            layers.append(layer_init(nn.Conv2d(current_channels, channels[1], kernel_size=4, stride=2)))
            layers.append(nn.ReLU())
            current_channels = channels[1]
            
        if len(channels) > 2:
            layers.append(layer_init(nn.Conv2d(current_channels, channels[2], kernel_size=3, stride=1)))
            layers.append(nn.ReLU())
            current_channels = channels[2]
            
        layers.append(nn.Flatten())
        
        # Dummy forward pass to calculate the output size of the conv+flatten layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 96, 96) # Default observation size
            dummy_backend = nn.Sequential(*layers)
            flattened_size = dummy_backend(dummy_input).shape[1]
            
        layers.append(layer_init(nn.Linear(flattened_size, fc_size)))
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BasePPOAgent(nn.Module):
    """Base class for PPO agents with shared utilities."""
    def __init__(self, envs, channels=[32, 64, 64], fc_size=512):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.backbone = CNNBackbone(obs_shape[0], channels, fc_size)
        self.critic = layer_init(nn.Linear(fc_size, 1), std=1)

    def _normalize_input(self, x):
        if x.max() > 1.5:
            return x / 255.0  # RGB case [0, 255]
        return x  # already [0, 1]

    def get_value(self, x):
        x = self._normalize_input(x)
        return self.critic(self.backbone(x))


class DiscreteConvolutionalPPO(BasePPOAgent):
    def __init__(self, envs, channels=[32, 64, 64], fc_size=512):
        super().__init__(envs, channels, fc_size)
        self.actor = layer_init(nn.Linear(fc_size, envs.single_action_space.n), std=0.01)
        self.is_continuous = False

    def get_action_and_value(self, x, action=None, deterministic=False):
        x = self._normalize_input(x)
        hidden = self.backbone(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(probs.probs, dim=-1)
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class ContinuousConvolutionalPPO(BasePPOAgent):
    def __init__(self, envs, channels=[32, 64, 64], fc_size=512):
        super().__init__(envs, channels, fc_size)
        action_dim = np.prod(envs.single_action_space.shape)
        self.actor_mean = layer_init(nn.Linear(fc_size, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.is_continuous = True

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Calculates action, log_prob, entropy, and value.
        
        ### Squashing Logic with Jacobian Correction:
        To bound actions within [0, 1] (gas/brake) and [-1, 1] (steering), we apply 
        Sigmoid and Tanh transformations. To maintain mathematical correctness 
        in the probability space, we subtract the log-determinant of the Jacobian 
        of these transformations from the raw Gaussian log-probabilities.
        """
        x = self._normalize_input(x)
        hidden = self.backbone(x)
        
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            if deterministic:
                raw_action = action_mean
            else:
                raw_action = probs.sample()
        else:
            # We assume 'action' passed during update is in the UNCONSTRAINED space 
            # if calculating loss, or we should have stored raw actions.
            # In our trainer, we store the result of get_action_and_value (processed_action).
            # This is a common pitfall. To keep PPO simple, we will calculate log_prob 
            # on the raw_action and the trainer will store this raw_action.
            raw_action = action

        # Environmental mapping (Squashing)
        gas = torch.sigmoid(raw_action[:, 0:1])
        steering = torch.tanh(raw_action[:, 1:2])
        brake = torch.sigmoid(raw_action[:, 2:3])
        processed_action = torch.cat([gas, steering, brake], dim=1)

        # log_prob calculation on the unconstrained action
        logprob = probs.log_prob(raw_action)

        # Jacobian correction: log_prob(squashed) = log_prob(raw) - log(|det(dy/dx)|)
        # For Tanh(x): d/dx = 1 - tanh^2(x)
        # For Sigmoid(x): d/dx = sigmoid(x) * (1 - sigmoid(x))
        
        # We subtract the log-determinant because log_prob is log(p(x)).
        # log p(y) = log p(x) - log |dy/dx|
        
        # Gas correction (Sigmoid)
        logprob[:, 0] -= (torch.log(gas + 1e-6) + torch.log(1.0 - gas + 1e-6)).squeeze(1)
        # Steering correction (Tanh)
        logprob[:, 1] -= torch.log(1.0 - steering.pow(2) + 1e-6).squeeze(1)
        # Brake correction (Sigmoid)
        logprob[:, 2] -= (torch.log(brake + 1e-6) + torch.log(1.0 - brake + 1e-6)).squeeze(1)

        logprob = logprob.sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(hidden)

        return raw_action, processed_action, logprob, entropy, value
