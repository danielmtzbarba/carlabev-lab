from torch import nn
import numpy as np

from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ConvolutionalPPO(nn.Module):
    def __init__(self, envs):
        super().__init__()

        # Automatically get number of input channels
        obs_shape = envs.single_observation_space.shape
        in_channels = obs_shape[0]  # works for (C, H, W)

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(4096, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def _normalize_input(self, x):
        if x.max() > 1.5:
            return x / 255.0  # RGB case
        else:
            return x  # already [0,1]

    def get_value(self, x):
        x = self._normalize_input(x)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None, deterministic=False):
        x = self._normalize_input(x)
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(probs.probs, dim=-1)
            else:
                action = probs.sample()

        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(hidden)

        return action, logprob, entropy, value
