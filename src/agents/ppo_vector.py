from torch import nn
import numpy as np
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class VectorPPO(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        input_dim = envs.single_observation_space.shape[0]  # assuming 1D vector input
        hidden_dim = 128

        self.network = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        self.actor = layer_init(nn.Linear(hidden_dim, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
