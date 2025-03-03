import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from .dqn import QNetwork
from .mpnet import MPNet


def build_agent(args, envs, device):
    if "dqn" in args.exp_name:
        q_network = QNetwork(envs).to(device)
        #q_network = MPNet(envs).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
        target_network = QNetwork(envs).to(device)
        #target_network = MPNet(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        return q_network, optimizer, target_network, rb
    else:
        exit()
