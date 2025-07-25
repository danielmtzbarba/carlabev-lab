import torch
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer

from .dqn import QNetwork
from .ppo import Agent
from .sac import SoftQNetwork, Actor
from .muzero import MuZeroAgent


def build_agent(args, envs, device):
    if "dqn" in args.exp_name:
        q_network = QNetwork(envs).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
        target_network = QNetwork(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        return q_network, optimizer, target_network, rb

    elif "ppo" in args.exp_name:
        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        return agent, optimizer
    
    elif "sac" in args.exp_name:

        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())


        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        q_optimizer = optim.Adam(
            list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4
        )
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )

        return actor, qf1, qf2, qf1_target, qf2_target, q_optimizer, actor_optimizer, rb

    elif "muzero" in args.exp_name:
        agent = MuZeroAgent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        return agent, optimizer
    else:
        exit()
