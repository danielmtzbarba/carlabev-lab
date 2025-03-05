from src.trainers.dqn import train_dqn
from src.trainers.ppo import train_ppo


def build_trainer(experiment):
    if "DQN" in experiment:
        return train_dqn
    if "PPO" in experiment:
        return train_ppo
    else:
        exit("Unimplemented DRL algorithm")
