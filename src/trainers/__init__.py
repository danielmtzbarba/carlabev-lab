from src.trainers.dqn import train_dqn
from src.trainers.ppo import train_ppo
from src.trainers.muzero import train_muzero


def build_trainer(experiment):
    if "DQN" in experiment:
        return train_dqn
    if "PPO" in experiment:
        return train_ppo

    if "MuZero" in experiment:
        return train_muzero
    else:
        exit("Unimplemented DRL algorithm")
