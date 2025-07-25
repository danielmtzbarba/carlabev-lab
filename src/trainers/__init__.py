from src.trainers.dqn import train_dqn
from src.trainers.ppo import train_ppo
from src.trainers.sac import train_sac
from src.trainers.muzero import train_muzero


def build_trainer(experiment):
    if "DQN" in experiment:
        return train_dqn
    elif "PPO" in experiment:
        return train_ppo
    elif "SAC" in experiment:
        return train_sac
    elif "MuZero" in experiment:
        return train_muzero
    else:
        exit("Unimplemented DRL algorithm")
