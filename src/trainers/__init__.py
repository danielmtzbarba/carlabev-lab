def build_trainer(experiment):
    if "dqn" in experiment:
        from src.trainers.dqn import train_dqn

        return train_dqn
    elif "ppo" in experiment:
        from src.trainers.ppo import train_ppo

        return train_ppo
    elif "sac" in experiment:
        from src.trainers.sac import train_sac

        return train_sac
    elif "muzero" in experiment:
        from src.trainers.muzero import train_muzero

        return train_muzero
    elif "debug" in experiment:
        return train_ppo
    else:
        exit("Unimplemented DRL algorithm")
