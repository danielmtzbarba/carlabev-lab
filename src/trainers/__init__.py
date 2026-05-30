def build_trainer(algorithm: str):
    trainer_builders = {
        "dqn": "train_dqn",
        "cnn-ppo": "train_ppo",
        "vector-ppo": "train_ppo",
        "sac": "train_sac",
        "muzero": "train_muzero",
    }

    try:
        trainer_name = trainer_builders[algorithm]
    except KeyError as exc:
        available = ", ".join(sorted(trainer_builders))
        raise ValueError(
            f"Unsupported trainer algorithm '{algorithm}'. "
            f"Available algorithms: {available}"
        ) from exc

    if algorithm == "dqn":
        from src.trainers.dqn import train_dqn

        return train_dqn
    if algorithm in {"cnn-ppo", "vector-ppo"}:
        from src.trainers.ppo import train_ppo

        return train_ppo
    if algorithm == "sac":
        from src.trainers.sac import train_sac

        return train_sac
    if algorithm == "muzero":
        from src.trainers.muzero import train_muzero

        return train_muzero

    raise RuntimeError(f"Trainer resolution failed for algorithm '{algorithm}' ({trainer_name})")
