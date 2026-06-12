def build_trainer(algorithm: str):
    trainer_builders = {
        "cnn-ppo": "train_ppo",
    }

    try:
        trainer_name = trainer_builders[algorithm]
    except KeyError as exc:
        available = ", ".join(sorted(trainer_builders))
        raise ValueError(
            f"Unsupported trainer algorithm '{algorithm}'. "
            f"Available algorithms: {available}"
        ) from exc

    if algorithm == "cnn-ppo":
        from src.trainers.ppo import train_ppo

        return train_ppo

    raise RuntimeError(f"Trainer resolution failed for algorithm '{algorithm}' ({trainer_name})")
