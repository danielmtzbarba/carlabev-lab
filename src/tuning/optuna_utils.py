from dataclasses import dataclass


DEFAULT_STUDY_PRIME_SEEDS: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def resolve_study_prime_seeds(num_seeds: int) -> list[int]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be positive.")
    if num_seeds > len(DEFAULT_STUDY_PRIME_SEEDS):
        raise ValueError(
            f"Requested num_seeds={num_seeds}, but only {len(DEFAULT_STUDY_PRIME_SEEDS)} "
            "shared prime study seeds are defined."
        )
    return list(DEFAULT_STUDY_PRIME_SEEDS[:num_seeds])


@dataclass
class OptunaArgs:
    study_id: str = "PPO_NAVIGATION"
    exp_id: int = 26
    stage: str = "policy_dynamics"
    n_trials: int | None = None
    total_timesteps: int | None = None
    eval_episodes: int | None = None
    eval_final_episodes: int | None = None
    num_seeds: int | None = None
