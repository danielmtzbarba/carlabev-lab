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
    phase: str = "1" # Choose "1", "2a", "2b", or "3"
    n_trials_phase_1: int = 100
    n_trials_phase_2a: int = 60
    n_trials_phase_2b: int = 100
    n_trials_phase_3: int = 50
    timesteps_phase_1: int = 1_100_000
    timesteps_phase_2a: int = 1_100_000
    timesteps_phase_2b: int = 1_100_000
    timesteps_phase_3: int = 2_100_000
    #
    eval_episodes: int = 30
    eval_final_episodes: int = 100
    num_seeds: int = 10
    #
    top_k_phase_1: int = 10 # Number of best trials to consider for Phase 2a
    top_k_phase_2a: int = 10 # Number of best trials to consider for Phase 2b
    top_k_phase_2b: int = 5 # Number of best trials to consider for Phase 3
