from dataclasses import dataclass

@dataclass
class OptunaArgs:
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
    num_seeds: int = 3
    #
    top_k_phase_1: int = 10 # Number of best trials to consider for Phase 2a
    top_k_phase_2a: int = 10 # Number of best trials to consider for Phase 2b
    top_k_phase_2b: int = 5 # Number of best trials to consider for Phase 3
