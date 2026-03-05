from dataclasses import dataclass

@dataclass
class OptunaArgs:
    exp_id: int = 26
    phase: str = "1" # Choose "1", "2a", "2b", or "3"
    n_trials_phase_1: int = 100
    n_trials_phase_2a: int = 50
    n_trials_phase_2b: int = 40
    n_trials_phase_3: int = 30
    timesteps_phase_1: int = 1_000_000
    timesteps_phase_2a: int = 2_000_000
    timesteps_phase_2b: int = 2_000_000
    timesteps_phase_3: int = 2_000_000
    save_every_phase_1: int = 25 
    save_every_phase_2a: int = 25
    save_every_phase_2b: int = 25
    save_every_phase_3: int = 25
    eval_episodes: int = 30
    eval_final_episodes: int = 100
    num_seeds: int = 3
    
    top_k_phase_1: int = 10 # Number of best trials to consider for Phase 2a
    top_k_phase_2a: int = 10 # Number of best trials to consider for Phase 2b
    top_k_phase_2b: int = 5 # Number of best trials to consider for Phase 3
