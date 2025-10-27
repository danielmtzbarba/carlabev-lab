from dataclasses import dataclass


@dataclass
class EnvConfig:
    env_id: str = "CarlaBEV-v0"
    size: int = 128
    map_name: str = "Town01"
    obs_space: str = "bev"  # "bev" or "vector"
    #
    action_space: str = "discrete"  # "discrete" or "continuous"
    render_mode: str = "rgb_array"
    max_actions: int = 5000
    vehicle_growth_start: int = 1000
    seed: int = 0
    record_videos: bool = False
    scenes_path: str = "assets/scenes"
    reward_params: dict = None


@dataclass
class PPOConfig:
    # PPO core
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4  # slightly higher, tune if unstable
    num_envs: int = 4  # match CPUs available
    num_steps: int = 1024  # rollout length per env → buffer size = 3072
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.97
    num_minibatches: int = 4  # equal to num_envs (1 minibatch per env)
    update_epochs: int = 8
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.003
    vf_coef: float = 0.6
    max_grad_norm: float = 0.5
    target_kl: float = 0.02  # small KL target helps stabilize

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    # Decay configuration
    ent_coef_start: float = 0.015
    ent_coef_end: float = 0.003
    vf_coef_start: float = 0.7
    vf_coef_end: float = 0.5
    clip_coef_start: float = 0.2
    clip_coef_end: float = 0.08
    decay_schedule: str = "cosine"  # or "linear"


@dataclass
class ArgsCarlaBEV:
    exp_name: str = "cnn-ppo-carlabev-debug"
    seed: int = 1
    cuda: bool = True
    save_model: bool = True
    capture_video: bool = True
    torch_deterministic: bool = True

    ppo: PPOConfig
    env: EnvConfig
