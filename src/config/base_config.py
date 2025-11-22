from dataclasses import dataclass, field


@dataclass
class LoggerConfig:
    enabled: bool = False
    dir: str = "results/carlabev/runs/"


@dataclass
class EnvConfig:
    seed: int = 0
    fps: int = 60
    size: int = 128
    env_id: str = "CarlaBEV-v0"
    map_name: str = "Town01"
    obs_space: str = "bev"
    obs_size: tuple = (96, 96)
    masked: bool = True
    frame_stack: int = 4

    action_space: str = "discrete"
    render_mode: str = "rgb_array"
    max_actions: int = 5000

    scenes_path: str = "assets/scenes"

    # Curriculum
    curriculum_enabled: bool = False
    curriculum_mode: str = "vehicles"
    route_dist_range: tuple = (30, 100)

    # Traffic
    traffic_enabled: bool = False
    max_vehicles: int = 25

    # Reward
    reward_type: str = "shaping"  # "shaping" | "carl"


@dataclass
class PPOConfig:
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4

    num_envs: int = 14
    num_steps: int = 256

    anneal_lr: bool = True
    gamma: float = 0.995

    gae_lambda: float = 0.9
    num_minibatches: int = 4
    update_epochs: int = 8

    clip_coef: float = 0.15
    ent_coef: float = 0.003
    vf_coef: float = 0.7

    max_grad_norm: float = 0.4
    target_kl: float = 0.015


@dataclass
class ArgsCarlaBEV:
    exp_id: int = 1
    exp_name: str = "default"

    cuda: bool = True
    seed: int = 1000

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)

    capture_video: bool = True
    capture_every: int = 50
    save_model: bool = True
    save_every: int = 200
    torch_deterministic: bool = True
