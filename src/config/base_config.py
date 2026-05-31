from dataclasses import dataclass, field

from CarlaBEV.config import EnvConfig as CarlaBEVEnvConfig
from CarlaBEV.config import RunConfig as CarlaBEVRunConfig


@dataclass
class LoggerConfig:
    enabled: bool = False
    dir: str = "results/carlabev/runs/"
    db_path: str = None
    trial_number: int = None


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
    fov_masked: bool = True
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
    reward_type: str = "carl"  # "shaping" | "carl"


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 14

    # Phase 1
    anneal_lr: bool = True
    learning_rate: float = 3e-4
    gae_lambda: float = 0.9
    gamma: float = 0.995

    # Phase 2a 
    num_steps: int = 256
    num_minibatches: int = 4
    update_epochs: int = 6

    # Phase 2b
    ent_coef: float = 0.015
    vf_coef: float = 0.65
    clip_coef: float = 0.18
    max_grad_norm: float = 0.4

    ent_coef_start: float = 0.015
    ent_decay_factor: float = 0.2
    ent_decay_schedule: str = "cosine"
    
    vf_coef_start: float = 0.65
    vf_decay_factor: float = 0.85
    vf_decay_schedule: str = "linear"
    
    clip_coef_start: float = 0.18
    clip_decay_factor: float = 0.65
    clip_decay_schedule: str = "cosine"
    
    # Phase 3: Network Architecture Policy
    channels: list = field(default_factory=lambda: [32, 64, 64])
    fc_size: int = 512

    # Other
    target_kl: float = 0.015
    norm_adv: bool = True
    clip_vloss: bool = True

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0



@dataclass
class ArgsCarlaBEV:
    study_id: str = "PPO_NAVIGATION"
    exp_id: int = 1
    exp_name: str = "default"
    algorithm: str = "cnn-ppo"
    train_protocol_id: str | None = None
    eval_protocol_ids: list[str] = field(default_factory=list)
    num_envs: int = 14

    cuda: bool = True
    seed: int = 1000
    torch_deterministic: bool = True

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    logging: LoggerConfig = field(default_factory=LoggerConfig)

    capture_video: bool = True 
    capture_every: int = 250

    save_model: bool = True 
    save_every: int = 200

    num_evals: int = 5
    eval_episodes: int = 100
    eval_final_episodes: int = 1000


def to_carlabev_env_config(env_cfg: EnvConfig) -> CarlaBEVEnvConfig:
    if env_cfg.obs_space == "vector":
        obs_mode = "vector"
    elif env_cfg.masked:
        obs_mode = "bev_semantic"
    else:
        obs_mode = "bev_rgb"

    return CarlaBEVEnvConfig(
        seed=env_cfg.seed,
        fps=env_cfg.fps,
        size=env_cfg.size,
        env_id=env_cfg.env_id,
        map_name=env_cfg.map_name,
        obs_size=env_cfg.obs_size,
        obs_mode=obs_mode,
        fov_masked=env_cfg.fov_masked,
        frame_stack=env_cfg.frame_stack,
        action_mode=env_cfg.action_space,
        render_mode=env_cfg.render_mode,
        max_actions=env_cfg.max_actions,
        scenes_path=env_cfg.scenes_path,
        reward_mode=env_cfg.reward_type,
        traffic_enabled=env_cfg.traffic_enabled,
        max_vehicles=env_cfg.max_vehicles,
    )


def to_carlabev_run_config(args: ArgsCarlaBEV) -> CarlaBEVRunConfig:
    return CarlaBEVRunConfig(
        env=to_carlabev_env_config(args.env),
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        seed=args.seed,
        capture_video=args.capture_video,
        capture_every=args.capture_every,
        cuda=args.cuda,
        torch_deterministic=args.torch_deterministic,
    )
