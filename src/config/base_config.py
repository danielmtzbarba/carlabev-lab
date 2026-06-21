from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

from CarlaBEV.config import EnvConfig as CarlaBEVEnvConfig
from CarlaBEV.config import RunConfig as CarlaBEVRunConfig

from src.config.studies.models import SemanticMaskMode, TemporalFusionMode


LEGACY_ACTION_PROFILE_IDS: dict[str, str] = {
    "discrete": "discrete9_v1",
    "continuous": "continuous_gsb_v1",
}
LEGACY_REWARD_PROFILE_IDS: dict[str, str] = {
    "carl": "carl_base_v1",
    "shaping": "shaping_base_v1",
}


def _warn_legacy_name(legacy: str, canonical: str):
    warnings.warn(
        f"`{legacy}` is deprecated in carlabev-lab; use `{canonical}` instead.",
        FutureWarning,
        stacklevel=3,
    )


@dataclass
class LoggerConfig:
    enabled: bool = False
    dir: str = "results/carlabev/runs/"
    db_path: str | None = None
    trial_number: int | None = None


@dataclass(init=False)
class EnvConfig:
    seed: int
    fps: int
    size: int
    env_id: str
    map_name: str
    obs_mode: str
    semantic_mask_ch: SemanticMaskMode
    temporal_fusion_mode: TemporalFusionMode
    obs_size: tuple[int, int]
    fov_masked: bool
    ego_anchor_x_frac: float
    ego_anchor_y_frac: float
    frame_stack: int

    action_mode: str
    action_profile_id: str
    render_mode: str
    max_actions: int
    scenes_path: str

    route_dist_range: tuple[int, int]

    traffic_enabled: bool
    route_direction_metrics_enabled: bool
    scene_library_enabled: bool
    scene_library_path: str
    scene_library_read_only: bool
    scene_library_require_hit: bool
    scene_library_generator_version: str

    reward_mode: str
    reward_profile_id: str

    def __init__(
        self,
        *,
        seed: int = 0,
        fps: int = 60,
        size: int = 128,
        env_id: str = "CarlaBEV-v0",
        map_name: str = "Town01",
        obs_mode: str | None = None,
        semantic_mask_ch: SemanticMaskMode = "6-class",
        temporal_fusion_mode: TemporalFusionMode = "stack",
        obs_size: tuple[int, int] = (96, 96),
        fov_masked: bool = True,
        ego_anchor_x_frac: float = 0.5,
        ego_anchor_y_frac: float = 0.5,
        frame_stack: int = 4,
        action_mode: str | None = None,
        action_profile_id: str | None = None,
        render_mode: str = "rgb_array",
        max_actions: int = 5000,
        scenes_path: str = "assets/scenes",
        route_dist_range: tuple[int, int] = (30, 100),
        traffic_enabled: bool = False,
        route_direction_metrics_enabled: bool = True,
        scene_library_enabled: bool = True,
        scene_library_path: str = "assets/scenes/scene_library.db",
        scene_library_read_only: bool = False,
        scene_library_require_hit: bool = False,
        scene_library_generator_version: str = "role_traffic_v1",
        reward_mode: str | None = None,
        reward_profile_id: str | None = None,
        obs_space: str | None = None,
        masked: bool | None = None,
        action_space: str | None = None,
        reward_type: str | None = None,
    ):
        if obs_space is not None:
            _warn_legacy_name("obs_space", "obs_mode")
        if masked is not None:
            _warn_legacy_name("masked", "obs_mode")
        if action_space is not None:
            _warn_legacy_name("action_space", "action_mode")
        if reward_type is not None:
            _warn_legacy_name("reward_type", "reward_mode")

        if obs_mode is None:
            if obs_space == "vector":
                obs_mode = "vector"
            elif masked is False:
                obs_mode = "bev_rgb"
            else:
                obs_mode = "bev_semantic"

        if action_mode is None:
            action_mode = action_space or "discrete"
        if action_profile_id is None:
            action_profile_id = LEGACY_ACTION_PROFILE_IDS.get(action_mode, "discrete9_v1")

        if reward_mode is None:
            reward_mode = "carl" if reward_type == "carl" else "shaping"
        elif reward_mode not in {"shaping", "carl"}:
            reward_mode = "carl" if reward_mode == "carl" else "shaping"
        if reward_profile_id is None:
            reward_profile_id = LEGACY_REWARD_PROFILE_IDS.get(reward_mode, "shaping_base_v1")

        self.seed = seed
        self.fps = fps
        self.size = size
        self.env_id = env_id
        self.map_name = map_name
        self.obs_mode = obs_mode
        self.semantic_mask_ch = semantic_mask_ch
        self.temporal_fusion_mode = temporal_fusion_mode
        self.obs_size = obs_size
        self.fov_masked = fov_masked
        self.ego_anchor_x_frac = ego_anchor_x_frac
        self.ego_anchor_y_frac = ego_anchor_y_frac
        self.frame_stack = frame_stack
        self.action_mode = action_mode
        self.action_profile_id = action_profile_id
        self.render_mode = render_mode
        self.max_actions = max_actions
        self.scenes_path = scenes_path
        self.route_dist_range = route_dist_range
        self.traffic_enabled = traffic_enabled
        self.route_direction_metrics_enabled = route_direction_metrics_enabled
        self.scene_library_enabled = scene_library_enabled
        self.scene_library_path = scene_library_path
        self.scene_library_read_only = scene_library_read_only
        self.scene_library_require_hit = scene_library_require_hit
        self.scene_library_generator_version = scene_library_generator_version
        self.reward_mode = reward_mode
        self.reward_profile_id = reward_profile_id

    @property
    def obs_space(self) -> str:
        return "vector" if self.obs_mode == "vector" else "bev"

    @obs_space.setter
    def obs_space(self, value: str):
        _warn_legacy_name("obs_space", "obs_mode")
        if value == "vector":
            self.obs_mode = "vector"
        elif value == "bev":
            if self.obs_mode == "vector":
                self.obs_mode = "bev_semantic"
        else:
            raise ValueError(f"Unsupported obs_space: {value}")

    @property
    def masked(self) -> bool:
        return self.obs_mode == "bev_semantic"

    @masked.setter
    def masked(self, value: bool):
        _warn_legacy_name("masked", "obs_mode")
        self.obs_mode = "bev_semantic" if value else "bev_rgb"

    @property
    def action_space(self) -> str:
        return self.action_mode

    @action_space.setter
    def action_space(self, value: str):
        _warn_legacy_name("action_space", "action_mode")
        self.action_mode = value

    @property
    def reward_type(self) -> str:
        return self.reward_mode

    @reward_type.setter
    def reward_type(self, value: str):
        _warn_legacy_name("reward_type", "reward_mode")
        self.reward_mode = "carl" if value == "carl" else "shaping"

    @property
    def input_type(self) -> str:
        return "masks" if self.obs_mode == "bev_semantic" else "rgb"

    @input_type.setter
    def input_type(self, value: str):
        if value == "masks":
            self.obs_mode = "bev_semantic"
        elif value == "rgb":
            self.obs_mode = "bev_rgb"
        else:
            raise ValueError(f"Unsupported input_type: {value}")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "fps": self.fps,
            "size": self.size,
            "env_id": self.env_id,
            "map_name": self.map_name,
            "obs_mode": self.obs_mode,
            "semantic_mask_ch": self.semantic_mask_ch,
            "temporal_fusion_mode": self.temporal_fusion_mode,
            "obs_size": self.obs_size,
            "fov_masked": self.fov_masked,
            "ego_anchor_x_frac": self.ego_anchor_x_frac,
            "ego_anchor_y_frac": self.ego_anchor_y_frac,
            "frame_stack": self.frame_stack,
            "action_mode": self.action_mode,
            "action_profile_id": self.action_profile_id,
            "render_mode": self.render_mode,
            "max_actions": self.max_actions,
            "scenes_path": self.scenes_path,
            "route_dist_range": self.route_dist_range,
            "traffic_enabled": self.traffic_enabled,
            "route_direction_metrics_enabled": self.route_direction_metrics_enabled,
            "scene_library_enabled": self.scene_library_enabled,
            "scene_library_path": self.scene_library_path,
            "scene_library_read_only": self.scene_library_read_only,
            "scene_library_require_hit": self.scene_library_require_hit,
            "scene_library_generator_version": self.scene_library_generator_version,
            "reward_mode": self.reward_mode,
            "reward_profile_id": self.reward_profile_id,
        }

    def legacy_aliases(self) -> dict:
        return {
            "obs_space": self.obs_space,
            "masked": self.masked,
            "action_space": self.action_space,
            "reward_type": self.reward_type,
        }


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 14

    anneal_lr: bool = True
    learning_rate: float = 3e-4
    gae_lambda: float = 0.9
    gamma: float = 0.995

    num_steps: int = 256
    num_minibatches: int = 4
    update_epochs: int = 6

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

    channels: list = field(default_factory=lambda: [32, 64, 64])
    fc_size: int = 512

    target_kl: float = 0.015
    norm_adv: bool = True
    clip_vloss: bool = True

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


@dataclass
class ArgsCarlaBEV:
    study_id: str = "PPO_NAVIGATION"
    exp_id: int = 1
    exp_name: str = "default"
    run_label: str = "default"
    run_id: str = "default"
    run_dir: str = "runs/default"
    algorithm: str = "cnn-ppo"
    run_mode: Literal["interactive", "headless"] = "interactive"
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
    video_output_dir: str | None = None
    video_episode_indices: list[int] | None = None
    video_name_prefix: str = "rl-video"
    train_video_count: int = 20
    eval_video_count: int = 5
    final_eval_video_count: int = 10

    save_model: bool = True
    save_every: int = 200

    num_evals: int = 5
    eval_episodes: int = 100
    eval_final_episodes: int = 1000

    def to_dict(self) -> dict:
        return {
            "study_id": self.study_id,
            "exp_id": self.exp_id,
            "exp_name": self.exp_name,
            "run_label": self.run_label,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "algorithm": self.algorithm,
            "run_mode": self.run_mode,
            "train_protocol_id": self.train_protocol_id,
            "eval_protocol_ids": list(self.eval_protocol_ids),
            "num_envs": self.num_envs,
            "cuda": self.cuda,
            "seed": self.seed,
            "torch_deterministic": self.torch_deterministic,
            "env": self.env.to_dict(),
            "ppo": self.ppo.__dict__.copy(),
            "logging": self.logging.__dict__.copy(),
            "capture_video": self.capture_video,
            "capture_every": self.capture_every,
            "video_output_dir": self.video_output_dir,
            "video_episode_indices": self.video_episode_indices,
            "video_name_prefix": self.video_name_prefix,
            "train_video_count": self.train_video_count,
            "eval_video_count": self.eval_video_count,
            "final_eval_video_count": self.final_eval_video_count,
            "save_model": self.save_model,
            "save_every": self.save_every,
            "num_evals": self.num_evals,
            "eval_episodes": self.eval_episodes,
            "eval_final_episodes": self.eval_final_episodes,
        }

    def legacy_aliases(self) -> dict:
        return {"env": self.env.legacy_aliases()}


def to_carlabev_env_config(env_cfg: EnvConfig) -> CarlaBEVEnvConfig:
    return CarlaBEVEnvConfig(
        seed=env_cfg.seed,
        fps=env_cfg.fps,
        size=env_cfg.size,
        env_id=env_cfg.env_id,
        map_name=env_cfg.map_name,
        obs_size=env_cfg.obs_size,
        obs_mode=env_cfg.obs_mode,
        semantic_mask_ch=env_cfg.semantic_mask_ch,
        temporal_fusion_mode=env_cfg.temporal_fusion_mode,
        fov_masked=env_cfg.fov_masked,
        ego_anchor_x_frac=env_cfg.ego_anchor_x_frac,
        ego_anchor_y_frac=env_cfg.ego_anchor_y_frac,
        frame_stack=env_cfg.frame_stack,
        action_mode=env_cfg.action_mode,
        action_profile_id=env_cfg.action_profile_id,
        render_mode=env_cfg.render_mode,
        max_actions=env_cfg.max_actions,
        scenes_path=env_cfg.scenes_path,
        reward_mode=env_cfg.reward_mode,
        reward_profile_id=env_cfg.reward_profile_id,
        traffic_enabled=env_cfg.traffic_enabled,
        route_direction_metrics_enabled=env_cfg.route_direction_metrics_enabled,
        scene_library_enabled=env_cfg.scene_library_enabled,
        scene_library_path=env_cfg.scene_library_path,
        scene_library_read_only=env_cfg.scene_library_read_only,
        scene_library_generator_version=env_cfg.scene_library_generator_version,
    )


def to_carlabev_run_config(args: ArgsCarlaBEV) -> CarlaBEVRunConfig:
    return CarlaBEVRunConfig(
        env=to_carlabev_env_config(args.env),
        exp_name=args.exp_name,
        num_envs=args.num_envs,
        seed=args.seed,
        capture_video=args.capture_video,
        capture_every=args.capture_every,
        video_output_dir=args.video_output_dir,
        video_episode_indices=args.video_episode_indices,
        video_name_prefix=args.video_name_prefix,
        cuda=args.cuda,
        torch_deterministic=args.torch_deterministic,
    )
