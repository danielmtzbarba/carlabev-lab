from dataclasses import dataclass

@dataclass
class ArgsCarlaBEV:
    exp_name: str = "cnn-ppo-discrete-carlabev-12env"
    seed: int = 1
    size: int = 128
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = True# disable in training, enable only for eval
    save_model: bool = True

    # Environment
    env_id: str = "CarlaBEV-v0"
    discrete: bool = True
    obs_space: str = "bev" 

    # PPO core
    total_timesteps: int = 20_000_000
    learning_rate: float = 3e-4      # slightly higher, tune if unstable
    num_envs: int = 12               # match CPUs available
    num_steps: int = 128             # rollout length per env → buffer size = 3072
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.92
    num_minibatches: int = 12        # equal to num_envs (1 minibatch per env)
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.015         # small KL target helps stabilize

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
