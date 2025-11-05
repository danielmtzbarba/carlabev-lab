from dataclasses import dataclass

@dataclass
class ArgsCarlaBEV:
    exp_name: str = "muzero-carlabev-test"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    size: int = 128
    """size of rendered image"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CarlaBEV-v0"
    """the id of the environment"""
    discrete: bool = True
    """Discrete or Continouous agent"""
    obs_space: str = "bev" 
    """RGB or vector Data"""
    action_space_size: int = 9
    """Number of Actions"""

    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4

    # Muzero Parameters
    batch_size: int = 16
    """the batch size"""
    hidden_dim: int = 128 
    """hidden_dim"""
    num_self_play_episodes: int = 20 
    """hidden_dim"""
    num_unroll_steps: int = 5 
    """hidden_dim"""
    buffer_size: int = 250 
    """hidden_dim"""
    learning_rate: float = 2.5e-4 
    """hidden_dim"""
    c_puct: float = 2.5e-4 
    num_simulations: int = 100
    iterations: int = 500 
    eval_episodes: int = 5
