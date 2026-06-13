from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorldModelDataConfig:
    dataset_paths: list[str] = field(default_factory=list)
    batch_size: int = 32
    num_workers: int = 0
    chunk_length: int = 8
    stride: int = 1
    val_ratio: float = 0.1
    expected_num_actions: int = 9
    include_metadata: bool = False


@dataclass
class WorldModelModelConfig:
    patch_size: int = 8
    encoder_dim: int = 256
    encoder_depth: int = 4
    predictor_depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    action_embed_dim: int = 64
    latent_dim: int = 256
    dropout: float = 0.0


@dataclass
class WorldModelOptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0


@dataclass
class WorldModelTrainLoopConfig:
    epochs: int = 10
    device: str = "cuda"
    save_every: int = 1
    sigreg_weight: float = 0.1


@dataclass
class WorldModelConfig:
    run_name: str = "lewm-phase1"
    data: WorldModelDataConfig = field(default_factory=WorldModelDataConfig)
    model: WorldModelModelConfig = field(default_factory=WorldModelModelConfig)
    optimizer: WorldModelOptimizerConfig = field(default_factory=WorldModelOptimizerConfig)
    training: WorldModelTrainLoopConfig = field(default_factory=WorldModelTrainLoopConfig)
