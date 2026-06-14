from __future__ import annotations

import pytest
import torch

from src.world_model.backbones import build_vit_backend
from src.world_model.config import WorldModelModelConfig


@pytest.mark.unit
def test_stable_pretraining_backend_projects_tokens_to_encoder_dim():
    cfg = WorldModelModelConfig(
        encoder_backend="stable_pretraining_vit_hf",
        encoder_size="small",
        patch_size=4,
        encoder_dim=16,
        encoder_depth=1,
        predictor_depth=1,
        num_heads=4,
        latent_dim=16,
    )

    backend = build_vit_backend(obs_shape=(3, 8, 8), cfg=cfg)
    output = backend(torch.randn(2, 3, 8, 8))

    assert output.last_hidden_state.shape[0] == 2
    assert output.last_hidden_state.shape[-1] == cfg.encoder_dim
