from __future__ import annotations

import types

import pytest
import torch

import src.world_model.backbones as backbones_mod
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


@pytest.mark.unit
def test_stable_pretraining_backend_restores_common_logging(monkeypatch):
    calls: list[str] = []

    class FakeVit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=16)

        def forward(self, pixels, *, interpolate_pos_encoding=True):
            del interpolate_pos_encoding
            batch = pixels.shape[0]
            return types.SimpleNamespace(last_hidden_state=torch.randn(batch, 4, 16))

    fake_module = types.SimpleNamespace(vit_hf=lambda **kwargs: FakeVit())

    def fake_import_module(name: str):
        assert name == "stable_pretraining.backbone.utils"
        return fake_module

    monkeypatch.setattr(backbones_mod.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(backbones_mod, "configure_logging", lambda: calls.append("configured"))

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
    _ = backend(torch.randn(1, 3, 8, 8))

    assert calls == ["configured"]
