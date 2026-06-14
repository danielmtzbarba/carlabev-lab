from __future__ import annotations

import torch

from src.world_model.config import WorldModelTrainLoopConfig
from src.world_model.runtime import amp_enabled, maybe_compile_model, resolve_amp_dtype


def test_resolve_amp_dtype_supports_bfloat16_and_float16():
    assert resolve_amp_dtype(WorldModelTrainLoopConfig(amp_dtype="bfloat16")) is torch.bfloat16
    assert resolve_amp_dtype(WorldModelTrainLoopConfig(amp_dtype="float16")) is torch.float16


def test_amp_enabled_only_on_cuda():
    cfg = WorldModelTrainLoopConfig(amp=True, amp_dtype="bfloat16")

    assert amp_enabled(cfg, torch.device("cpu")) is False
    assert amp_enabled(cfg, torch.device("cuda")) is True


def test_maybe_compile_model_is_noop_when_disabled():
    model = torch.nn.Linear(4, 4)

    compiled = maybe_compile_model(model, WorldModelTrainLoopConfig(compile_model=False))

    assert compiled is model
