from __future__ import annotations

from dataclasses import dataclass
import importlib

import torch
from torch import nn

from src.world_model.config import WorldModelModelConfig
from src.world_model.encoders import ViTEncoder


@dataclass
class ViTOutput:
    last_hidden_state: torch.Tensor


class LeWMCompatibleViT(nn.Module):
    def __init__(self, *, obs_shape: tuple[int, int, int], cfg: WorldModelModelConfig) -> None:
        super().__init__()
        self.encoder = ViTEncoder(obs_shape=obs_shape, cfg=cfg)

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        interpolate_pos_encoding: bool = True,
    ) -> ViTOutput:
        del interpolate_pos_encoding
        token_sequence = self.encoder.forward_tokens(pixels)
        return ViTOutput(last_hidden_state=token_sequence)


class StablePretrainingViTHFAdapter(nn.Module):
    def __init__(self, *, obs_shape: tuple[int, int, int], cfg: WorldModelModelConfig) -> None:
        super().__init__()
        channels, height, width = obs_shape
        if height != width:
            raise ValueError(
                "stable_pretraining_vit_hf backend currently expects square observations, "
                f"got {(height, width)}."
            )
        try:
            backbone_utils = importlib.import_module("stable_pretraining.backbone.utils")
        except ImportError as exc:
            raise ImportError(
                "The 'stable_pretraining' package is required for encoder_backend="
                "'stable_pretraining_vit_hf'. Install the dependency first, or switch "
                "back to encoder_backend='lewm_compatible_vit'."
            ) from exc

        vit_hf = getattr(backbone_utils, "vit_hf", None)
        if vit_hf is None:
            raise ImportError(
                "stable_pretraining.backbone.utils.vit_hf is not available in the installed "
                "stable_pretraining package."
            )

        # LeWM's public path assumes RGB-like inputs. Our semantic BEV stacks often have
        # many channels, so we project them to three channels before feeding the HF-style ViT.
        self.channel_adapter = (
            nn.Identity()
            if channels == 3
            else nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0)
        )
        self.vit = vit_hf(
            size=cfg.encoder_size,
            patch_size=cfg.patch_size,
            image_size=height,
            pretrained=False,
            use_mask_token=False,
        )
        hidden_size = getattr(getattr(self.vit, "config", None), "hidden_size", None)
        self.token_projection = (
            nn.Identity()
            if hidden_size in (None, cfg.encoder_dim)
            else nn.Linear(hidden_size, cfg.encoder_dim)
        )

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        interpolate_pos_encoding: bool = True,
    ) -> ViTOutput:
        pixels = self.channel_adapter(pixels)
        output = self.vit(pixels, interpolate_pos_encoding=interpolate_pos_encoding)
        if not hasattr(output, "last_hidden_state"):
            raise ValueError(
                "stable_pretraining vit_hf backend did not return an object with "
                "'last_hidden_state'."
            )
        return ViTOutput(last_hidden_state=self.token_projection(output.last_hidden_state))


def build_vit_backend(*, obs_shape: tuple[int, int, int], cfg: WorldModelModelConfig) -> nn.Module:
    if cfg.encoder_backend == "lewm_compatible_vit":
        return LeWMCompatibleViT(obs_shape=obs_shape, cfg=cfg)
    if cfg.encoder_backend == "stable_pretraining_vit_hf":
        return StablePretrainingViTHFAdapter(obs_shape=obs_shape, cfg=cfg)
    raise ValueError(f"Unsupported encoder_backend={cfg.encoder_backend!r}")
