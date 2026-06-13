from __future__ import annotations

import torch
from torch import nn

from src.world_model.config import WorldModelModelConfig


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(self.norm(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, *, obs_shape: tuple[int, int, int], cfg: WorldModelModelConfig) -> None:
        super().__init__()
        channels, height, width = obs_shape
        patch_size = cfg.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Observation size {(height, width)} must be divisible by patch_size={patch_size}"
            )
        num_patches = (height // patch_size) * (width // patch_size)
        self.patch_embed = nn.Conv2d(
            channels,
            cfg.encoder_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.encoder_dim))
        mlp_dim = int(cfg.encoder_dim * cfg.mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=cfg.encoder_dim,
                    heads=cfg.num_heads,
                    mlp_dim=mlp_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.encoder_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(obs)
        patches = patches.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(obs.size(0), -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]
