from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.world_model.config import WorldModelModelConfig


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if not (heads == 1 and dim_head == dim)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(t.size(0), t.size(1), self.heads, -1).transpose(1, 2)
            for t in qkv
        )
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=False)
        x = x + self.mlp(self.norm2(x))
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
                Block(
                    dim=cfg.encoder_dim,
                    heads=cfg.num_heads,
                    dim_head=cfg.dim_head,
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

    def forward_tokens(self, obs: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(obs)
        patches = patches.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(obs.size(0), -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(obs)[:, 0]
