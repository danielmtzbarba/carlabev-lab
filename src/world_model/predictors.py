from __future__ import annotations

import torch
from torch import nn

from src.world_model.config import WorldModelModelConfig
from src.world_model.encoders import FeedForward


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class CausalAttention(nn.Module):
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
        seq_len = x.size(1)
        attn_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        return out


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = self.cond(cond).chunk(6, dim=-1)
        x = x + gate_attn * self.attn(_modulate(self.norm1(x), shift_attn, scale_attn))
        ff_input = _modulate(self.norm2(x), shift_ff, scale_ff)
        x = x + gate_ff * self.ff(ff_input)
        return x


class ActionEncoder(nn.Module):
    def __init__(self, *, num_actions: int, cfg: WorldModelModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_actions, cfg.action_embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(cfg.action_embed_dim, cfg.encoder_dim),
            nn.SiLU(),
            nn.Linear(cfg.encoder_dim, cfg.encoder_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        embedded = self.embedding(actions.long())
        return self.proj(embedded)


class ARLatentPredictor(nn.Module):
    def __init__(self, *, cfg: WorldModelModelConfig, max_frames: int) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_frames, cfg.encoder_dim))
        mlp_dim = int(cfg.encoder_dim * cfg.mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                ConditionalTransformerBlock(
                    dim=cfg.encoder_dim,
                    heads=cfg.num_heads,
                    mlp_dim=mlp_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.predictor_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.encoder_dim)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, latents: torch.Tensor, action_cond: torch.Tensor) -> torch.Tensor:
        x = latents + self.pos_embedding[:, : latents.size(1)]
        for block in self.blocks:
            x = block(x, action_cond)
        return self.norm(x)
