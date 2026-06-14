from __future__ import annotations

import torch
from torch import nn

from src.world_model.config import WorldModelModelConfig
from src.world_model.encoders import Attention, Block, FeedForward


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
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
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.ada_ln_modulation[-1].weight, 0)
        nn.init.constant_(self.ada_ln_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal=True)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class=Block,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        self.layers = nn.ModuleList(
            [
                block_class(
                    hidden_dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            if isinstance(block, Block):
                x = block(x)
            else:
                x = block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        smoothed_dim: int,
        emb_dim: int,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ARPredictor(nn.Module):
    def __init__(self, *, cfg: WorldModelModelConfig, num_frames: int) -> None:
        super().__init__()
        mlp_dim = int(cfg.encoder_dim * cfg.mlp_ratio)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, cfg.encoder_dim))
        self.dropout = nn.Dropout(cfg.emb_dropout)
        self.transformer = Transformer(
            input_dim=cfg.encoder_dim,
            hidden_dim=cfg.encoder_dim,
            output_dim=cfg.encoder_dim,
            depth=cfg.predictor_depth,
            heads=cfg.num_heads,
            dim_head=cfg.dim_head,
            mlp_dim=mlp_dim,
            dropout=cfg.dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        time_steps = x.size(1)
        x = x + self.pos_embedding[:, :time_steps]
        x = self.dropout(x)
        return self.transformer(x, c)
