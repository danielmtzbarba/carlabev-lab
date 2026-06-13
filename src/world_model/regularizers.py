from __future__ import annotations

import torch
from torch import nn


class GaussianLatentRegularizer(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-(t.square()) / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        proj = torch.randn(latents.size(-1), self.num_proj, device=latents.device)
        proj = proj / proj.norm(p=2, dim=0, keepdim=True)
        projected = (latents @ proj).unsqueeze(-1) * self.t
        err = (projected.cos().mean(-3) - self.phi).square() + projected.sin().mean(-3).square()
        statistic = (err @ self.weights) * latents.size(-2)
        return statistic.mean()
