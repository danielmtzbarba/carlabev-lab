from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.world_model.config import WorldModelModelConfig
from src.world_model.encoders import ViTEncoder
from src.world_model.predictors import ARLatentPredictor, ActionEncoder
from src.world_model.regularizers import GaussianLatentRegularizer


class LeWorldModel(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: tuple[int, int, int],
        num_actions: int,
        context_frames: int,
        cfg: WorldModelModelConfig,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(obs_shape=obs_shape, cfg=cfg)
        self.action_encoder = ActionEncoder(num_actions=num_actions, cfg=cfg)
        self.predictor = ARLatentPredictor(cfg=cfg, max_frames=context_frames)
        self.projector = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.pred_proj = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.regularizer = GaussianLatentRegularizer()
        self.num_actions = num_actions
        self.obs_shape = obs_shape

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        batch, steps = obs.shape[:2]
        flat_obs = obs.reshape(batch * steps, *obs.shape[2:])
        enc = self.encoder(flat_obs)
        return enc.reshape(batch, steps, -1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        context_latents = self.encode_sequence(obs)
        target_latents = self.projector(self.encode_sequence(next_obs)).detach()
        action_cond = self.action_encoder(actions)
        pred_latents = self.predictor(context_latents, action_cond)
        pred_latents = self.pred_proj(pred_latents)
        reg_latents = self.projector(context_latents)
        return {
            "pred_latents": pred_latents,
            "target_latents": target_latents,
            "regularizer_latents": reg_latents,
        }

    def loss(self, batch: dict[str, torch.Tensor], *, sigreg_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
        outputs = self.forward(batch["obs"], batch["action"], batch["next_obs"])
        pred_loss = F.mse_loss(outputs["pred_latents"], outputs["target_latents"])
        reg_loss = self.regularizer(outputs["regularizer_latents"])
        total = pred_loss + sigreg_weight * reg_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "pred_loss": float(pred_loss.detach().cpu()),
            "reg_loss": float(reg_loss.detach().cpu()),
        }
        return total, metrics
