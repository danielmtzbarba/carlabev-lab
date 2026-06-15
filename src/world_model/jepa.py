from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.world_model.backbones import build_vit_backend
from src.world_model.config import WorldModelModelConfig
from src.world_model.predictors import ARPredictor, Embedder, MLP
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
        self.encoder = build_vit_backend(obs_shape=obs_shape, cfg=cfg)
        self.action_encoder = Embedder(
            input_dim=num_actions,
            smoothed_dim=num_actions,
            emb_dim=cfg.encoder_dim,
            mlp_scale=cfg.action_mlp_scale,
        )
        self.predictor = ARPredictor(cfg=cfg, num_frames=context_frames)
        self.projector = MLP(
            input_dim=cfg.encoder_dim,
            hidden_dim=cfg.latent_dim,
            output_dim=cfg.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.pred_proj = MLP(
            input_dim=cfg.encoder_dim,
            hidden_dim=cfg.latent_dim,
            output_dim=cfg.latent_dim,
            norm_fn=nn.BatchNorm1d,
        )
        self.regularizer = GaussianLatentRegularizer()
        self.num_actions = num_actions
        self.obs_shape = obs_shape

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(actions.long(), num_classes=self.num_actions).float()
        return self.action_encoder(one_hot)

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        batch, steps = obs.shape[:2]
        flat_obs = obs.reshape(batch * steps, *obs.shape[2:])
        output = self.encoder(flat_obs, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        return pixels_emb.reshape(batch, steps, -1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        context_latents = self.encode_sequence(obs)
        with torch.no_grad():
            next_latents = self.encode_sequence(next_obs)
            flat_next_latents = next_latents.reshape(-1, next_latents.size(-1))
            target_latents = self.projector(flat_next_latents).reshape(
                next_latents.size(0),
                next_latents.size(1),
                -1,
            )
        action_cond = self._encode_actions(actions)
        pred_latents = self.predictor(context_latents, action_cond)
        flat_pred_latents = pred_latents.reshape(-1, pred_latents.size(-1))
        pred_latents = self.pred_proj(flat_pred_latents).reshape(pred_latents.size(0), pred_latents.size(1), -1)
        flat_context_latents = context_latents.reshape(-1, context_latents.size(-1))
        reg_latents = self.projector(flat_context_latents).reshape(context_latents.size(0), context_latents.size(1), -1)
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
