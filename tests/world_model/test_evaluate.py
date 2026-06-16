from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.world_model.config import (
    WorldModelConfig,
    WorldModelDataConfig,
    WorldModelModelConfig,
    WorldModelOptimizerConfig,
    WorldModelTrainLoopConfig,
)
from src.world_model.evaluate import WorldModelCheckpointEvalConfig, evaluate_world_model_checkpoint
from src.world_model.train import train_world_model
from tests.world_model.test_train import _collect_train_dataset


@pytest.mark.integration
def test_evaluate_world_model_checkpoint_smoke(monkeypatch, tiny_cfg, tmp_workdir):
    output_dir = _collect_train_dataset(monkeypatch, tiny_cfg, tmp_workdir, total_transitions=8)

    train_result = train_world_model(
        WorldModelConfig(
            run_name="wm-eval-smoke",
            data=WorldModelDataConfig(
                dataset_paths=[str(output_dir)],
                batch_size=2,
                num_workers=0,
                chunk_length=2,
                stride=1,
                val_ratio=0.25,
                expected_num_actions=5,
            ),
            model=WorldModelModelConfig(
                patch_size=4,
                encoder_dim=16,
                encoder_depth=1,
                predictor_depth=1,
                num_heads=4,
                mlp_ratio=2.0,
                action_embed_dim=8,
                latent_dim=16,
                dropout=0.0,
            ),
            optimizer=WorldModelOptimizerConfig(
                learning_rate=1e-3,
                weight_decay=0.0,
                max_grad_norm=1.0,
            ),
            training=WorldModelTrainLoopConfig(
                epochs=1,
                device="cpu",
                save_every=1,
                sigreg_weight=0.01,
            ),
        ),
        show_progress=False,
    )

    result = evaluate_world_model_checkpoint(
        WorldModelCheckpointEvalConfig(
            checkpoint_path=train_result.best_checkpoint_path,
            dataset_paths=[str(output_dir)],
            device="cpu",
            batch_size=2,
            include_train_split=True,
        )
    )

    assert result.val_metrics.loss >= 0.0
    assert result.val_metrics.pred_loss >= 0.0
    assert result.val_metrics.reg_loss >= 0.0
    assert result.train_metrics is not None
    assert result.train_metrics.loss >= 0.0
    assert Path(result.output_path).exists()
    payload = json.loads(Path(result.output_path).read_text(encoding="utf-8"))
    assert payload["checkpoint_path"].endswith("world_model_best.pt")
