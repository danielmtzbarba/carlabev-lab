import pytest
import torch

from src.agents.cnn_ppo import CNNBackbone


@pytest.mark.unit
def test_backbone_output_shape_matches_fc_size():
    backbone = CNNBackbone(in_channels=3, channels=[8, 16, 16], fc_size=64)
    batch = torch.zeros((2, 3, 96, 96), dtype=torch.float32)

    output = backbone(batch)

    assert output.shape == (2, 64)
