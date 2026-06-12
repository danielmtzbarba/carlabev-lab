import pytest

from src.trainers import build_trainer
from src.trainers.ppo import train_ppo


@pytest.mark.unit
def test_build_trainer_returns_ppo_trainer():
    assert build_trainer("cnn-ppo") is train_ppo


@pytest.mark.unit
def test_build_trainer_rejects_removed_algorithms():
    with pytest.raises(ValueError, match="Unsupported trainer algorithm"):
        build_trainer("sac")
