from __future__ import annotations

from pathlib import Path

from src.utils.common_logging import _format_value, _render_message, event_message


def test_event_message_formats_stage_phase_and_payload() -> None:
    message = event_message("train", "epoch_done", epoch="1/5", val_loss=0.353609)
    assert message == "EPOCH_DONE | epoch=1/5 val_loss=0.354"


def test_render_message_colors_level_stage_phase_and_payload() -> None:
    rendered = _render_message(
        {
            "level": type("Level", (), {"name": "INFO"})(),
            "time": type("Time", (), {"strftime": lambda self, _fmt: "13:41:48"})(),
            "message": event_message("TRAIN", "TRAIN_BATCH", batch="0050/2716", loss=0.308),
        },
        color=True,
    )

    assert "\033[94m[INFO]\033[0m" in rendered
    assert "\033[38;2;255;255;255m13:41:48\033[0m" in rendered
    assert "\033[93mTRAIN_BATCH\033[0m" in rendered
    assert "TRAIN - " not in rendered
    assert "\033[92mbatch\033[0m=\033[38;2;255;255;255m0050/2716\033[0m" in rendered
    assert "\033[92mloss\033[0m=\033[38;2;255;255;255m0.308\033[0m" in rendered


def test_render_message_plain_mode_keeps_readable_text() -> None:
    rendered = _render_message(
        {
            "level": type("Level", (), {"name": "INFO"})(),
            "time": type("Time", (), {"strftime": lambda self, _fmt: "13:41:48"})(),
            "message": event_message("TRAIN", "DONE", best_val_loss=0.353609, final_val_loss=0.353609),
        },
        color=False,
    )

    assert "\033[" not in rendered
    assert "[INFO]" in rendered
    assert "DONE | best_val_loss=0.354 final_val_loss=0.354" in rendered


def test_render_message_escapes_literal_braces_in_freeform_messages() -> None:
    rendered = _render_message(
        {
            "level": type("Level", (), {"name": "INFO"})(),
            "time": type("Time", (), {"strftime": lambda self, _fmt: "08:27:11"})(),
            "message": "Created ViT-small from scratch with config: {'hidden_size': 384, 'patch_size': 8}",
        },
        color=False,
    )

    assert "{{'hidden_size': 384, 'patch_size': 8}}" in rendered


def test_format_value_shortens_workspace_and_home_paths(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "runs" / "world_model" / "demo"
    target.mkdir(parents=True)
    monkeypatch.chdir(workspace)
    assert _format_value(target) == "demo"

    home_like = Path.home() / "demo" / "artifact.txt"
    assert _format_value(home_like) == "artifact.txt"
