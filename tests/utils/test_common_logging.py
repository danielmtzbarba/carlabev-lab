from __future__ import annotations

import logging

from src.utils.common_logging import _ColorFormatter, event_message


def test_event_message_formats_stage_phase_and_payload() -> None:
    message = event_message("train", "epoch_done", epoch="1/5", val_loss=0.353609)
    assert message == "TRAIN - EPOCH_DONE | epoch=1/5 val_loss=0.354"


def test_color_formatter_colors_level_stage_phase_and_payload() -> None:
    formatter = _ColorFormatter("[%(levelname)s] %(asctime)s | %(message)s", datefmt="%H:%M:%S", use_color=True)
    record = logging.LogRecord(
        name="carlabev_lab.world_model.train",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg=event_message("TRAIN", "TRAIN_BATCH", batch="0050/2716", loss=0.308),
        args=(),
        exc_info=None,
    )
    record.created = 0.0

    rendered = formatter.format(record)

    assert "\033[34m[INFO]\033[0m" in rendered
    assert "\033[36mTRAIN\033[0m" in rendered
    assert "\033[35mTRAIN_BATCH\033[0m" in rendered
    assert "\033[94mbatch\033[0m=\033[92m0050/2716\033[0m" in rendered
    assert "\033[94mloss\033[0m=\033[92m0.308\033[0m" in rendered


def test_color_formatter_plain_mode_keeps_readable_text() -> None:
    formatter = _ColorFormatter("[%(levelname)s] %(asctime)s | %(message)s", datefmt="%H:%M:%S", use_color=False)
    record = logging.LogRecord(
        name="carlabev_lab.world_model.train",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg=event_message("TRAIN", "DONE", best_val_loss=0.353609, final_val_loss=0.353609),
        args=(),
        exc_info=None,
    )
    record.created = 0.0

    rendered = formatter.format(record)

    assert "\033[" not in rendered
    assert "[INFO]" in rendered
    assert "TRAIN - DONE | best_val_loss=0.354 final_val_loss=0.354" in rendered
