from __future__ import annotations

import json

import pytest

import src.carlabev_lab.cli.main as drl_cli


@pytest.mark.unit
def test_main_dispatches_train_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_train(argv):
        captured["argv"] = list(argv)
        return "trained"

    monkeypatch.setattr(drl_cli, "run_train_command", fake_train)

    result = drl_cli.main(["train", "exp", "--exp-id", "26"])

    assert result == "trained"
    assert captured["argv"] == ["exp", "--exp-id", "26"]


@pytest.mark.unit
def test_main_dispatches_tune_run_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_tune(argv):
        captured["argv"] = list(argv)
        return "tuned"

    monkeypatch.setattr(drl_cli, "run_tune_command", fake_tune)

    result = drl_cli.main(["tune", "run", "--study-id", "PPO_NAVIGATION"])

    assert result == "tuned"
    assert captured["argv"] == ["--study-id", "PPO_NAVIGATION"]


@pytest.mark.unit
def test_main_accepts_legacy_run_train_alias(monkeypatch):
    captured: dict[str, object] = {}

    def fake_train(argv):
        captured["argv"] = list(argv)
        return "trained"

    monkeypatch.setattr(drl_cli, "run_train_command", fake_train)

    result = drl_cli.main(["run", "train", "exp", "--exp-id", "26"])

    assert result == "trained"
    assert captured["argv"] == ["exp", "--exp-id", "26"]


@pytest.mark.unit
def test_main_dispatches_results_leaderboard_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_results(argv):
        captured["argv"] = list(argv)
        return "leaderboard"

    monkeypatch.setattr(drl_cli, "run_results_leaderboard_command", fake_results)

    result = drl_cli.main(["results", "leaderboard", "--study-id", "PPO_NAVIGATION"])

    assert result == "leaderboard"
    assert captured["argv"] == ["--study-id", "PPO_NAVIGATION"]


@pytest.mark.unit
def test_main_dispatches_seed_scene_diagnostics_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_diag(argv):
        captured["argv"] = list(argv)
        return "diagnostics"

    monkeypatch.setattr(drl_cli, "run_diagnostics_seed_scenes_command", fake_diag)

    result = drl_cli.main(
        ["diagnostics", "seed-scenes", "visualize", "--output-dir", "results/demo"]
    )

    assert result == "diagnostics"
    assert captured["argv"] == ["visualize", "--output-dir", "results/demo"]


@pytest.mark.unit
def test_main_dispatches_world_model_collect_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_collect(argv):
        captured["argv"] = list(argv)
        return "collected"

    monkeypatch.setattr(drl_cli, "run_world_model_collect_command", fake_collect)

    result = drl_cli.main(
        ["world-model", "collect", "exp", "--study-id", "PPO_NAVIGATION"]
    )

    assert result == "collected"
    assert captured["argv"] == ["exp", "--study-id", "PPO_NAVIGATION"]


@pytest.mark.unit
def test_main_dispatches_world_model_inspect_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_inspect(argv):
        captured["argv"] = list(argv)
        return "inspected"

    monkeypatch.setattr(drl_cli, "run_world_model_inspect_command", fake_inspect)

    result = drl_cli.main(["world-model", "inspect", "--path", "datasets/world_model/demo"])

    assert result == "inspected"
    assert captured["argv"] == ["--path", "datasets/world_model/demo"]


@pytest.mark.unit
def test_main_dispatches_world_model_summary_command(monkeypatch):
    captured: dict[str, object] = {}

    def fake_summary(argv):
        captured["argv"] = list(argv)
        return "summarized"

    monkeypatch.setattr(drl_cli, "run_world_model_summary_command", fake_summary)

    result = drl_cli.main(["world-model", "summary", "--path", "datasets/world_model/demo"])

    assert result == "summarized"
    assert captured["argv"] == ["--path", "datasets/world_model/demo"]


@pytest.mark.unit
def test_parse_experiment_args_applies_study_config(monkeypatch):
    monkeypatch.setattr(drl_cli, "validate_run_config", lambda _cfg: None)

    cfg = drl_cli._parse_experiment_args(
        ["exp", "--study-id", "PPO_NAVIGATION", "--exp-id", "30", "--seed", "7"]
    )

    assert cfg.study_id == "PPO_NAVIGATION"
    assert cfg.exp_id == 30
    assert cfg.seed == 7
    assert cfg.env.ego_anchor_y_frac == pytest.approx(0.75)


@pytest.mark.unit
def test_run_eval_command_uses_latest_run_pointer(monkeypatch, tmp_path):
    monkeypatch.setattr(drl_cli, "validate_run_config", lambda _cfg: None)

    latest_run_dir = (
        tmp_path / "runs" / "PPO_NAVIGATION" / "exp_26" / "trial_0" / "seed_7"
    )
    latest_run_dir.mkdir(parents=True)
    latest_pointer = tmp_path / "runs" / "PPO_NAVIGATION" / "exp_26" / "LATEST_RUN.json"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(
        json.dumps({"run_dir": str(latest_run_dir)}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs):
        captured["model_path"] = kwargs["model_path"]
        captured["run_dir"] = kwargs["cfg"].run_dir
        return {"aggregate": {"normalized_score": 0.5}}

    class FakeLogger:
        def __init__(self, cfg):
            captured["logger_run_dir"] = cfg.run_dir

        def log_evaluation(self, payload, step):
            captured["logged"] = (payload, step)

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(drl_cli, "evaluate_ppo", fake_evaluate)
    monkeypatch.setattr(drl_cli, "DRLogger", FakeLogger)

    payload = drl_cli.run_eval_command(
        ["exp", "--study-id", "PPO_NAVIGATION", "--exp-id", "26", "--seed", "7"]
    )

    assert payload["aggregate"]["normalized_score"] == pytest.approx(0.5)
    assert captured["run_dir"] == str(latest_run_dir)
    assert captured["model_path"] == str(
        latest_run_dir / "checkpoints" / "ppo_final.pt"
    )
    assert captured["logged"] == ({"normalized_score": 0.5}, 0)
    assert captured["closed"] is True


@pytest.mark.unit
def test_main_rejects_unknown_command():
    with pytest.raises(SystemExit, match="Unknown command"):
        drl_cli.main(["unknown"])
