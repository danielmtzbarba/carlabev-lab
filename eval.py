import os
import json
from src.eval.eval_ppo import evaluate_ppo
from src.config.experiment_loader import load_experiment
from src.utils.logger import DRLogger
from src.utils.run_paths import RunPaths

if __name__ == "__main__":
    cfg = load_experiment()
    latest_pointer = RunPaths(
        study_id=cfg.study_id,
        exp_id=cfg.exp_id,
        trial_number=None,
        seed=cfg.seed,
    ).latest_pointer_path
    if latest_pointer.exists():
        with open(latest_pointer, "r", encoding="utf-8") as handle:
            latest = json.load(handle)
        cfg.run_dir = latest["run_dir"]
    model_path = os.path.join(cfg.run_dir, "checkpoints", "ppo_final.pt")
    payload = evaluate_ppo(
        cfg=cfg,
        model_path=model_path,
        num_episodes=1000,
        num_envs=14,
        render=False,  # turn True for visualization
        device="cuda",
        file_name="final/manual_eval.npy",
    )
    logger = DRLogger(cfg)
    logger.log_evaluation(payload["aggregate"], 0)
