import os
from src.eval.eval_ppo import evaluate_ppo
from src.config.experiment_loader import load_experiment
from src.utils.logger import DRLogger

base_path = "/home/danielmtz/Data/results/carlabev/runs_final"
base_path = "runs"

if __name__ == "__main__":
    cfg = load_experiment()
    model_path = os.path.join(base_path, cfg.exp_name, "ppo_final.pt")
    results = evaluate_ppo(
        cfg=cfg,
        model_path=model_path,
        num_episodes=1000,
        num_envs=14,
        render=False,  # turn True for visualization
        device="cuda",
    )
    logger = DRLogger(cfg)
    logger.log_evaluation(results, 0)
