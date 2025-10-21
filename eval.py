import os
from src.eval.eval_ppo import evaluate_ppo
from src.exp import get_experiment

base_path = "/home/danielmtz/Data/results/runs"
exp_name = "cnn-ppo-discrete-carlabev-12env-lr3e-4"

if __name__ == "__main__":
    args, _ = get_experiment(exp_name)
    model_path = os.path.join(base_path, exp_name, "ppo_last.pt")

    results = evaluate_ppo(
        args=args,
        model_path=model_path,
        num_episodes=30,
        render=True,  # turn True for visualization
        device="cuda"
    )
