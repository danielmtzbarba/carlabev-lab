import os
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from torch import nn

from src.agents import build_agent
from src.envs import make_eval_env


def evaluate_ppo(args, model_path, num_episodes=20, render=False, device="cuda"):
    """
    Evaluate a trained PPO model and report statistics.

    Args:
        model_path: Path to the trained model .pt file
        num_episodes: Number of evaluation episodes
        render: Whether to render environment visually
        device: "cuda" or "cpu"
    """
    exp_name = args.exp_name 
    console = Console()

    # --- Setup environment ---
    console.print(f"[green]Initializing evaluation environment...[/green]")
    eval_env = make_eval_env(exp_name, render=render)

    # --- Load model ---
    console.print(f"[cyan]Loading model from[/cyan] [bold]{model_path}[/bold]")
    agent, _ = build_agent(args, eval_env, device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # --- Evaluation storage ---
    all_returns, all_lengths = [], []
    causes, success_count, collision_count, unfinished_count = [], 0, 0, 0

    for ep in track(range(num_episodes), description="Running evaluation..."):
        obs, _ = eval_env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        done, total_reward, steps = False, 0.0, 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)

            obs, reward, terminated, truncated, info = eval_env.step(action.cpu().numpy())
            done = terminated or truncated
            total_reward += reward
            steps += 1
            obs = torch.tensor(obs, dtype=torch.float32, device=device)

            if render:
                eval_env.render()

        # --- Extract termination cause ---
        cause = info.get("termination", {}).get("termination", "unknown")
        if isinstance(cause, (list, np.ndarray)):
            cause = cause[0]  # handle vectorized info
        causes.append(cause)

        if cause == "success":
            success_count += 1
        elif cause == "collision":
            collision_count += 1
        else:
            unfinished_count += 1

        all_returns.append(total_reward)
        all_lengths.append(steps)

    eval_env.close()

    # --- Compute aggregate statistics ---
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    mean_length = np.mean(all_lengths)
    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    unfinished_rate = unfinished_count / num_episodes

    # --- Rich summary ---
    table = Table(title=f"Evaluation Results ({num_episodes} episodes)", show_header=True, header_style="bold magenta")
    table.add_column("Metric", justify="left", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Mean Return", f"{mean_return:.2f} ± {std_return:.2f}")
    table.add_row("Mean Length", f"{mean_length:.1f} steps")
    table.add_row("Success Rate", f"{success_rate*100:.1f}%")
    table.add_row("Collision Rate", f"{collision_rate*100:.1f}%")
    table.add_row("Unfinished Rate", f"{unfinished_rate*100:.1f}%")
    console.print(table)

    # --- Save results ---
    results = {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "unfinished_rate": unfinished_rate,
        "returns": all_returns,
        "lengths": all_lengths,
        "causes": causes,
    }

    save_path = os.path.join("runs", exp_name, "eval_results.npy")
    np.save(save_path, results)
    console.print(f"[green]✅ Saved evaluation results to:[/green] {save_path}")

    return results
