import os
import torch
import numpy as np
from copy import deepcopy
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from random import choice
from src.agents import build_agent
from CarlaBEV.envs import make_env


def evaluate_ppo(
    cfg, model_path, num_episodes=1000, num_envs=14, render=False, device="cuda"
):
    """
    Evaluate a trained PPO model and report statistics.

    Uses (possibly) vectorized environments, following the same
    reset/options pattern as in train_ppo.
    """
    console = Console()

    # --- Copy config to avoid mutating original ---
    cfg_eval = deepcopy(cfg)
    exp_name = cfg_eval.exp_name

    # --- Setup environment (vectorized) ---
    # Assumes make_env returns a SyncVectorEnv-like object when eval=True
    eval_env = make_env(cfg_eval, eval=True)

    # --- Load model ---
    agent, _ = build_agent(cfg_eval, eval_env, device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # --- Evaluation storage across all episodes ---
    all_returns, all_lengths = [], []
    causes = []
    success_count = 0
    collision_count = 0
    unfinished_count = 0

    # Per-env episode trackers
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    # Initial reset for all envs
    options = {
        #   "scene": choice(["lead_brake", "jaywalk"]),
        "scene": "rdm",
        "num_vehicles": 25,
        "route_dist_range": [250, 500],
        "reset_mask": np.full((num_envs,), True, dtype=bool),
    }
    obs, info = eval_env.reset(seed=cfg_eval.seed, options=options)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    episodes_finished = 0

    # We don't know how many "iterations" we need in advance, so we loop until
    # we've collected `num_episodes` finished episodes.
    with Progress() as progress:
        task = progress.add_task("[green]Evaluating...", total=num_episodes)
        while episodes_finished < num_episodes:
            # --- Agent action ---
            with torch.no_grad():
                # obs_t shape: (num_envs, *obs_shape)
                action, _, _, _ = agent.get_action_and_value(obs_t)

            # Step vector env
            next_obs, reward, terminated, truncated, info = eval_env.step(
                action.cpu().numpy()
            )

            # Ensure numpy arrays
            reward = np.array(reward, dtype=np.float32)
            terminated = np.array(terminated, dtype=bool)
            truncated = np.array(truncated, dtype=bool)

            done = np.logical_or(terminated, truncated)

            # Accumulate rewards / lengths for all envs
            ep_returns += reward
            ep_lengths += 1

            if render:
                eval_env.render()

            # --- Handle finished episodes in each env ---
            if "episode_info" in info:
                ep_info = info["episode_info"]
            else:
                ep_info = None

            for i, d in enumerate(done):
                if not d:
                    continue

                if episodes_finished >= num_episodes:
                    # We've already collected enough episodes; ignore extras
                    continue

                # Extract termination cause if available
                cause_i = None
                if ep_info is not None and "termination" in ep_info:
                    # ep_info["termination"] is vectorized over envs
                    cause_i = ep_info["termination"][i]

                if cause_i is None:
                    cause_i = "unknown"

                causes.append(cause_i)

                if cause_i == "success":
                    success_count += 1
                elif cause_i == "collision":
                    collision_count += 1
                else:
                    unfinished_count += 1

                all_returns.append(float(ep_returns[i]))
                all_lengths.append(int(ep_lengths[i]))
                episodes_finished += 1
                progress.update(task, advance=1)

                # Reset per-env accumulators for next episode
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

            # --- If we still need more episodes, reset finished envs ---
            if np.any(done) and episodes_finished < num_episodes:
                reset_mask = done.copy()
                options["reset_mask"] = reset_mask
                next_obs, reset_info = eval_env.reset(
                    seed=cfg_eval.seed, options=options
                )

            # Update obs tensor for next step
            obs = next_obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    eval_env.close()

    # --- Compute aggregate statistics ---
    all_returns = np.array(all_returns, dtype=np.float32)
    all_lengths = np.array(all_lengths, dtype=np.float32)

    mean_return = float(all_returns.mean()) if len(all_returns) > 0 else 0.0
    std_return = float(all_returns.std()) if len(all_returns) > 0 else 0.0
    mean_length = float(all_lengths.mean()) if len(all_lengths) > 0 else 0.0

    success_rate = success_count / num_episodes
    collision_rate = collision_count / num_episodes
    unfinished_rate = unfinished_count / num_episodes

    # --- Rich summary table ---
    table = Table(
        title=f"Evaluation Results ({num_episodes} episodes)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", justify="left", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Mean Return", f"{mean_return:.2f} ± {std_return:.2f}")
    table.add_row("Mean Length", f"{mean_length:.1f} steps")
    table.add_row("Success Rate", f"{success_rate*100:.1f}%")
    table.add_row("Collision Rate", f"{collision_rate*100:.1f}%")
    table.add_row("Unfinished Rate", f"{unfinished_rate*100:.1f}%")

    # If you want to see it in console, uncomment:
    # console.print(table)

    # --- Save results ---
    results = {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "unfinished_rate": unfinished_rate,
        # still compatible with your previous code
    }

    save_path = os.path.join("runs", exp_name, "eval-results-1000.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, results)
    console.print(f"[green]✅ Saved evaluation results to:[/green] {save_path}")

    return results
