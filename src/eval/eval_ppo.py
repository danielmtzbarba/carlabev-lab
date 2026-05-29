import os
from copy import deepcopy

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from CarlaBEV.envs import make_env
from src.agents import build_agent
from src.config.reset_protocol import build_eval_protocol_samplers


def _run_eval_protocol(eval_env, agent, cfg_eval, protocol_id, sampler, num_episodes, render, device):
    all_returns, all_lengths = [], []
    success_count = 0
    collision_count = 0
    unfinished_count = 0

    num_envs = cfg_eval.num_envs
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    options = sampler.initial_options(num_envs)
    obs, _ = eval_env.reset(seed=cfg_eval.seed, options=options)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    episodes_finished = 0

    with Progress() as progress:
        task = progress.add_task(f"[green]Evaluating {protocol_id}...", total=num_episodes)
        while episodes_finished < num_episodes:
            with torch.no_grad():
                out = agent.get_action_and_value(obs_t)
                if agent.is_continuous:
                    _, action, _, _, _ = out
                else:
                    action, _, _, _ = out

            next_obs, reward, terminated, truncated, info = eval_env.step(
                action.cpu().numpy()
            )

            reward = np.array(reward, dtype=np.float32)
            terminated = np.array(terminated, dtype=bool)
            truncated = np.array(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            ep_returns += reward
            ep_lengths += 1

            if render:
                eval_env.render()

            ep_info = info["episode_info"] if "episode_info" in info else None

            for i, d in enumerate(done):
                if not d or episodes_finished >= num_episodes:
                    continue

                cause_i = None
                if ep_info is not None and "termination" in ep_info:
                    cause_i = ep_info["termination"][i]
                if cause_i is None:
                    cause_i = "unknown"

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

                ep_returns[i] = 0.0
                ep_lengths[i] = 0

            if np.any(done) and episodes_finished < num_episodes:
                next_obs, _ = eval_env.reset(
                    seed=cfg_eval.seed,
                    options=sampler.next_options(reset_mask=done.copy()),
                )

            obs = next_obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

    all_returns = np.array(all_returns, dtype=np.float32)
    all_lengths = np.array(all_lengths, dtype=np.float32)

    results = {
        "protocol_id": protocol_id,
        "episodes": num_episodes,
        "mean_return": float(all_returns.mean()) if len(all_returns) > 0 else 0.0,
        "std_return": float(all_returns.std()) if len(all_returns) > 0 else 0.0,
        "mean_length": float(all_lengths.mean()) if len(all_lengths) > 0 else 0.0,
        "success_rate": success_count / num_episodes,
        "collision_rate": collision_count / num_episodes,
        "unfinished_rate": unfinished_count / num_episodes,
    }
    return results


def _aggregate_protocol_results(protocol_results):
    total_episodes = sum(item.get("episodes", 0) for item in protocol_results.values())
    if total_episodes == 0:
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "mean_length": 0.0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "unfinished_rate": 0.0,
        }

    weighted = {}
    for key in ("mean_return", "mean_length", "success_rate", "collision_rate", "unfinished_rate"):
        weighted[key] = sum(
            result[key] * result["episodes"] for result in protocol_results.values()
        ) / total_episodes

    weighted["std_return"] = sum(
        result["std_return"] * result["episodes"] for result in protocol_results.values()
    ) / total_episodes
    weighted["evaluated_protocol_ids"] = list(protocol_results.keys())
    return weighted


def evaluate_ppo(
    cfg,
    model_path,
    num_episodes=1000,
    num_envs=14,
    render=False,
    device="cuda",
    file_name="ppo-eval.npy",
    eval_protocol_ids=None,
):
    """
    Evaluate a trained PPO model against the experiment's configured eval protocols.
    """
    console = Console()

    cfg_eval = deepcopy(cfg)
    exp_name = cfg_eval.exp_name
    cfg_eval.num_envs = num_envs
    eval_env = make_env(cfg_eval, eval=True)

    agent, _ = build_agent(cfg_eval, eval_env, device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    samplers = build_eval_protocol_samplers(cfg_eval, eval_protocol_ids)
    protocol_results = {}

    for protocol_id, sampler in samplers.items():
        protocol_results[protocol_id] = _run_eval_protocol(
            eval_env=eval_env,
            agent=agent,
            cfg_eval=cfg_eval,
            protocol_id=protocol_id,
            sampler=sampler,
            num_episodes=num_episodes,
            render=render,
            device=device,
        )

    eval_env.close()

    aggregate = _aggregate_protocol_results(protocol_results)

    table = Table(
        title=f"Evaluation Results ({num_episodes} episodes per protocol)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", justify="left", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Mean Return", f"{aggregate['mean_return']:.2f} ± {aggregate['std_return']:.2f}")
    table.add_row("Mean Length", f"{aggregate['mean_length']:.1f} steps")
    table.add_row("Success Rate", f"{aggregate['success_rate']*100:.1f}%")
    table.add_row("Collision Rate", f"{aggregate['collision_rate']*100:.1f}%")
    table.add_row("Unfinished Rate", f"{aggregate['unfinished_rate']*100:.1f}%")
    table.add_row("Protocols", ", ".join(aggregate.get("evaluated_protocol_ids", [])))
    console.print(table)

    payload = {
        "aggregate": aggregate,
        "protocols": protocol_results,
    }

    save_path = os.path.join("runs", exp_name, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, payload, allow_pickle=True)
    console.print(f"[green]✅ Saved evaluation results to:[/green] {save_path}")

    return payload
