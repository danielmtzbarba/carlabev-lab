import os
from copy import deepcopy

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from CarlaBEV.envs import make_env
from src.agents import build_agent
from src.config.base_config import to_carlabev_run_config
from src.config.reset_protocol import build_eval_protocol_samplers
from src.utils.storage_paths import resolve_artifact_path
from src.eval.scoring import compute_comfort_score, compute_eval_score


def _initial_reset_seeds(sampler, num_envs: int):
    if hasattr(sampler, "initial_reset_seeds"):
        return sampler.initial_reset_seeds(num_envs)
    return None


def _next_reset_seeds(sampler, reset_mask):
    if hasattr(sampler, "next_reset_seeds"):
        return sampler.next_reset_seeds(reset_mask)
    return None


def _evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    positions = np.linspace(0, total - 1, num=count)
    return sorted({int(round(value)) for value in positions})


def _capture_protocol_videos(cfg, agent, protocol_id, num_episodes, output_dir, video_count, video_name_prefix, device):
    if video_count <= 0:
        return

    cfg_capture = deepcopy(cfg)
    cfg_capture.num_envs = 1
    cfg_capture.capture_video = True
    cfg_capture.video_output_dir = str(output_dir)
    cfg_capture.video_episode_indices = _evenly_spaced_indices(num_episodes, video_count)
    cfg_capture.video_name_prefix = video_name_prefix

    capture_env = make_env(to_carlabev_run_config(cfg_capture), eval=True)
    sampler = build_eval_protocol_samplers(cfg_capture, [protocol_id])[protocol_id]

    options = sampler.initial_options(1)
    reset_seeds = _initial_reset_seeds(sampler, 1)
    obs, _ = capture_env.reset(seed=reset_seeds, options=options)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    episodes_finished = 0

    while episodes_finished < num_episodes:
        with torch.no_grad():
            out = agent.get_action_and_value(obs_t)
            if agent.is_continuous:
                _, action, _, _, _ = out
            else:
                action, _, _, _ = out

        next_obs, _, terminated, truncated, _ = capture_env.step(action.cpu().numpy())
        done = bool(np.logical_or(np.array(terminated, dtype=bool), np.array(truncated, dtype=bool))[0])

        if done:
            episodes_finished += 1
            if episodes_finished < num_episodes:
                next_obs, _ = capture_env.reset(
                    seed=_next_reset_seeds(sampler, np.array([True], dtype=bool)),
                    options=sampler.next_options(reset_mask=np.array([True], dtype=bool)),
                )

        obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

    capture_env.close()


def _run_eval_protocol(eval_env, agent, cfg_eval, protocol_id, sampler, num_episodes, render, device):
    all_returns, all_lengths = [], []
    success_count = 0
    collision_count = 0
    unfinished_count = 0
    route_direction_metrics = {
        "straight_fraction": [],
        "left_turn_fraction": [],
        "right_turn_fraction": [],
    }
    comfort_metrics = {
        "mean_abs_accel_long": [],
        "mean_abs_accel_lat": [],
        "mean_abs_jerk_long": [],
        "mean_abs_jerk_lat": [],
        "mean_abs_yaw_rate": [],
        "mean_abs_yaw_acc": [],
        "comfort_violation_rate": [],
        "harsh_brake_rate": [],
    }

    num_envs = cfg_eval.num_envs
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    options = sampler.initial_options(num_envs)
    reset_seeds = _initial_reset_seeds(sampler, num_envs)
    obs, _ = eval_env.reset(seed=reset_seeds, options=options)
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
                if ep_info is not None:
                    for key in route_direction_metrics:
                        route_direction_metrics[key].append(float(ep_info.get(key, [0.0] * num_envs)[i]))
                    for key in comfort_metrics:
                        comfort_metrics[key].append(float(ep_info.get(key, [0.0] * num_envs)[i]))
                episodes_finished += 1
                progress.update(task, advance=1)

                ep_returns[i] = 0.0
                ep_lengths[i] = 0

            if np.any(done) and episodes_finished < num_episodes:
                next_obs, _ = eval_env.reset(
                    seed=_next_reset_seeds(sampler, done.copy()),
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
    for key, values in route_direction_metrics.items():
        results[key] = float(np.mean(values)) if values else 0.0
    for key, values in comfort_metrics.items():
        results[key] = float(np.mean(values)) if values else 0.0
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
            "straight_fraction": 0.0,
            "left_turn_fraction": 0.0,
            "right_turn_fraction": 0.0,
            "mean_abs_accel_long": 0.0,
            "mean_abs_accel_lat": 0.0,
            "mean_abs_jerk_long": 0.0,
            "mean_abs_jerk_lat": 0.0,
            "mean_abs_yaw_rate": 0.0,
            "mean_abs_yaw_acc": 0.0,
            "comfort_violation_rate": 0.0,
            "harsh_brake_rate": 0.0,
        }

    weighted = {}
    for key in (
        "mean_return",
        "mean_length",
        "success_rate",
        "collision_rate",
        "unfinished_rate",
        "straight_fraction",
        "left_turn_fraction",
        "right_turn_fraction",
        "mean_abs_accel_long",
        "mean_abs_accel_lat",
        "mean_abs_jerk_long",
        "mean_abs_jerk_lat",
        "mean_abs_yaw_rate",
        "mean_abs_yaw_acc",
        "comfort_violation_rate",
        "harsh_brake_rate",
    ):
        weighted[key] = sum(
            result.get(key, 0.0) * result["episodes"]
            for result in protocol_results.values()
        ) / total_episodes

    weighted["std_return"] = sum(
        result.get("std_return", 0.0) * result["episodes"]
        for result in protocol_results.values()
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
    capture_video_count: int = 0,
    video_output_dir: str | None = None,
    video_name_prefix: str = "eval",
    save_payload: bool = True,
):
    """
    Evaluate a trained PPO model against the experiment's configured eval protocols.
    """
    console = Console()

    cfg_eval = deepcopy(cfg)
    run_dir = str(
        resolve_artifact_path(
            getattr(cfg_eval, "run_dir", os.path.join("runs", cfg_eval.exp_name))
        )
    )
    cfg_eval.num_envs = num_envs
    eval_env = make_env(to_carlabev_run_config(cfg_eval), eval=True)

    ppo_artifacts = build_agent(cfg_eval, eval_env, device)
    agent = ppo_artifacts.agent
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
    aggregate["comfort_score"] = compute_comfort_score(aggregate)
    aggregate["normalized_score"] = compute_eval_score(aggregate)

    if capture_video_count > 0 and video_output_dir is not None and protocol_results:
        first_protocol_id = next(iter(protocol_results.keys()))
        _capture_protocol_videos(
            cfg=cfg,
            agent=agent,
            protocol_id=first_protocol_id,
            num_episodes=num_episodes,
            output_dir=video_output_dir,
            video_count=capture_video_count,
            video_name_prefix=video_name_prefix,
            device=device,
        )

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
    table.add_row("Straight Fraction", f"{aggregate['straight_fraction']*100:.1f}%")
    table.add_row("Left-Turn Fraction", f"{aggregate['left_turn_fraction']*100:.1f}%")
    table.add_row("Right-Turn Fraction", f"{aggregate['right_turn_fraction']*100:.1f}%")
    table.add_row("Comfort Violation Rate", f"{aggregate['comfort_violation_rate']*100:.1f}%")
    table.add_row("Harsh Brake Rate", f"{aggregate['harsh_brake_rate']*100:.1f}%")
    table.add_row("Mean |Jerk Long|", f"{aggregate['mean_abs_jerk_long']:.3f}")
    table.add_row("Mean |Jerk Lat|", f"{aggregate['mean_abs_jerk_lat']:.3f}")
    table.add_row("Mean |Yaw Rate|", f"{aggregate['mean_abs_yaw_rate']:.3f}")
    table.add_row("Comfort Score", f"{aggregate['comfort_score']:.3f}")
    table.add_row("Normalized Score", f"{aggregate['normalized_score']:.3f}")
    table.add_row("Protocols", ", ".join(aggregate.get("evaluated_protocol_ids", [])))
    console.print(table)

    payload = {
        "aggregate": aggregate,
        "protocols": protocol_results,
    }

    if save_payload:
        save_path = os.path.join(run_dir, "eval", file_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, payload, allow_pickle=True)
        console.print(f"[green]✅ Saved evaluation results to:[/green] {save_path}")

    return payload
