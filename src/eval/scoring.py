import numpy as np


def _bounded_inverse(value, limit):
    if limit <= 0:
        return 1.0
    return float(np.clip(1.0 - (float(value) / float(limit)), 0.0, 1.0))


def compute_comfort_score(eval_results):
    component_scores = [
        _bounded_inverse(eval_results.get("mean_abs_accel_long", 0.0), 2.0),
        _bounded_inverse(eval_results.get("mean_abs_accel_lat", 0.0), 2.0),
        _bounded_inverse(eval_results.get("mean_abs_jerk_long", 0.0), 3.0),
        _bounded_inverse(eval_results.get("mean_abs_jerk_lat", 0.0), 3.0),
        _bounded_inverse(eval_results.get("mean_abs_yaw_rate", 0.0), 20.0),
        _bounded_inverse(eval_results.get("mean_abs_yaw_acc", 0.0), 120.0),
        _bounded_inverse(eval_results.get("comfort_violation_rate", 0.0), 1.0),
        _bounded_inverse(eval_results.get("harsh_brake_rate", 0.0), 0.25),
    ]
    return float(np.mean(component_scores))


def compute_eval_score(eval_results):
    success = float(eval_results.get("success_rate", 0.0))
    collision = float(eval_results.get("collision_rate", 0.0))
    unfinished = float(eval_results.get("unfinished_rate", 0.0))
    comfort = compute_comfort_score(eval_results)

    score = 100.0 * (
        0.45 * success
        + 0.30 * (1.0 - collision)
        + 0.10 * (1.0 - unfinished)
        + 0.15 * comfort
    )
    if collision > 0.6:
        score *= 0.5
    elif collision > 0.4:
        score *= 0.75
    return float(np.clip(score, 0.0, 100.0))
