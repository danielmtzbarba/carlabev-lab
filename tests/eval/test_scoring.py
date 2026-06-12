import pytest

from src.eval.scoring import compute_comfort_score, compute_eval_score


@pytest.mark.unit
def test_comfort_score_decreases_with_harsher_motion():
    smooth = {
        "mean_abs_accel_long": 0.1,
        "mean_abs_accel_lat": 0.1,
        "mean_abs_jerk_long": 0.1,
        "mean_abs_jerk_lat": 0.1,
        "mean_abs_yaw_rate": 0.1,
        "mean_abs_yaw_acc": 0.1,
        "comfort_violation_rate": 0.0,
        "harsh_brake_rate": 0.0,
    }
    harsh = dict(smooth, mean_abs_jerk_long=3.0, harsh_brake_rate=0.25)

    assert compute_comfort_score(smooth) > compute_comfort_score(harsh)


@pytest.mark.unit
def test_eval_score_rewards_better_outcomes():
    weaker = {"success_rate": 0.4, "collision_rate": 0.4, "unfinished_rate": 0.2}
    stronger = {"success_rate": 0.8, "collision_rate": 0.1, "unfinished_rate": 0.1}

    assert compute_eval_score(stronger) > compute_eval_score(weaker)
