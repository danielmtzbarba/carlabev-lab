import optuna
import pandas as pd


def check_median():
    study = optuna.load_study(
        study_name="carlabev",
        storage="sqlite:///results/carlabev_optuna.db",
    )

    # Pruner thresholds are based on the historically best trials at a given step.
    # The MedianPruner looks at ALL previous trials at step N, and if the current trial
    # is worse than the median of those trials at step N, it prunes it.
    p1_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and str(t.user_attrs.get("phase")) == "1"
    ]

    step_0_vals = []
    step_1_vals = []
    step_2_vals = []

    for trial in p1_trials:
        if 0 in trial.intermediate_values:
            step_0_vals.append(trial.intermediate_values[0])
        if 1 in trial.intermediate_values:
            step_1_vals.append(trial.intermediate_values[1])
        if 2 in trial.intermediate_values:
            step_2_vals.append(trial.intermediate_values[2])
        if 3 in trial.intermediate_values:
            step_2_vals.append(trial.intermediate_values[3])

    print("Phase 1 Medians (The cutoff hurdle Phase 2a faces):")
    print(
        f"Eval Step 0 (~100k steps): Median Score = {pd.Series(step_0_vals).median():.4f}"
    )
    if step_1_vals:
        print(
            f"Eval Step 1 (~200k steps): Median Score = {pd.Series(step_1_vals).median():.4f}"
        )
    if step_2_vals:
        print(
            f"Eval Step 2 (~300k steps): Median Score = {pd.Series(step_2_vals).median():.4f}"
        )

    print("\nRecent Phase 2a Trials:")
    p2_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED
        and str(t.user_attrs.get("phase")) == "2a"
    ]
    for trial in p2_trials[-5:]:
        print(
            f"Trial {trial.number}: Pruned at step {len(trial.intermediate_values) - 1}. "
            f"Values: {trial.intermediate_values}"
        )


check_median()
