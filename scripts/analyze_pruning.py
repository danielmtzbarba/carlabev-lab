import optuna
import pandas as pd
import sqlite3

def check_median():
    study = optuna.load_study(study_name="carlabev", storage="sqlite:///results/carlabev_optuna.db")
    df = study.trials_dataframe()
    
    # Pruner thresholds are based on the historically best trials at a given step.
    # The MedianPruner looks at ALL previous trials at step N, and if the current trial
    # is worse than the median of those trials at step N, it prunes it.
    
    p1_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and str(t.user_attrs.get("phase")) == "1"]
    
    # Collect all intermediate values from Phase 1 at step 1
    step_0_vals = []
    step_1_vals = []
    step_2_vals = []
    
    for t in p1_trials:
        if 0 in t.intermediate_values: step_0_vals.append(t.intermediate_values[0])
        if 1 in t.intermediate_values: step_1_vals.append(t.intermediate_values[1])
        if 2 in t.intermediate_values: step_2_vals.append(t.intermediate_values[2])
        if 3 in t.intermediate_values: step_2_vals.append(t.intermediate_values[3])
        
    print(f"Phase 1 Medians (The cutoff hurdle Phase 2a faces):")
    print(f"Eval Step 0 (~100k steps): Median Score = {pd.Series(step_0_vals).median():.4f}")
    if step_1_vals: print(f"Eval Step 1 (~200k steps): Median Score = {pd.Series(step_1_vals).median():.4f}")
    if step_2_vals: print(f"Eval Step 2 (~300k steps): Median Score = {pd.Series(step_2_vals).median():.4f}")

    print("\nRecent Phase 2a Trials:")
    p2_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED and str(t.user_attrs.get("phase")) == "2a"]
    for t in p2_trials[-5:]:
        print(f"Trial {t.number}: Pruned at step {len(t.intermediate_values)-1}. Values: {t.intermediate_values}")

check_median()
