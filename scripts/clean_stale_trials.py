import optuna
import tyro

def main(exp_id: int = 26):
    study_name = f"carlabev_optuna_{exp_id}"
    storage_name = f"sqlite:///results/{study_name}.db"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except Exception:
        study = optuna.load_study(study_name="carlabev", storage="sqlite:///results/carlabev_optuna.db")

    running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    
    print(f"Found {len(running_trials)} stuck RUNNING trials.")
    for t in running_trials:
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        print(f"Marked Trial {t.number} as FAILED.")
        
    print("Cleanup complete.")

if __name__ == "__main__":
    tyro.cli(main)
