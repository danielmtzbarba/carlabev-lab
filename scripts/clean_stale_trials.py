import optuna
import tyro

from src.config.experiment_loader import get_study_db_path, get_study_name


def main(study_id: str = "PPO_NAVIGATION"):
    study = optuna.load_study(
        study_name=get_study_name(study_id),
        storage=f"sqlite:///{get_study_db_path(study_id)}",
    )

    running_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
    ]

    print(f"Found {len(running_trials)} stuck RUNNING trials.")
    for t in running_trials:
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        print(f"Marked Trial {t.number} as FAILED.")

    print("Cleanup complete.")


if __name__ == "__main__":
    tyro.cli(main)
