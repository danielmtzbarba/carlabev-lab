import os
import tyro
import optuna
from datetime import datetime
import time
import random

from src.config.base_config import ArgsCarlaBEV
from src.config.experiment_loader import (
    apply_experiment_config,
    get_study_db_path,
    get_study_name,
)
from src.config.studies.registry import get_study_config
from src.tuning.optuna_utils import OptunaArgs
from src.tuning.engine import (
    completed_stage_trials,
    inherited_stage_overrides,
    normalize_stage_name,
    resolved_stage_config,
    run_stage_objective,
    stage_title,
    trial_stage_name,
)


def main():
    cli_args = tyro.cli(OptunaArgs)
    stage_name = normalize_stage_name(cli_args.stage)
    study_config = get_study_config(cli_args.study_id)
    if study_config.tuning is None:
        raise ValueError(f"Study {cli_args.study_id!r} does not declare a tuning configuration.")
    tuning_config, stage_cfg = resolved_stage_config(study_config.tuning, stage_name, cli_args)

    # Create base ArgsCarlaBEV to get environment configuration
    base_args = ArgsCarlaBEV(study_id=cli_args.study_id, exp_id=cli_args.exp_id)
    base_args = apply_experiment_config(base_args, cli_args.exp_id, study_id=cli_args.study_id)
    print(
        f"⚙️ Running Optuna Hyperparameter tuning for "
        f"Study = {cli_args.study_id}, Base Experiment ID = {cli_args.exp_id}, "
        f"Stage = {stage_name}"
    )
    
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
        if tuning_config.pruner == "median"
        else optuna.pruners.NopPruner()
    )
    # Implement SQLite storage with concurrency support
    os.makedirs("results", exist_ok=True)
    db_path = get_study_db_path(cli_args.study_id)
    storage_name = f"sqlite:///{db_path}"
    
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"connect_args": {"timeout": 60}}
    )

    sampler = optuna.samplers.TPESampler(constant_liar=True)
    
    study = None
    
    # Attempt to create or load the study robustly to avoid race conditions when many Slurm nodes start simultaneously
    for _ in range(20):
        try:
            study = optuna.create_study(
                storage=storage,
                load_if_exists=True,
                direction="maximize", 
                pruner=pruner,
                sampler=sampler,
                study_name=get_study_name(cli_args.study_id)
            )
            break
        except Exception as e:
            print(f"Study creation collision or lock detected: {e}. Retrying in a few seconds...")
            time.sleep(random.uniform(2, 6))
            
    if study is None:
        raise RuntimeError("Failed to create or load the Optuna study after multiple attempts due to database locking.")

    try:
        print(
            f"--- Starting {stage_title(stage_name)} "
            f"({stage_name}, budget={stage_cfg.total_timesteps}, target_trials={stage_cfg.n_trials}) ---"
        )
        while True:
            stage_trials = completed_stage_trials(study, stage_name)
            if len(stage_trials) >= stage_cfg.n_trials:
                print(
                    f"{stage_title(stage_name)} reached target of {stage_cfg.n_trials} completed trials."
                )
                break

            inherited_overrides = inherited_stage_overrides(study, tuning_config, stage_cfg)
            if inherited_overrides:
                print(
                    f"Using inherited overrides from: {', '.join(stage_cfg.inherits_from)}"
                )

            study.optimize(
                lambda trial: run_stage_objective(
                    trial,
                    base_args=base_args,
                    study_id=cli_args.study_id,
                    tuning_config=tuning_config,
                    stage_cfg=stage_cfg,
                    inherited_overrides=inherited_overrides,
                ),
                n_trials=1,
            )
    except KeyboardInterrupt:
        print("\nInterrupted early! Saving study results so far...")
        
    print(f"Number of finished trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print("Best trial overall:")
        trial = study.best_trial
        
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        print("  Stage: ", trial_stage_name(trial))
    else:
        print("No trials completed. Skipping best trial extraction.")
        
    # Save study statistics to CSV
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = (
        f"results/optuna_study_{base_args.study_id}_exp_{base_args.exp_id}_{timestamp}.csv"
    )
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved Optuna statistics to: {csv_path}")


if __name__ == "__main__":
    main()
