from src.config.authored_scenarios import (
    authored_scenario_entries_for_families,
    authored_scenario_entries_for_family,
)
from src.config.studies.models import (
    ExperimentSpec,
    ScenarioCatalogProtocol,
    StudyConfig,
)


EDGE_CASE_SCENARIOS = StudyConfig(
    study_id="EDGE_CASE_SCENARIOS",
    description="Scenario-focused study for structured edge-case evaluation and training around curated hazardous situations.",
    optuna_study_name="EDGE_CASE_SCENARIOS",
    db_path="results/edge_case_scenarios_optuna.db",
    metadata={
        "owner": "carlabev-lab",
        "kind": "scenario",
        "notes": "Structured catalogue for edge-case runs. Trainer-side scenario selection can consume scene/scenario_preset_id from the experiment spec.",
    },
    train_protocols={
        "jaywalk_only_train": ScenarioCatalogProtocol(
            protocol_id="jaywalk_only_train",
            mode="scenario_catalog",
            entries=authored_scenario_entries_for_family("jaywalk"),
            variation_enabled=True,
            variation_seed_mode="random_per_reset",
        ),
        "lead_brake_only_train": ScenarioCatalogProtocol(
            protocol_id="lead_brake_only_train",
            mode="scenario_catalog",
            entries=authored_scenario_entries_for_family("lead_brake"),
            variation_enabled=True,
            variation_seed_mode="random_per_reset",
        ),
        "red_light_only_train": ScenarioCatalogProtocol(
            protocol_id="red_light_only_train",
            mode="scenario_catalog",
            entries=authored_scenario_entries_for_family("red_light_runner"),
            variation_enabled=True,
            variation_seed_mode="random_per_reset",
        ),
        "all_edge_cases_train": ScenarioCatalogProtocol(
            protocol_id="all_edge_cases_train",
            mode="scenario_catalog",
            entries=authored_scenario_entries_for_families(
                ["jaywalk", "lead_brake", "red_light_runner"]
            ),
            sample_strategy="random",
            variation_enabled=True,
            variation_seed_mode="random_per_reset",
        ),
    },
    eval_protocols={
        "all_edge_cases_eval": ScenarioCatalogProtocol(
            protocol_id="all_edge_cases_eval",
            mode="scenario_catalog",
            entries=authored_scenario_entries_for_families(
                ["jaywalk", "lead_brake", "red_light_runner"]
            ),
            sample_strategy="cycle",
            variation_enabled=False,
            variation_seed_mode="none",
        ),
    },
    experiments={
        1: ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="masks",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="jaywalk_only_train",
            eval_protocol_ids=["all_edge_cases_eval"],
            tags=["edge-case", "jaywalk"],
        ),
        2: ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="masks",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="lead_brake_only_train",
            eval_protocol_ids=["all_edge_cases_eval"],
            tags=["edge-case", "lead-brake"],
        ),
        3: ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="masks",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="red_light_only_train",
            eval_protocol_ids=["all_edge_cases_eval"],
            tags=["edge-case", "red-light"],
        ),
        4: ExperimentSpec(
            action_mode="discrete",
            traffic="on",
            input_type="masks",
            reward_mode="carl",
            curriculum="off",
            fov_mask="off",
            train_protocol_id="all_edge_cases_train",
            eval_protocol_ids=["all_edge_cases_eval"],
            tags=["edge-case", "all-scenarios"],
        ),
    },
)
