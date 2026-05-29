from __future__ import annotations

from pathlib import Path

from src.config.studies.models import ScenarioEntry


SCENES_DIR = Path("assets/scenes")

AUTHOR_SCENARIO_FAMILIES = {
    "jaywalk": "jaywalk-*.json",
    "lead_brake": "leadbrake-*.json",
    "red_light_runner": "redlightrunner-*.json",
}


def authored_scenario_paths_for_family(family: str) -> list[str]:
    try:
        pattern = AUTHOR_SCENARIO_FAMILIES[family]
    except KeyError as exc:
        available = ", ".join(sorted(AUTHOR_SCENARIO_FAMILIES))
        raise KeyError(
            f"Unknown authored scenario family '{family}'. Available families: {available}"
        ) from exc

    return sorted(str(path) for path in SCENES_DIR.glob(pattern))


def authored_scenario_entries_for_family(family: str) -> list[ScenarioEntry]:
    return [
        ScenarioEntry(config_file=path, notes=f"Authored scenario from family '{family}'")
        for path in authored_scenario_paths_for_family(family)
    ]


def authored_scenario_entries_for_families(families: list[str]) -> list[ScenarioEntry]:
    entries: list[ScenarioEntry] = []
    for family in families:
        entries.extend(authored_scenario_entries_for_family(family))
    return entries
