from src.config.scene_benchmarks.navigation_medium_v1 import NAVIGATION_MEDIUM_V1
from src.config.studies.models import SceneBenchmarkConfig, SceneBenchmarkProfile


SCENE_BENCHMARK_REGISTRY: dict[str, SceneBenchmarkConfig] = {
    NAVIGATION_MEDIUM_V1.benchmark_id: NAVIGATION_MEDIUM_V1,
}


def list_scene_benchmark_ids() -> list[str]:
    return sorted(SCENE_BENCHMARK_REGISTRY.keys())


def get_scene_benchmark_config(benchmark_id: str) -> SceneBenchmarkConfig:
    try:
        return SCENE_BENCHMARK_REGISTRY[benchmark_id]
    except KeyError as exc:
        available = ", ".join(list_scene_benchmark_ids())
        raise KeyError(
            f"Unknown scene benchmark {benchmark_id!r}. Available benchmarks: {available}"
        ) from exc


def get_scene_benchmark_profile(
    benchmark_id: str,
    scene_profile_id: str,
) -> SceneBenchmarkProfile:
    benchmark = get_scene_benchmark_config(benchmark_id)
    try:
        return benchmark.profiles[scene_profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(benchmark.profiles))
        raise KeyError(
            f"Unknown scene_profile_id {scene_profile_id!r} for benchmark {benchmark_id!r}. "
            f"Available profiles: {available}"
        ) from exc
