from __future__ import annotations

from dataclasses import dataclass

import tyro

from src.visualization.study_results import STUDY_SPECS, generate_all_plots


@dataclass
class Args:
    runs_dir: str = "runs"
    output_dir: str = "docs/study_plots"
    dpi: int = 180
    formats: tuple[str, ...] = ("png",)


def main() -> None:
    args = tyro.cli(Args)
    generated = generate_all_plots(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        dpi=args.dpi,
        formats=args.formats,
    )

    print("Generated study result plots:")
    for study_id in [*STUDY_SPECS.keys(), "cross_study"]:
        paths = generated.get(study_id, [])
        if not paths:
            continue
        print(f"\n## {study_id}")
        for path in paths:
            if path.is_file():
                print(path)


if __name__ == "__main__":
    main()
