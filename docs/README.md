# Medium Scene Visual Assets

These assets were generated from a representative `rt_medium_v1` simulator scene with visible vehicles. They are intended to support the medium-difficulty report set:

- [difficulty temporal fusion report](/home/danielmtz/Projects/carlabev-lab/results/reports/ppo_navigation_difficulty_temporal_fusion.md)
- [medium temporal fusion report](/home/danielmtz/Projects/carlabev-lab/results/reports/ppo_navigation_medium_temporal_fusion.md)
- [medium FOV anchor report](/home/danielmtz/Projects/carlabev-lab/results/reports/ppo_navigation_medium_fov_anchor.md)

## Overview Figures

- [medium_scene_representative_rgb.png](/home/danielmtz/Projects/carlabev-lab/docs/medium_scene_representative_rgb.png)
- [medium_scene_semantic_modes.png](/home/danielmtz/Projects/carlabev-lab/docs/medium_scene_semantic_modes.png)
- [medium_scene_anchor_modes.png](/home/danielmtz/Projects/carlabev-lab/docs/medium_scene_anchor_modes.png)
- [medium_scene_temporal_fusion_modes.png](/home/danielmtz/Projects/carlabev-lab/docs/medium_scene_temporal_fusion_modes.png)

## Per-Mode Directories

- [representative_rgb](/home/danielmtz/Projects/carlabev-lab/docs/representative_rgb)
  - current RGB frame only
- [semantic_modes](/home/danielmtz/Projects/carlabev-lab/docs/semantic_modes)
  - raw RGB and each semantic mask mode exported channel-by-channel
- [anchor_modes](/home/danielmtz/Projects/carlabev-lab/docs/anchor_modes)
  - `center` and `lookahead_75` frames on the same seeded world state
  - after the env render-padding fix, anchor changes affect framing only, not spawn
- [temporal_fusion](/home/danielmtz/Projects/carlabev-lab/docs/temporal_fusion)
  - `rgb_history`
  - `current_stack`
  - `vehicle_temporal`
  - `vehicle_weighted`

## Regeneration

Regenerate all assets with:

```bash
uv run carlabev-lab results medium-report
```

## Study Artifact Layout

Study runs referenced by the reports now resolve through the short scaffold:

```text
runs/<study_id>/exp_<exp_id>/trial_<trial_id>/seed_<seed>/
```

with study-local outputs under:

- `checkpoints/`
- `eval/`
- `videos/train/`
- `videos/eval/intermediate/`
- `videos/eval/final/`
