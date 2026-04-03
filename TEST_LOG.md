# lerobot-doctor Test Log

Tested against 12 real HuggingFace datasets. All checks ran without crashes.

## Results Summary

| # | Dataset | Eps | FPS | Overall | Key Findings |
|---|---------|-----|-----|---------|-------------|
| 1 | lerobot/pusht | 206 | 10 | WARN | Short episodes for ACT chunk_size |
| 2 | lerobot/pusht_image | 206 | 10 | WARN | Short episodes for ACT chunk_size; `next.success` zero-variance |
| 3 | lerobot/aloha_sim_insertion_human | 50 | 50 | WARN | Clean (only video skip warning) |
| 4 | lerobot/aloha_sim_insertion_scripted | 50 | 50 | WARN | Clean (only video skip warning) |
| 5 | lerobot/aloha_sim_transfer_cube_human | 50 | 50 | WARN | Clean (only video skip warning) |
| 6 | lerobot/aloha_static_cups_open | 50 | 50 | WARN | Clean (only video skip warning) |
| 7 | lerobot/aloha_mobile_shrimp | 18 | 50 | WARN | Effort sensor outliers (real torque spikes) |
| 8 | lerobot/xarm_lift_medium | 800 | 15 | WARN | All episodes too short for ACT/Diffusion chunk_size |
| 9 | lerobot/droid_100 | 100 | 15 | WARN | Frozen actions (27 consecutive), gripper clipping at 1.0 |
| 10 | lerobot/koch_pick_place_5_lego | 50 | 30 | WARN | Frozen actions (22 consecutive) from teleop pauses |
| 11 | lerobot/unitreeh1_fold_clothes | 38 | 50 | WARN | 40-DOF action space flagged; distribution shift in observation dim 5 |
| 12 | lerobot/columbia_cairlab_pusht_real | 136 | 10 | WARN | **6 zero-variance obs dims + 5 zero-variance action dims**; normalization will produce NaN |

## False Positive Fixes Applied

After first round of testing, fixed:
- `next.done`, `next.success`, `next.reward` no longer trigger outlier warnings (binary signals)
- Action jump threshold raised from 5-std-max to 8-std-mean (high-DOF robots at 50fps were all flagged)
- Stuck actuator threshold raised from 30% to 80% of episodes (gripper DOFs are normally mostly static)
- `next.*` columns excluded from "broken sensor" anomaly check
- Video check skips gracefully for HF Hub datasets (videos not downloaded)

## Real Issues Found

1. **columbia_cairlab_pusht_real**: 6 observation dims and 5 action dims are padded zeros. Normalization will NaN.
2. **droid_100**: Gripper action (dim 6) is 52% at max value (1.0), 27 consecutive frozen actions in episode 7.
3. **koch_pick_place_5_lego**: 22 consecutive frozen actions in episode 0 (teleop pause).
4. **unitreeh1_fold_clothes**: Distribution shift in observation dim 5 between first/last recording sessions.
5. **aloha_mobile_shrimp**: Effort sensor outliers (20 extreme values in dim 7).
6. **xarm_lift_medium**: All episodes 25 frames -- too short for any chunked policy.

## Crashes / Errors

None. All 12 datasets completed without errors.
