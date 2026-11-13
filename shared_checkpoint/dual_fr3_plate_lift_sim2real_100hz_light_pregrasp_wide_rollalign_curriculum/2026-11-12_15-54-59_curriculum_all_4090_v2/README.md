# Box lift Curriculum v2, model_5999 (4090, 2026-11-12)

- task: `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Curriculum-v0`, code 21ef198, from scratch, 4096 envs, 6000 it
- actor 56D (same observation layout as GoalXY 5499), critic + DR values 14D
- export: `exports/dual_fr3_boxlift_pregrasp_wide_rollalign_curriculum_v2_model5999.npz` (max |diff| vs torch 3.3e-6)
  - `start.mode` pregrasp_table (3956 rows) plus `start.hover` (fixed joint position, box x 0.40-0.50, y +-0.10, yaw +-0.1)
  - goal `xy_range_base` x 0.40-0.50, y +-0.10, z 0.15-0.35; episode 6 s
  - the last ~1000 it trained on hover starts only; table starts still pass (below)

Deterministic eval (`scripts/tools/eval_dual_fr3_plate_robustness.py`, seed 42, final curriculum state: low friction, independent goal xy).
Success = goal distance, tilt < 15 deg and both plates in contact, held for 25 steps (1.25 s).

| start | envs | success (goal 10 cm) | success (goal 5 cm) | drops | time to success (median) |
|---|---|---|---|---|---|
| table (`-Play-Table-v0`) | 128 | 97.7 % | 87.5 % | 0 | 3.2 s |
| hover (`-Play-Hover-v0`) | 128 | 99.2 % | 86.7 % | 0 | 3.45 s |

Plate force per plate median 18 N. Slip probe (friction 0 for 0.3 s): 99 % fall, 53-73 % lifted again.

Hover start on the robot: move both arms to `start.hover.joint_position` (left 1-7, right 1-7), then `start_policy`.
Watch the first plate contacts (sim: one plate can touch up to 0.4 s before the other, p95).
