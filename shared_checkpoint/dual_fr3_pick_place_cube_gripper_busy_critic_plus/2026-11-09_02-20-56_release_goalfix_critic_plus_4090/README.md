# Pick-place cube, CriticPlus model_4749 (4090, 2026-11-09)

- task: `Isaac-PickPlace-Cube-DualFR3-UMI-GripperBusy-CriticPlus-v0` (Play: `...-Play-v0`), code 75ba17b
- actor: same 37D as `-GripperBusy-v0` (last obs = gripper busy bit: 1 from sending the gripper goal until its result).
  CriticPlus only adds critic inputs (gripper internal state 8D, DR values 7D), so deployment is the GripperBusy actor.
- gripper: action < 0 close, >= 0 open (deploy node `gripper_hysteresis` 0); sim command delay 0.45-0.9 s
- training: `2026-11-09_01-28-57_critic_plus_4090` (from scratch, 11-09 fix code) to it 750, then resumed with the
  release reward fix (1ad1ebb) and the goal-resample fix (75ba17b) to it 4749

Eval (`scripts/tools/eval_dual_fr3_pick_place_grasp.py --keep_terminations`, 128 envs, deterministic, Play task):

| | |
|---|---|
| success | 97/128 (held 121, placed 111, released 107) |
| cube xy error at placed | median 0.9 cm |
| cube tilt at grasp / max while held | median 3.9 deg / 11-17 deg |
| fingertip above table at grasp | median 3.9 cm (target 5, floor 3) |
| finger-table contact > 1 N | 30/128 envs, max 41 N |

Chosen over Tilt model_4749 (109/128 but contact 60/128, max 94 N, fingertip 2.9 cm) and CriticPlus model_4500
(107/128 but one env hit 2564 N on the table) for the lowest table contact.
Known issue: still grasps lower than the 5 cm target.
