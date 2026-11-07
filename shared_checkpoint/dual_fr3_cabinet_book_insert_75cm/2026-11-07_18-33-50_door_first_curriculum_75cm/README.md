# Table75 DoorFirst curriculum, model_2000 (4090, 2026-11-07)

- task: `Isaac-Cabinet-BookInsert-DualFR3-UMI-Table75-DoorFirstCurriculum-v0` (Play: `...-Play-v0`, curriculum at progress 1)
- 75 cm table, Upright rewards, DoorFirst ramped in by `mdp.door_first_ramp` (from scratch, 4096 envs, seed 42)
- curriculum at it 2000: progress 1 (book_wait -20, book gates 80-90 deg, success door >= 85 deg)

door-entry eval (121 start rows, Play task):

| | |
|---|---|
| success | 121/121 |
| door at book front entry | median 99, min 95 deg |
| rows past the wait line before door 80 deg | 0 |
| door max | 112 deg |
| tilt at entry / final | median 14 deg / 1 deg |
| final depth | 11.8-12.4 cm |

Known issue: the book tilts 31-35 deg while it is carried beside the opening door (record_views, 3 episodes);
the follow-up run adds a tilt penalty (`...-DoorFirstCurriculum-Tilt-v0`).
