# 박스 리프트·큐브 pick-and-place 커리큘럼 환경: 구성과 평가 방법 (2026-11-10)

실물에 올린 정책(박스 5499, 큐브 4000)은 여러 체크포인트를 이어 학습해 만들었다. 이 문서의 두 task는 같은 보상·관측·행동을
**처음부터 한 run**으로 배우는 커리큘럼이고, 11-10에 논의한 후보 항목을 **전부 켠 채** 학습한 뒤 결과를 보고 나쁜 항목을 빼는
ablation 절차를 따른다. 5070과 4090이 같은 코드로 서로 다른 ablation을 돌릴 수 있게 항목마다 on/off 플래그가 있다.

- 박스: `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Curriculum-v0`
  (`lift_plate/joint_pos_env_cfg.py` 끝 `_apply_box_curriculum`, 플래그 `BOX_CURRICULUM_FLAGS`)
- 큐브: `Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-v0`
  (`pick_place_cube/pick_place_env_cfg.py` 끝 `_apply_cube_curriculum`, 플래그 `CUBE_CURRICULUM_FLAGS`)
- actor 관측·행동은 각각 5499(policy 그룹)·GripperBusy(37D)와 같아 **배포 노드를 그대로 쓴다**. critic만 커져서 옛 checkpoint는 못 읽는다.

## 1. 박스 리프트 Curriculum

기준은 5499의 최종 cfg(GoalXY)이고, 계보가 이어 학습용으로 즉시 켰던 항을 되돌려 처음부터 배우게 한다.

| 구분 | 항목 | 켜지는 시점 | 플래그 |
|---|---|---|---|
| 5499 보상 ①a | plate_face_tilt(worst) −20, offcenter −20, min_squeeze 16 N −0.5 | 단계 shaping_a: 성공률 ≥ 0.10, 200 iter | 항상 |
| 5499 보상 ①b | box_lift_accel −20, plate_box_slip −10, box_yaw_drift −10 (+ 파지 높이 −10) | 단계 shaping_b: ①a 끝 + 성공률 ≥ 0.20, 200 iter | 항상 |
| 5499 보상 | yawtrack(success의 yaw 계수) | EMA ≥ 1.5 뒤 2000 iter ramp (Scratch 2999는 3000 iter) | 항상 |
| 환경 ramp ② | goal xy: 박스 위(수직) → 독립 샘플 x 0.40~0.50, y ±0.10 (`MixedGoalXYPoseCommand`, `env._goalxy_scale`) | 단계 goalxy: ①b 끝 + 성공률 ≥ 0.30, 400 iter | `goalxy_ramp` |
| 환경 ramp ③ | 마찰: 박스 0.4~1.2 / 판 0.5 → SOFTLIFT(박스 0.1~0.8, 판 0.3~0.5). reset마다 재배정 | 단계 friction: ② 끝 + 성공률 ≥ 0.30, 800 iter | `friction_ramp` |
| 환경 ramp ④ | 호버 시작 혼합 0 → 1 (`env._hover_mix`), 에피소드 6 s, 판이 박스 윗면 누르면 −5/s | 단계 hover: ③ 끝 + 성공률 ≥ 0.30, 800 iter | `hover` |
| DR | 박스 크기 ±1 cm (USD scale이라 커리큘럼 불가) | 처음부터 | `size_dr` |
| DR | 관절 상태 0~10 ms, 명령 0~1 step 지연 (DR task의 절반) | 처음부터 | `joint_delay` |
| DR | 박스 pose 관측 에피소드 고정 오프셋 ±1 cm, yaw ±3° (BoxDrop 지연·끊김·잡음에 추가) | 처음부터 | `camera_bias` |
| 벌점 | 파지 높이: 판 중심−면 중심 높이 차 3 cm 초과분 −10/(m·s) | 단계 shaping_b | `grasp_height` |
| 벌점 | j1 거울: (q_L + q_R)의 에피소드 시작 대비 변화 0.05 rad 초과분 −1/(rad·s) | 처음부터 | `j1_mirror` |
| critic | DR 값 14D: 박스·양 판 정지 마찰, 질량, 크기(y, z), 양팔 강성·감쇠 배율, 관절·명령·비전 지연, 비전 끊김 | − | `critic_dr` |
| actor | 관측 history 4 step(0.2 s, 56 × 4 = 224D). **기본 off**, `-History-v0`/`-History-Play-v0` ablation | − | `obs_history` |

- **단계 스케줄러(`BOX_STAGES`, 11-12 4090 run 1 뒤)**: 기준은 `success_rate` = success 합 / (weight × dt × 에피소드 step) = 0~1로
  에피소드 길이와 무관하다(계보 수렴치 0.46~0.62, 4090 run 1 최고 0.28). 단계는 순서대로, 이전 단계가 1.0이고 성공률이 그 단계 기준 이상인
  step에서만 진행한다(아니면 **멈춤**, 되돌림 없음). 한 단계가 끝나면 다음 단계는 성공률이 기준을 **다시** 넘어야 움직이므로 단계 사이에
  재안정 구간이 생긴다. yawtrack(EMA 1.5)·외란·q′(EMA 1→3)는 계보 그대로 독립 EMA다.
- **호버 자세**: 명목 박스(0.45, 0)의 접근 끝 자세에서 판 TCP를 0.25 m 올리고 판마다 7.5 cm 바깥(간격 +15 cm)으로 옮긴 IK 해,
  좌우 거울 대칭. 박스 위치와 무관한 고정 관절값(`HOVER_Q_LEFT/RIGHT`, `scripts/tools/solve_plate_hover_pose.py`).
  박스는 x 0.40~0.50, y ±0.10, yaw ±0.1 rad에서 뽑는다. 0.25 m는 논의 범위(0.2~0.3)의 기본값이고 `--dz`로 바꿔 다시 풀면 된다.
- **j1 거울**을 절대값이 아니라 시작 대비 변화량으로 건 이유: 표(접근 끝) 시작의 |q_L + q_R|가 박스 y에 비례(3.1·y, p90 0.29 rad)해서
  4090의 J1Mirror(deadband 0.15 절대값)를 그대로 쓰면 시작 자세가 정해 둔 비대칭을 벌한다.
- 평가 cfg(커리큘럼 **최종 상태**로 고정: SOFTLIFT 마찰, goal xy 독립, shaping 최종 weight):
  `-Play-v0`(표 50 % + 호버 50 %), `-Play-Table-v0`(표만, 5499 평가와 같은 조건), `-Play-Hover-v0`(호버만).
- ablation 예: `-NoHover-v0` (호버만 끔, 에피소드 4 s). 다른 항목은 `_flags`를 덮어쓴 하위 클래스를 같은 식으로 추가한다.
- `-History-v0`(전부 on + 관측 history): 메모리 없는 정책은 미끄러지기 전엔 μ를 알 수 없으므로, 전부-on run 뒤 두 번째 ablation으로 돌린다.
  비교 지표는 낮은 μ(0.1~0.3)만 고른 성공률과 `box_slip_probe` 재파지율. 쌓기 레이아웃은 Isaac Lab 그룹 history: 항(term)마다 (4, dim)을
  오래된 것 → 최신 순으로 펴서 항 순서대로 이어 붙이고, 리셋 직후는 첫 관측을 4번 반복. 배포하려면 노드에 같은 링 버퍼가 필요하다.

## 2. 큐브 pick-and-place Curriculum

기준은 GripperBusy(4000 계보: busy 비트, 실물 그리퍼 지연, 큐브 pose 오프셋, 11-09 release shaping·goal resample 수정).

| 항목 | 켜지는 시점 | 플래그 |
|---|---|---|
| CriticPlus critic(그리퍼 내부 상태 8D + DR 값 7D) | − | `critic_plus` |
| 기울기 15° 초과 벌점 −5 | reach_held EMA ≥ 0.5 (들기를 배운 뒤) | `tilt` |
| 끌기 벌점 −5/s: 들었던 큐브를 쥔 채 책상 높이로 목표 5 cm 밖에서 끌기 (`cube_drag`) | reach_placed EMA ≥ 0.3 | `drag_penalty` |
| (보류) 끌기 금지: 같은 조건이면 HELD 단계 박탈 (`env._pp_carry_strict`). run 1에서 placed 붕괴, 기본 off | reach_placed EMA ≥ 0.3 | `carry_strict` |
| 손가락−책상 접촉력 비례 벌점 (20 N → −2/s, 100 N → −10/s; 닿음 0/1 −5는 유지) | 처음부터 | `finger_force` |
| 그리퍼 멈춤 → result 대기 0.8~1.2 s (sim 기본 0.3~0.4) | 처음부터 | `gripper_settle` |
| 책상 −Near(앞 모서리 x 0.30) 고정, 큐브 시작·목표 x 0.45~0.65 → 0.35~0.65 ramp | reach_released EMA ≥ 0.3 뒤 200 iter (run 1은 placed 0.3) | `near_ramp` |

평가 cfg `-Curriculum-Play-v0`은 최종 상태(기울기 벌점 최종, 끌기 금지 켬, x 0.35~0.65)다.

## 3. 학습 중 보는 것 (TensorBoard)

`python3 ~/.claude/skills/analyze-training/scripts/parse_tb.py logs/rsl_rl/<experiment>/<run>` 또는 TB에서:

| 항목 | 박스 | 큐브 |
|---|---|---|
| 커리큘럼 진행 | `Curriculum/schedule/success_rate`, `schedule/stage_index`(끝난 단계 수 0~5), `schedule/shaping_a … hover`(0 → 1); 항별 `Curriculum/plate_face_tilt` 등 weight, `goalxy`, `hover_mix`, `yawtrack` | `Curriculum/progress/reach_*`(단계 도달률 EMA), `cube_tilt_excess`(−1e-4 → −5), `carry_strict`(0 → 1), `near_x`(0.45 → 0.35) |
| 성공 | `Episode_Reward/success`(sustained, 20/s) | `Episode_Reward/success`, `reach_success` |
| 후보 항목의 비용 | `Episode_Reward/plate_face_z_excess`, `j1_mirror`, `plate_on_box_top` | `Episode_Reward/cube_tilt_excess`, `finger_table_force` |
| 종료 | `Episode_Termination/object_out_of_bounds` | `cube_dropped`, `cube_toppled`, `cube_off_table` |

- 단계가 **멈춰 있으면**(`schedule/success_rate`가 그 단계 기준 아래) 뒤 단계는 안 켜진다. 2000 iter에 `stage_index`가 2(shaping만)면 0.30이
  높은 것이니 0.20으로 낮춘다. 어떤 단계 ramp 중 성공률이 떨어져 멈춘 채 회복이 없으면 그 단계가 ablation 1순위다. 호버(④)는 6000 iter 안에
  못 올 수 있으니 TB를 보고 `--resume`으로 연장한다(11-12 결정).
- 박스의 `hover_mix`가 오르는 동안 `Episode_Reward/plate_on_box_top`과 `object_out_of_bounds`가 같이 오르면 내려오는 동작을 못 배운 것이다
  (에피소드 6 s 안에 못 잡는지는 `Episode_Length`로 본다).

## 4. 학습 후 sim 평가 (체크포인트마다 같은 순서)

### 4.1 박스

1. **결정적 성공률** — `-Play-Table-v0`(5499와 같은 조건)과 `-Play-Hover-v0`에서 각각:
   ```bash
   python scripts/tools/eval_dual_fr3_plate_robustness.py --headless --num_envs 128 --episodes 1 --seed 42 \
       --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Curriculum-Play-Table-v0 \
       --checkpoint logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_curriculum/<run>/model_N.pt
   ```
   보는 값: sustained success, 운반 중 기울기, 양판 squeeze 유지율, drop, 판 힘·높이 비대칭. 5499 기준값은
   `~/etri_ws/src/fr3_husky_etri/docs/dual_fr3_boxlift_5499_report.md`의 sim 표.
2. **낮은 마찰 미끄럼** — `scripts/tools/box_slip_probe.py --task ...-Play-Table-v0 --checkpoint ... --mu 0.0 --viz none`:
   들다가 미끄러질 때 낙하율·재파지율. 실물 실패(판 3~8 cm 미끄럼 → 박스 회전)를 재현하는 지표.
3. **goal 정확도** — 1의 출력에서 goal 거리·yaw 결합 성공률(goal x 0.40~0.50, y ±0.10 전 범위).
4. **호버 시작 품질**(호버 run만) — `-Play-Hover-v0`에서 1을 돌리고 `plate_on_box_top` 에피소드 합, 첫 양판 접촉 시각, 접촉 시 판 높이
   (`eval_dual_fr3_plate_robustness.py --joint_trace_csv`로 env 몇 개의 판/관절 궤적 저장).
5. **배포 전 관문** — `scripts/tools/dump_boxlift_deploy_trace.py`(Play-Table) → etri `verify_boxlift_obs_parity.py`,
   `scripts/tools/export_boxlift_actor.py`. 주의: export 도구는 goal 커맨드를 `VerticalLift`/`UniformPoseCommand`로 구분하므로
   `MixedGoalXYPoseCommand`(최종 상태 = Uniform) 처리를 export 때 추가해야 하고, 호버 시작은 manifest에 새 start 종류가 필요하다.

### 4.2 큐브

1. **파지·운반 품질** — 4090 세 run을 잰 것과 같은 도구:
   ```bash
   python scripts/tools/eval_dual_fr3_pick_place_grasp.py --headless --num_envs 128 \
       --task Isaac-PickPlace-Cube-DualFR3-UMI-Curriculum-Play-v0 \
       --checkpoint logs/rsl_rl/dual_fr3_pick_place_cube_curriculum/<run>/model_N.pt
   ```
   보는 값: 쥘 때 손가락 끝 높이·겹침(깊은 파지), 손가락−책상 접촉 env 수·최대 힘, 쥘 때·운반 중 기울기, 단계 도달, PLACED에서 OPEN 요청률,
   목표 xy 오차. 기준: 4000(`docs/dual_fr3_pick_place_cube.md`), 4090 Tilt 107/128·CriticPlus 114/128.
2. **성공률(종료 포함)** — 같은 도구에 `--keep_terminations` (학습 성공률과 같은 정의).
3. **끌기 여부** — 1의 출력에서 "들린 뒤 다시 책상 높이로 내려온 시간"과 운반 중 큐브 z. 실물 model_4000은 책상 위로 끌고 가 목표 앞에서만 들었다.
4. **범위 끝** — 큐브 시작 x 0.35~0.45만(`reset_cube.x_range`를 바꾼 짧은 Play cfg)으로 1을 다시 돌려 Near 범위 끝의 성공률.

## 5. 실물 평가 (sim이 통과한 뒤)

- 박스: `~/etri_ws/src/fr3_husky_etri/docs/dual_fr3_box_experi_step.md` 절차, 10회. bag에서 조임(판 법선 힘), 들기 순간 판−박스 상대 이동,
  박스 yaw 변화를 5499 분석(11-05 bag)과 같은 방법으로 뽑는다. 호버 정책은 노드가 pregrasp 표 대신 `HOVER_Q_*`로 이동한 뒤 정책을 켠다.
- 큐브: `dual_fr3_cube_pickplace` 실험 md 절차, 10회. 운반 중 큐브 z(ArUco)로 끌기 여부, 쥘 때 손가락 끝 높이(FK).

## 6. ablation 절차 (5070·4090 공통)

1. 전체 on run을 두 PC에서 seed만 다르게 1개씩(또는 4090 2개) 돌린다. 체크포인트는 500 iter마다 4.1/4.2의 1번만 돌려 곡선을 그린다.
2. 전체 on이 기준(5499 / 4000)보다 나쁘면, TB에서 비용이 큰 항목(§3 "후보 항목의 비용")과 ramp 시점에 success가 꺾인 항목을 고른다.
3. 그 항목만 끈 하위 클래스(`_flags` 덮어쓰기, 예 `-NoHover`)를 등록해 같은 iter 수로 돌리고 §4를 반복한다. 한 번에 한 항목씩 뺀다.
4. 항목을 빼서 좋아지면 그 항목은 "보류"로 두고 이유(어느 지표)를 이 문서 §7에 적는다. 두 PC 결과는 run 이름에 PC를 넣어 구분한다
   (`--run_name curriculum_all_5070`, `curriculum_nohover_4090`).
5. 최종 후보는 §4 전부 + 배포 관문(4.1-5)을 통과한 것 하나만 실물로 간다.

## 7. 결과 기록

### 박스 run 1 (4090, 2026-11-12 14:00, `curriculum_all_4090`, ec8c1b6, 4096 env, 3531 iter에서 중단) — 스케줄러 도입
- 290~330 it 사이에 success EMA 합이 3/4/5를 모두 넘어 shaping(−20/−10, 100~200 iter)·goal xy·마찰·호버가 동시에 켜짐. 400 it 성공 5.5/s
  (6 s 합 33, 성공률 0.28)가 최고였고, 600 it 2.7로 떨어졌는데도 ramp는 step 수만 세며 끝까지 감(700 it 마찰 0.99, 호버 0.94).
- 800 it부터 조이기·들기 0, reaching ~7만 받는 "다가가기만" 국소해. std 0.39 → 0.11. 3531 it까지 회복 없음(체크포인트 model_3500).
- 원인: (1) 발동 기준이 성공률 ~0.04의 한 점이라 순서가 없었음, (2) 멈춤·되돌림 없음, (3) 벌점과 환경 난이도가 겹침.
- 조치(run 2): §1 단계 스케줄러(성공률 기준 0.10/0.20/0.30, 순서, 멈춤, shaping a/b 분리, ramp 2배). 4090이 pull 뒤 `-Curriculum-v0` 재시작,
  `-History-v0`는 그 뒤.

### 박스 run 2 (4090, 2026-11-12 15:54, `curriculum_all_4090_v2`, 21ef198, 4096 env, 6000 iter) — 실로봇 후보

`logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_curriculum/2026-11-12_15-54-59_curriculum_all_4090_v2`

| iter | 학습 성공률(`schedule/success_rate`) | 끝난 단계 | 비고 |
|---|---|---|---|
| 500 | 0.43 | 0 | shaping ramp 중 |
| 1000 | 0.66 | 2 | goal xy 0.66 |
| 1500 | 0.66 | 3 | 마찰 0.46 |
| 2000 | **0.25** | 3 | 마찰 0.95: 낮은 마찰이 병목 |
| 2500~3750 | 0.26~0.30 | 4 | 호버 0.40 → 0.50, 기준 0.30 근처에서 멈춤·재개 |
| ~5000 | 0.33 | 5 | 호버 1.0 (이후 학습은 전부 호버 시작, `hover_mix`는 호버 확률) |
| 5999 | 0.51 | 5 | |

model_5999 결정적 평가(`eval_dual_fr3_plate_robustness.py`, seed 42, 커리큘럼 최종 상태, 성공 = goal 거리·기울기 15°·양판 접촉을 연속 25 step(1.25 s)):

| 조건 | env | 성공 (goal 10 cm) | 성공 (goal 5 cm, `--success_d 0.05`) | drop / 놓침 | TTS 중앙 (10 / 5 cm) | 판 힘 판당 중앙 / 최대 | 판 높이 차 중앙 |
|---|---|---|---|---|---|---|---|
| `-Play-Table` (표 시작, 5499와 같은 시작) | 128 | **97.7%** | **87.5%** | 0 / 1 | 3.20 / 3.50 s | 18.6 / 36 N | 3.6 cm |
| `-Play-Hover` (호버 시작) | 128 | **99.2%** | **86.7%** | 0 / 1 | 3.45 / 3.80 s | 18.0 / 29 N | 2.1 cm |
| `-Play` (반반) | 256 | 98.8% | | 0 / 3 | 3.35 s | 18.3 / 63 N | 3.0 cm |

- 들고 있을 때 goal 오차 중앙 7.7~7.8 cm, yaw 오차 1.7~2.2°. 들고 있는 시간 중 goal 5 cm 안 비율 0.50~0.53, 10 cm 안 0.63~0.66.
- 호버 시작은 첫 양판 접촉 시차가 크다(중앙 0.15 s, p95 0.40 s; 표 시작 0.10 / 0.15 s). 실물에서 한 판이 먼저 박스를 미는지 볼 것.
- `box_slip_probe.py --mu 0.0`(512 env, 1.5 s에 0.3 s): 99%가 바닥까지 낙하, **다시 듦 53%**(Table) / **73%**(반반). 이전 계열(2999, Regrasp 포함) 0%.
- 영상(`record_views.py`, 각 4 에피소드): run 폴더 `videos/model_{4200,5999}_{table,hover}_ep4.mp4`. 5999는 8 에피소드 모두 유지, goal 최소 1.6~3.7 cm, 기울기 평균 1~2°.
- export: `exports/dual_fr3_boxlift_pregrasp_wide_rollalign_curriculum_v2_model5999.npz`(torch 대비 max |diff| 3.3e-6). manifest `start`는
  `pregrasp_table`(3956줄)에 `start.hover`(고정 관절값 14, 박스 x 0.40~0.50, y ±0.10, yaw ±0.1)가 붙고, goal은 `xy_range_base`
  (x 0.40~0.50, y ±0.10), 에피소드 6 s. 배포 노드에 호버 시작 모드(고정 관절값으로 이동 뒤 `start_policy`)가 필요하다.

### 박스 History (4090, 2026-11-13 00:53, `curriculum_history_4090`, 21ef198, 6000 iter) — 관측 history 4 step

- 학습: 낮은 마찰 직후 성공률 하락이 작았다(2000 it v2 0.25 vs History 0.53). 모든 단계 완료가 약 1500 iter 빨랐다(3500 vs ~5000). 최종 학습 성공률은 같다(0.51).
- model_5999, v2와 같은 조건(`-Play` / `-History-Play` 반반, 256 env, seed 42): 성공 100% vs v2 98.8%(3 env 차이, 유의하지 않음), drop 0.
- 미끄럼(`--mu 0.0`, 반반): 다시 듦 **40%** vs v2 73%, 미끄럼 뒤 최대 기울기 중앙 21° vs 13°, 끝에 성공 구역 16% vs 63%.
- 판단: history 4 step은 학습 속도에만 이득, 최종 정책은 미끄럼 대처가 나쁘다(Peng 2018의 FF+history ≈ FF와 같은 방향). 실물 후보는 v2.
  배포하려면 노드에 4 step 관측 버퍼도 필요하다.

### 박스 지연 ablation (KRoC, 2026-11-13 task 추가, 학습은 4090) — v2에서 지연 DR만 끈다

지연 항은 둘로 묶는다. 제어 구조(20 Hz 목표 유지, 토크 변화율 1000 Nm/s, ±0.1 rad clip)는 지연 DR이 아니라 그대로 둔다.

| 묶음 | 내용 | 플래그 |
|---|---|---|
| 비전 (박스 pose 추정) | 에피소드마다 0~150 ms 지연(0~3 policy step) + step마다 2 %로 0.1~1.0 s 끊김(마지막 값 유지) + 에피소드 10 %는 도중부터 끝까지 끊김 | `vision_delay` (11-13 추가, BoxDrop 계보 항을 플래그로) |
| 관절 | 관절 상태 0~10 ms(`q − q̇·d`, 정책 관측과 목표 기준 q) + 명령 0~1 물리 step(0~10 ms) | `joint_delay` |

| task | 비전 | 관절 | run_name (4090) |
|---|---|---|---|
| `-Curriculum-v0` (v2, 기준) | on | on | `curriculum_all_4090_v2` (끝남, model_5999) |
| `-Curriculum-NoDelay-v0` | off | off | `curriculum_nodelay_4090` |
| `-Curriculum-NoVisionDelay-v0` | off | on | `curriculum_novisiondelay_4090` |
| `-Curriculum-NoJointDelay-v0` | on | off | `curriculum_nojointdelay_4090` |

- 카메라 고정 오프셋(±1 cm, yaw ±3°)과 관측 노이즈는 지연이 아니라 모든 run에서 켠다. critic DR 14D의 지연 값은 꺼진 항만 0(차원은 같아 checkpoint 호환).
- 학습: v2와 같은 seed 42, 4096 env, 6000 iter.
  `python scripts/rsl_rl/train.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-Curriculum-NoDelay-v0 --headless --num_envs 4096 --seed 42 --run_name curriculum_nodelay_4090`
- sim 평가: 각 model_5999를 (a) 자기 `-<Ablation>-Play-v0`(학습 조건), (b) v2 `-Play-v0`(지연 전부 on) 두 곳에서 같은 도구·seed로 본다. 실물 측정 지연 분포를 넣은 평가 env는 bag 분석 뒤 추가.
- 확인(5070, 11-13): cfg 8개(v2/3 ablation × 학습/Play), NoDelay·NoJointDelay env smoke(관측 56/70D, NoDelay 박스 관측 = 실제 pose + 오프셋 오차 0, 꺼진 지연의 critic 값 0), NoVisionDelay 학습 3 iter.

### 큐브 run 1 (5070, 2026-11-12 14:00, `curriculum_all_5070`, 2048 env, 1380 iter에서 중단) — carry_strict 보류
- iter ~500 reach_held 0.9 (들기 습득, 기울기 벌점 −5 켜짐), 560~680 reach_placed 0.10 → 0.31.
- iter 680 placed EMA 0.3을 넘자 carry_strict·Near x ramp·action/joint_vel/arm_contact 벌점(−0.1/−1)이 같은 iter에 켜짐 →
  40 iter 뒤 placed 0.003, 이후 700 iter 동안 0.000 (held 0.92 유지, 운반 중 기울기 −0.07 → −0.18/s).
- 해석: 그때까지의 placing이 "책상 위로 끌어서" 놓는 동작이라, 단계를 박탈하는 carry_strict가 켜지자 그 동작 전체가 stage 0이 되고
  정책은 공중에 든 채 버티는 국소해로 갔다. action 벌점 latch는 4000 계보와 같으므로 원인에서 제외.
- 조치(run 2): carry_strict → 끌기 벌점 `cube_drag` −5/s(placed 0.3 뒤)로 대체, Near ramp는 reach_released 0.3 뒤로 미룸.

### 큐브 run 2 (5070, 2026-11-12 14:34, `curriculum_all_5070_v2`, 2048 env, 6000 iter 완주) — 전부 on(carry_strict 대신 drag_penalty)
- 학습: held 0.9(iter 500) → 끌기 벌점 −5가 placed 0.3 전(iter ~700)에 켜져 placed 0.12 → 0.006 dip → iter 1700 회복(0.89) → 손 떼기
  iter 2000(released 0.47) → Near ramp 2000~2200(x 0.35 도달) → 이후 placed 0.75~0.84, success EMA 0.53~0.69(최고 5200)에서 평평.
- 결정적 평가(`eval_dual_fr3_pick_place_grasp.py`, 128 env, Near 전체 범위 x 0.35~0.65, 종료 포함):

| 체크포인트 | success | placed → released → success | 첫 파지 손가락 끝 높이(책상 위) | 쥘 때 기울기 중앙값 | 운반 최대 기울기 | 손가락−책상 접촉 env / 최대 힘 중앙값 |
|---|---|---|---|---|---|---|
| model_5999 | **103/128 (80 %)** | 115 → 114 → 103 | 2.6 cm (목표 5 cm) | 6.7° | 11~16° | 78/128 / 6.9 N |
| model_5200 | 88/128 (69 %) | 113 → 107 → 88 | 2.6 cm | 10.0° | 12~17° | 89/128 / 5.9 N |

- 비교: 4000 계보 기준 배치 91 %, Near 이어 학습 5999 92.6 %; 4090 scratch(기준 배치, 5000 it) Tilt 84 %, CriticPlus 89 %.
  → 전부 on + Near 범위 scratch 6000 it는 **80 %**로 낮다. 파지 깊이는 여전히 깊고(2.6 cm, 손가락 힘 벌점이 못 고침; 접촉 env는 103 → 78,
  최대 힘 136 → 7 N으로 세기만 줄었음), 기울기는 A(28~33°)보다 낫고 B(8~12°)보다는 크다.
- x 0.35~0.45만(`-Curriculum-Play-NearEnd-v0`, model_5999): success **93/128 (73 %)**, held 116(못 집음 7), 손가락−책상 접촉 95/128 — Near 앞쪽 끝에서
  집기·책상 찍기가 더 나쁘다. 영상 `…/curriculum_all_5070_v2/videos/play/curriculum_all_model_5999_1env_840step.mp4`.
- 다음 ablation 후보: (1) `near_ramp` off(기준 배치)로 4000·4090 B와 같은 조건 비교 → 아래 NoNear, (2) x 0.35~0.45 구간만 평가 → 위 NearEnd,
  (3) 깊은 파지는 벌점으로 안 잡히니 파지점 자체(GRASP_OFFSET)나 tip_floor 상향을 검토.

### 큐브 NoNear ablation (5070, 2026-11-12 18:28, `curriculum_nonear_5070`, 21ef198, 2048 env, 6000 iter 완주) — near_ramp만 off
- 학습: run 2와 같은 길(held iter 500, 끌기 벌점 dip, 회복)을 가되 손 떼기가 iter 1500~2000에 한 번에 올라옴(released 0.51 → 0.86, run 2는 0.18 → 0.78).
  success EMA(reach_success) 5999에 0.69(run 2 0.64), Episode_Reward/success 0.52/s(run 2 0.48/s), value loss 130(run 2 165). 둘 다 6000에서 아직 완만히 오르는 중.
- 결정적 평가(128 env, model_5999). x 범위는 학습한 범위(0.45~0.65)이고 run 2는 Near 전체(0.35~0.65)라 조건이 같지 않다:

| 평가 | NoNear 5999 | run 2 5999 (Near 범위) |
|---|---|---|
| 종료 포함(`--keep_terminations`) success | **99/128 (77 %)** (held 116 → placed 108 → released 106) | 103/128 (80 %) |
| 종료 시 리셋(기본) success | **108/128 (84 %)** (117 → 111 → 111) | 95/128 (74 %) |
| 첫 파지 손가락 끝 높이 / 큐브 윗면부터 겹침 | **4.5 cm / 5.0 cm** (목표 5 cm) | 2.0 cm / 7.7 cm |
| 쥘 때 기울기 중앙값 / 최대 | 5.5° / 28° | 6.7° / 21° |
| 손가락−책상 접촉 env / 최대 힘 중앙값 | 62/128 / 10.3 N | 78/128 / 6.9 N |
| 운반(HELD) 시간 중앙값 / 목표까지 xy | 0.8 s / 6.3 cm | 1.1 s / 8.4 cm |
| x 0.35~0.45만(`-Play-NearEnd-v0`, 학습 범위 밖) | 61/128 (48 %), held 88 | 93/128 (73 %) |

- 해석: (1) 성공률은 run 2와 같은 급(77~84 % vs 74~80 %)이고 4000 계보(91 %)·4090 B(89 %)에는 둘 다 못 미친다 → 80 % 언저리는 Near 때문이 아니라
  전부-on 항목(기울기·손가락 힘·끌기 벌점·settle 1.2 s) 또는 scratch 6000 it 부족 쪽이다. (2) **파지 깊이는 Near가 원인**: Near를 빼자 2.0 → 4.5 cm로
  목표(5 cm)에 거의 맞고, 책상 접촉 env도 78 → 62. 큐브가 몸쪽(x 0.35~0.45)에 있을수록 손목을 세워 깊게 찍어 쥐는 자세가 되고 그것이 전 범위로 번진 것으로 본다.
  (3) NoNear는 학습 범위 밖(x 0.35~0.45)에서 48 %라 Near를 쓰려면 ramp 자체는 필요하다.
- 다음: 실물 책상 배치가 Near(앞 모서리 0.30)라면 Near를 유지하되 파지 깊이를 보상으로 직접 잡아야 한다(GRASP_OFFSET를 큐브 윗면 기준으로 올리거나
  tip_floor 기준 높이 상향). 그 전에 항목 하나씩 끄는 ablation(tilt → finger_force → gripper_settle 순)은 NoNear 조건(기준 배치)에서 돌려 91 %와 비교한다.

### 큐브 NoNear+NoTilt ablation (5070, 2026-11-13 00:49, `curriculum_nonear_notilt_5070`, 2048 env, 6000 iter 완주) — tilt 보류
- task `-Curriculum-NoNear-NoTilt-v0`(+`-Play`): NoNear 위에 `tilt`만 off(기울기 15° 초과 벌점 −5 커리큘럼 없음, 기본 `cube_tilt` −2는 유지).
- 학습: 놓기(reach_placed)를 NoNear보다 1000 iter 빨리 배움(0.85 at 1000 vs 0.50 at 2000). 끝 EMA reach_success 0.75(NoNear 0.71), success 0.55/s(0.52).
- 결정적 평가(128 env, model_5999):

| 평가 | NoTilt 5999 | NoNear 5999 | 4000 계보 |
|---|---|---|---|
| 종료 포함 success | **118/128 (92 %)** (held 127 → placed 125 → released 124) | 99/128 (77 %) | 91 % |
| 종료 시 리셋 success | **115/128 (90 %)** | 108/128 (84 %) | |
| 첫 파지 손가락 끝 높이 / 윗면부터 겹침 | 2.0 cm / 7.7 cm | 4.5 cm / 5.0 cm | 2.6 cm |
| 쥘 때 기울기 중앙값 / 최대 | 5.3° / 28° | 5.5° / 28° | |
| 손가락−책상 접촉 env / 최대 힘 중앙값 | 78/128 / 7.1 N | 62/128 / 10.3 N | |
| 운반 시간 중앙값 / 목표까지 xy | 0.9 s / 3.6 cm | 0.8 s / 6.3 cm | |
| NearEnd(x 0.35~0.45, 학습 범위 밖) | 69/128 (54 %) | 61/128 (48 %) | |

- 해석: (1) **tilt 벌점을 빼자 92 %로 4000 계보(91 %)와 같은 급**이 됐다. 기울기는 벌점 유무와 무관하게 중앙값 5°/최대 28°로 같다 → 벌점이
  기울기를 줄이지 못한 채 성공률만 10~15 %p 깎고 있었다. tilt는 **보류**. (2) 파지 깊이 재해석: NoTilt는 2.0 cm로 run 2와 같다. 즉 NoNear의 4.5 cm는
  Near를 뺀 효과가 아니라 "tilt 벌점 + 기준 배치"의 조합이었다(Near + tilt는 2.0, NoNear + tilt는 4.5, NoNear + no tilt는 2.0). 앞 절의 "Near가
  깊은 파지 원인"은 철회한다. 깊은 파지는 이 계보의 기본 전략이고, 성공률과는 무관하게 책상 접촉(78/128)으로 남는다.
- 다음: 남은 항목(finger_force → gripper_settle)을 NoNear+NoTilt 위에서 하나씩 뺀다. finger_force를 빼서 책상 접촉이 늘지 않으면 그것도 보류.

### 큐브 NoNear+NoTilt+NoFinger ablation (5070, 2026-11-13 07:02, `curriculum_nonear_notilt_nofinger_5070`, 2048 env, 6000 iter 완주) — finger_force 유지
- task `-Curriculum-NoNear-NoTilt-NoFinger-v0`(+`-Play`): 손가락−책상 힘 비례 벌점(−2)만 더 off, 기본 `finger_table` −5는 유지.
  기준은 NoTilt 5999(92/90 %, 접촉 78/128, 최대 힘 7.1 N) — 성공률이 유지되고 접촉 env·힘이 늘지 않으면 finger_force도 보류.
- 학습: 곡선은 NoTilt와 거의 같다(끝 200 iter 평균 reach_success EMA 0.70 vs 0.73, success 0.51/s vs 0.53, 기본 `finger_table` 벌점 −0.041/s vs −0.034/s).
- 결정적 평가(128 env, model_5999):

| 평가 | NoFinger 5999 | NoTilt 5999 |
|---|---|---|
| 종료 포함 success | 106/128 (83 %) (held 126 → placed 118 → released 115) | **118/128 (92 %)** |
| 종료 시 리셋 success | 118/128 (92 %) | 115/128 (90 %) |
| 첫 파지 손가락 끝 높이 | 2.4 cm (리셋 2.1) | 2.0 cm |
| 쥘 때 기울기 중앙값 / 최대 | 4.8° / 21° | 5.3° / 28° |
| 손가락−책상 접촉 env / 최대 힘 중앙값 | 73/128 / **11.3 N** (리셋 72/128 / **14.9 N**) | 78/128 / 7.1 N (리셋 79 / 8.5 N) |
| 접촉 시간 중앙값 | 0.2 s | 0.1 s |
| 운반 시간 중앙값 / 목표까지 xy | 0.9 s / 4.8 cm | 0.9 s / 3.6 cm |
| NearEnd(x 0.35~0.45, 학습 범위 밖) | 99/128 (77 %), 접촉 107/128 / 14.7 N | 69/128 (54 %), 접촉 118/128 / 12.0 N |

- 해석: 접촉하는 env 수(73 vs 78)는 그대로지만 **접촉력이 1.6~1.8배**(7.1 → 11.3 N, 리셋 모드 8.5 → 14.9 N)로 늘고 접촉 시간도 2배다. 힘 비례 벌점이
  "닿더라도 세게 누르지 않게" 하는 역할은 하고 있었다. 종료 포함 성공률도 92 → 83 %로 떨어졌다(놓기 뒤 떨어뜨림·넘어뜨림 +7 env; 리셋 모드는 92 %로 같다).
  NearEnd 77 %는 NoTilt 54 %보다 높지만 학습 범위 밖 평가라 한 run 차이로는 결론 안 냄. → **finger_force는 유지**. 실물 책상 접촉 안전(손가락 끝이
  책상을 누르는 힘) 때문에도 남겨 둔다.
- 다음: gripper_settle을 NoNear+NoTilt(finger_force on) 위에서 뺀다.

### 큐브 NoNear+NoTilt+NoSettle ablation (5070, 2026-11-13 10:14, `curriculum_nonear_notilt_nosettle_5070`, 2048 env, 6000 iter 완주) — gripper_settle 유지
- task `-Curriculum-NoNear-NoTilt-NoSettle-v0`(+`-Play`): 그리퍼 result 대기 0.8~1.2 s를 sim 기본 0.3~0.4 s로 되돌림(실물 그리퍼 지연 모사 off). finger_force 유지.
  기준은 NoTilt 5999(92/90 %, 운반 0.9 s). settle이 짧으면 sim 성공률은 오르기 쉬우나 실물 그리퍼(0.8~1.2 s)와 어긋나므로, sim에서 같은 급이면
  실물 지연이 있는 쪽(settle on)을 유지하고, settle on이 성공률을 크게 깎는 경우에만 폭을 좁히는 쪽으로 판단.
- 학습: 2589 iter 시점에 NoTilt와 같은 곡선(mean reward 330 vs 339)이었으나 놓기(reach_placed ≥ 0.5)를 900 iter 늦게 배움(1513 vs 620).
- 결정적 평가(128 env, model_5999):

| 평가 | NoSettle 5999 | NoTilt 5999 |
|---|---|---|
| 종료 포함 success | 111/128 (87 %) (held 123 → placed 115 → released 114) | **118/128 (92 %)** |
| 종료 시 리셋 success | 113/128 (88 %) | 115/128 (90 %) |
| 쥘 때 기울기 중앙값 / 최대 | **12.9° / 41°** | 5.3° / 28° |
| 손가락−책상 접촉 env / 최대 힘 중앙값 | 89/128 / 13.1 N (리셋 101 / 13.9 N) | 78/128 / 7.1 N |
| HELD 중 OPEN 요청 비율 / PLACED 시간 | 21 % / 0.3 s | 74 % / 0.9 s |
| NearEnd(x 0.35~0.45, 학습 범위 밖) | 24/128 (19 %) | 69/128 (54 %) |

- 해석: settle을 짧게 해도 sim 성공률이 오르지 않고 오히려 5 %p 낮다. 기울기가 2배(13°), 책상 접촉도 늘었다. 그리퍼가 빨리 반응하니
  정책이 놓기 전 미리 OPEN을 내는 습관(NoTilt 74 %)을 안 배우고 거칠게 다룬다. 실물 그리퍼 지연과도 어긋나므로 **gripper_settle(0.8~1.2 s) 유지**.
- **ablation 결론(NoNear 기준 순서 tilt → finger_force → gripper_settle)**: 끌 것은 tilt뿐. 최종 조합 = `-Curriculum-NoNear-NoTilt-v0`
  (critic_plus, drag_penalty, finger_force, gripper_settle on; tilt·carry_strict·near_ramp off) = 5999 92/90 %, 4000 계보와 같은 급.
- 남은 질문: Near ramp는 run 2(전부 on, 80 %)가 낮았던 원인인지 tilt와 겹쳐서 못 가린 것인지 모른다. NoTilt 위에 Near만 켠
  `-Curriculum-NoTilt-v0`(Near + tilt off) 한 run이면 Near를 배포에 쓸 수 있는지(x 0.35~0.65) 답이 나온다. 사용자 결정 뒤 진행.

### 큐브 NoTilt (Near on) ablation (5070, 2026-11-13 12:42, `curriculum_notilt_5070`, 2048 env, 6000 iter) — 1250 iter에서 중단(13:14 메모리 부족으로 VSCode째 종료), 보류
- task `-Curriculum-NoTilt-v0`(+`-Play`): 전부 on에서 tilt만 off = NoNear-NoTilt + Near 배치(앞 모서리 0.30)·x ramp(0.45~0.65 → 0.35~0.65, released 0.3 뒤).
  cfg 확인: settle (0.8, 1.2), finger_force on, tilt_excess 없음, near_x ramp 있음, drag on.
- 판단: 기본 범위 평가가 NoNear-NoTilt(92/90 %)와 같은 급이고 NearEnd(x 0.35~0.45)가 54 %보다 뚜렷이 높으면 Near를 최종 조합에 넣는다.
  run 2처럼 80 %대로 떨어지면 Near ramp가 원인으로 확정되고 최종은 NoNear-NoTilt.
