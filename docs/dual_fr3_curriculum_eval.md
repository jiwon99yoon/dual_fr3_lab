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

### 큐브 run 1 (5070, 2026-11-12 14:00, `curriculum_all_5070`, 2048 env, 1380 iter에서 중단) — carry_strict 보류
- iter ~500 reach_held 0.9 (들기 습득, 기울기 벌점 −5 켜짐), 560~680 reach_placed 0.10 → 0.31.
- iter 680 placed EMA 0.3을 넘자 carry_strict·Near x ramp·action/joint_vel/arm_contact 벌점(−0.1/−1)이 같은 iter에 켜짐 →
  40 iter 뒤 placed 0.003, 이후 700 iter 동안 0.000 (held 0.92 유지, 운반 중 기울기 −0.07 → −0.18/s).
- 해석: 그때까지의 placing이 "책상 위로 끌어서" 놓는 동작이라, 단계를 박탈하는 carry_strict가 켜지자 그 동작 전체가 stage 0이 되고
  정책은 공중에 든 채 버티는 국소해로 갔다. action 벌점 latch는 4000 계보와 같으므로 원인에서 제외.
- 조치(run 2): carry_strict → 끌기 벌점 `cube_drag` −5/s(placed 0.3 뒤)로 대체, Near ramp는 reach_released 0.3 뒤로 미룸.
