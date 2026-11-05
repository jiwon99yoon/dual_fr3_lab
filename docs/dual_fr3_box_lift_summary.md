# 듀얼 FR3 판 박스 들기: 현황 정리 (2026-11-05)

판(plate) 박스 들기 작업을 잠시 멈추면서, 지금까지 한 것과 배운 것, 실로봇 후보, 다음에 할 일을 한곳에 정리한다.
날짜별 상세 기록은 `docs/dual_fr3_lift_sim2real.md`, 6697 보고서는 `fr3_husky_etri/docs/dual_fr3_boxlift_6697_report.md`에 있다.
이 문서는 MoveIt pregrasp 시작(`-Light-Pregrasp-Wide-*`) 계열이다. ready 자세 시작 계열(`-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-*`, 4090, 실로봇 후보 LMSL 5800)의 sim2real 요소는 [`dual_fr3_box_sim2real_summary.md`](dual_fr3_box_sim2real_summary.md)에 있다.

## 1. 현황

- 실로봇 최고: **SqueezeLift 6697**, 성공 80%(2026-11-05, 시도 수는 실험 기록 참고).
- sim 기준 최고였던 FlatSqueeze 3998은 실물에서 12회 중 11회 실패했다. **sim 지표만으로는 실물 순위를 고를 수 없다.**
- 실물 실패 방식은 정책과 상관없이 같다: 들기 순간 판이 박스 면을 타고 3~8 cm 미끄러지고, 이어서 박스가 잡는 축 둘레로 20~70° 돈다.
  sim에서는 접촉 마찰 0.22~0.25에서 같은 미끄럼이 나온다(학습 범위 0.45~0.85의 절반).
- 다음 실로봇 후보는 6697 계보에서 낮은 마찰로 이어 학습한 LowMuFlat 7200·7100이다(6절, export 완료).

## 2. 문제 설정 (지금 정해진 것)

| 항목 | 값 |
|---|---|
| 박스 | 310(앞뒤) × 410(잡는 방향) × 280 mm, 0.4~0.6 kg(실측 501~561 g) |
| 판 | 각 팔 link7에 140 × 140 × 25 mm, 그리퍼 없음 |
| 역할 분담 | 접근은 MoveIt(시작 자세 표 한 줄로 이동, 판이 면 4 cm 밖, 박스 중심 +5 cm), RL은 조이기 ~ 들기 4 s |
| 시작 자세 표 | `assets/pregrasp/plate_box410wide_pregrasp_gap004_margin005_snapj7.npz` (3956줄, 박스 x 0.40~0.50, y ±0.11, yaw ±6.6°) |
| 목표 | 수직 들기, 박스 중심 base z 0.15~0.35 m (GoalXY는 목표 x 0.40~0.50, y ±0.10) |
| 제어 계약 | 20 Hz, q_target = q_측정 + clip(0.1a, ±0.1 rad), FCI 한계 − 0.05 rad로 자름, 팔 PD 80/4, 87/12 Nm, 1000 Nm/s |
| 관측 / 행동 | 56 (관절각 14, 관절속도 14, 박스 위치 3, 목표 7, 직전 행동 14, 박스 자세 4) / 14 |
| 물리 | Isaac Sim 6.0.1, Isaac Lab 3.0.0b2, dt 0.01, decimation 5, 박스 `max_angular_velocity` 1000 deg/s |
| 배포 | `fr3_husky_etri` `ppo_boxlift_policy_node`(NumPy), NPZ manifest로 계약 검사, `export_boxlift_actor.py` + `pregrasp_start_poses.py` |

## 3. 거쳐 온 길

| 단계 | 바꾼 것 | 배운 것 |
|---|---|---|
| Sim2Real-100Hz (10-22) | reach와 같은 제어 계약 | ±0.1 rad면 PD 토크 상한 8 Nm라 4 kg 박스를 못 든다 |
| Offset03 / Level / PlateOnly (10-23~25) | ±0.3 rad, 수평 계수, 판 외 접촉 벌점, 관절 한계 | 박스를 굴려 들기, 관절 한계 넘기, 커리큘럼 threshold 10은 도달 불가 |
| Light (11-02) | 실물 박스가 1 kg 미만 → ±0.1 rad로 복귀 | 0.5~1 kg은 ±0.1로 충분, 대신 박스를 yaw로 돌려 든다 |
| Pregrasp (11-03) | 접근을 rule-based/MoveIt으로 분리 | 접근 문제가 사라지고 3000 iter에 100% |
| Wide (11-03) | 실물 박스는 긴 변(410)이 좌우 | 박스 치수 축 순서를 확인 없이 정한 실수 |
| Wide-RollAlign 2999 | 판 모서리 정렬 시작 자세 | 판을 V자로 젖혀 아래 모서리로 받친다(25°) |
| `max_angular_velocity` 수정 (11-04) | 30 → 1000 deg/s(단위가 deg/s) | 그 전 정책은 박스가 못 도는 물리에서만 버텼다. 2999는 79%로 무너짐 |
| FaceGrip 3700 / 4699 | 판-면 각도 벌점 | V자는 줄었지만 조임 11 N(안 떨어질 만큼만) |
| MinSqueeze 5698 | 판별 14 N 부족분 벌점 | 조임 16 N. 그래도 다 조이기 전에 든다 |
| **SqueezeLift 6697** | 들기 보상 × 조임 factor, 접근 기울기, 판-박스 미끄럼 벌점 | V자 사라짐, **실물 80%** |
| FaceGrip-BoxDrop 2999 (4090) | 박스 pose 지연·끊김, privileged critic | sim 평행·대칭 최고, 실물에서 미끄러짐 |
| FlatSqueeze 3998 | 더 기운 판 기준 평행, 판 법선 조임 16 N | sim 최고(낮은 마찰 99.8%), **실물 12회 중 11회 실패** |
| LowMuFlat 7100 / 7200 | 6697 + 접촉 마찰 0.3~0.85 | 6697보다 낮은 마찰·약한 Kp에 강함, 7696은 박스 yaw 33°로 붕괴 |
| 실물 bag 분석 | trace + `/joint_states` effort | 조임 토크는 sim과 같음, 부족한 건 마찰 × 법선력 |
| LowMuSoftLift 4500 (3998에서) | 접촉 마찰 0.21~0.64, 들기 가속 1.5 m/s² 초과 벌점, 박스 크기 ±1 cm, yaw drift 벌점 | 미끄럼 줄었지만 오래 학습하면 판을 박스 중심 아래로 내려 잡음(4997) |
| GoalXY 4700 (4500에서) | 목표 x/y를 박스와 따로 뽑기 | 옮기며 들기 98%, 5100 이후 관절 한계에 붙음 |

자주 걸린 함정:

- 이어 학습하면 커리큘럼이 처음부터 다시 돈다. yawtrack 등은 `ramp_steps=1`로 바로 켜야 원래 보상과 같다.
- 성공에 걸린 ramp(조임 factor 등)는 성공이 0이면 영영 안 켜진다(SqueezeLift-B 실패).
- 함수 기본값의 `SceneEntityCfg`는 RewardManager가 resolve하지 않는다. params에 명시한다.
- 오래 학습하면 sim에서만 통하는 자세(낮게 잡기, 관절 한계 붙기, 박스 yaw 돌리기)로 간다. 중간 checkpoint를 같이 평가한다.

## 4. 실로봇 결과

| 정책 | 결과 | 비고 |
|---|---|---|
| FaceGrip 4699 | 들기는 되지만 조임이 빠듯함 | 11-04 |
| MinSqueeze-Scratch 2999 | 그나마 괜찮음 | 11-04 |
| **SqueezeLift 6697** | **성공 80%** | 11-05 |
| FaceGrip-BoxDrop 2999 | 판-박스 미끄럼 | 11-05 |
| FlatSqueeze 3998 | 12회 중 11회 미끄럼 + 회전 | 11-05, `fr3_husky_etri/data/dual_fr3_boxlift_real` |
| LowMuSoftLift 4500·4997, GoalXY 4700·5499 | 기록 필요 | |

실물 측정값(11-05 세션 14개):

- 조임 토크는 sim과 같다. 들고 있을 때 j1 5.7~7.9 Nm, j1·j3 PD 오차 0.08~0.099 rad.
- 지연: 정책 주기 50.0 ms, 관절 상태 나이 < 1 ms, 추론 0.2 ms, 박스 pose stamp 나이 35~90 ms.
- 비전 박스 z가 sim보다 2.8 cm 높다. 노드 `box_position_offset_m: [0, 0, -0.03]`로 보정한다.

## 5. sim2real에서 배운 것

1. **sim 지표로 실물 순위를 못 고른다.** 낮은 마찰, 약한 Kp, 관절 마찰 세 대리 평가 모두 3998 ≥ 6697이었는데 실물은 반대였다.
2. **실물 마찰 × 법선력은 학습 범위의 절반쯤이다.** 판-박스 마찰 실측이 없다. 판 고무 패드(하드웨어)나 낮은 마찰 DR이 필요하다.
3. **Isaac은 판 사이 박스 회전을 MuJoCo보다 2~3배 잘 버틴다.** 미끄러진 뒤의 회전을 sim이 작게 본다. MuJoCo sim2sim을 순위 확인에 같이 쓴다.
4. **6697이 버틴 이유(가설).** 6697은 학습 마찰에서도 들 때 판이 2~4 cm 미끄러져, 판이 박스 중심 위 약 11 cm인 상태로 버티는 법을 배웠다.
   실물에서 미끄러진 뒤의 상태가 학습 분포 안이다. 3998은 학습 중 미끄럼 0.1 cm(판 +5.8 cm)라 그 상태가 처음이다. 또 3998은 실물에서 들기 전부터 j1·j3가 행동 한계에 붙어 더 조일 수 없다.
   "천천히 들어서 덜 미끄러진다"는 sim이 반박했다(마찰 0.25에서 6697 8.8 cm, 3998 4.4 cm).
5. **박스 pose 지연·끊김은 6697도 sim에서 96%로 견딘다.** 관절·명령 지연에는 민감하다(50 ms 85.9%, 90 ms 14.6%). 실물 관절 지연은 1 ms 미만이라 문제가 아니다.

## 6. 실로봇 후보 checkpoint (export 완료)

모두 `fr3_husky_etri/fr3_husky_nn_policy/models/`, 시작 자세 후보는 `fr3_husky_etri/model/…_start_poses.yaml`(박스 A 0.45/0, B 0.42/0, C 0.48/0, D 0.45/+0.05, E 0.45/−0.05).
같은 시작 자세 표를 쓰므로 MoveIt 자세는 어느 정책이나 같다.

| 순서 | npz (`fr3_husky_nn_policy/models/`) | 시작 자세 후보 (`model/`) |
|---|---|---|
| 기준 | `facegrip_squeezelift14_model_6697.npz` | `…_facegrip_squeezelift_model6697_start_poses.yaml` |
| 1 | `facegrip_squeezelift_lowmuflat_model_7200.npz` (11-05 export) | `…_facegrip_squeezelift_lowmuflat_model7200_start_poses.yaml` |
| 2 | `facegrip_squeezelift_lowmuflat_model_7100.npz` (11-05 export) | `…_facegrip_squeezelift_lowmuflat_model7100_start_poses.yaml` |
| 3 | `facegrip_boxdrop_flatsqueeze_lowmusoftlift_goalxy_model_4700.npz` | `…_goalxy_model4700_start_poses.yaml` |
| 3' | `facegrip_boxdrop_flatsqueeze_lowmusoftlift_model_4500.npz` | `…_lowmusoftlift_model4500_start_poses.yaml` |

checkpoint 원본:

- 6697: `logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift/2026-11-04_22-13-23_squeezelift14_from_ms5698/model_6697.pt`
- 7100·7200: `logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_squeezelift_lowmuflat/2026-11-05_01-21-10_lowmuflat_from_sl6697/`
- 4500: `logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatsqueeze_lowmusoftlift/2026-11-05_04-14-11_lowmusoftlift_from_fs3998/`
- 4700: `logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip_boxdrop_flatsqueeze_lowmusoftlift_goalxy/2026-11-05_05-20-47_goalxy_from_sl4500/`

sim 비교 (Isaac, 1024 env, 4 s, DR·외란. 미끄럼은 512 env `sim_slip_probe`):

| | 6697 | LowMuFlat 7200 | LowMuFlat 7100 | GoalXY 4700 | LowMuSoftLift 4500 | FlatSqueeze 3998 |
|---|---|---|---|---|---|---|
| 실물 | **80%** | 아직 | 아직 | 기록 필요 | 기록 필요 | 12회 중 1회 |
| 학습 | SqueezeLift | 6697 + 접촉 마찰 0.3~0.85, 503 iter | 같은 run, 403 iter | 4500 + goal x/y, 200 iter | 3998 + 접촉 마찰 0.21~0.64, 들기 가속 제한, 502 iter | FlatSqueeze |
| 성공 기본 / 낮은 마찰 (접촉 0.3~0.85) | 99.5 / 92.5% | 100 / 99.7% | 100 / 99.9% | 100 / 99.9% | 100 / 100% | 100 / 99.8% |
| Kp × 0.7 / × 0.5 | 82.7 / 6.9% | 98.8 / 49.3% | 98.2 / 51.0% | 99.9 / 88.8% | 100 / 87.7% | 99.7 / 65.1% |
| 박스 pose 지연 0~150 ms + 끊김 | 96.0% | 94.8% | 95.9% | 학습에 포함 | 학습에 포함 | 학습에 포함 |
| 들 때 최대 기울기 median / p90 | 7.8 / 10.5° | 8.2 / 10.5° | 10.2 / 13.2° | 6.6 / 9.8° | 6.3 / 10.8° | 3.6 / 6.0° |
| 학습 마찰: 미끄럼 (0.25→0.6 s) / 0.6 s 판 중심 − 박스 중심 | 2.3 / +10.8 cm | 0.5 / +9.7 cm | 0.4 / +9.2 cm | −0.2 / −1.3 cm | 0.0 / +1.5 cm | 0.1 / +5.8 cm |
| 접촉 마찰 0.25: 미끄럼 median / 최대 기울기 > 20° | 8.8 cm / 50% | 6.3 cm / 17% | 6.4 cm / 22% | 0.5 cm / 4% | 1.4 cm / 7% | 4.4 cm / 25% |

- 7200·7100은 6697의 파지(판이 박스 중심 위 약 10 cm)를 그대로 두고, 낮은 마찰·약한 Kp·접촉 마찰 0.25에서 6697보다 낫다. 실물에서 통한 동작을 유지하므로 다음 후보다.
  7200은 들 때 기울기와 마찰 0.25 기울기가 7100보다 작고, 7100은 goal 정확도(2.3 vs 3.5 cm)가 조금 낫다.
- 4700·4500은 sim에서 미끄럼이 가장 적다. 하지만 실물에서 실패한 3998 계보이고, 판이 박스 중심 높이에서 잡는다. sim 지표가 실물 순위를 못 맞췄으므로(6697 보고서 6절) 7xxx 다음에 본다.
- 쓰지 않는 것: 3998(실물 12회 중 11회 실패), 4997·5499(판을 박스 중심 아래까지 내리고 관절 한계 여유 0.05 rad), LowMuFlat 7696(박스 yaw 33°, export 안 함), 2999·3700·4699·5698(이전 단계).

## 7. 다음에 할 일

확인:

- 80% 실험 세션을 `fr3_husky_etri/data/dual_fr3_boxlift_real/`에 올리고 같은 분석(들기 시작, 미끄럼, 판 높이, 기울기)을 한다.
- MuJoCo sim2sim에서 6697과 3998을 같은 조건으로 돌려, 실물 순위와 맞는지 본다. 맞으면 MuJoCo를 후보 선별에 쓴다.
- 판-박스 마찰을 실측한다(경사 시험 정도로 충분).

학습:

- 6697 계보 유지: LowMuFlat처럼 6697에서 이어 학습하되, 접촉 마찰을 실물 상당(0.2~0.3)까지 내리고 박스 pose 지연을 넣는다.
- critic에 마찰·질량·크기를 준다(LowMuSoftLift는 value loss가 50배, critic이 마찰을 못 봄).
- 판 잡는 높이 유지(박스 중심 위) 벌점, 관절 한계 여유 벌점 강화(오래 학습하면 내려 잡고 한계에 붙음).
- 미끄러진 상태에서 시작하는 에피소드를 넣어 "미끄러진 뒤 버티기"를 직접 학습시킨다(가설 4 검증 겸).

배포:

- 정지 시 hold 모드(지금은 멈추면 조임이 풀려 떨어짐), 낙하 감지 → MoveIt 복귀 → 재시작.

## 8. 주요 파일

| 무엇 | 경로 |
|---|---|
| task cfg | `source/dual_fr3_lab/dual_fr3_lab/tasks/manager_based/lift_plate/joint_pos_env_cfg.py`, `__init__.py`, `agents/rsl_rl_ppo_cfg.py` |
| export | `scripts/tools/export_boxlift_actor.py`, `scripts/tools/pregrasp_start_poses.py` |
| 시작 자세 표 생성 (rule-based) | `scripts/tools/eval_dual_fr3_plate_rule_based.py --save_table` |
| 영상 | `scripts/rsl_rl/record_views.py` |
| 상세 기록 | `docs/dual_fr3_lift_sim2real.md` |
| 배포 노드·실험 절차 | `fr3_husky_etri/fr3_husky_nn_policy/fr3_husky_nn_policy/ppo_boxlift_policy_node.py`, `fr3_husky_etri/docs/dual_fr3_box_experi_step.md` |
| 6697 보고서 | `fr3_husky_etri/docs/dual_fr3_boxlift_6697_report.md` |

평가 스크립트(`box_rl_eval_dr.py`, `sim_slip_probe.py`, `real_slip_table.py` 등)는 아직 저장소 밖(임시 작업 폴더)에 있다. 다시 쓰려면 저장소로 옮겨야 한다.
