# Dual FR3 box-lift: 학습 환경의 sim2real 요소 정리

> 기준 task: `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410-Wide-J1Mirror-FaceGrip-BoxDrop-v0` 계열
> (ready 자세 시작, 바닥 박스 접근부터 들기까지 전부 RL). 갱신 2026-11-05.
>
> 상속: `DualFR3PlateBoxLiftEnvCfg` → V1 → V2 → P39 → P44 → P47 → Sim2Real100Hz → Offset03 → Offset03-Level →
> Offset03-PlateOnly → Light → Light-Margin → Light-Box410 → T6 → J1Mirror → Wide-J1Mirror → FaceGrip-BoxDrop.
> 실험별 경과는 [`dual_fr3_lift_sim2real.md`](dual_fr3_lift_sim2real.md), 공통 원칙은 [`sim2real_gap.md`](sim2real_gap.md).
> MoveIt pregrasp 시작 계열(실로봇 10회 중 8회인 GoalXY 5499 등)의 현황·실로봇 후보는 [`dual_fr3_box_lift_summary.md`](dual_fr3_box_lift_summary.md).

실물에서 통하는 이유는 세 가지로 묶인다.

1. sim의 제어 경로를 실로봇 컨트롤러와 같게 만들었다.
2. 실물에서 생기는 관측·물성 오차를 학습 중에 넣었다.
3. 실로봇이 할 수 없거나 실물에서 깨지기 쉬운 동작을 보상과 한계로 막았다.

## 1. 제어 경로 (Reach Sim2Real v2 계약, `apply_sim2real_control`)

| 항목 | sim 설정 | 실물 대응 / 이유 |
|---|---|---|
| 주기 | 정책 20 Hz, 물리·PD 100 Hz (decimation 5) | 배포 노드와 같은 주기로 행동을 낸다 |
| 행동 | Δq = clip(0.1·a, ±0.1 rad). 정책 시점에 q_target = q(t_k) + Δq를 만들고 다음 정책 step까지 고정 (`reference_mode="policy_step"`) | 배포 노드와 같은 식. 한 step에 0.1 rad 넘게 움직이는 명령이 나올 수 없다 |
| 액추에이터 | explicit PD Kp 80 / Kd 4, 토크 상한 87/12 Nm, 토크 변화율 1000 Nm/s | FR3 토크 제어 한계. sim에서만 가능한 순간 큰 토크를 막는다 |
| 수동 특성 | armature, 관절 마찰, 점성 마찰을 MuJoCo 모델 값과 맞춤 | 실제 관절은 이상적인 강체 관절이 아니다 |
| 관절 순서 | left 1~7, right 1~7 고정 (`preserve_order`) | 배포 노드의 관측·행동 순서와 같다 |
| actor 분포 | tanh-squashed Gaussian (runner cfg) | 출력이 항상 (−1, 1) 안이라 clip에 붙어 생기는 이상 거동이 없다 |

## 2. 관측 오차

| 항목 | 설정 | 실물 대응 |
|---|---|---|
| 관절 위치 / 속도 | Gaussian σ 0.005 rad / 0.05 rad/s | 엔코더, 캘리브레이션 오차 |
| 박스 위치 | σ 1 cm | 비전(ArUco) 위치 추정 오차 |
| 박스 자세 | σ 0.02 (약 2°) | ArUco는 각도 오차가 위치보다 크다 |
| 비전 지연 | 에피소드마다 0~3 step(0~150 ms)을 뽑아 고정 (`BOX_OBS_DELAY_STEPS`) | 카메라·인식 파이프라인 지연 |
| 비전 끊김 | step마다 2%로 0.1~1 s 끊김(마지막 값 유지), 에피소드 10%는 0.05~2 s 시점부터 끝까지 끊김 (`BOX_OBS_*`) | 마커 가림, 인식 실패. 배포 노드는 0.30 s 넘게 끊기면 fault (`box_pose_timeout_s`) |
| 판 힘 관측 없음 | actor 관측에서 plate force 제거 (V1, 58 → 56D) | tau_ext 추정은 노이즈가 크고 payload 보정이 필요하다. 그 의존을 없앴다 |
| 비대칭 critic | critic은 노이즈·지연 없는 실제 박스 pose, actor는 오염된 관측만 (`_add_privileged_critic`) | 학습은 정확한 정보로 안정시키고, 배포되는 actor는 실물과 같은 조건만 본다 |

## 3. 물성 DR과 외란

| 항목 | 범위 | 이유 |
|---|---|---|
| 박스 질량 | 0.4~0.6 kg 절대값 (실측 501~561 g) | 측정 오차 |
| 박스 마찰 | static 0.4~1.2, dynamic 0.3~1.0 | 판-박스 마찰이 파지력의 전부. 판은 PhysX 기본 0.5와 average 결합이라 실제 접촉 마찰은 0.45~0.85 |
| PD 게인 | Kp, Kd 각 ±15% | 실제 게인·관절 마찰 불확실성. 같은 명령에 다른 응답을 피드백으로 보정하게 된다 |
| 박스 스폰 | x 0.40~0.50(박스 중심), y ±0.10, yaw ±10° | 실제 배치가 매번 다르다 |
| goal | base 기준 x 0.40~0.50, y ±0.10 (`UniformPoseCommand`), z 0.15~0.35 | 한 위치에 과적합하지 않는다 |
| 외란 | 0.4~0.8 s마다 박스에 랜덤 힘·토크, 질량에 비례해 축소(약 ±0.6 N, ±0.13 Nm), success EMA 1→3에서 점진 ON | 들고 있는 동안의 흔들림, 모델링 오차 |

cube 진단(2026-07-27): 노이즈만, 물성 DR만 넣으면 떨어뜨리지 않았고 둘을 함께 넣었을 때만 미끄러져 떨어졌다.
그래서 두 축을 항상 같이 학습에 넣는다.

## 4. 실물에서 깨지는 동작 막기

| 항목 | 설정 | 이유 |
|---|---|---|
| 관절 한계 | FR3 datasheet 한계 − 0.05 rad로 sim 한계와 q_target을 자름 + 한계 근처 벌점 | 실로봇 관절 한계 fault 방지 |
| 판으로만 들기 | 판 외 접촉력(link1~6, link7 원통, 베이스)이 10 N이면 들기·goal·success 보상 0 | 팔로 끼워 드는 sim 전용 해를 막는다 |
| 수평 들기 | 들기·goal 보상에 수평 factor(45°에서 0), success는 15° 이내 | 기울어진 파지는 실물에서 미끄러진다 |
| 판 면 평행 (FaceGrip) | `plate_face_tilt` 벌점 | 모서리(V자) 접촉 대신 면 접촉으로 마찰을 안정시킨다 |
| 좌우 대칭 | 판 높이 대칭, j1 거울 대칭(deadband 0.15 rad) | 팔이 한쪽으로 꼬이는 자세 방지 |
| 매끄러움 | action rate, 관절 속도 벌점 | 떨림, 급가속 감소 |
| 시작 상태 | 실물 ready 자세에서 시작 | 학습 시작 분포 = 실물 시작 상태. 접근 단계를 따로 둘 필요가 없다 |

## 5. 근거

- FaceGrip 4699: 실로봇 들기 확인.
- A(FaceGrip-BoxDrop ready 시작) 4999, sim: 비전 1 s 끊김에도 3/3 유지, 들고 있는 동안 판-면 각도 median 2.5° (p90 5.2°).
- FlatSqueeze 3998(다른 PC): 판 평평, 판당 약 18 N.

## 6. 아직 다루지 않은 것

| 항목 | 지금 | 비고 |
|---|---|---|
| 들기 시작 순간 조임 | A 4999: 들기 시작 7~8 N. LMSL 5800: 12 N(약한 판 p10 9.4 N), 대신 판 각도 ~30° | `-FlatSqueeze-LowMuSoftLift` 5800이 낮은 마찰에서 100% (A·B2는 77~80%). 결과는 [`dual_fr3_lift_sim2real.md`](dual_fr3_lift_sim2real.md) "box ready 시작" |
| 박스 크기 오차 | A·B2 계열은 크기 DR 꺼짐 (V2에서 scale 1.0) | `-LowMuSoftLift`부터 ±1 cm |
| 판 재질 마찰 | A·B2 계열은 판 0.5 고정 | `-LowMuSoftLift`부터 판 0.3~0.5, 박스 0.1~0.8 (접촉 0.2~0.65) |
| 관절 상태·명령 지연 | 없음 | `-Wide-RollAlign-DR`에 0~20 ms / 0~30 ms 구현. 2999 평가에서 40+50 ms까지 성공 97% |
| 카메라 고정 바이어스 | 매 step Gaussian만 | 카메라-base 캘리브레이션 오차처럼 한 방향으로 계속 틀어진 오차는 없다 |
| 물리 주기 | 100 Hz | 실제 FR3 토크 루프는 1 kHz. `-Sim2Real1kHz` cfg는 있다 |
