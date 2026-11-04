# Dual FR3 Lift Sim2Real (cube / box)

`Isaac-Lift-Cube-DualFR3-Sim2Real-*-v0`와 `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-*-v0`는
cube v3와 box P47의 observation·reward·물체 DR을 유지하면서, 팔 제어 경로를
[Reach Sim2Real v2](dual_fr3_reach_v2.md)와 같은 계약으로 바꾼 재학습용 환경이다 (2026-10-22).

`*-Friction` 버전(cube `Friction-v3`, box `P47-Friction-v0`)은 이 중 관절 passive dynamics만
맞춘 것이고, implicit PD·무제한 action은 그대로다. 실로봇 배포를 목표로 하면 이 Sim2Real 버전을 쓴다.

## Reach v2 계약 대비 비교

| 항목 | Reach Sim2Real v2 | cube v3 / box P47 | **lift Sim2Real** |
|---|---|---|---|
| 팔 액추에이터 | explicit PD Kp80/Kd4 | implicit PD Kp80/Kd4 | explicit PD Kp80/Kd4 |
| 토크 한계 / 변화율 | 87·12 Nm / 1000 Nm/s | 87·12 Nm / 없음 | 87·12 Nm / 1000 Nm/s |
| passive dynamics | armature 0.1, friction 0.2, viscous 0.003 | USD friction 0.2만 | Reach와 동일 |
| 물리·PD 주기 | 1 kHz | 100 Hz | 1 kHz (`-1kHz`) / 100 Hz (`-100Hz`) |
| 정책 주기 | 20 Hz | cube 20 Hz / box **50 Hz** | 20 Hz |
| actor | tanh-squashed Gaussian, `mean_bound=5` | 무제한 Gaussian | Reach와 동일 |
| 팔 offset | `clip(0.1·a, ±0.1 rad)` | `0.1·a`, 제한 없음 | Reach와 동일 |
| target 기준 | `--reach_policy_step_action` 시 policy-step | physics-step | **policy-step 고정** |
| PD 게인 DR | ±15% | cube 없음 / box ±15% | ±15% |
| box 관절 순서 | left1~7, right1~7 | articulation 순서(좌우 교차) | left1~7, right1~7 |

액추에이터 값은 `robots/dual_fr3.py`의 `with_torque_rate_limited_arms()` 한 곳에서 정의하며,
Reach Sim2Real robot cfg도 같은 함수로 만든다(리팩터 전후 cfg 동일 확인).

policy-step target은 정책 시점에 한 번 만든 절대 목표를 decimation 동안 유지한다.

```text
q_target[k] = q_measured(t_k) + clip(0.1 * a_k, -0.1, 0.1)   # 50 ms 고정
tau = Kp (q_target - q) - Kd qdot  -> 87/12 Nm saturation -> 1000 Nm/s rate limit   # 매 physics step
```

현재 Reach v2 학습(`--reach_policy_step_action`)과 같은 의미다. 이 lift task들은 플래그 없이
cfg에 고정되어 있으므로 train/play에 플래그를 줄 필요가 없다.

## Task별 차이

**cube** (`DualFR3CubeLiftEnvCfgSim2Real{1kHz,100Hz}`, base v3)
- obs 36D / action 8D(오른팔 7 + 그리퍼 1) 순서는 v3 그대로라 배포 obs builder를 재사용할 수 있다.
- 그리퍼는 implicit PD 그대로다. binary action이라 tanh 출력의 부호만 쓴다(`a<0` close).
- episode 8 s(160 step).

**box** (`DualFR3PlateBoxLiftEnvCfg_Sim2Real{1kHz,100Hz}`, base P47)
- 정책 50 Hz -> 20 Hz. episode 5 s는 유지하므로 100 step이다.
- success 진입 조건 0.5 s 유지: `hold_steps` 25 -> 10. 다른 reward는 RewardManager가 dt를 곱하고,
  커리큘럼은 EMA/iteration 기준이라 주기와 무관하다.
- action과 joint obs 순서를 left1~7, right1~7로 고정했다(P47 checkpoint와 obs 순서가 다름).

## 주의

- 제어 경로를 한 번에 여러 축 바꿨다(PD 종류·주기·action bound·target 기준, box는 정책 주기까지).
  기존 lift 문서의 "한 run에 한 축" 원칙과 다르므로, 실패하면 `-100Hz`와 비교해 물리 주기 영향을 먼저 분리한다.
- box는 정책 주기가 바뀌어 P47 보상 튜닝이 20 Hz에서도 통하는지 학습으로 확인해야 한다.
- 모든 v3/P47 checkpoint와 호환되지 않는다. from-scratch 학습만 가능하다.
- actor가 tanh 분포라 rsl_rl 진입점만 등록했다(skrl/rl_games/sb3 cfg는 무제한 Gaussian용).
- **box는 2026-11-04 전까지 박스가 30°/s보다 빨리 못 도는 물리였다**(`max_angular_velocity=30`, 단위 deg/s).
  그 전 box 정책과 평가는 모두 이 제한 아래에서 나왔다. 아래 "박스 회전 제한 버그" 절.

## 배포 측 남은 작업 (`fr3_husky_etri`)

- `ppo_liftcube_policy_node`는 아직 v3용 예전 경로다(`isaac_relative_control=False` + velocity/acceleration
  smoothing, `|0.1·a| > 0.15 rad`이면 fault). Reach 노드처럼 `isaac_relative_control=True` + policy-step
  target 경로로 바꿔야 학습 계약과 맞는다. tanh actor라 `|0.1·a| < 0.1`이므로 fault 조건에는 걸리지 않는다.
- export는 `export_ppo_actor.py <ckpt> <out.npz> --output-activation tanh`로 한다.
- box는 배포 노드가 아직 없다. 만들 때 아래 "box action bound ±0.3 rad" 체크리스트를 반영한다.

## box action bound ±0.3 rad (`-Offset03`, 2026-10-23)

±0.1 rad 계약의 box(`Sim2Real-100Hz`)는 10000 iter 동안 한 번도 들지 못했다(`lifting_object` 0,
reach+squeeze만 학습, tanh 포화로 entropy −8). measured-relative PD에서 판이 박스에 막힌 관절의 최대
토크는 `Kp·Δq = 80 × 0.1 = 8 Nm`이라 4 kg 박스를 마찰로 잡고 들기에 부족하다. FR3 하드웨어 한계가
아니라 계약에서 나오는 상한이다. cube는 그리퍼가 파지력을 내므로 해당 없다.

A/B로 확인했다. offset만 ±0.3 rad로 바꾼 `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-v0`는
139 iter에 lift를 시작했고 1000 iter에서 `lifting_object` 10.2였다. 같은 iter의 P47(Isaac 5.0)은 8.1이었다.
로그는 `dual_fr3_plate_lift_sim2real_100hz_offset03`에 있다. tanh 출력이 (-1, 1)이라 clip만 넓히면 효과가
없으므로 scale과 clip을 둘 다 0.3으로 바꿨다.

```text
q_target[k] = q_measured(t_k) + clip(0.3 * a_k, -0.3, 0.3)   # 나머지(Kp80/Kd4, 87·12 Nm, 1000 Nm/s, 20 Hz)는 동일
```

**box 배포 노드 체크리스트.** 제어기 구조와 게인은 그대로 두고 파라미터만 맞춘다.

- [ ] action scale `0.1` -> **`0.3`**, clip `±0.1` -> **`±0.3 rad`**. 0.1을 그대로 두면 모든 동작이 1/3로 줄어 정책이 동작하지 않는다.
- [ ] `|Δq|` fault 기준(cube 노드는 `0.15 rad`) -> **약 `0.35 rad`**. 그대로 두면 0.15 rad를 넘는 정상 action에서 정지한다.
- [ ] libfranka collision behavior 임계값을 확인한다. squeeze 시 joint1~4에 약 25 Nm 외력이 걸린다. joint5~7은 12 Nm로 포화된다. 임계값이 이보다 낮으면 reflex로 정지한다.
- [ ] 첫 실로봇 테스트는 접촉 토크가 8 Nm에서 24 Nm으로 커진 점을 감안해 저속·보호 조건에서 한다.
- [ ] 관절 순서는 left1~7, right1~7이다. 나머지 obs와 export(`--output-activation tanh`)는 위와 같다.

Reach와 cube는 ±0.1 rad 계약 그대로다. 이 변경은 box에만 해당한다.

## box 수평 들기 + FR3 관절 한계 (`-Offset03-Level`, 2026-10-24)

Offset03 `model_2000`을 `scripts/rsl_rl/record_views.py`로 녹화해 보니 박스를 굴려서 들었다. 들고 있는 동안
기울기는 평균 46~57°(최대 105°)였고, 왼판이 오른판보다 약 29 cm 낮았다. 원인은 두 가지로 봤다.

- **보상**: `lifting_object`, `lift_progress`, `object_goal_tracking`에 기울기 조건이 없다. 기울기 조건은 15° 이내를
  요구하는 `success`에만 있어서, `box_upright`(−3) 페널티를 받더라도 기울여서 드는 쪽이 이득이었다.
- **관절 한계**: sim 관절 한계(URDF 값)가 FR3 datasheet 한계보다 넓다. 정책의 왼팔 j1·j2는 env-step의 약 80%에서
  datasheet 한계를 넘어 sim 한계에 붙어 있었다(j6은 46%). 오른팔은 한계 안에 있었다. 실로봇이었다면 한계 위반으로
  정지했을 자세다. 한계 구속력이 24 Nm 상한과 무관하게 힘을 내므로 비대칭 들기와도 관련이 있을 수 있다(미확인).

| 관절 | sim(URDF) | FR3 datasheet → 정정: FCI 문서 한계 |
|---|---|---|
| j1 | ±2.9007 | ±2.7437 |
| j2 | ±1.8361 | ±1.7837 |
| j4 | −3.0770 ~ −0.1169 | −3.0421 ~ −0.1518 |
| j5 | ±2.8763 | ±2.8065 |
| j6 | 0.4398 ~ 4.6216 | 0.5445 ~ 4.5169 |
| j7 | ±3.0508 | ±3.0159 |

> **2026-11-03 정정**: 오른쪽 열을 "datasheet"라고 불렀지만 틀렸다.
> - 제품 datasheet(R02212 2.6, 2026-07)의 관절 위치 한계는 A1 ±166°, A2 ±105°, A3 ±166°, A4 −176/−7°, A5 ±165°, A6 25/265°, A7 ±175°다. 이는 왼쪽 열(franka_description URDF = sim USD)과 같다.
> - 오른쪽 열은 Franka FCI 문서 "Robot and interface specifications"의 FR3 q_min/q_max이고, datasheet보다 좁다.
> - 배포 노드(`policy_control_action_server.hpp`, `ppo_*_policy_node.py`)는 왼쪽 열 − 0.02 rad로 검사한다.
> - 실로봇이 어느 값에서 멈추는지는 실물에서 확인하지 않았다. 위의 "실로봇이었다면 정지했을 자세"도 확인되지 않은 주장이다.
> - 코드 이름(`FR3_JOINT_POS_LIMITS`, `set_fr3_datasheet_joint_limits`)은 그대로 두고 주석만 고쳤다. 둘 중 좁은 쪽으로 자르는 설정은 유지한다.

물리적으로 수평 들기가 가능한지는 따로 확인했다. 판 마찰은 μ 0.5이고 PhysX가 두 마찰을 평균하므로 실제 마찰은
μ_eff = (μ_box + 0.5)/2다. 스크립트로 판을 수평으로 대고 조였을 때(판당 약 44 N, 관절의 7%만 0.3 rad 상한 도달)
3.5 kg은 μ_box 0.3까지, 4.5 kg은 μ_box 0.6 이상에서 수평으로 들렸다. 4.5 kg·μ_box ≤ 0.45만 미끄러졌다.
성공 여부는 판당 필요 힘 `m·g / (2·μ_eff)`와 일치했으므로 기울여 들기는 대부분 보상 문제다.

`Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-Level-v0`(Offset03 기반 from-scratch)의 변경점:

- 위 세 보상과 `object_goal_tracking_fine_grained`에 수평 factor `clamp(1 − box_upright / 0.293, 0, 1)`을 곱한다.
  수평이면 1이고 45°에서 0이 된다. 발견 단계에서 약간 기울이는 동작은 부분 보상으로 남긴다.
- `fr3_joint_limit_margin`(weight −1, 처음부터 적용): datasheet 한계까지 거리가 0.1 rad 미만이면
  `((0.1 − 거리)/0.1)²`를 양팔 14관절에 대해 합한다. 관절 공간 대부분에서는 0이다.
- 같은 `model_2000`을 새 보상으로 채점하면 `lifting_object` 10.65 → 1.57, `lift_progress` 5.88 → 0.99,
  `object_goal_tracking` 3.19 → 0.41이 되고 관절 한계 페널티가 −9.7/s다.

한 run에서 두 축을 바꿨으므로, 실패하면 수평 gate와 관절 한계를 나눠서 확인한다. 로그는
`dual_fr3_plate_lift_sim2real_100hz_offset03_level`에 있다.

배포 체크리스트 추가 항목: 배포할 정책은 datasheet 한계 안에서 동작해야 한다. 배포 전에 play로 관절 범위를 확인한다.

**결과(10000 iter).** 들기는 약 300 iter에 시작했다. 영상에서 들고 있는 동안 기울기는 평균 7~12°였다. 다만
`lifting_object`에 수평 factor를 곱하자 "들기 습득" 판정(`lifting_object` 초당 EMA > 10)이 끝까지 넘지 않았다
(최대 약 8.7). 그래서 action_rate·대칭·reaching 강화 커리큘럼이 하나도 켜지지 않았다. success는 4000 iter에
2.3까지 올랐다가 0.02로 떨어졌다. 관절 한계 페널티로 j1은 해결됐지만 왼팔 j7은 env-step의 32%에서 USD 한계에
붙어 있었다. 참고로 수평 조건 없이 이어 학습한 Offset03(`..._resume2050/model_9999`)도 기울기는 평균 10~13°로
줄었지만, 왼팔 j1(74%)·j7(59%)이 datasheet 한계를 넘었다.

## box 판으로만 들기 + 관절 한계 강제 (`-Offset03-PlateOnly`, 2026-10-24)

판 외 부위가 박스에 닿는 비율을 따로 쟀다(64 env × 300 step, 박스가 5 cm 이상 들린 스텝에서 1 N 초과). 측정
스크립트는 robot body마다 박스 filter 접촉 센서를 붙인다. link7은 판 collider와 FR3 link7 원통 collider가 같은
body라서 접촉점 단위로 나눈다. 판은 link7 z 0.107~0.132, 원통은 z 0.050~0.105에 있다. 판은 link7 z축 기준으로
45° 돌아가 있어서 모서리가 x/y축 위 0.099 m에 온다. 처음에 이 점을 놓쳐 판 모서리 접촉을 원통 접촉으로 셌다.

| 정책 | 왼팔 link2 | 왼팔 link4 | 왼팔 link5 | 베이스 | link7 원통 |
|---|---:|---:|---:|---:|---:|
| Offset03 resume `model_9999` | 5% | 0% | 0% | 13% | 0% |
| Level `model_4500` | 0% | 14% | 2% | 1% | 0% |
| Level `model_9999` | 0% | 0% | 11% | 2% | 0.1% |

기존 `undesired_contact_penalty`(link1~6, 개수)는 −1e-4로 시작해 들기 습득 후 −1로 커지므로 들기 보상(15/s)을
이기지 못했다. 베이스-박스 접촉은 아예 보지 않았다.

`Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Offset03-PlateOnly-v0`(Level 기반 from-scratch)의 변경점:

- **판으로만 들기**: `lifting_object`, `lift_progress`, `object_goal_tracking`(+fine), `success`에 판 전용 factor
  `clamp(1 − F_nonplate / 10 N, 0, 1)`을 곱한다. `F_nonplate`는 양팔 link1~6 net 접촉력(상대 무관), 베이스-박스,
  link7 원통-박스 접촉력의 합이다. 수평 factor는 Level 그대로다. `undesired_contact_penalty`는 같은 출처를
  세는 `nonplate_contact_count`로 바꿨다(커리큘럼 −1e-4 → −1은 그대로).
- **관절 한계 강제**: startup event로 sim 관절 한계를 datasheet 값으로 좁히고, PD 목표 `q_target`도 같은
  한계로 자른다(`ReachRelativeJointPositionActionCfg.target_limits`, 기본 None이라 다른 task는 그대로).
  `fr3_joint_limit_margin`(−1)은 유지한다.
- **커리큘럼 발동 수정**: gate가 없는 `lift_monitor` 항(weight 1e-3)을 두고 페널티 커리큘럼이 이 항으로
  "들기 습득"을 판정한다. threshold는 같은 의미(들린 시간 약 67%)가 되도록 1e-3/15로 줄였다.

배포 체크리스트 추가 항목:

- [ ] 배포 노드에서 `q_target`을 FR3 datasheet 관절 한계(위 표)로 자른다. 학습과 같은 clamp다.
- [ ] 판 외 접촉 측정(위 표와 같은 방식)으로 배포할 checkpoint를 확인한다.

로그는 `dual_fr3_plate_lift_sim2real_100hz_offset03_plateonly`, stdout은 repo 루트 `box_plateonly.log`.

결과(10000 iter, `2026-10-24_21-11-54`, 4시간 35분). 접촉은 같은 측정(64 env × 300 step)이고 영상은 `videos/` 3 에피소드다.

| | 판 외 접촉 최대(들린 스텝) | 판 factor | 들린 동안 기울기 평균/최대 | 판 높이차 평균 | play success |
|---|---|---:|---|---:|---:|
| Level `model_9999` | 왼팔 link5 11% | 0.91 | — | — | — |
| PlateOnly `model_3050` | 베이스 1.5%, 왼팔 link5 1.4% | 0.98 | 6~11° / 14~21° | 6~12 cm | 6.5 |
| PlateOnly `model_9999` | 오른팔 link5 2.6%, 베이스 1.8% | 0.96 | 9~14° / 20~28° | 11~13 cm | 3.1 |

- 판 외 접촉과 관절 한계 문제는 해결했다. link7 원통 접촉은 0이고, 관절이 datasheet 한계를 넘은 스텝도 0이다.
- 커리큘럼은 이번에도 발동하지 않았다. gate가 없는 `lift_monitor`도 최대 6.58e-4(lifting 단위 9.87)라
  threshold 6.667e-4(=10)를 넘지 못했다. 그래서 action_rate·판 높이 대칭·box_upright 강화가 끝까지 꺼져
  있었고, 판 높이가 10 cm 이상 어긋난 대각선 파지와 기울기가 남았다.
- 학습 중 success는 3000 iter에 3.0으로 가장 높았고 이후 0.3~2.2 사이를 오갔다. 지금 후보는 `model_3050`이다.

이어 학습(2026-10-25): 커리큘럼 기준을 `LIFT_LEARNED_THRESHOLD = 8`(lifting 단위)로 낮추고 `model_3050`에서
3000 iter를 이어 학습한다(run `2026-10-25_*_resume3050_th8`, stdout `box_plateonly_resume.log`). 페널티 커리큘럼은
약 20 iter 만에 모두 켜졌다. disturbance·qprime은 바로 1이 됐고, yawtrack은 0부터 다시 3000 iter에 걸쳐 1까지 올라간다.

이어 학습 결과 (run `2026-10-25_19-14-27_resume3050_th8`, 6050 iter에서 끝남, Traceback 0):

- 학습 중 success(250 iter 평균)는 3.98(3000대) → 2.0(4000대) → 0.4(5000~6000대)로 떨어졌고 회복하지 않았다.
- lifting_object는 8.2를 유지했다. 들기는 계속하지만 success 조건(15° 이내·goal·판 전용)을 못 채운다.

평가는 PlateOnly-Play에서 했다. 커리큘럼은 처음부터 다시 시작하므로 모든 checkpoint를 같은 조건에서 비교한다.

| ckpt | 판 외 접촉 최대 (들린 스텝 비율) | factor | play success | 기울기 평균/최대 (영상 3 ep) | 판 높이차 \|dz\| 평균 |
|---|---|---|---|---|---|
| 첫 run 3050 | base 1.5%, L link5 1.4% | 0.98 | 6.5 | 6~11° / 14~21° | 6~12 cm |
| resume 3150 | L link5 2.9%, base 2.0% | 0.96 | 5.5 | 6~11° / 14~23° | 4.7~9.4 cm |
| resume 6049 | L link5 5.0%, base 1.9%, L link2 1.5%, R link1 1.3% | 0.92 | 0.6 | 7~15° / 14~22° | 3.7~7.4 cm |

- 페널티가 켜진 뒤 판 높이차(대각선 파지)는 줄었다(6~12 → 3.7~7.4 cm).
- 반면 판 외 접촉은 늘었고 success는 무너졌다. 기울기는 비슷하다.
- 배포 후보는 여전히 첫 run `model_3050`이다. 다음 수정은 페널티를 한꺼번에 latch하지 않고 하나씩 켜거나, success가 떨어질 때 강도를 되돌리는 방식이 필요해 보인다(추측).

## box 가벼운 실물 박스 + ±0.1 rad (`-Light`, 2026-11-02)

사용자 확인: 실물 박스는 **1 kg 미만**이다. 7/29에 정한 4 kg(`V1_BOX_MASS`)은 실물과 맞지 않는다.
±0.1 rad로 못 들었던 원인은 4 kg이었으므로, 가벼운 박스라면 reach·cube와 같은 배포 기준(`max_policy_target_delta_rad`
0.11)을 그대로 쓸 수 있는지 확인했다.

**조임-들기 시험** (scratchpad `squeeze_test3.py`, CPU 물리):
- 정책은 `Sim2Real-100Hz` model_9999다(±0.1, 들기는 못 배우고 판을 박스에 대는 것까지만 배움).
- 판이 둘 다 닿으면 스크립트로 넘긴다. 스크립트는 판을 박스 중간 높이(0.15 m)로 옮긴 다음 5 cm 더 밀어 조이고, 20 cm 들어 올려 2초 들고 있는다.
- 처음에는 정책이 판을 댄 높이(박스 윗면 위, z 0.29 m) 그대로 들었다. 이때는 0.5 kg도 모서리로 들리다가 돌아 빠졌다.

| 박스 질량 | 판당 조임 | 결과 (마찰 0.4 / 0.6 / 0.8 / 1.0, gain 2~4) |
|---|---|---|
| 0.5 kg | 16~20 N | 0.4~1.0 전부 12~16 cm 들림, 기울기 2.5~9° |
| 1.0 kg | 16~20 N | 0.4는 실패, 0.6 이상은 12~14 cm 들림 |
| 1.5~2.0 kg | 16~21 N | 0.8 이상에서 3~7 cm까지만 들리고 미끄러짐 |
| 4.0 kg | 16~21 N | 전부 0 |

- 조임은 ±0.1 rad(관절당 8 Nm)에서 판당 약 16~22 N이 한계다. 필요한 조임 m·g/(2·μ_eff)와 맞는다.
- gain 8(action이 clip에 60% 넘게 붙음)은 무게와 상관없이 전부 실패했다.

**task**: `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light[-Play]-v0`, 로그 `dual_fr3_plate_lift_sim2real_100hz_light`
- PlateOnly 그대로다: 판으로만 들기, datasheet 관절 한계, 커리큘럼 threshold 8.
- action scale/clip을 0.1 / ±0.1로 되돌렸다.
- 박스 질량은 명목 0.75 kg이고 DR은 0.5~1.0 kg 절대값이다(`LIGHT_BOX_MASS[_RANGE]`). 실측하면 좁힌다.
- 외란은 질량 비율만큼 줄였다(±0.94 N / ±0.19 Nm). 4 kg 기준 ±5 N / ±1 Nm와 같은 가속도다.
- probe: 질량 DR 0.51~0.88 kg, 10 step 랜덤 action에서 보상이 유한하다.
- 학습: 핸드오버 v1 두 run이 끝나면 6000 iter로 자동 시작한다(scratchpad `queue_box_light.sh`, stdout은 repo 루트 `box_light.log`).

## box rule-based 조임-들기 (`-Light` 환경, 2026-11-02)

RL 대신 스크립트로 접근부터 들기까지 전부 할 수 있는지 시험했다. 도구는 `scripts/tools/eval_dual_fr3_plate_rule_based.py`다.

- 8월 데모(`run_dual_fr3_plate_lift.py`)를 Light 환경의 배포 계약으로 다시 짰다.
  - 팔 명령은 env의 arm action을 그대로 거친다: 20 Hz, `q_target = q_measured + clip(0.1·a, ±0.1 rad)`, explicit PD, datasheet 한계.
  - 박스 질량·마찰·위치·yaw DR과 PD 게인 DR도 그대로 켠다.
  - 쓰는 정보는 박스 pose, 박스 크기, goal 높이뿐이다(정책 obs와 같다).
- 순서
  1. 판을 박스 y면 8 cm 바깥, 윗면 10 cm 위로 보낸다.
  2. 잡는 높이(박스 중심 + 5 cm)로 내린다.
  3. 면 안쪽 5 cm까지 기준을 밀어 넣는다. 첫 접촉 전까지는 박스 pose를 계속 따라간다.
  4. 0.3 s 조인 뒤 0.15 m/s로 goal 높이까지 든다.
- 제어: 기준 pose가 제한 속도로 움직이고, DLS IK(`2·J⁺e`)로 따라간다.
  - 관절 step은 벡터 전체를 같은 비율로 줄여 ±0.1 안에 넣는다. 관절마다 따로 자르면 IK 방향이 틀어져 팔이 엉뚱하게 움직였다.
  - 판 법선축 회전은 접근 전까지만 자유롭게 둔다.
  - 잡은 뒤에도 자유롭게 두면 그 축이 박스 pitch 축이라 박스가 15~26° 기울었다. 고정하자 기울기 중앙값이 3.4°가 됐다.

**결과** (CPU, 128 env, success = env의 success 항: goal 10 cm, 기울기 15°, 양판 접촉, 0.5 s 유지)

| 조건 | success | 비고 |
|---|---|---|
| 박스 x 0.45~0.60 (env 기본), 외란 0 | 86.7% | 잡는 높이 +5 cm |
| 같은 조건, 잡는 높이 +6.5 cm | 81.2% | |
| 같은 조건, 들기 0.25 m/s | 63.3% | 빨리 들면 기울고 미끄러짐 |
| 박스 x 0.45~0.60, 외란 1.0 | 84.4% | x < 0.50: 100%, 0.50~0.55: 91%, ≥ 0.55: 64% |
| **박스 x 0.45~0.52, 외란 1.0** | **98.4%** | 판당 조임 중앙값 16.5/17.1 N, 들고 있을 때 기울기 중앙값 3.6° (p90 6.2°) |
| 박스 1~2 kg, x 0.45~0.60 | 23.4% | ±0.1의 조임 한계(판당 약 16 N) |

- **실패는 거의 다 먼 박스에서 난다. 작업공간 한계가 아니라 이 스크립트의 한계다.**
  - 박스가 멀면(x ≥ 0.55) 판을 잡는 높이까지 못 내린다. j2가 FR3 한계(1.78 rad)에 붙고 DOWN이 수렴하지 않는다.
  - 그러면 판이 윗모서리 근처를 잡고, 조임이 약해져 미끄러진다.
  - null-space 자세를 바꿔도(ready 자세로 당기기 끄기, 한계 회피 3배) 64~77%에 그쳤다.
  - 처음에는 "x 0.6에서 중간 높이를 잡으려면 어깨~손목 0.75 m 이상 필요, FR3는 약 0.72 m"라는 어림으로 작업공간 밖이라고 결론냈다. **틀렸다.** 같은 계약의 Light RL 정책은 x ≥ 0.55 박스를 89~99% 든다(아래 "Light RL 결과" 절). RL도 j2를 한계에 붙여 쓰지만 다른 관절로 보완한다.
- **실물 적용 시 주의**
  - 박스를 로봇 기준 x ≤ 약 0.52 m에 두거나 Husky를 그만큼 접근시켜야 한다.
  - 닿는 범위 안에서도 env의 약 30%에서 j2가 datasheet 한계에 닿는다. 실물에서는 한계 여유(예: 0.05 rad)를 강제해야 하고, 그만큼 판이 높아진다.
  - 걸리는 시간은 success까지 중앙값 8.0 s다. RL의 에피소드 5 s보다 길다. 들기 속도를 0.15 m/s로 둔 것이 크다.
- **배포 경로(안)**
  - 새 서버 없이 `ppo_liftcube_policy_node.py`처럼 20 Hz로 관절·박스 pose를 받아 `RunPolicyControl`에 Δq를 보내면 된다.
  - 신경망 대신 이 스크립트의 IK 단계를 넣는다. FK/Jacobian은 Python에서 필요하다.
  - Δq가 ±0.1 안이라 배포 기준 0.11을 그대로 쓴다.

## box Light RL 결과와 rule-based 비교 (2026-11-02)

run `dual_fr3_plate_lift_sim2real_100hz_light/2026-11-02_15-17-33_light`, 6000 iter(15:17~19:51).
- TB: `Episode_Reward/success`(sustained라 성공률 아님)는 1800·2500 근처가 최고(약 2.9)이고 최종은 약 1.8이다.
- 페널티 커리큘럼(threshold 8)은 이번에도 발동하지 않았다. action_rate·joint_vel·대칭 항이 0인 채로 끝났다.
- entropy는 −8까지 내려갔다(정책 std 0.32).

**평가** (scratchpad `box_rl_eval.py`, GPU, 1024 env, 외란 1.0, 관측 노이즈 끔, 결정적 정책, 5 s = 학습 에피소드)
- success 판정은 rule-based와 같다(env success 항, 0.5 s 유지). rule-based도 같은 조건으로 GPU 1024 env에서 다시 돌렸다.

| | x 0.45~0.60 (env 기본) | x 0.45~0.52 | x ≥ 0.55 | success까지 | 들린 동안 최대 기울기 중앙값 | 판 외 접촉 > 1 N |
|---|---|---|---|---|---|---|
| RL model_5999 | 93.4~96.4% | 98.1% | 89~94% | 3.8 s | 12.4° | 5% |
| RL model_2500 | 98.6% | 97.8% | 99% | 3.75 s | 9.6° | 56% (> 10 N 27%) |
| RL model_1800 | 95.8% | 96.1% | 97% | 3.75 s | 11.9° | 64% |
| rule-based | 78.4% | 98.2% | 43% | 8.2 s | 4.1° | 4% |

- 5999는 같은 조건 두 번(시드 다름)에 96.4%와 93.4%가 나왔다.
- 닿는 범위(x ≤ 0.52)에서는 RL과 rule-based가 비슷하다. 먼 박스는 RL만 든다.
- RL은 rule-based보다 빠르지만 더 기운다(기울기 중앙값 10~14° vs 3.5~4°). 판 높이차는 5999가 4.7 cm, 2500이 2.4 cm다.
- 판 외 접촉은 5999가 가장 적다. 2500·1800은 팔 링크가 박스에 자주 닿는다.
- **관절 한계**: 셋 다 j2를 datasheet 상한에 붙여 쓴다. 5999는 env의 62~75%가 한 번 이상 붙고, 에피소드 시간의 3~9%를 붙어 있다. rule-based도 j2가 한계에 붙는다(29~49%). 실물에서는 j2 한계 여유를 두고 다시 확인해야 한다.
- 판 접촉력은 에피소드 중 최대값이 판당 31~41 N이지만 순간값이다. 들고 있는 끝 시점에는 판당 약 13~14 N으로 rule-based(17 N)와 비슷하다. 조임 한계(판당 16~22 N)는 RL에서도 그대로다.
- **박스를 수직축으로 돌려서 든다.** 끝 시점의 박스 yaw가 5999는 중앙값 38°(p90 46°), 2500은 27°다. 시작 yaw는 약 3°다. 영상에서도 박스가 비스듬히 돌아간 채 몸쪽으로 들려 있다.
  - P31~P38 때도 있던 문제다(24~37°). 지금 success에 곱하는 yaw 계수(`yaw_alpha` 0.5)는 38°에서도 보상을 절반만 깎으니 막기에 약하다.
  - rule-based는 판 자세를 박스에 맞춰 고정해서 돌리지 않는다.
- 영상(각 4 에피소드, seed 0): `<run>/videos/model_5999_views_ep4.mp4`, `model_2500_views_ep4.mp4`.
- 배포 후보는 판 외 접촉이 적은 model_5999다. 기울기·판 높이차·yaw를 줄이려면 커리큘럼 threshold를 더 낮추거나 latch 상태로 이어 학습해야 한다.

## box 관절 한계 여유 (`-Light-Margin`, 2026-11-03)

`Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Margin[-Play]-v0`: Light와 같고, 관절 한계 세 곳을 FCI 한계보다 `FR3_JOINT_LIMIT_INSET`(0.05 rad) 안쪽으로 옮겼다.
- 세 곳은 sim 관절 한계(startup event), PD 목표 clamp(`target_limits`), 한계 페널티(`fr3_joint_limit_margin`)다.
- j2 상한은 1.7337이 된다. URDF/datasheet 한계(1.8361)와 0.10 rad, 배포 노드 검사 값과 0.08 rad 떨어진다.
- 재학습 전 확인: model_5999를 이 환경에서 그대로 돌리면(512 env, 외란 1.0) success 92.8%다. Light에서는 93~96%였다. 관절 최소 여유는 정확히 0.050 rad로, 한계가 적용된 것을 확인했다.
- 이어 학습: Light model_5999에서 +1500 iter를 돌렸다(run `2026-11-03_00-14-53_margin005_from5999`, 로그 폴더는 Light와 같다, stdout은 repo 루트 `box_light_margin.log`, 약 1.75 s/iter, 00:14~00:56).
- 결과(model_7498, 1024 env, 외란 1.0): success 92.9%로 재학습 전과 같다. x ≥ 0.55에서는 83%다.
  - 끝 시점에 성공 구간에 있는 비율은 78.5%로, 재학습 전 91.6%보다 낮다.
  - 판 외 접촉 > 1 N이 20%다(재학습 전 9%). 박스 yaw는 41°로 그대로다.
  - 관절 여유는 모든 env에서 0.05 rad를 지켰다(구조상 그렇다).
  - 결론: 한계를 좁혀도 5999는 거의 그대로 동작하고, 1500 iter 재학습은 나아진 게 없다.

## box 접근 분리: 접근은 rule-based/MoveIt, RL은 조이기~들기 (`-Pregrasp`, 2026-11-03)

사용자 제안: 비전으로 박스 pose를 잡고 rule-based처럼 목표 자세를 정해 MoveIt으로 접근한다. RL은 거기서 시작해 드는 것까지만 맡는다.
- 근거: Light RL의 박스 yaw(38~41°), 판 높이차, j2 한계 문제는 판을 어디에 대느냐(접근)에서 생긴다. P37에서도 yaw의 원인이 "kinematic placement"였다.
- 넘겨받는 지점: 판이 박스 면에서 4 cm 바깥이고, 높이는 박스 중심 + 6.5 cm, 판 법선이 박스를 향하고 판 roll이 고정된 상태다.
  - 판이 이미 닿은 상태에서 넘기지 않는 이유: 비전·플래너 오차가 그대로 박스를 미는 힘이 되기 때문이다.

**1. 접근 끝 자세 표** (`scripts/tools/eval_dual_fr3_plate_rule_based.py --pregrasp_gap 0.04 --save_table ...`)
- rule-based 접근(HIGH → DOWN → APPROACH)을 면 바깥 4 cm에서 멈춘다. 팔이 멈추면(관절 속도 < 0.05 rad/s가 10 step) 관절각과 박스 pose를 기록한다.
- 관절 한계에 막혀 목표에 못 간 자세도 기록한다. 판 법선 방향 잔차 3 cm 이내, 높이 잔차 −4~+1 cm(높게 멈춤 허용), 각도 0.15 rad 이내이고, 접촉이 없고 박스가 움직이지 않은 것만 남긴다.
- 잡는 높이 비교(512 env, x 0.45~0.52, Light-Margin 한계): 중심 + 5 cm는 유효 71%, null-space 설정을 바꿔도 72%였다. **중심 + 6.5 cm는 88.5%**라 이 값을 쓴다. Light RL도 6~7 cm에서 잡았다.
- 최종 표: 4096 env, seed 0 → 3568줄(87.1%). `source/dual_fr3_lab/dual_fr3_lab/assets/pregrasp/plate_box_pregrasp_gap004_margin005.npz`
  - 박스 x 0.45~0.50은 94%, 0.50~0.52는 70%가 유효하다. y가 ±0.05 밖이면 80%다. 탈락은 대부분 판이 4 cm 넘게 높게 멈춘 경우다(482개).
  - 관절 여유 중앙값은 0.084 rad, 최소 0.050 rad(Margin 한계)다.
- 실물에서도 같은 IK(ready 자세 시드, 같은 null-space 설정)로 목표 관절각을 구하고 MoveIt/MoveToJoint는 경로만 맡기면 sim과 시작 분포가 같아진다(아직 구현 안 함).

**2. task**: `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp[-Upright][-Play]-v0`
- Light-Margin과 같다(관측 56D, 행동, 보상, 관절 한계). 다른 점은 아래와 같다.
  - reset 이벤트 `reset_from_pregrasp_table`이 표에서 한 줄을 뽑아 양팔 관절(PD 목표 포함)과 박스 pose를 같이 놓는다.
  - 오차를 흉내 내려고 박스 xy ±1 cm, yaw ±0.05 rad, 관절 ±0.01 rad 노이즈를 넣는다.
  - 에피소드는 4 s(80 step)다.
- reset 확인(256 env): 판-면 간격 중앙값 4.1/4.0 cm(2.1~6.8), 높이 6.8 cm다. 10 step 동안 접촉 0 N, 박스 이동 0이었다.
- `-Upright`: P47에서 들기 습득 뒤에 켜지던 자세 품질 항을 처음부터 최종 weight로 둔다(box_upright −3, plate_height_symmetry −8, plate_approach_symmetry −20, one_sided_contact −5). success yaw 계수 ramp도 바로 시작한다(1000 iter).
- 첫 학습(555 × 475 × 275 박스, 01:00 시작)은 사용자가 박스를 바꾸기로 해서 01:22에 멈췄다. 폴더 이름에 `_box555_cancelled`를 붙였다.

**박스 교체: 410 × 310 × 280 mm, 0.4~0.6 kg** (2026-11-03 사용자: 실물 2차 박스, 빈 박스 501~561 g)
- Pregrasp 계열에만 적용한다(`BOX410_SIZE`, `BOX410_MASS_RANGE`). Light/Light-Margin과 그 체크포인트는 555 박스 그대로다. 외란은 질량에 비례해 줄였다(명목 0.5 kg).
- rule-based 조이기-들기 비교(`--box_size`, CPU 64 env, seed 0, 외란 1.0, x 0.45~0.60, 질량은 Light DR 0.5~1.0 kg):

  | 박스 | 성공 | x < 0.50 | x 0.50~0.55 | x ≥ 0.55 | 조임 L/R | 들린 동안 최대 기울기 median/p90 | DOWN 수렴 |
  |---|---|---|---|---|---|---|---|
  | 555 × 475 × 275 | 76.6% | 95% | 89% | 52% | 12.8/13.2 N | 4.2/9.1° | 9% |
  | 410 × 310 × 280 | 82.8% | 90% | 94% | 68% | 15.5/14.8 N | 4.6/20.6° | 28% |

  - 실패는 두 박스 모두 대부분 x ≥ 0.57이다. j2가 한계에 걸려 판이 잡는 높이까지 못 내려가고(10~13 cm 높게) 박스 윗부분을 잡다가 미끄러진다.
  - 새 박스는 폭이 좁아 판이 몸 쪽으로 더 들어오므로 DOWN 수렴이 늘었다. 들린 동안 기울기 p90이 커졌다(64 env라 p90은 6개 정도).
  - 영상: `logs/rule_based/box410/rule_box410_seed1_3env.mp4`(env 3개를 차례로, 각 10 s, 3개 모두 성공). 이 영상은 박스 3개가 모두 x 0.55~0.60이고 0.62~0.86 kg이라 배포 조건이 아니다.
  - 배포 조건 영상: `logs/rule_based/box410/rule_box410_near_seed2_3env.mp4`(`--box_dx -0.05 0.02 --mass 0.4 0.6`, 외란 켬). 박스 x 0.45~0.50, 0.41~0.48 kg이고 3개 모두 성공했다. 화면 위에 박스 x와 질량을 표시한다.
- 들기 전 판 떨림(사용자가 영상에서 발견, `--log`로 step별 기록, 256 env, x 0.45~0.52, 0.4~0.6 kg).
  - 떨림은 DOWN과 APPROACH(접촉 전)에서 생긴다. 판 위치 2차 차분 중앙값은 3~5 mm이고, 손목 j5·j6 속도 방향이 초당 5~7번 바뀐다. 조이기 시작한 뒤(LIFT/HOLD)에는 0.6 mm다.
  - 원인은 IK gain 2다. 명령이 q_meas + 2·(필요한 관절 변화)인데, 가벼운 손목 관절은 PD(Kp 80)로 50 ms 안에 거의 다 따라가서 매 step 넘쳤다가 되돌아온다.
    - gain 1: DOWN 떨림이 3.0 → 1.4 mm, 방향 전환이 4.7 → 0.7회/s로 줄었다. 하지만 조임힘도 17 → 9 N으로 줄어, 64 env 중 4개가 기울기 15° 초과로 실패했다. 조임힘은 막힌 판의 명령 포화(0.1 rad × Kp 80)에서 나오기 때문이다.
    - 속도를 반으로(v_pre 0.15) 줄여도 방향 전환은 그대로였다. null-space를 꺼도 거의 같았다.
  - 해결: `--gain_free 1.0`(접촉 전까지만 gain 1, 조이기부터 gain 2).

    | 설정 | 성공 | 조임 | 떨림 DOWN | 떨림 APPROACH 접촉 전 |
    |---|---|---|---|---|
    | gain 2 | 100% | 17.9 N | 2.9 mm, 4.7/s | 5.0 mm, 7.5/s |
    | gain_free 1 | 100% | 18.1 N | 1.3 mm, 0.7/s | 2.9 mm, 2.2/s |

  - 남는 것: 첫 접촉 순간 판 감속(약 1 cm, v_app 0.15 m/s 충돌)과 조이기 중 1.7~2.7 mm 접촉 떨림이다. 먼저 닿은 판이 다른 판이 올 때까지 박스를 1~2 cm 민다.
  - 같은 장면 영상: `logs/rule_based/box410/rule_box410_near_seed2_3env_gainfree1.mp4`.
  - 기본값은 이전 결과와 맞추려고 gain 2 그대로 둔다. 실물 접근은 MoveIt 궤적이라 이 떨림과는 관계없다. 이 스크립트 제어기를 실물에 쓰면 `--gain_free 1.0`을 쓴다.
- gain_free 1 뒤에도 남는 흔들림(사용자 지적, 같은 256 env 조건).
  - DOWN 목표에 도착한 뒤 팔이 멈추지 못하고 목표를 지나쳤다가 돌아온다. 예: env 140은 판이 37 cm/s로 목표를 지나 7 cm 넘어갔고 0.7 s 주기로 흔들렸다.
    - 큰 관절(j2, j4)의 PD 감쇠 Kd 4가 팔 관성에 비해 작아서다. 여기에 50 ms마다 목표를 측정값 기준으로 다시 잡는 명령 방식이 겹친다.
  - 첫 접촉에서 판 힘이 한 step 만에 0 → 25 N(p90 157 N)으로 튀고 0.15 s 되튄다.
  - 시험한 옵션(모두 기본 꺼짐):

    | 옵션 | 결과 |
    |---|---|
    | `--damp 0.05` (q_target에서 0.05·q̇를 빼서 Kd를 +4) | 도착 뒤 판 속도 21 → 14 cm/s, 접촉 힘 p90 157 → 45 N, 접촉 전 떨림 p95 4.4 → 3.0 mm, DOWN 수렴 56 → 80%, 성공 100%. 0.1 이상은 손목이 불안정해 성공 20% 이하 |
    | `--v_contact 0.05` (면 2 cm 앞부터 0.05 m/s) | 접촉 힘 p90 → 34 N, 성공까지 +0.9 s |
    | `--a_ref 1.0` (기준 위치 가감속) | 효과 없음. 팔이 기준보다 늦어서 기준이 서도 팔은 계속 간다 |
    | `--limit_aware` (한계에 막힌 관절을 빼고 IK 재계산) | 효과 없음 |
  - 사용자 요청으로 rule-based는 여기서 멈췄다(2026-11-03). 다음에 쓸 조합 후보는 `--gain_free 1.0 --damp 0.05 --v_contact 0.05`다.
- 외란·질량 분리(410 박스, CPU 256 env, seed 0, x 0.45~0.60):

  | 질량 | 외란 | 성공 | x < 0.55 | x ≥ 0.55 | 실패 내역 |
  |---|---|---|---|---|---|
  | 0.4~0.6 kg (실물) | 끔 | 96.1% | 100% | 88% | 기울기 4, goal 못 감 6 |
  | 0.4~0.6 kg (실물) | 켬 | 96.1% | 100% | 88% | 기울기 3, goal 못 감 7 |
  | 0.5~1.0 kg | 끔 | 93.4% | 99.4% | 81% | 못 듦 3, 떨굼 4, 기울기 6, goal 4 |
  | 0.5~1.0 kg | 켬 | 90.6% | 96.5% | 79% | 못 듦 3, 떨굼 8, 기울기 10, goal 3 |

  - 위 64 env 결과(82.8%)는 표본이 작아서 낮게 나왔다.
  - 외란은 실물 질량에서는 영향이 없고, 1 kg 가까운 박스에서만 떨굼·기울기를 조금 늘린다.
  - 실물 질량의 실패 10개는 전부 x0 0.56~0.60이다. DOWN에서 j2가 한계에 걸려 한쪽 판이 11~14 cm 높게 멈추고, 판 높이차 6~8 cm로 박스 윗부분을 잡는다. 그래서 들면서 박스가 기울거나 판 사이로 미끄러져 goal보다 10 cm 이상 낮게 끝난다.
  - 이 스크립트 IK(ready 자세로 당기는 null-space)의 한계다. Light RL은 x ≥ 0.55에서 89~99%였다.
  - 시간: 성공까지 중앙값 8 s(속도 제한 v_pre 0.3, v_app/v_lift 0.15 m/s 때문)다. RL은 3.75 s였다.
- 접근 끝 자세 표를 새 박스로 다시 만들었다(512 env 비교: 잡는 높이 중심 + 5 cm 97.5%, + 6.5 cm 97.9%).
  - 손이 닿는 범위가 더 이상 문제가 아니라서 rule-based 들기와 같은 + 5 cm를 쓴다.
  - 박스 중심 높이는 손이 안 닿는다(11-03 추가 확인, 512 env): 중심 + 0 cm 유효 3.7%(판이 목표보다 5.8 cm 높게 멈추고 관절이 Margin 한계에 붙음), + 2.5 cm 63.5%, + 5 cm 97.5%.
  - 최종: 4096 env, seed 0 → 3976줄(97.1%), `assets/pregrasp/plate_box410_pregrasp_gap004_margin005.npz`. 555 박스 표는 지웠다.
  - x 0.45~0.50은 99%, 0.50~0.52는 92%가 유효하다. 관절 여유 중앙값은 0.089 rad다.
- reset 확인: 판-면 간격 중앙값 4.1 cm(2.0~7.1), 판 높이는 박스 중심 + 5.3 cm다. 접촉 0, 박스 이동 0이었다.
- 학습(01:25 시작, 두 run 동시, 각 3000 iter from scratch, 약 2.8 s/iter, 약 2시간 20분): run_name `pregrasp_box410`, `pregrasp_upright_box410`. stdout은 repo 루트 `box_pregrasp_box410.log`, `box_pregrasp_upright_box410.log`다.
- 중간 평가(02:30, `pregrasp_box410` model_1200, 1024 env, 외란 1.0, 4 s):

  | 항목 | 값 |
  |---|---|
  | 성공 | 99.9%(끝 시점 성공 구간 98.6%) |
  | 성공까지 시간 | 중앙값 1.65 s (Light 3.75 s, rule-based 8 s) |
  | 끝 기울기 | 6.5° |
  | 판 높이차 | 1.2 cm (Light 2.7) |
  | 끝 박스 yaw | **36.7°(p90 44°)** |

  - 박스 yaw는 Light(38°)와 거의 같다. 접근을 분리해도 조이고 들면서 박스가 돌아간다. 이때 yaw 페널티 ramp(yawtrack)는 0.36이었다.
  - 영상: `<run>/videos/model_1200_views_ep4.mp4`.
- 최종(model_2999, 03:45 종료, 같은 평가):

  | 항목 | model_1200 | **model_2999** |
  |---|---|---|
  | 성공 | 99.9% | **100%**(끝 시점 성공 구간 100%) |
  | 성공까지 시간 | 1.65 s | **1.40 s** (p90 1.55) |
  | 끝 기울기 | 6.5° | **4.7°** (들린 동안 최대 6.8°, p90 9.1°) |
  | 판 높이차 | 1.2 cm | **0.8 cm** (p90 1.9) |
  | 끝 박스 yaw | 36.7° | **0.4°** (p90 1.0°) |
  | 판 외 접촉 > 1 N | 15% | **0%** |
  | 끝 조임 | | 12.2/12.1 N |

  - yaw 페널티 ramp가 끝까지 가면서 박스 회전이 사라졌다. P38(2.7°)보다 낫다.
  - 영상: `<run>/videos/model_2999_views_ep4.mp4`.
  - 배포 후보다. 실물에서는 비전 박스 pose → 같은 IK로 접근 끝 관절각 → MoveIt 경로 → 정책 시작 순서다.
  - **시작 자세(11-03 측정, 1024 env, 정책 없이 리셋 직후)**:
    - 판 중심 앞뒤는 박스 중심 그대로다(dx 0, p10~p90 ±1.3 cm는 박스 xy 노이즈). 높이는 박스 중심보다 5.3 cm 위다(표 grip_dz 0.05, 설계값). 면 간격 4.1 cm다.
    - 판 면은 박스 면과 평행하다(1.5°, p90 2.8°).
    - **정사각형 판이 면 안에서 박스 모서리 대비 약 18°(p10~p90 11~25°) 돌아가 있다.** 표를 만든 IK가 판 법선축 회전을 정하지 않고 팔 자세에 맡기기 때문이다(`fin_q` = 처음 자세에서 법선만 돌림, APPROACH에서 그때 회전을 고정). `--fixed_roll`도 박스 모서리에 맞추는 옵션은 아니다.
  - **실로봇 export(11-03)**: `fr3_husky_etri/model/dual_fr3_boxlift_pregrasp_box410_model2999.npz`.
    - `export_boxlift_actor.py`는 `-Pregrasp` run에서 `manifest["start"]`를 `pregrasp_table`로 쓴다. 학습에 쓴 표를 `pregrasp_joint_position`(3976 × 14, left1~7·right1~7 순서)과 `pregrasp_box_pose_base`(3976 × 7, base frame)로 함께 넣는다.
    - 박스 시작 범위는 표 범위 + 리셋 노이즈다: x 0.437~0.530, y ±0.117, yaw ±0.16 rad. Pregrasp env의 object reset 범위(x 0.45~0.60)는 표 reset이 덮어쓰므로 쓰지 않는다.
    - 실물 순서: ready 자세에서 박스 pose를 잰다 → 표에서 가장 가까운 줄을 고른다 → 그 관절각으로 MoveIt 이동 → 정책을 시작한다(에피소드 4 s).
    - 범위 안 임의 박스 pose 20000개 모두 학습 노이즈(±1 cm, ±0.05 rad) 안에 표 줄이 있었다(중앙값은 노이즈의 1/4).
    - `boxlift_core.load_boxlift_policy`로 읽힌다. npz 출력은 torch actor와 1e-7까지 같다.
    - 노드의 시작 검사(`check_start_conditions`)는 아직 ready 자세와 비교한다. 실험 순서(`ready_tolerance_rad` 2.6, 시작 자세 후보, 터미널별 명령)는 `fr3_husky_etri/docs/dual_fr3_box_experi_step.md`에 있다.
  - **판 회전 정렬 표(11-03)**: `eval_dual_fr3_plate_rule_based.py --roll_align`은 처음부터 판 방향 전체를 따라간다. 판 모서리를 박스 x축·수직에 맞추고, 그런 네 방향 중 지금 자세에 가장 가까운 것을 쓴다.
    - 512 env 유효율: + 5 cm 97.9%(정렬 전 97.5%), + 6.5 cm 99.6%. 회전을 고정해도 손이 닿는 범위는 줄지 않았다. 관절 여유 중앙값은 0.077 rad다(정렬 전 0.092).
    - 표: 4096 env, seed 0, + 5 cm → 4003줄(97.7%), `assets/pregrasp/plate_box410_pregrasp_gap004_margin005_rollalign.npz`. 기존 표는 그대로 둔다(model_2999가 그 표로 학습됨).
    - 리셋 직후 판 모서리 회전 −0.1°(p10~p90 −1.1~+0.7°, 정렬 전 −18°)이다. 위치·간격·면 평행도는 정렬 전과 같다.
    - Pregrasp task는 기존 표를 쓴다. 이 표로는 RollAlign task를 학습했다(아래, 실패).
    - 비교 그림(같은 박스 pose, 양쪽 판): `docs/figures/pregrasp_start_old_vs_rollalign.png`.
    - 재학습 task: `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-RollAlign[-Play]-v0`(로그 `dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_rollalign`). Pregrasp와 표만 다르다. 4090에서 3000 iter from scratch로 돌린다.
    - 다른 PC에서 학습한 run의 env.yaml에는 그 PC의 표 절대경로가 들어간다. `export_boxlift_actor.py`는 그 경로가 없으면 이 checkout의 `assets/pregrasp/<같은 파일명>`을 쓴다.
    - **결과(로컬, 12:03~13:30, `2026-11-03_12-03-06_pregrasp_rollalign_box410`)**: 판은 평행해졌지만 박스를 비튼다. 실로봇 후보가 아니다.

      | 항목(1024 env, 외란 1.0, 4 s) | RollAlign 600 | 1200 | 2100 | 2999 | 참고: Pregrasp 2999 |
      |---|---|---|---|---|---|
      | 성공 | 100% | 100% | 99.9% | **31.8%** | 100% |
      | 끝 박스 yaw | 63° | 58° | 48° | 41° | 0.4° |
      | 들기 중 최대 기울기 | 13.1° | 10.4° | 7.7° | 11.8° | 6.8° |
      | 최대 판 힘 | 29 N | 22 N | 21 N | 18 N | 17 N |

      - 판 V자는 사라졌다. 판 법선-면 차이가 0.5 s에 5.9°(기존 28.9°), pitch −2.9°(기존 +26.7°)다.
      - 대신 모든 checkpoint가 박스를 수직축으로 41~67° 비튼다. 학습 곡선에서 성공 보상이 600 iter 9.7 → 2999 iter 1.0으로 계속 떨어졌다(기존 run은 12.8로 회복).
      - 관절: 오른팔 j2가 모든 env에서 Margin 한계(0.05)에 붙고, 왼팔 j7이 한계 0.11까지 간다. 기존 정책은 모든 관절이 0.13 이상 남는다.
      - 원인 후보: `--roll_align`은 네 정렬 방향 중 ready 자세 판에 가까운 것을 골라서 j7이 기존 표보다 72° 돌아간 자세(왼 2.06/오 −0.49, 기존 0.81/0.76)에서 시작한다. 판은 정사각형이라 −18°만 돌려도 정렬되는데 반대쪽으로 돈 것이다. run 하나라 seed 차이일 수도 있다.
      - 영상: `<run>/videos/model_2999_views_ep4.mp4`(박스를 비틀어 41 cm까지만 들고 goal에서 6~14 cm 멈춤).
  - **j7 정렬 표 `--snap_j7`(11-03)**: 기본 접근(판 회전은 기존처럼 둠)으로 줄을 기록한 뒤, j7만 최대 45° 돌려 판 모서리를 박스에 맞춘다. j7 축이 판 법선이라 판 중심·법선·간격은 그대로다.
    - Wide 박스 표에서 j7 회전 왼 −15.7°, 오 +15.8°(중앙값), 결과 j7 왼 0.56/오 1.01(기존 표 근처). 리셋 직후 판 모서리 회전 0.0°(p10~p90 ±0.8°).
  - **판 평행도(11-03 측정, 512 env)**: 리셋 때는 판 법선과 박스 면 법선 차이가 1.6°(p90 2.8°)로 평행하다(yaw ±2.4°는 박스 yaw 노이즈).
    정책이 움직이면서 판을 아래 끝이 박스 쪽, 위 끝이 바깥인 V자로 기울인다. pitch는 첫 접촉(0.15 s) 11°, 들기 중(0.5~0.75 s) 27°, 유지(1.5 s) 5.5°다.
    들기 중 판 중심은 면에서 3.6 cm 떨어져 있어 아래 모서리로 잡는 셈이다. 오른판은 접촉 전 yaw도 9° 돈다(왼판 0°).
    판 방향을 직접 보는 항은 `reaching_aligned`(거리 × cos) 하나뿐이라 cos 27° = 0.89, 손해가 작다. 종이 박스 실물에서는 모서리 눌림·미끄럼 위험이 있다.
- `pregrasp_upright_box410`은 3000 iter 끝까지 한 번도 들지 못했다(들기·성공 보상 0, 03:34 종료). 자세 페널티를 처음부터 켜면 들기 발견을 막는다는 P7 교훈과 같다. 영상: `<run>/videos/model_1350_views_ep2.mp4`(판을 대기만 하고 가만히 있음).

## 학습

```bash
python scripts/rsl_rl/train.py --task Isaac-Lift-Cube-DualFR3-Sim2Real-1kHz-v0 --headless          # 5000 iter
python scripts/rsl_rl/train.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-1kHz-v0 --headless     # 10000 iter
```

학습 속도 (RTX 5070, 4096 env, 처음 6 iter 측정. 접촉이 늘면 더 느려질 수 있음):

| task | s/iter | 기본 iter | 예상 시간 |
|---|---:|---:|---:|
| cube 100Hz | ~1.9 | 5000 | ~2.6 h |
| cube 1kHz | 11~13 (증가 중) | 5000 | 18 h 이상 |
| box 100Hz | ~1.5 | 10000 | ~4.2 h |
| box 1kHz | ~8.8 | 10000 | 24 h 이상 |

로그: `dual_fr3_lift_sim2real_{1khz,100hz}`, `dual_fr3_plate_lift_sim2real_{1khz,100hz}`.
Play는 같은 rate의 `-Play-v0` / `-Play-Noisy-v0`를 쓴다.

- 실로봇 박스 배치 기준(좌표계, 학습 범위 그림): [dual_fr3_box_real_setup.md](dual_fr3_box_real_setup.md)

## box Light-Box410: 410 박스를 바닥부터 전부 RL (2026-11-03)

- `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Box410[-Play]-v0`: Light-Margin + `_apply_box410()` + 박스 x 0.45~0.52. Pregrasp와 비교하는 기준선이다.
- 학습: 5000 iter from scratch, run_name `light_box410`, stdout은 repo 루트 `box_light_box410.log`, launcher는 scratchpad `train_light_box410.sh`다.
- 결과(model_4999, 12:15 종료, 512 env, 외란 1.0, 5 s):

  | 항목 | Light-Box410 model_4999 | 참고: Pregrasp box410 model_2999 |
  |---|---|---|
  | 성공 | 99.4% | 100% |
  | 성공까지 시간 | 3.25 s | 1.40 s |
  | 끝 박스 yaw | **64°**(p90 69°) | 0.4° |
  | 끝 기울기 | 8.6° | 4.7° |
  | 잡는 높이(판 중심 − 박스 중심) | **12.9 cm** | 8.8 cm |
  | 최대 판 힘 | 33 N | 17 N |
  | 판 외 접촉 > 1 N | 7% | 0% |

  - 성공 판정은 통과하지만 박스 윗모서리 근처를 잡고 박스를 수직축으로 64° 돌린다. 판 중심이 윗면에서 1 cm 아래이므로 판의 절반이 박스 위로 나온다. 3300 iter(yaw 61°, 12.4 cm)에서 나아지지 않았다.
  - 4090 run 영상의 "잘 못 든다"는 이 거동으로 보인다. 410 박스는 접근을 분리한 Pregrasp 쪽이 맞다.

## box Wide: 긴 변(410 mm)을 좌우로 (`-Light-Pregrasp-Wide`, 2026-11-03)

- 실물 박스는 긴 변이 로봇 좌우(판이 잡는 방향)다. Box410/Pregrasp/RollAlign은 410을 앞뒤(x)로 두고 학습했다.
  - 원인: 11-02에 "410 310 280"을 이전 박스(555 앞뒤 × 475 잡는 방향 × 275)와 같은 (x 앞뒤, y 잡는 방향, z) 순서로 읽었다. 답변에 "410(앞뒤) × 310(판으로 잡는 방향)"이라고 적었지만 확인을 받지 않았다.
  - 지금 정책(Pregrasp model_2999)을 쓰려면 실물 박스를 90° 돌려 410 변을 앞뒤로 놓는다.
- 시작 자세 표 두 가지로 나눠 학습한다. 같은 seed로 만들어 박스 pose와 j1~j6이 똑같고 j7만 다르다.

  | task (`Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-…`) | 표 (`assets/pregrasp/`) | 리셋 직후 판 모서리 회전 | j7 중앙값 L/R |
  |---|---|---|---|
  | `Wide[-Play]-v0` (로그 `…_pregrasp_wide`) | `plate_box410wide_pregrasp_gap004_margin005.npz` (기존 Pregrasp 방식) | −15.9° (p10~p90 −22~−9°) | 0.83 / 0.74 |
  | `Wide-RollAlign[-Play]-v0` (로그 `…_pregrasp_wide_rollalign`) | `…_snapj7.npz` (`--snap_j7`) | 0.0° (±0.7°) | 0.56 / 1.01 |

  - 박스 `BOX410_WIDE_SIZE` = (x 0.31, y 0.41, z 0.28), 나머지(노이즈·보상·에피소드 4 s)는 Pregrasp와 같다.
  - 표: `--box_size 0.31 0.41 0.28 --box_dx -0.10 0.0 [--snap_j7]`, 4096 env, seed 0, + 5 cm → 각 3956줄(96.6%). 박스 중심 x 0.40~0.50.
  - x 범위를 앞당긴 이유: 판이 바깥(y ±0.245)으로 벌어져서 j2 한계에 더 빨리 걸린다. x 0.45~0.52로 만들면 86%(x ≥ 0.50은 66%)다. 0.36~0.50에서는 x ≤ 0.48까지 거의 100%다. 박스 깊이가 31 cm라 x 0.40이어도 로봇 쪽 면이 0.245(Husky 앞에서 13.6 cm)다.
  - 리셋 직후: 판-면 간격 3.9 cm, 판 중심 박스 중심 + 5.2 cm, 판 모서리 회전 0.0°.
- 학습: 로컬에서 두 run을 같이, 각 3000 iter from scratch(16:32~18:55, iter당 2.8 s, traceback 0).
  - `logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide/2026-11-03_16-32-43_pregrasp_wide_box410` → stdout `box_pregrasp_wide.log`
  - `logs/rsl_rl/dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign/2026-11-03_16-33-43_pregrasp_wide_rollalign_box410` → stdout `box_pregrasp_wide_rollalign.log`
  - 16:25에 정렬 표 하나로 시작한 run은 165 iter에서 멈췄다(두 표로 나누기로 함).

### Wide 결과 (model_2999)

1024 env, 학습 노이즈 그대로, 질량·마찰 DR, 외란 1.0, 80 step(4 s). Pregrasp 2999는 410 앞뒤 박스(같은 평가)다.

| | Pregrasp 2999 (410 앞뒤) | Wide 2999 | Wide-RollAlign 2999 |
|---|---|---|---|
| 성공 (0.5 s 유지) | 100% | 100% | 100% |
| 성공까지 median / p90 | 1.40 / 1.55 s | 1.65 / 1.90 s | 1.40 / 1.60 s |
| 끝 goal 거리, 기울기 | 1.7 cm, 4.7° | 3.7 cm, 3.4° | 2.4 cm, 2.9° |
| 들 때 최대 기울기 median (p90) | 6.8° (9.1) | 9.3° (11.9) | 12.0° (14.0) |
| 끝 박스 yaw median (p90), 시작 yaw 2.9° | 0.4° (1.0) | 2.0° (4.8) | 1.5° (2.9) |
| 판 최대 힘 L/R | 16.6 / 16.7 N | 17.5 / 17.1 N | 20.3 / 20.6 N |
| 들 때 판 중심 − 박스 중심 dz, 좌우 판 높이 차 | 8.8 cm, 0.8 cm | 7.5 cm, 2.3 cm | 9.0 cm, 0.5 cm |
| 판 V자 기울기 (판-면 각, 0.5 s / 0.75 s) | 28.9° / 28.6° | 14.0° / 16.4° | 24.4° / 25.7° |
| FR3 한계 여유 median / min | 0.107 / 0.064 rad | 0.123 / 0.056 rad | 0.135 / 0.064 rad |
| 판 외 접촉 > 1 N | 0 | 0 | 0 |

- 관절 이동량(에피소드 중 시작 대비 최대 |Δq| median, 512 env):
  - Wide-RollAlign: 양팔 대칭. j2 1.5~1.6, j4 1.0, j1 ≤ 0.6 rad.
  - Wide: **비대칭**. 왼팔은 j2 1.40·j4 0.93으로 들지만, 오른팔은 j2 0.50만 쓰고 j1 1.06·j3 1.43·j5 1.03·j7 1.28 rad를 돌린다. 영상에서 오른팔 팔꿈치가 옆으로 눕는다. 관절 경로 합 L 5.5 / R 7.3 rad(RollAlign 6.0 / 5.8).
- Wide는 V자 기울기가 작지만 좌우 판 높이가 2.3 cm 어긋나고 박스 yaw가 덜 맞는다. RollAlign은 판이 수평으로 맞고 빠르지만 V자가 크다(옛 Pregrasp와 비슷).
- 실물 첫 후보: **Wide-RollAlign 2999**(대칭 동작, 한계 여유 큼). Wide 2999는 두 번째 후보.
  - **(2026-11-04 정정)** 위 표는 박스 회전 제한(30°/s) 물리 결과다. 고친 물리에서 Wide-RollAlign 2999는 성공 79%,
    들 때 박스 기울기 p90 42°라 실물 후보에서 뺐다. 후속 후보는 FaceGrip 4699("FaceGrip 4699" 절).
- 실물 시작 자세 다섯 줄(박스 x 0.419~0.477, y ±0.05)에서 노이즈 없이 500 env: 두 정책 모두 성공 100%, 한계 여유 최소 RollAlign 0.129 / Wide 0.112 rad.
- export: `fr3_husky_etri/model/dual_fr3_boxlift_pregrasp_wide[_rollalign]_model2999.npz` (+ `_start_poses.yaml`).
  - 실로봇 노드 launch에 `box_size_m:="[0.310, 0.410, 0.280]"`을 준다(이번에 launch 인자 추가). 기본값 [0.41, 0.31, 0.28]이면 매니페스트 검사에서 거부된다.
- 영상: `<run>/videos/model_2999_views_ep4.mp4` (두 run 각각).
- 배치 그림: `docs/figures/box410wide_workspace_{top,side,front}.png` ([dual_fr3_box_real_setup.md](dual_fr3_box_real_setup.md)). `draw_box_workspace.py`가 Pregrasp task면 표의 박스 범위(+ 리셋 노이즈)를 그리고 팔을 ready로 되돌리게 고쳤다.

### 박스 관측 고정 (`-Wide-RollAlign-FrozenBox`, 2026-11-04)

- 비전이 완전히 끊기거나 `/box_pose`를 놓은 위치로 고정 publish하면, 들기 시작한 뒤 정책이 보는 박스 pose가 실제와 달라진다. 정책은 매 step 박스 pose를 본다.
- sim 확인: 박스 관측을 에피소드 시작 값으로 고정하고 model_2999를 평가했다(1024 env, 같은 DR).

  | | 성공 | 끝 goal 거리 | 끝 기울기 / 들 때 최대 | 판 외 접촉 > 1 N |
  |---|---|---|---|---|
  | Wide-RollAlign 2999 | 0% | 20.2 cm | 24.8° / 29.6° | 57% |
  | Wide 2999 | 8.1% | 21.6 cm | 56.1° / 65.6° | 57% |

  - 둘 다 박스는 들지만(양 판 들기 100%) goal을 지나쳐 들고 박스가 기운다. 들기 중 박스 관측에 크게 기댄다.
  - → 비전 없이(고정 publish) 명령 모드로 들면 안 된다(etri 실험 문서에 경고 추가).
- 대책 task `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FrozenBox[-Play]-v0`:
  - policy의 `object_position`·`object_orientation`을 `object_position_at_reset`·`object_orientation_at_reset`(에피소드 첫 관측 값 유지)으로 바꾼다.
  - `critic` 관측 그룹을 따로 두고 실제 박스 pose를 노이즈 없이 준다(asymmetric actor-critic, `obs_groups = {"actor": ["policy"], "critic": ["critic"]}`). 차원이 같아(56) Wide-RollAlign checkpoint에서 이어 학습할 수 있다.
  - export: `export_boxlift_actor.py`가 이 항을 받아 매니페스트 `observation.box_pose = "held_at_start"`로 적는다. 실로봇 노드는 아직 이 항을 모른다(관측 항 검사에서 거부). 받으려면 `start_policy` 때 박스 pose를 잡아 두고 그 값을 계속 넣게 고쳐야 한다. 고정 publish라면 지금도 결과는 같다.
- **학습은 중단했다 (2026-11-04).** 실물에 Azure Kinect 비전이 연동되어 있어 박스 pose를 계속 받을 수 있다. 비전이 전혀 없는 경우를 학습할 필요는 없고, 비전이 잠깐 끊기는 경우만 학습하기로 했다.
  - 13:53에 두 run을 시작해 14:00 무렵 멈췄다. Wide-RollAlign 2999에서 이어 학습한 run은 iter 3179/3999, 처음부터 학습한 run은 101/3000까지 갔다. 평가하지 않았고 run 폴더는 삭제했다.
  - task와 exporter 지원(`held_at_start`)은 코드에 남아 있다.
- 대신 끊김 학습 task `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-BoxDrop[-Play]-v0`를 추가했다. **아직 실행하지 않았다.**
  - policy의 박스 위치·자세에 비전 지연과 끊김을 넣는다(`object_position_dropout`·`object_orientation_dropout`, 설정은 `BOX_OBS_*` 상수).
    - 에피소드마다 0~3 step(0~150 ms) 지연.
    - step마다 2% 확률로 0.1~1.0 s 끊김. 끊긴 동안 마지막 값을 유지한다.
    - 에피소드의 10%는 도중(0.05~2 s)부터 끝까지 끊김.
  - critic은 FrozenBox와 같이 실제 박스 pose를 받는다.
  - 실로봇 노드는 지금 박스 pose가 0.30 s 넘게 안 들어오면 fault를 낸다(`box_pose_timeout_s`). 이 정책을 쓰려면 끊긴 동안 마지막 pose를 계속 넣게 고쳐야 한다.
  - 확인(2026-11-04): 5 iter 처음부터, 2999에서 이어 3 iter 둘 다 정상. PLAY env 256개 × 240 step에서 끊긴 step 23.6%, 끊긴 동안 값 유지 14487/14487, 지연 0·끊김 없음이면 실제 pose와 같음 12435/12435, 끊김 길이 중앙값 10 step(최대 30, 연달아 끊기면 이어짐), 끝까지 끊김 11.7%.
  - 4090에서 2999로부터 이어 1000 iter 학습 예정. 이어 학습할 때는 `env.curriculum.yawtrack.params.trigger=0.0 env.curriculum.yawtrack.params.ramp_steps=1`을 준다(FaceGrip과 같은 이유).

### 판 면 파지 (`-Wide-RollAlign-FaceGrip`, 2026-11-04)

> 이 절의 학습(2999 → 3998)과 평가는 박스 회전 제한(30°/s) 물리에서 했다. 고친 물리에서 3700을 다시 평가하고
> 이어 학습한 결과는 "FaceGrip 4699" 절에 있다.

- 목적: Wide-RollAlign 2999는 들 때 판을 V자로 젖혀(판 법선이 위로, 0.75 s에 중앙값 25.7°) 아래 모서리로 박스를 받친다. 선 접촉이라 MuJoCo·실물에서 미끄러질 수 있다.
- task `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip[-Play]-v0`:
  - `plate_face_tilt`: 판 법선과 담당 박스 면 법선 사이 각도(rad)의 양 판 평균, 양 판 접촉(> 1 N) 중에만. `pg_parallel`의 `tot`과 같은 정의.
  - `ramp_reward_weight_on_success`: success EMA ≥ 3이 된 뒤 200 iter에 걸쳐 weight를 −1e-4 → −10(`FACEGRIP_TILT_WEIGHT`).
  - yawtrack은 바로 1(`trigger=0, ramp_steps=1`). 이어 학습하면 curriculum이 처음부터 다시 돌고 yawtrack은 기본 3000 iter ramp라, 그대로 두면 2999와 success 보상이 달라진다.
  - 그 밖의 보상·관측·행동은 Wide-RollAlign 그대로다.
- run `dual_fr3_plate_lift_sim2real_100hz_light_pregrasp_wide_rollalign_facegrip/2026-11-04_14-16-51_facegrip_w10_from_wra2999`: 2999에서 이어 1000 iter(마지막 checkpoint `model_3998`). stdout `box_facegrip.log`.
  - 학습 중 판-면 각도(에피소드 평균, 비접촉은 0): 3200대 6.7° → 3400대 4.5° → 3600대 5.6° → 3800대 10.9° → 3900대 9.4°. 줄었다가 다시 늘었다.
- 평가(1024 env, 외란 1.0, 80 step, `-Wide-RollAlign-Play-v0`):

  | | 2999 | FaceGrip 3400 | FaceGrip 3998 |
  |---|---|---|---|
  | 성공 (0.5 s 유지) | 100% | 100% | 99.7% |
  | 성공까지 중앙값 / p90 | 1.40 / 1.60 s | 1.55 / 1.70 s | 1.55 / 1.85 s |
  | 끝 goal 거리 | 2.4 cm | **7.2 cm** | 2.6 cm |
  | 끝 박스 기울기 / 들 때 최대 | 2.9° / 12.0° | 5.3° / 10.2° | 6.2° / 9.6° |
  | 판 최대 힘 L/R | 20.3 / 20.6 N | 17.3 / 17.1 N | 15.0 / 14.4 N |
  | 판-면 각도 0.2 s | 10.8° | 8.5° | 7.5° |
  | 판-면 각도 0.5 s | 24.4° | 14.1° | 17.9° |
  | 판-면 각도 0.75 s | **25.7°** | **8.9°** | **13.9°** (p90 23.1°) |
  | 판-면 각도 1.5 s (들고 있는 중) | 6.6° | 1.8° | 4.6° (p90 16.1°) |
  | 판 yaw(수직축 회전) 1.0 s, p10 | −6.2° | −0.7° | **−18.9°** |

  - 들 때 V자 최대는 25.7°에서 15°(3400)·18°(3998)로 줄었다. 각도는 거의 다 pitch(V자) 성분이다.
  - 3400은 판이 가장 평평하지만 goal에 덜 올라간다(끝 goal 거리 7.2 cm).
  - 3998은 goal은 2999 수준으로 돌아왔다. 대신 일부 env에서 판이 수직축으로 15~19° 돌아간다(yaw p10). 학습 후반에 평균 각도가 다시 오른 것과 맞는다.
  - 둘 다 박스 들 때 최대 기울기와 판 힘은 2999보다 작다.
- 중간 checkpoint (같은 평가):

  | ckpt | 성공 | 끝 goal 거리 | 판-면 각도 0.5 / 0.75 / 1.5 s | 판 yaw 1.0 s p10 |
  |---|---|---|---|---|
  | 2999 | 100% | 2.4 cm | 24.4 / 25.7 / 6.6° | −6.2° |
  | 3400 | 100% | 7.2 cm | 14.1 / 8.9 / 1.8° | −0.7° |
  | 3500 | 100% | 8.3 cm | 14.2 / 8.2 / 1.3° | −0.5° |
  | 3600 | 100% | 6.4 cm | 14.5 / 7.9 / 1.7° | −2.2° |
  | **3700** | 100% | 3.4 cm | 16.2 / 11.3 / 4.4° | −2.5° |
  | 3800 | 100% | 3.0 cm | 22.2 / 19.1 / 8.3° | −8.9° |
  | 3998 | 99.7% | 2.6 cm | 17.9 / 13.9 / 4.6° | −18.9° |

  - 판을 평평하게 할수록 goal에서 멀어진다(3400~3600). 정책이 두 쪽을 오갔다.
  - 3700이 절충점이다: goal 3.4 cm, V자 최대 약 16°(2999 26°), 판 비틀림 없음.
- 영상: `<run>/videos/model_3998_views_ep4.mp4`.

### 외란 확대 (`-Wide-RollAlign-Disturb`, 2026-11-04)

- 목적: 박스가 흔들려도 더 단단히 조이고 미끄러짐을 바로잡게 해서 덜 놓치게 한다(MuJoCo·실물 접촉 차이 대비).
- 지금 외란은 질량 비례로 줄인 ±0.625 N, ±0.125 Nm(4 kg 박스 ±5 N 기준)라 박스 무게(4~6 N)의 13%이고, 판 마찰 한계(12~36 N)에 비해 거의 영향이 없다.
- task: 세기만 300 iter(`DISTURB_BIG_RAMP_STEPS` 7200 step)에 걸쳐 ±2.5 N, ±0.3 Nm(`DISTURB_BIG_*`)로 올린다(`ramp_event_params_by_steps`). 방향·주기(0.4~0.8 s)와 success 기반 on/off는 그대로, yawtrack은 바로 1. `-Play`는 처음부터 최대 세기.
- run `..._wide_rollalign_disturb/2026-11-04_15-08-51_disturb4x_from_wra2999`: 2999에서 이어 1000 iter(마지막 `model_3998`). stdout `box_disturb.log`.
- **결과는 쓰지 않는다.** 박스 회전 제한(30°/s) 물리에서 학습했고, 그 물리에서는 외란 토크가 박스를 거의 못 돌렸다
  ("박스 회전 제한 버그" 절). 외란을 다시 볼 때는 고친 물리에서 FaceGrip 계열로 새로 학습한다.

### 다시 잡기 (`-Wide-RollAlign-Regrasp`, 2026-11-04)

- 진단(`scripts/tools/box_slip_probe.py`, Wide-RollAlign 2999, 512 env, 8 s):
  - 미끄러짐 없이 8 s: 0.5 m 들어 끝까지 유지, success zone 100%(학습 4 s보다 길어도 문제없음).
  - 1.5 s에 박스·판 마찰을 0으로 0.3 s: 들고 있던 env 100%가 0.35 s(중앙값) 만에 바닥까지 떨어진다. 박스는 서서 착지(넘어짐 0%), 7 cm(p90 13 cm) 밀리고 yaw 2° 바뀜.
  - 남은 6.5 s 동안 다시 든 env **0%**. 보상은 이미 다시 잡기를 막지 않는다(낙하 종료 없음, success 매 step, reaching 항상 켜짐). 놓친 경험과 다시 잡을 시간이 없었다.
  - 마찰은 PhysX tensor API(`root_view.set_material_properties`)로 에피소드 중에도 바뀐다. 판은 `fr3_link7` shape 2개씩(마찰 0.5), PhysX 기본 average 결합이라 박스만 0으로 하면 0.25가 남아서 판도 같이 낮춘다.
- task:
  - `box_slip`(interval, 매 step): 에피소드마다 50%로 미끄러짐 한 번을 예약, 1~4 s 뒤 처음 양 판으로 들고 있을 때 박스·판 마찰 0.02를 0.05~0.4 s(`BOX_SLIP_*`). 짧으면 다시 받을 수 있고 0.3 s 넘으면 바닥까지 떨어진다.
  - `reset_from_drop_states`: reset의 30%를 놓친 직후 상태 표에서 시작(`assets/regrasp/plate_box410wide_dropstates_wra2999.npz`, 1536줄: 2999로 seed·미끄러짐 시각·저장 시각을 바꿔 3번 만든 것, 박스 x 0.30~0.75, y ±0.27). 표 형식은 Pregrasp 표와 같다(`reset_from_pregrasp_table`, 표 캐시를 경로별로 바꿈).
  - 에피소드 8 s, goal 재샘플 9 s(기본 5 s면 중간에 goal이 바뀜). 보상·관측·행동·외란은 Wide-RollAlign 그대로, yawtrack 바로 1.
  - 확인(`-Play`, 256 env, 400 step): 미끄러짐 272번, 끝난 뒤 마찰이 startup 값으로 돌아옴, 놓친 상태 시작 27%.
- run `..._wide_rollalign_regrasp/2026-11-04_15-16-18_regrasp_from_wra2999`: 2999에서 이어 학습, iter 3997에서 멈춤(마지막 `model_3950`). stdout `box_regrasp.log`.
- 결과(model_3300, 회전 제한 물리):
  - 바닥에서 다시 들기 **0%** 그대로.
  - 놓친 직후 두 판이 안쪽으로 닫혀 판 간격 6 cm(박스 41 cm), 판이 박스 중심 위 50 cm, x로 20~28 cm 어긋난다.
    pregrasp 시작과 전혀 다른 상태라 RL이 다시 잡는 길을 못 찾는다.
  - 받아내기는 배웠다: 0.1 s 미끄러짐 낙하 20% → 4%. 0.2 s 이상은 2999·3300 모두 100% 낙하.
  - action std 0.29 → 0.55로 커졌다.
- 결론: 바닥 재파지는 RL 대신 노드에서 한다(낙하 감지 → MoveIt으로 pregrasp 자세 복귀 → `start_policy` 다시).
  이 결과도 회전 제한 물리라 받아내기 수치는 다시 재야 한다.

### DR: 박스 크기·판 마찰·지연 (`-Wide-RollAlign-DR`, 2026-11-04)

- 목적: 실물 차이를 학습에 넣는다. 박스 치수 오차, 판에 붙일 고무 패드 마찰, 관절 상태·명령 전달 지연.
- task `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-DR[-Play]-v0` (Wide-RollAlign 2999에서 이어 학습용, yawtrack 바로 1):
  - 박스 크기: 축마다 ±1 cm(`BOX_SIZE_DR_M`). 시작 자세 표는 기준 크기 그대로라 판-면 간격이 4 cm ± 크기 차/2가 된다.
    `reset_from_pregrasp_table`은 env별 반높이로 리셋 높이를 맞춘다(크기 1.0이면 변화 없음).
  - 판 마찰: `fr3_link7` shape(판)에 startup DR, 정지 0.6~1.2 / 동 0.5~1.0(`PLATE_FRICTION_*`, 고무 패드 가정).
  - 지연: `DelayedReachRelativeJointPositionAction`(`reach/mdp/actions.py`)이 env마다 리셋 때 뽑는다.
    - 관절 상태 0~20 ms(`JOINT_STATE_DELAY_S`): 1차 근사 `q(t−d) = q(t) − qd(t)·d`. joint_pos 관측(`joint_pos_rel_delayed`)과 목표 기준 q가 같은 지연을 쓴다(실물 노드와 같음).
    - 명령 0~3 physics step = 0~30 ms(`COMMAND_DELAY_STEPS`): 그동안 이전 목표를 유지한다.
  - export: `export_boxlift_actor.py`는 `joint_pos_rel_delayed`를 `joint_pos_rel`로 적고(실물은 지연이 실제로 있고 노드는 일반 항을 계산),
    manifest `action`에 `trained_joint_state_delay_s`, `trained_command_delay_s`를 남긴다(DR 없는 task는 [0, 0]).
- 2999 평가(회전 제한 물리, 1024 env): 이 DR 범위에서 이미 성공 100%, goal 2.5 cm.
- 지연 한계(2999, 관절 상태 + 명령):

  | 지연 | 성공 |
  |---|---|
  | 20 + 30 ms | 100% |
  | 40 + 50 ms | 97% |
  | 60 + 50 ms | 15% (조임 35 N으로 두 배, 붕괴) |

- **학습은 취소했다.** 회전 제한 버그를 찾은 뒤 시작 전에 접었다. task 코드는 남아 있다.
- 주의: 판 마찰 범위(0.6~1.2)는 지금 판(PhysX 기본 0.5, average 결합이라 판-박스 0.45~0.85)보다 높다. 마찰이 높으면 정책이 덜 조인다.
  고무 패드를 붙이기 전에는 쓰지 않는다.

### 박스 회전 제한 버그 (`max_angular_velocity`, 2026-11-04)

- lift_plate 박스 `rigid_props.max_angular_velocity=30.0`의 단위는 **deg/s**다(Isaac Lab docstring: "in deg/s"). m/s처럼 읽고 넣은 값이라, 박스가 30°/s(0.52 rad/s)보다 빨리 돌 수 없었다.
- 확인(판·팔 없이 박스만, scratch `wrench_api_test*.py`):
  - +10 N 힘: 0.3 s에 48 cm 상승(계산 46 cm). 병진은 정상.
  - 3 Nm 토크, 공중: 30°/s에서 더 안 빨라진다. 바닥 위: 0°.
  - → 판 사이에서 박스가 거의 못 돌았고, 외란 토크도 사실상 효과가 없었다.
- 수정: `max_angular_velocity=1000.0`(17 rad/s, lift_cube와 같음, 상한 역할만). `joint_pos_env_cfg.py` box spawn에 주석.
- 고친 물리로 다시 평가(1024 env, 외란 1.0, 80 step):

  | | 제한 30°/s | 제한 1000°/s |
  |---|---|---|
  | Wide-RollAlign 2999 성공 / 들 때 기울기 median, p90 | 100% / 12.1°, 14.1° | **79.3% / 16.8°, 42.1°** |
  | FaceGrip 3700 성공 / 들 때 기울기 median, p90 | 100% / 11.0°, 13.1° | 99.7% / 13.0°, 16.5° |

  - MuJoCo에서 RollAlign 2999가 약 55° 돌던 것과 맞는다. V자로 판 아래 모서리를 받치는 잡기는 박스가 못 돌 때만 버텼다.
- 영향: 그 전 box 정책은 전부 제한 아래에서 학습했다. Wide-RollAlign 2999, FaceGrip 3700, Disturb, Regrasp, DR 평가, 4090의 BoxDrop.
  이 커밋을 pull하면 BoxDrop도 고친 물리로 바뀐다.

### FaceGrip 4699: 고친 물리로 이어 학습 (2026-11-04)

- run `..._wide_rollalign_facegrip/2026-11-04_16-05-53_facegrip_angvel1000_from_fg3700`: FaceGrip 3700에서 이어 1000 iter(→ 4699), 고친 물리.
  보상은 FaceGrip 그대로, `env.curriculum.plate_face_tilt.params.ramp_steps=1`(판 각도 벌점 바로 −10). stdout `box_facegrip_av.log`.
- 평가(고친 물리, 1024 env, 외란 1.0, 80 step):

  | | 3700 | 4200 | 4699 |
  |---|---|---|---|
  | 성공 | 99.7% | 100% | 100% |
  | 끝 goal 거리 | 3.7 cm | 2.8 cm | 3.2 cm |
  | 들 때 최대 기울기 median / p90 | 13.0 / 16.5° | 8.9 / 12.4° | 8.2 / 11.1° |
  | 판 V자 0.5 s | 16° | 15° | 9.5° |
  | 끝 박스 기울기 | 4.0° | 5.5° | 7.7° |
  | 판 힘 2 s | 13.2 N | 10.1 N | 10.8 N |
  | 회전 시작 토크 +y / −y | 0.93 / 0.80 Nm | 0.70 / 0.57 Nm | 0.73 / 0.65 Nm |

  - 회전 시작 토크: MuJoCo 쪽 방식(들고 있는 목표 고정, 박스에 0.5 Nm/s로 토크를 올려 2° 돌 때), scratch `torque_ramp_probe.py`(저장소에 없음).
    MuJoCo 3700은 0.27(pyramidal), 0.26~0.56 Nm(elliptic)이다. 같은 조임(13~14 N)에서 Isaac이 2~3배 더 버틴다.
  - 첫 접촉 때 판 기울기: 2999·3700 8.6°, 4699 4.4°. 판 각도 벌점은 양 판이 닿은 뒤에만 걸린다.
  - Isaac 3700, 2 s: 박스 pitch +0.8°, TCP가 박스 중심 위 10.5~10.8 cm, 평균 접촉점 z +9.3 / +7.3 cm(L/R).
    MuJoCo는 0.45 s 뒤에도 판이 계속 미끄러져 올라가 TCP +17 cm, 박스 pitch 23~26°.
- export: `fr3_husky_etri/fr3_husky_nn_policy/models/facegrip_angvel1000_model_4699.npz`,
  시작 자세 `model/dual_fr3_boxlift_pregrasp_wide_rollalign_facegrip_model4699_start_poses.yaml`(표는 RollAlign과 같음).
- 영상: `<4699 run>/videos/model_4699_views_ep4.mp4`(4 에피소드 모두 유지, goal 1.1~2.9 cm).
  고친 물리로 다시 찍은 `model_2999_views_ep4_angvel1000.mp4`(Wide-RollAlign run), `model_3700_views_ep4_angvel1000.mp4`(FaceGrip run).
- 실로봇(2026-11-04): 들기는 되지만 빠듯했다. 판 V자가 약간 남고, 박스를 충분히 세게 조이지 않는 것처럼 보였다(2999도 같은 경향).

### 최소 조임 (`-Wide-RollAlign-FaceGrip-MinSqueeze`, 2026-11-04)

- 과조임 벌점 `excessive_squeeze`(판별 100 N 초과분, lifting 학습 후 −0.05)는 4699에서 꺼져 있는 것과 같다. 들고 있을 때 11 N이다.
- 조임을 더 하라는 보상이 없다. 양 판 접촉 기준은 1 N이고, `squeezing_object`는 접촉 여부 × 면 접촉 품질만 본다.
  그래서 정책은 sim에서 안 떨어질 만큼만 조인다.
  - sim 최악(0.6 kg, 판-박스 마찰 0.45): 판당 필요 6.5 N, 11 N이면 1.7배 여유.
  - 실물 마찰이 0.3이면 판당 9.8 N이 필요해 여유가 거의 없다.
- 행동 범위 확인(scratch `squeeze_sat_probe.py`, 저장소에 없음, 4699, 512 env, 1.5 s 뒤 들고 있는 동안):

  | 관절 | `q_target − q` median (L / R) | 토크 median (L / R) |
  |---|---|---|
  | j1 | 0.064 / 0.072 rad | 5.0 / 5.6 Nm |
  | j3 | 0.060 / 0.064 rad | 4.9 / 5.3 Nm |
  | 나머지 | ≤ 0.023 rad | ≤ 1.3 Nm |

  - 판 힘 11.2 / 10.7 N(합력과 박스 y 법선이 같음). clip(±0.1)에 닿는 step은 없다.
  - policy-step에서 PD 오차는 행동(≤ 0.1 rad)과 같아 관절 토크 상한은 80 × 0.1 = 8 Nm다. 조임은 거의 j1·j3이 만들고 이미 64~72%를 쓴다.
    → 이 자세로는 판당 약 15 N(오른팔 j1)이 sim 상한이다.
  - 더 조이려면 action scale이나 PD 강성을 키워야 한다. 실물 `policy_control_action_server.hpp`의 `kIsaacStiffness`(reach와 공유)와 노드 `max_policy_target_delta_rad`(0.20)도 같이 바뀐다.
- task `Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-MinSqueeze[-Play]-v0` (FaceGrip 4699에서 이어 학습):
  - `min_squeeze_penalty`: 판별 접촉력이 14 N(`MINSQUEEZE_FORCE_N`)에 모자란 만큼의 합 [N]. 양 판이 닿아 있거나(들기 전 미리 조이기) 박스가 2 cm 넘게 들려 있을 때만.
    들린 채 한 판이 떨어지면 그 판은 14 N 전체가 부족분이다. 힘은 `excessive_squeeze`와 같은 합력 norm.
  - weight: success EMA ≥ 3 뒤 100 iter에 걸쳐 −1e-4 → −0.5(`MINSQUEEZE_*`). 4699 기준 판당 3 N 부족이면 −3/s(총 보상 약 52/s).
  - `plate_face_tilt`는 바로 −10. 과조임 벌점(100 N)과 나머지는 FaceGrip 그대로, checkpoint 호환.
- run `..._facegrip_minsqueeze/2026-11-04_19-38-04_minsqueeze14_from_fg4699`: 4699에서 이어 1000 iter(마지막 `model_5698`). stdout `box_minsqueeze.log`.
  - 학습 중 `min_squeeze` 보상(초당): weight가 −0.5에 닿은 직후(약 5000 iter) −3.3 → 끝 −2.2. success 보상은 10.4까지 내려갔다가 11.1로 회복.
  4699는 `init_facegrip_4699/`에 복사해 두고 `--resume --load_run init_facegrip_4699 --checkpoint model_4699.pt`로 시작했다.
- 결과(고친 물리, 4699와 같은 평가):

  | | 4699 | MinSqueeze 5698 |
  |---|---|---|
  | 성공 (1024 env) | 100% | 100% |
  | 끝 goal 거리 | 3.2 cm | 1.9 cm |
  | 들 때 최대 기울기 median / p90 | 8.2 / 11.1° | 8.7 / 11.7° |
  | 끝 박스 기울기 | 7.7° | 3.7° |
  | 들고 있을 때 판 힘 L / R (1.5 s 뒤) | 11.2 / 10.7 N | **16.1 / 16.3 N** |
  | j1 PD 오차 L / R (p90) | 0.064 / 0.072 (0.079) rad | 0.077 / 0.086 (0.092) rad |
  | j3 PD 오차 L / R (p90) | 0.060 / 0.064 (0.073) rad | 0.085 / 0.080 (0.090) rad |
  | 회전 시작 토크 +y / −y median | 0.73 / 0.65 Nm | **1.18 / 1.33 Nm** |
  | 판 V자 0.5 s median (p90) | 9.5° (14.5°) | 8.3° (14.9°) |
  | 판 각도 1.5 s median (p90) | 2.5° (5.2°) | 4.8° (11.8°) |

  - 조임 약 1.5배, 회전 시작 토크 약 1.7배. 성공·goal·끝 기울기는 같거나 좋아졌다.
  - 예상 상한 15 N을 넘었다. j1·j3만 보던 추정이었고, 정책이 j4·j6(오른팔 j6 2.8 Nm)도 더 쓴다.
    대신 j1·j3이 행동 한계의 77~90%(p90 0.087~0.092 rad)라 남은 여유는 약 10%다. 실물 관절 마찰이 더 크면 여기서 막힌다.
  - 들고 있을 때 판 각도 꼬리(1.5 s p90 11.8°)가 4699보다 길다. 영상(`<run>/videos/model_5698_views_ep4.mp4`) 4 에피소드에서
    판 높이 차(L−R)가 +3.0~+4.5 cm로 4699(−1.6~−2.8 cm)와 반대쪽이다. 2 s 때 오른판 접촉점이 박스 중심 −x로 3.8 cm 치우친다.
- 처음부터 학습하는 task `...-FaceGrip-MinSqueeze-Scratch[-Play]-v0`. run `..._facegrip_minsqueeze_scratch/2026-11-04_20-10-17_facegrip_minsqueeze14_scratch`(3000 iter, 학습 중), stdout `box_minsqueeze_scratch.log`.
  - 이유: 4699 계보(2999 → 3700 → 4699)는 2999·3700을 회전 제한 물리에서 배웠다. 약 11 N으로 판 아래 모서리를 받치는 습관이 남아 있을 수 있다.
  - Wide-RollAlign을 바로 상속한다. 보상은 MinSqueeze와 똑같고, 이어 학습용 설정은 쓰지 않는다:
    yawtrack은 기본(success EMA 1.5 뒤 3000 iter), `plate_face_tilt`는 success EMA 3 뒤 200 iter, `min_squeeze`는 300 iter(`MINSQUEEZE_SCRATCH_RAMP_STEPS`)에 걸쳐 올린다.
  - 확인: 설정 비교로 보상 항목·weight가 MinSqueeze와 같음, env 16개 × 0 행동 5 step에서 관측 56 / 행동 14, 보상 유한.
  - 학습은 `--max_iterations 3000` from scratch(Wide-RollAlign 2999와 같은 길이).
  - 행동 한계(PD 오차 ≤ 0.1 rad)는 scratch여도 같다. MinSqueeze 5698처럼 j1·j3에 덜 기대는 자세를 찾으면 여유가 생길 수 있다.
- Scratch 결과(3000 iter, 21:25 끝, 같은 평가):

  | | 1500 | 1750 | 2999 |
  |---|---|---|---|
  | 성공 (1024 env) | 98.7% | **0.2%** | 100% |
  | 끝 goal 거리 | 6.3 cm | 19.8 cm | 2.7 cm |
  | 들 때 최대 기울기 median / p90 | 4.5 / 7.3° | 9.2 / 13.0° | 8.7 / 12.2° |
  | 들고 있을 때 판 힘 L / R | 16.1 / 16.1 N | 13.4 / 13.1 N | 17.2 / 17.2 N |
  | 조임 관절 PD 오차 median (p90 최대) | j1·j3 0.081~0.095 (0.098) | j1·j3 0.068~0.091 | j1·j4 0.081~0.084 (0.091) |

  - 1500~1750에 성공 보상이 11 → 0으로 무너졌다가 2999에 9.8로 회복했다. 들기 보상(13.5)은 그대로, goal 추종 10 → 5.5, min_squeeze −0.8 → −2.5.
    1500은 오른팔 j1 p90 0.098 rad로 행동 한계에 붙어 있었다. 조임과 goal로 옮기기를 같은 j1·j3로 해야 해서 부딪힌 것으로 본다(추정).
    2999는 j3 대신 j4로 조이는 자세로 바뀌어 회복했다. yawtrack은 3000 iter 안에 0.96까지만 올랐다.
  - 2999는 4699·5698 대비 조임이 가장 세고 좌우 같다(17.2 N), 첫 접촉 판 기울기 2.0°(5698 5.7°), 좌우 판 높이 차 1.1~2.5 cm.
    회전 시작 토크 1.27 / 1.18 Nm(5698과 비슷). 판 V자는 그대로다: 0.3 s 2.0° → 0.5 s 10.8°(p90 15.5°), 1.5 s 5.5°.
    접촉 전 기울기보다 들 때 박스 무게에 판이 돌아가는 것이 V자의 주원인이다 → SqueezeLift.
  - 영상 `<run>/videos/model_2999_views_ep4.mp4`(4/4 유지, goal 0.6~1.7 cm).

### 조인 다음 들기 (`-FaceGrip-SqueezeLift`, `-SqueezeLift-LowMuHeavy-Act015-Scratch`, 2026-11-04)

- 진단: 5698도 판이 닿자마자(0.2 s) 다 조이기 전에 든다. 판 힘(min L/R) 0.55 s 7 N → 1 s 11 N → 2 s 16 N, 0.55 s에 이미 100% 들림.
  그동안 판이 V자로 돌아간다(4699 판 pitch 0.3 s 3.7° → 0.5 s 8° → 1.5 s 0°). 접촉 전에도 판이 3~6° 기울어 있다(판 각도 벌점은 양 판 접촉 뒤에만).
- 공통 추가(`_apply_squeeze_lift`):
  - 조임 factor: 들기 보상 5개(lifting_object, lift_progress, object_goal_tracking[_fine_grained], success)에 `1 − s + s·clamp(min(F_L, F_R) / squeeze_force)`를 곱한다.
    s(`env._squeezelift_scale`)는 success EMA 3 뒤 ramp로 0 → 1(`ramp_env_scale_on_success`). lift_monitor(커리큘럼 판정)는 그대로.
  - `plate_face_tilt_approach`: 양 판이 다 닿기 전, 판 중심이 담당 면 바깥 6 cm 안인 판의 기울기 [rad]. weight −10(접촉 뒤 `plate_face_tilt`와 같음).
  - `plate_box_slip`: 1 N 넘게 닿은 판마다 판 중심점의 박스 대비 접선 속도 + 0.1 m × 상대 각속도 [m/s], 양 판 합. weight −10.
  - 5698로 잰 값(s = 1, 256 env): 들고 있을 때 factor median 1.0, 미끄럼 median 0.055 / p90 0.16 m/s(−0.5~−1.6/s),
    접촉했는데 아직 바닥일 때 미끄럼 median 0.35 m/s, 접근 기울기 median 3.4° / p90 6.3°.
- `-FaceGrip-SqueezeLift`: MinSqueeze + 위 추가, 조임 기준 14 N, ramp 100 iter. 행동(0.1)·관측 그대로라 5698에서 이어 학습한다.
  init `..._facegrip_squeezelift/init_minsqueeze_5698/model_5698.pt`, `--resume --load_run init_minsqueeze_5698 --checkpoint model_5698.pt`.
- `-FaceGrip-SqueezeLift-LowMuHeavy-Act015-Scratch`: MinSqueeze-Scratch + 위 추가 + 다음, 처음부터 학습:
  - 박스·판(fr3_link7) 마찰 정지 0.25~0.5 / 동 0.2~0.45(average 결합이라 접촉도 0.25~0.5), 박스 0.4~0.8 kg.
    버티는 데 판당 m g / (2μ) 최대 15.7 N(0.8 kg, 0.25).
  - action scale·clip 0.15(PD 상한 8 → 12 Nm). 노드는 manifest의 scale/clip을 쓰고 노드 상한 0.20 rad 안이라 노드 수정 없음.
  - min_squeeze 목표와 조임 기준 20 N, 조임 factor·두 벌점 ramp 300 iter.
- 확인: 두 task 모두 `train.py` 3 iter(64 env) 정상, A는 5698 로드 후 squeeze_lift ramp 시작. 샘플된 박스 질량 0.41~0.80 kg, 박스 마찰 0.25~0.49.
- A 결과: run `..._facegrip_squeezelift/2026-11-04_22-13-23_squeezelift14_from_ms5698`(5698 → 6697, stdout `box_squeezelift.log`), 같은 평가:

  | | MinSqueeze 5698 | SqueezeLift A 6697 |
  |---|---|---|
  | 성공 / 끝 goal 거리 | 100% / 1.9 cm | 99.5% / 2.0 cm |
  | 들 때 최대 기울기 median / p90, 끝 기울기 | 8.7 / 11.7°, 3.7° | 7.8 / 10.5°, 6.2° |
  | 0.55 s 판 힘 min(L, R) (조임 factor) | 7.1 N (0.53) | 10.8 N (0.75) |
  | 들고 있을 때 판 힘 L / R | 16.1 / 16.3 N | 17.2 / 17.2 N |
  | 판 V자(pitch) 0.5 s / 1.0 s median | +5.7° / +4.7° | **+0.3° / 0.0°** |
  | 판 각도 합 0.5 s / 1.5 s median (p90) | 8.3° (14.9°) / 4.8° | 6.3° (14.8°) / 3.1° |
  | 판 yaw 첫 접촉 / 0.5 s median | | −4.8° / −4.8° |
  | 들 때 판 높이 차 \|zL − zR\| median (p90) | 1.5 (2.6) cm | **3.4 (4.6) cm** |
  | 회전 시작 토크 +y / −y | 1.18 / 1.33 Nm | 1.18 / 1.52 Nm |
  | j1·j3 PD 오차 median, clip 닿는 step 비율 p90 | 0.077~0.086 rad, 0 | 0.082~0.091 rad, 최대 0.18 |

  - V자(판 아래 모서리로 받치기)는 없어졌다. 남은 판 각도는 yaw(수직축 둘레, 첫 접촉부터 −5°)다. 접근 기울기 벌점으로는 줄지 않았다(접근 중 median 3.4° → 4.6°).
  - 대신 좌우 판 높이 차가 커졌고(영상 L − R +6.6~+7.8 cm) 끝 박스 기울기가 6.2°다. j1·j3이 행동 한계에 닿기 시작했다(왼 j3 p90 0.095).
  - 영상 `<run>/videos/model_6697_views_ep4.mp4`.
- B 결과: run `..._squeezelift_lowmuheavy_act015_scratch/2026-11-04_22-40-53_squeezelift_lowmuheavy_act015_scratch`(stdout `box_squeezelift_b.log`), 3000 iter 동안 success 0.
  - 1550 평가(B Play): 양 판으로 100% 들지만 1 s 26 cm → 4 s 22 cm로 미끄러져 내려와 goal(50 cm)에 못 간다(goal 거리 36 cm). 판 힘 끝 18 N, 최대 36 N, 관절 한계는 안 닿는다.
  - 조임 쪽 항(min_squeeze 20 N, 조임 factor)이 전부 success EMA 3 뒤에 켜지는데 success가 0이라 끝까지 꺼져 있다. yawtrack·외란·qprime도 같은 이유로 0이다.

### FaceGrip-BoxDrop 2999와 판 yaw (2026-11-05)

`-Wide-RollAlign-FaceGrip-BoxDrop-Scratch`(4090, BoxDrop + `plate_face_tilt`, `min_squeeze` 없음, privileged critic) 2999를 같은 방식으로 평가했다(박스 관측은 실제 값).

| | FaceGrip-BoxDrop 2999 | MinSqueeze-Scratch 2999 | SqueezeLift A 6697 |
|---|---|---|---|
| 성공 / 끝 goal 거리 | 100% / 1.3 cm | 100% / 2.7 cm | 99.5% / 2.0 cm |
| 들 때 최대 기울기 median / p90, 끝 기울기 | 4.7 / 7.1°, 1.9° | 8.7 / 12.2°, 3.7° | 7.8 / 10.5°, 6.2° |
| 판 V자(pitch) 0.5 s median | +1.0° | +10.7° | +0.3° |
| 들고 있을 때 판 힘 L / R | 13.4 / 13.5 N | 17.2 / 17.2 N | 17.2 / 17.2 N |
| 들 때 판 높이 차 \|zL − zR\| median | 0.8 cm | 0.8 cm | 3.4 cm |
| 회전 시작 토크 +y / −y | 1.10 / 1.09 Nm | 1.27 / 1.18 Nm | 1.18 / 1.52 Nm |
| j1·j3 PD 오차 median (행동 한계 0.1) | 0.068~0.071 rad, clip 0 | j1·j4 ~0.09 | 0.082~0.091, clip p90 최대 0.18 |

- 판 yaw(수직축 둘레, 판 법선 vs 면 법선)는 한쪽 판만 돌아간다. 두 판을 합친 median(0.5 s −5°)은 이것을 반으로 보여 준다.
  - FaceGrip-BoxDrop: 왼판이 접촉 −1.4° → 0.6 s −11.4° → 1.5 s −1.6°, 오른판은 0°. 박스는 1도 안쪽으로만 돈다.
    - 접촉 ~ 0.5 s 왼판 회전을 관절로 나누면 j4 −2.8°, j5 −3.8°, j6 −1.7°, j1·j3 ~0°. 오른팔은 같은 때 j5 +4.7°로 j4·j6 몫을 되돌린다.
    - j1·j3의 PD target은 6~19° 더 돌아가 있지만 박스 면이 막아서 실제 j1·j3는 거의 안 돈다.
  - A 6697: 오른판이 닿기 전부터 −10°로 와서 들 때 −12~13°, 1.5 s −7.6°. 왼판은 0°. 접근 기울기 벌점으로 안 고쳐진 접근 문제다.
- `plate_face_tilt`는 두 판 평균이라 11°/0°를 5.5°로 세고 ~1 s뿐이라 에피소드당 ~−1(총 보상 ~200)이다. V자(20° 넘게 들고 있는 내내)는 같은 weight로 고쳐졌다.
- 측정 도구: scratchpad `yaw_probe.py`(판별 yaw, 판 회전 vs 박스 회전, 관절별 기여, PD target에서의 yaw).
- 실로봇용 export: `fr3_husky_nn_policy/models/facegrip_boxdrop_scratch_model_2999.npz`, `facegrip_squeezelift14_model_6697.npz`(+ start poses yaml, 5698과 같은 5박스).

### 더 기운 판 기준 평행 (+ 판 법선 조임) (`-FaceGrip-BoxDrop-FlatOnly`, `-FlatSqueeze`, 2026-11-05)

둘 다 FaceGrip-BoxDrop-Scratch 2999에서 이어 학습한다. 관측·행동·critic 그룹·박스 관측 지연·끊김은 그대로라 실물 노드도 그대로 쓴다.
FlatOnly는 아래 표의 앞 두 항만, FlatSqueeze는 FlatOnly에 `min_squeeze`를 더한다. 나란히 학습해 조임 페널티가 평행·대칭에 주는 영향을 가른다.

| 항 | 바꾼 것 | 최종 weight |
|---|---|---|
| `plate_face_tilt` | 두 판 평균 → 더 기운 판(`plate_face_tilt_worst`). 학습된 −10에서 시작 | −20 |
| `plate_face_offcenter` (새) | 판 중심이 박스 면 중심에서 앞뒤(박스 x)로 벗어난 거리, 양 판 합, 양 판 접촉 중 | −20 /m |
| `min_squeeze` (FlatSqueeze만) | 합력 norm → 판 법선(TCP z) 성분(`min_squeeze_normal_penalty`), 14 → 16 N | −0.5 |

- 세 항 모두 success EMA 3 뒤 100 iter에 걸쳐 올린다. 이어 학습이라 yawtrack은 FaceGrip처럼 바로 1로 둔다.
- 높이(박스 z)는 중앙으로 끌지 않는다. 판이 박스 중심보다 위를 잡아야 박스가 판-판 축에 매달린 진자처럼 바로 선다. 중앙성을 들기 보상에 곱한 P30은 정밀 유지가 무너졌으므로 따로 둔 페널티다.
- 2999에서 잰 값(양 판 접촉·들린 동안): 더 기운 판 0.5~0.6 s 11.8~12.2°(평균식 6.4°), 앞뒤 오차 합 ~2 cm, 판 법선 힘 9.5~13.9 N(norm과 차이 0.3 N 안쪽), 16 N 부족분 평균 8.2 N.
  - 최종 weight에서 들고 있는 동안 초당 tilt −1.5(평균식 −10이면 −0.4), offcenter −0.45, min_squeeze −4.1.
- 확인: 두 task 모두 `train.py` 3 iter(64 env, 각 experiment의 `init_fgbd_2999/model_2999.pt` 로드) 정상, curriculum이 바로 ramp를 시작한다.
  reward·curriculum 설정 차이는 FlatOnly vs FaceGrip-BoxDrop-Scratch가 위 표의 앞 두 항과 yawtrack, FlatSqueeze vs FlatOnly가 `min_squeeze` 하나뿐이다.
- 학습 명령(이 PC, 각 1000 iter, `<V>`는 `FlatOnly` 또는 `FlatSqueeze`, run 이름은 `flatonly_from_fgbd2999` / `flatsqueeze_from_fgbd2999`):

  ```bash
  python scripts/rsl_rl/train.py --task Isaac-Lift-Box-DualFR3-Plate-Sim2Real-100Hz-Light-Pregrasp-Wide-RollAlign-FaceGrip-BoxDrop-<V>-v0 \
      --viz none --max_iterations 1000 --run_name <run 이름> \
      --resume --load_run init_fgbd_2999 --checkpoint model_2999.pt
  ```
