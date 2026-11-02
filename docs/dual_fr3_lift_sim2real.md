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
  - 최종: 4096 env, seed 0 → 3976줄(97.1%), `assets/pregrasp/plate_box410_pregrasp_gap004_margin005.npz`. 555 박스 표는 지웠다.
  - x 0.45~0.50은 99%, 0.50~0.52는 92%가 유효하다. 관절 여유 중앙값은 0.089 rad다.
- reset 확인: 판-면 간격 중앙값 4.1 cm(2.0~7.1), 판 높이는 박스 중심 + 5.3 cm다. 접촉 0, 박스 이동 0이었다.
- 학습(01:25 시작, 두 run 동시, 각 3000 iter from scratch, 약 2.8 s/iter, 약 2시간 20분): run_name `pregrasp_box410`, `pregrasp_upright_box410`. stdout은 repo 루트 `box_pregrasp_box410.log`, `box_pregrasp_upright_box410.log`다.

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
