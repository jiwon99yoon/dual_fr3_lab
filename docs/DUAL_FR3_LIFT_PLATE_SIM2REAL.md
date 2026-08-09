# Dual FR3 Plate-EE Box Lift — Sim2Real 전략과 문헌 정리

> 현재 개발 태스크: `Isaac-Lift-Box-DualFR3-Plate-P39-v0`
> (동결 기준점: `Isaac-Lift-Box-DualFR3-Plate-v2`, P38)
>
> 현재 기준: **2026-08-08, P38 완료** (`2026-08-08_15-18-11`, 선택 checkpoint `model_9850.pt`;
> 상세 결과는 [`log_per_task.txt`](log_per_task.txt))
>
> 코드 기준: `source/dual_fr3_lab/dual_fr3_lab/tasks/manager_based/lift_plate/`
>
> 역할: 이 문서는 **실물 전이 계약, calibration, 문헌 근거, P39 이후 실험 설계와 안전 평가**를 관리한다.
> reward/P1~P38 설계 이력은 [`DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md`](DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md),
> 공통 원칙은 [`sim2real_gap.md`](sim2real_gap.md)를 본다.

---

## 0. 결론과 현재 결정

P38 이후의 우선순위는 관찰된 좌우 비대칭을 한 축씩 분리하는 다음 순서다.

1. **P38 완료:** late-checkpoint sweep 결과 `model_9850.pt`를 현재 sim 기준점으로 선택했다.
2. **P39:** 실제 Object 접촉의 좌/우 XOR 지속시간만 새 reward로 추가한다.
3. **P40:** raw action 범위와 physics-substep 재기준화를 bounded/snapshot action contract로 정리한다.
4. **P41:** q7 mirror 초기화를 먼저 검증한 뒤, 검증된 reflection만 mirror loss/augmentation에 사용한다.
5. 그 뒤 **asymmetric critic → 실측 temporal DR → history/GRU**를 각각 분리해 검증한다.
6. 병행해 실물 FR3·ArUco의 지연, 보간, bias, 좌우 skew와 plate/box 물성을 측정한다.
7. 질량·마찰 범위를 무작정 넓히지 않고 실측한 CoM/TCP/contact posterior로 DR을 조정한다.

현재 유지할 결정:

- **P38의 14D delta policy output은 유지**한다. 다만 현재 sim은 같은 delta를 100 Hz physics
  substep마다 최신 `q_current`에 다시 더하므로, 이를 50 Hz held target으로 해석해 바로 배포하지 않는다.
- **actor는 실기에 존재하는 56D 관측만 사용**한다.
- **V1/V2 actor에 plate force를 다시 넣지 않는다.** `tau_ext`는 우선 독립 안전 감시용이다.
- retry를 보존하기 위해 time/phase clock이나 late-contact termination을 넣지 않는다.
- P38의 soft base-yaw outcome은 유지하되 더 강화하지 않는다. Yaw는 해결됐고 남은 것은 contact timing/force다.
- 새로운 DR·history·critic·reward를 한 run에서 동시에 바꾸지 않는다. P39가 실패하면 그 reward를
  P40/P41에 누적하지 않고 마지막 합격 기준점에서 분기한다.

### P38 최종 판정

Corrected evaluator의 `Play-Noisy-v2`, full disturbance(±5 N/±1 Nm), 128 env × 1 episode × seeds
42/43/44에서 `model_9850.pt`를 평가했다.

| 지표 | P37 기준(seed 42) | P38 `model_9850` (3-seed 평균) |
|---|---:|---:|
| precise / upright / yaw-aligned hold | 0.606 / 0.662 / 0.103 | **0.737 / 0.787 / 0.786** |
| box yaw / grip-axis offset | 23.6° / 23.6° | **2.7° / 6.4°** |
| 0.5 s sustained SR: base / yaw<10° | 0.922 / 0.000 | **0.997 / 0.997** |
| plate-height difference p95 | 21.0 cm | **8.4 cm** |
| drop / OOB | 0 / 0 | **0 / 0** |

따라서 **base-yaw, position, upright를 동시에 유지하는 P38 outcome은 해결**됐다. 반면 최초 좌우 접촉
시차는 median 40~60 ms, 40 ms 이내 비율 0.477로 P37보다 개선되지 않았고, hold force p95 112 N 및
transient grasp-loss 3~4회/128 episodes가 남았다. 다음 단계는 yaw reward 추가가 아니라 접촉 timing과
force outlier를 실측·교정하는 것이다.

---

## 1. P38 기준 배포 계약

### 1-1. Task와 물리 설정

| 항목 | P38 / Plate-v2 기준 |
|---|---|
| 선택 checkpoint | `2026-08-08_15-18-11/model_9850.pt` |
| 물체 | 555 × 475 × 275 mm 박스, nominal 4.0 kg |
| 질량 DR | 3.5~4.5 kg, startup에서 env별 샘플 |
| 크기 DR | **OFF** (`scale=(1,1,1)`) — P25 confound 제거 후 v2에서 고정 |
| 마찰 DR | box static 0.4~1.2 / dynamic 0.3~1.0; plate USD 재질과 결합 |
| actuator DR | 양팔 14 joint PD stiffness/damping 각각 nominal의 0.85~1.15 |
| reset offset | nominal world center `(0.5,0,0.16)` 기준 x −0.05~+0.10 m, y ±0.10 m, yaw ±0.1 rad |
| 목표 | 초기 x/y를 유지하는 수직 lift, base-frame z 0.15~0.35 m |
| 외란 | success 기반 scale이 점진 활성화; 이후 phase gate 없이 각 축 ±5 N / ±1 Nm를 0.4~0.8 s마다 재샘플 |
| 제어 | physics 100 Hz, policy 50 Hz (`dt=0.01`, decimation 2) |
| episode | 5 s = 250 policy steps |
| robot gravity | robot link gravity OFF, object gravity ON — 실물 FR3 중력보상 근사용 잠정 baseline |

실제 box center x/y reset 범위는 0.45~0.60 m / -0.10~0.10 m다. 외란 curriculum의 활성 조건은
success지만, scale이 0보다 커진 뒤 wrench 자체에는 grasp/lift gate가 없어 접근 전 바닥 phase에도 적용된다.
따라서 현재 구현은 grasp-only 안정성 시험이 아니라 whole-episode disturbance다. 외란은 contact/latency
model의 대체물이 아니다.

### 1-2. Actor observation: 56D, force-free

| 순서 | 관측 | Dims | 실물 소스 |
|---|---|---:|---|
| 0 | 양팔 `joint_pos` (default 대비 상대값) | 14 | FR3 encoder |
| 1 | 양팔 `joint_vel` | 14 | FR3 state / filtered derivative |
| 2 | `object_position` in robot-root frame | 3 | 외부 카메라 + ArUco 6DoF |
| 3 | goal pose command | 7 | 초기 box x/y + 지정 z + identity quaternion `(1,0,0,0)` |
| 4 | previous action | 14 | policy runtime 내부 상태 |
| 5 | `object_orientation` quaternion, `w>=0` | 4 | 같은 ArUco pose frame |
| 합계 |  | **56** | |

현재 noise는 joint position 0.005 rad, joint velocity 0.05 rad/s, object position 0.01 m,
quaternion component 0.02 Gaussian이다. 마지막 방식은 회전군에 맞는 noise가 아니므로 sim2real branch에서는
**axis-angle/SO(3) noise 후 normalize + `w>=0`** 방식으로 교체한다.

Goal quaternion은 roll/pitch/yaw가 모두 0으로 고정된 **dead 4D input**이며 arbitrary yaw command를 학습한
것이 아니다. P38의 yaw 목표는 command quaternion이 아니라 robot-root/base-frame zero yaw다. 실물 runtime은
goal을 `(x_initial, y_initial, z_goal, 1, 0, 0, 0)`으로 만들고, ArUco/ROS의 흔한 `xyzw`를 actor가 쓰는
Isaac `wxyz`로 변환한 뒤 object quaternion의 부호를 `w>=0`으로 맞춘다.

주의: v0에는 `plate_forces` 2D 관측이 있었지만, V1이 이를 제거하고 V2가 V1을 상속한다. 따라서
**P38 actor는 `tau_ext`, contact force, true grasp flag를 보지 않는다.** Contact sensor truth는 reward,
curriculum, 향후 critic에서만 사용할 수 있다.

### 1-3. Action과 실기 의미

```text
policy @ 50 Hz: processed_delta = 0.1 * a              # 20 ms 동안 같은 delta
physics @ 100 Hz, substep k: q_target[k] = q[k] + processed_delta
```

`RelativeJointPositionAction.process_actions()`는 policy tick마다 한 번 실행되지만 `apply_actions()`는 두
physics substep 각각에서 최신 q로 target을 다시 만든다. 따라서 이것은 50 Hz에서 absolute target을 한 번
만들어 hold/interpolate하는 구조가 아니다. `a=1`을 단순히 5 rad/s rate limit으로 환산할 수도 없으며,
실제 속도는 PD gain·관성·effort/rate saturation에 따라 정해진다. Actor Gaussian output도 hard-bound되지 않는다.

실물 배포 전 둘 중 하나를 명시적으로 택한다.

1. **현재 sim semantics 재현:** raw delta를 policy interval 동안 hold하고 100 Hz adapter가 최신 q에 다시
   더해 target을 갱신한다. 실물 tracking·limit에서 안전한지 먼저 검증한다.
2. **snapshot target으로 변경:** policy tick의 q에 delta를 한 번 더한 absolute target을 hold/interpolate한다.
   이 방식은 현재 P38과 다른 action semantics이므로 sim action term을 같은 방식으로 바꾸고 재학습 또는
   검증된 fine-tuning을 거친다.

어느 경우든 이전 target 누적이나 neutral-offset action은 쓰지 않고, delta/action clamp와 joint별
position/velocity/acceleration/jerk hard limit을 둔다. Physics-substep `q`, processed delta, resulting target을
먼저 trace한 뒤 실물 adapter unit test의 기준으로 사용한다.

### 1-4. PPO 기준

- 4096 env, rollout 24 step/env, actor/critic `[256,128,64]` ELU.
- observation normalization OFF, `gamma=0.98`, `lambda=0.95`, initial action std 1.0.
- P38은 command-line override로 10,000 iterations를 실행한다.
- 1 iteration = 98,304 policy transitions. 10,000 iterations는 약 9.83억 transitions다.

---

## 2. 현재 커버리지와 남은 gap

| 축 | 현재 반영 | 남은 핵심 gap | 우선순위 |
|---|---|---|---:|
| Perception | position/orientation iid noise, pose를 base frame으로 관측 | frame rate, end-to-end latency, zero-order hold, dropout/occlusion, persistent extrinsic bias, correlated drift | 1 |
| Actuation | delta를 policy rate에 갱신하고 target을 physics substep마다 current-q로 재기준화, PD gain ±15% | substep semantics 재현 여부, command hold/interpolation, action delay/jitter, left/right differential skew, tracking bandwidth, rate/acc/jerk saturation | 1 |
| Robot geometry/inertia | CAD/USD kinematics, fixed base | joint zero, left/right base transform, plate TCP offset·normal·parallelism 오차; 현재 link7 모델에 추가 plate payload mass/inertia가 반영되지 않음 | 1 |
| Object/contact | mass·friction DR, external wrench | actual CoM/inertia, plate-box friction posterior, cardboard/plate compliance, rigid penetration exploit | 2 |
| Policy memory | previous action 1 frame | stale pose와 hidden dynamics를 추론할 history/GRU 없음 | 2 |
| Value learning | actor와 critic이 같은 관측 | clean state·hidden randomization을 보는 privileged critic 없음 | 1 |
| Force safety | excessive-squeeze reward, sim contact truth | 실물 `tau_ext` scale/bias/filter/threshold와 독립 watchdog | 1 (배포) |

특히 yaw는 reward만의 문제가 아닐 수 있다. 좌우 command latency, joint zero, plate normal, base extrinsic 중
하나가 비대칭이면 sim에서 학습한 동시 접촉도 실물에서는 한쪽 선접촉으로 바뀐다. 따라서 P37/P38의
접촉 대칭 보상과 별개로 **좌우 real tracking skew**를 측정해야 한다.

---

## 3. 관련 문헌과 직접 적용점

| 문헌 | 결과/기법 | 이 프로젝트에 적용 | 그대로 복사하지 않을 것 |
|---|---|---|---|
| [Lin et al., Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids, 2025](https://proceedings.mlr.press/v305/lin25c.html) | bimanual box lift 실물 80%; joint-target trajectory로 4분 미만 real-to-sim tuning; deployable actor + privileged critic; action/frame lag DR | 후속 asymmetric critic, FR3 tracking autotune, pose/action lag 분리의 가장 직접적인 근거 | 저가 humanoid hand의 DR 수치와 5 Hz policy를 FR3에 복사하지 않음 |
| [Dao et al., Sim-to-Real Learning for Humanoid Box Loco-Manipulation, 2023](https://arxiv.org/abs/2310.03191) | 1~10 kg box 학습; current-q relative action이 neutral-offset보다 빠르고 덜 forceful; hand waypoint가 sample efficiency 향상 | 현 relative action 유지, contact geometry prior와 state-reset curriculum | fixed clock contact/lift choreography와 strict late termination은 retry 목표와 충돌 |
| [Kataoka et al., Bi-Manual Block Assembly via Sim-to-Real RL, 2023](https://arxiv.org/abs/2303.14870) | dual xArm에서 noise와 action interpolation을 함께 맞춰야 실기 성공; current+previous observation 사용 | command hold/interpolation/latency와 observation history를 물성 DR보다 먼저 검증 | 논문의 force limit 단위·control rate를 FR3에 직접 이식하지 않음 |
| [DemoStart, 2024](https://arxiv.org/html/2409.06613) | demo action 대신 full simulator state를 reset seed로 사용; action space가 달라도 됨 | rule-based trajectory의 near-contact/stable-squeeze/early-lift state 사용 가능 | closest box-lift 연구에서는 generic human-init 이득이 90% vs 90%였으므로 1순위로 과대평가하지 않음 |
| [DeXtreme, 2023](https://arxiv.org/html/2210.13702) | deployable actor, large privileged critic, delay/bias/noise 분리, ADR와 memory | critic state 설계, persistent bias와 white noise 분리, reversible DR | dexterous in-hand cube의 DR 폭을 large box에 복사하지 않음 |
| [Learning Dexterous In-Hand Manipulation, 2018](https://arxiv.org/abs/1808.00177) | recurrent policy와 dynamics/vision randomization으로 zero-shot transfer; hard-to-model sensors는 actor에서 제외 | force-free actor와 history/GRU 방향 지지 | 막대한 randomization을 calibration 없이 먼저 적용하지 않음 |
| [IndustReal, 2023](https://arxiv.org/abs/2305.17110) | contact-rich Franka task에서 비물리적 penetration rollout을 식별·downweight; action integrator의 tracking 개선과 windup 보고 | plate-box signed penetration/impulse validity metric, 필요 시 rollout filter | contact에서 unclamped integrator 사용 금지 |
| [SimOpt, 2019](https://arxiv.org/abs/1810.05687), [AdaptSim, 2023](https://proceedings.mlr.press/v229/ren23b.html) | 소수 real rollout으로 task-relevant simulation distribution을 갱신 | zero-shot 실패 뒤 friction/compliance/delay posterior 보정 | inferred parameter를 물리적 ground truth로 단정하지 않음 |
| [SYMDEX, 2025](https://proceedings.mlr.press/v305/li25d.html) | bilateral symmetry를 inductive bias로 사용해 bimanual sample efficiency·robustness 향상 | 좌우 mirror augmentation/consistency loss 후순위 후보 | joint sign·base reflection·quaternion mapping 검증 없이 적용하지 않음 |
| [Bi-Touch, 2023](https://arxiv.org/abs/2307.06423) | sim policy의 excessive squeeze가 실물에서 실패 모드로 발현 | 접촉 강성·마찰 실측과 independent force watchdog 필요 | raw sim contact force를 real `tau_ext`와 등치하지 않음 |
| [Beltran-Hernandez et al., 2020](https://arxiv.org/abs/2003.00628) | rigid position robot에서 RL과 admittance/force control을 결합하고 safety filter 사용 | joint policy가 실물 squeeze를 못 맞출 때 1~2D squeeze-axis compliance를 후순위 검토 | 처음부터 14 joint별 gain을 policy action으로 열지 않음 |

이 문헌들로부터 이 프로젝트가 취하는 종합 결론은 **배포 신호를 제한한 actor, calibration, 실제
시간구조, structured noise, 필요할 때의 memory**를 먼저 검증한다는 것이다. Extensive DR가 핵심인 연구도
있지만 그 수치 범위를 calibration 없이 Dual FR3에 복사하지 않는다.

---

## 4. 실물 calibration 계획

### 4-1. Robot tracking과 timing

박스 없이 안전한 workspace에서 양팔 각각과 양팔 동시로 작은 step/multisine joint target을 실행한다.

필수 로그:

```text
t_sensor, t_policy_start, t_policy_end, t_command
q_des, q, dq, tau, tau_ext
controller mode, command accepted/rejected, watchdog state
```

추정할 값:

- joint별 command→motion delay, rise time, bandwidth, steady-state error.
- policy tick jitter와 저수준 보간/hold 방식.
- velocity/acceleration/jerk saturation과 명령 reject 조건.
- 왼팔과 오른팔의 delay·gain·zero offset 차이.
- 단독 동작과 동시 동작에서 tracking이 달라지는지.

먼저 sim의 physics-substep `q`, processed delta, resulting target을 trace해 실제 semantics를 고정한다.
DelayBuffer를 넣을 때도 **raw delta의 전송 지연**과 이미 계산된 **absolute target의 지연**은 다른 모델이므로
구분한다. 첫 모델은 `ImplicitActuator`를 바로 교체하지 않고 선택한 action adapter에 측정된
hold/delay/interpolation을 넣으며, actuator class 교체는 별도 실험으로 둔다.

### 4-2. Perception timing과 error structure

필수 로그:

```text
camera capture timestamp, pose publication timestamp, policy-consumed timestamp
object pose, covariance/quality, marker IDs, valid/stale flag, frame name
```

확인할 값:

- 실제 update rate와 capture→policy end-to-end delay 분포.
- 동일 frame을 몇 policy step 재사용하는지(zero-order hold).
- marker 가림·motion blur 때 dropout 길이와 recovery 방식.
- camera-to-base calibration의 episode-persistent translation/rotation bias.
- position과 orientation이 같은 timestamp·같은 valid flag를 쓰는지.

Sim에서는 object position과 quaternion에 각각 독립 iid noise를 주지 않고, 하나의 stateful pose-estimator
proxy가 **SE(3) bias + frame noise + delay + sample-hold + dropout**을 함께 적용하도록 한다. Actor에는
`pose_age` 또는 stale age를 주는 방안을 history와 분리해 검증한다.

### 4-3. Geometry와 contact

- left/right arm base transform, joint zero, plate TCP 위치를 측정한다.
- 두 plate 작업면 normal과 평행도, mount compliance를 측정한다.
- 현재 sim link7에 별도 반영되지 않은 plate payload의 실제 질량·CoM·inertia를 측정하고 모델링 방식을 결정한다.
- 박스 질량뿐 아니라 CoM offset과 가능한 inertia 범위를 확인한다.
- 저속·저하중 press/slip 또는 pull/tilt test로 plate-box static/dynamic friction 범위를 얻는다.
- 대표 자세에서 low-force contact를 수행해 sim/real의 `q_des-q`, `tau_ext`, box motion trajectory를 비교한다.
- 실물 controller에 plate payload mass/CoM을 등록한 뒤 gravity compensation residual을 다시 측정한다. Sim의
  robot-link gravity OFF는 중력 항 근사일 뿐 payload inertia와 tracking dynamics를 보상하지 않는다.

`excessive_squeeze`의 sim 100 N threshold는 학습용 PhysX contact 값이다. 이를 실물 `tau_ext` threshold로
직접 사용하지 않는다. 실물 force watchdog threshold는 robot·plate·box 안전 한계와 calibration 결과로
별도 결정한다.

---

## 5. P39 이후 학습 로드맵

### 5-1. P39 확정: actual one-sided-contact XOR duration

P37의 `plate_approach_symmetry`는 TCP-to-face 거리차만 보므로 실제 접촉 onset을 보장하지 못했다.
P38 `model_9850`의 384 episodes에서 최초 접촉은 left/right/tie=`371/3/10`이었고,
`|dt|<=20/40 ms` 비율은 `0.164/0.477`이었다. 양판 접촉 뒤 actual joint path도 R/L=`1.35`였다.

P39는 P38/V2를 상속하고 다음 항 하나만 추가한다.

```text
C_L = 1[||F_L,Object|| > 1 N]
C_R = 1[||F_R,Object|| > 1 N]
X   = C_L XOR C_R
r_xor = w_xor * X

w_xor = 0.0                         # discovery 중 P38 reward와 정확히 동일
w_xor = -5.0  if lifting EMA > 10  # 기존 lifting-learned latch
```

별도 timestamp/latch 없이 RewardManager가 매 20 ms마다 적분한다. 따라서 최초 선접촉뿐 아니라 운반 중
한쪽 판 접촉을 잃은 시간도 비용에 포함된다. `plate_approach_symmetry=-20`을 포함한 P38의 reward,
action, observation, DR, PPO는 모두 유지한다.

| 항목 | 설정 |
|---|---|
| train task | `Isaac-Lift-Box-DualFR3-Plate-P39-v0` |
| play / noisy eval | `Isaac-Lift-Box-DualFR3-Plate-P39-Play-v0` / `Isaac-Lift-Box-DualFR3-Plate-P39-Play-Noisy-v0` |
| env cfg | `DualFR3PlateBoxLiftEnvCfg_P39` |
| train seed / iterations | 42 / 10,000, from scratch |
| log root | `logs/rsl_rl/dual_fr3_plate_lift_p39/` |
| checkpoint protocol | `model_9850.pt` primary, `model_9999.pt` late-regression sentinel |

50 Hz에서 full weight의 one-sided step은 return `-5*0.02=-0.1`이다. 기존 2.68 step(53.6 ms)
진단값이면 약 `-0.268`, 0.5 s면 `-2.5`, 1 s면 `-5`다. 효과가 작고 task가 안정적일 때만
별도 P39b에서 -10을 검토하며 첫 run에 -20을 사용하지 않는다.

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Lift-Box-DualFR3-Plate-P39-v0 \
  --headless --seed 42 --max_iterations 10000
```

P39 평가(아래 명령을 seed 42/43/44로 반복):

```bash
python scripts/tools/eval_dual_fr3_plate_robustness.py \
  --task Isaac-Lift-Box-DualFR3-Plate-P39-Play-Noisy-v0 \
  --checkpoint logs/rsl_rl/dual_fr3_plate_lift_p39/<run>/model_9850.pt \
  --num_envs 128 --episodes 1 --seed 42 \
  --disturbance_scale 1.0 --headless
```

합격은 train return이 아니라 updated evaluator의 동일 full-episode 정의로 판정한다. 기존 2.68 step은
first-contact 기반 진단값이므로 P38도 새 evaluator로 다시 재어 공정한 full-episode 기준값을 고정한다.

- mean one-sided duration: P38 대비 50% 이상 감소.
- first-contact `<=20 ms >=0.50`, `<=40 ms >=0.75`, p95 `<=60 ms`.
- actual bilateral-contact episode `>=0.99`, dominant first side `<=0.80`.
- precise hold `>=0.70`; upright/yaw-aligned hold `>=0.75`; box yaw `<=5 deg`.
- sustained yaw10 SR `>=0.95`; drop/OOB=0; hold-force p95 `<=115 N`.
- grasp-loss `<=4 events/128 episodes`.
- post-bilateral R/L joint-path ratio는 1.35보다 악화되면 안 되며, `<=1.15`는 stretch goal이다.

접촉 timing이 합격했지만 joint-path ratio가 남으면 “접촉 원인 해결, motion symmetry 미해결”로 판정한다.

### 5-2. P40: bounded/snapshot action interface

현재 deterministic raw action은 대부분 `[-1,1]` 밖이고, 같은 delta를 100 Hz physics substep마다 최신
`q_current`에 다시 더한다. 단순 post-hoc clip은 P38/P39와 다른 정책이므로 P40은 scratch로 재학습한다.

- physics-substep target trace로 현재 semantics를 먼저 확정한다.
- policy tick에서 absolute `q_target`을 snapshot하고 다음 tick까지 hold하는 contract를 구현한다.
- raw policy output을 명시 범위로 제한하고 saturation 비율, max joint travel, force tail을 기록한다.
- P39가 합격한 경우에만 selected P39 위에 누적한다. 실패하면 P38에서 분기한다.
- action contract 변경과 latency/interpolation DR는 같은 run에 넣지 않는다.

### 5-3. P41: q7 mirror 검증 후 mirror loss/augmentation

현재 양팔 q7 default는 모두 `+pi/4`지만 exact reflection은 right q7 `-pi/4`를 요구한다. 먼저 q7
초기화만 바꾼 P41a로 reset/FK/contact 영향을 분리한다. 그 뒤에만 P41b mirror loss/augmentation를 검토한다.

- 실제 joint/action order `[L1,R1,L2,R2,...,L7,R7]`를 사용한다.
- `M(M(x))=x`, FK-reflected TCP, joint limit/default, object/goal y, quaternion을 unit test한다.
- reward/termination/contact label invariance까지 통과하지 않으면 symmetry 학습을 시작하지 않는다.
- mirror loss와 data augmentation는 서로 다른 ablation으로 둔다.

### 5-4. P42 이후: asymmetric critic과 temporal Sim2Real

P42 asymmetric critic은 actor의 noisy 56D와 reward/action을 유지하고 critic에만 clean object velocity,
plate pose/contact, mass/friction/CoM, actuator gain, disturbance/curriculum scale을 준다. 이는 value
estimation을 돕지만 memoryless actor의 online system identification을 대신하지 않는다.

그 뒤 실측 범위로 P43 temporal factorial을 수행한다.

| 후보 | 단일 변경 | 질문 |
|---|---|---|
| P43-A | measured action hold/delay/interpolation | actuation timing만으로 성능이 얼마나 변하는가? |
| P43-B | pose sample-hold/lag/dropout/bias | perception timing만으로 성능이 얼마나 변하는가? |
| P43-C | A+B | 두 gap의 조합에서 policy가 유지되는가? |

P44에서 measured maximum latency를 덮는 short history를 먼저 비교하고, 부족할 때만 작은 GRU/LSTM을
검토한다. RSL-RL checkpoint는 custom curriculum scale/EMA를 자동 복원하지 않을 수 있으므로 resume/fine-tune
실험은 scale을 저장·복원하거나 시작값을 명시적으로 고정한다.

### 5-5. 후순위: targeted reset

학습 병목이 계속 `first contact -> stable bilateral squeeze`에 있을 때만 rule-based/success rollout의
full simulator state를 저장해 reset seed로 쓴다.

- native reset 70~80%, assisted reset 20~30%로 시작.
- `near-face`, `first-contact`, `stable-squeeze`, `early-lift` anchor를 구분.
- 현재 policy가 항상 실패하거나 항상 성공하는 anchor보다 중간 성공률 anchor에 가중.
- policy가 안정되면 earlier/native reset 비율을 높임.
- demo action은 behavior cloning에 쓰지 않으므로 rule-based action space가 달라도 무관.

### 5-6. PPO 후순위 screen

구조 실험과 섞지 않고 다음을 한 축씩 본다.

- `gamma 0.98 -> 0.99`: 50 Hz에서 3초 뒤 reward weight가 약 0.048에서 0.221로 증가한다.
- rollout `24 -> 32/48`: 현재 24 step은 0.48 s로 success hold 0.5 s보다 짧다.
- observation normalization ON/OFF: actor/critic을 분리해 비교한다.

---

## 6. 구현 위치

| 변경 | 코드 위치 / 방식 |
|---|---|
| actor observation | `lift_plate/joint_pos_env_cfg.py`의 policy observation group |
| privileged critic | 같은 cfg에 별도 `critic` observation group 추가 + RSL-RL `obs_groups` 연결 |
| action semantics/delay | custom action term에서 raw delta와 substep-rebased/absolute target을 분리하고, 지연시킬 신호를 명시한 뒤 `DelayBuffer` 적용 |
| pose delay | object position/orientation을 한 stateful observation term에서 함께 생성 |
| history | Isaac Lab `ObservationTermCfg.history_length` 또는 group history 사용 |
| CoM DR | mass/friction startup event 옆에 rigid-body CoM randomization 추가 |
| plate compliance | welded rigid plate USD이므로 단순 cfg 변경이 아님; contact calibration 뒤 후순위 |
| run 재현 | 매 run의 resolved env/agent cfg, custom curriculum scale, git commit/diff를 함께 보존 |

`play.py`는 checkpoint 폴더의 historical env를 자동 재구성하지 않고 현재 task cfg를 사용할 수 있다.
과거 checkpoint 비교 시 저장 당시 env cfg를 복원하지 않으면 코드 변화가 정책 차이처럼 보일 수 있다.

---

## 7. 평가 프로토콜

### 7-1. Simulation checkpoint selection

모든 checkpoint를 다음 고정 seed suite에서 비교한다.

1. nominal clean.
2. current training distribution.
3. one-factor sweeps: mass, friction, CoM, left/right gain skew, action delay, pose delay, pose bias/dropout, wrench.
4. measured posterior의 corner combinations.

필수 stage metric:

- both-near-face / both-contact / lifted / goal-hold / sustained success 비율.
- left/right first-contact time difference와 contact onset pose.
- yaw mean·median·p95, yaw<10° 비율, roll/pitch tilt.
- slip/drop, peak·impulse squeeze proxy.
- action rate/jerk, joint tracking residual, saturation 횟수.
- nonfinite observation/value/action과 out-of-bounds 비율.

선택 기준은 마지막 checkpoint나 train mean reward가 아니라 **held-out 성공·정렬·안전 지표의 Pareto**다.

`eval_dual_fr3_plate_robustness.py`는 P38 평가 후 v2 기본 task, seed, 고정 disturbance scale, bilateral
contact+upright+goal의 0.5 s sustained episode SR, yaw 10°/15° 결합 SR, 최초 접촉 시차, force/height
비대칭과 실제 done/OOB를 출력하도록 보완했다. Mass·friction·gain은 startup event이므로 32 env × 4
episodes는 128회가 아니라 **32개의 독립 dynamics draw**다. 최종 비교는 `--num_envs 128 --episodes 1`을
seed 42/43/44로 반복하고, 학습 말기 물리는 `--disturbance_scale 1.0`으로 명시한다.

### 7-2. Real deployment ladder

1. **No object / slow free-space:** policy 또는 scripted target을 낮은 scale로 실행하고 tracking·frame 확인.
2. **Object present / no contact:** 접근까지만 허용하고 좌우 TCP·ArUco·workspace 확인.
3. **Low-force contact / no lift:** symmetric squeeze, contact onset과 `tau_ext` 확인.
4. **Low lift:** 2~5 cm만 들어 slip·yaw·watchdog 확인.
5. **Nominal lift/hold:** 목표 높이·0.5 s hold.
6. **Robustness:** 초기 pose, 날짜, 마찰 조건, 안전한 외란을 점진 확대.

각 condition은 checkpoint 순서를 섞고 최소 20~30 trials, 가능하면 여러 날짜에 평가한다. 실패는
perception stale, one-side contact, insufficient/excessive squeeze, slip, yaw/tilt, controller reject,
collision/workspace로 분류한다.

### 7-3. 실물 safety supervisor

Safety는 actor 학습과 독립이어야 한다.

- object pose stale/invalid watchdog.
- joint position/velocity/acceleration/jerk hard limit.
- payload tare 뒤 joint별 posture-dependent `tau_ext` envelope 또는 검증된 Jacobian wrench threshold와 sustained-duration 조건.
- self/inter-arm/workspace collision check.
- box drop·out-of-workspace abort.
- command heartbeat, controller fault, physical E-stop.

Force observation 삭제는 force safety 삭제가 아니다. 다만 FR3 `tau_ext_hat_filtered`는 payload/model error와
모든 외부 접촉이 섞인 **joint-space residual**이어서 per-plate normal force와 1:1 대응하지 않는다. Payload
등록 후 bias/tare를 수행하고, joint별 posture-dependent envelope 또는 검증된 Jacobian wrench estimate를
사용한다. Robot 내부 reflex가 가장 낮은 층에서 동작하고, 그 위에 외부 watchdog과 policy command limiter를 둔다.

---

## 8. 배포 전 체크리스트

- [x] P38 late-checkpoint sweep과 3-seed full-disturbance held-out 평가 완료 (`model_9850.pt` 선택).
- [ ] run별 resolved env/agent cfg와 curriculum scale 저장.
- [ ] actor 56D 순서·단위·frame을 sim/real unit test로 일치시킴.
- [ ] current-q relative action semantics를 실물에서 확인.
- [ ] sim physics-substep target trace와 실물 100 Hz adapter 또는 snapshot-action branch의 의미를 일치시킴.
- [ ] 양팔 tracking latency·interpolation·left/right skew 측정.
- [ ] ArUco capture→policy latency, rate, bias, dropout 측정.
- [ ] base/camera/arm/TCP frame chain 검증.
- [ ] plate payload 질량·CoM·inertia 측정 및 sim/실물 controller 양쪽의 반영 여부 확인.
- [ ] payload tare 후 `tau_ext` posture envelope/Jacobian wrench safety threshold 검증.
- [ ] 4 kg 박스 질량·CoM·마찰·접촉 변형 측정.
- [ ] action/pose temporal factorial 통과.
- [ ] nominal 및 measured-posterior sim suite 통과.
- [ ] independent safety supervisor dry-run 완료.
- [ ] §7-2의 real ladder를 순서대로 통과.

실측값과 run 결과는 [`log_per_task.txt`](log_per_task.txt)에 남기고, 전략 변경은 이 문서와
[`DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md`](DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md)의 해당 roadmap에 교차 기록한다.
