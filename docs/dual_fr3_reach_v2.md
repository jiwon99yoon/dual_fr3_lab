# Dual FR3 Reach Sim2Real v2

`Isaac-Reach-DualFR3-Sim2Real-v2`는 Reach v1의 observation, bounded actor,
relative action 의미와 20 Hz policy 주기를 유지하면서, simulation physics와
explicit PD/torque 계산을 100 Hz에서 1 kHz로 올린 재학습용 환경이다.

목표는 MuJoCo 및 실제 FR3 controller의 정상적인 1 kHz torque 제어 구조와
학습환경의 시간 이산화를 가깝게 맞추는 것이다. v1에서 사용했던 100 Hz
torque 계산을 배포 controller에 인위적으로 재현하는 torque gating은 사용하지
않고, 양쪽 모두 1 kHz에서 torque를 새로 계산하도록 한다.

## 설계 목표

v2는 v1의 다음 policy contract를 그대로 유지한다.

- observation 58D, action 14D
- `left joint1~7, right joint1~7` 관절 순서
- tanh-squashed Gaussian actor output in `(-1, 1)`
- 최대 `+-0.1 rad`의 measured-relative joint-position offset
- 20 Hz policy inference와 50 ms action hold
- Kp=80, Kd=4 explicit PD
- FR3 관절별 torque magnitude limit과 1000 Nm/s torque-rate limit
- 별도의 velocity/acceleration trajectory limiter 없음

핵심 변경은 action을 생성하는 주기가 아니라, held action을 target과 torque로
변환하고 physics에 반영하는 내부 주기다.

```text
policy inference                       20 Hz (50 ms)
  -> tanh action a_t in (-1, 1)
  -> processed offset Delta q = 0.1 * a_t
  -> 같은 Delta q를 50 physics step 동안 hold

매 physics step                        1 kHz (1 ms)
  -> q_desired = q_measured + Delta q
  -> tau_PD = Kp * (q_desired - q_measured) - Kd * qdot
  -> torque magnitude saturation
  -> torque-rate limit (최대 1 Nm/update)
  -> PhysX integration
```

`RelativeJointPositionAction`은 50 ms 동안 절대 `q_desired`를 고정하는 것이
아니다. processed offset `Delta q`를 유지하고, 매 1 ms physics substep마다
현재 joint position에 더해 `q_desired`를 다시 계산한다. 따라서 배포
controller도 같은 measured-relative semantics를 구현해야 한다.

## v1 대비 변경점

| 항목 | Reach Sim2Real v1 | Reach Sim2Real v2 |
|---|---|---|
| Gym ID | `Isaac-Reach-DualFR3-Sim2Real-v1` | `Isaac-Reach-DualFR3-Sim2Real-v2` |
| physics dt / rate | 0.01 s / 100 Hz | **0.001 s / 1 kHz** |
| policy decimation | 5 | **50** |
| policy rate | 20 Hz | 20 Hz |
| policy action hold | 50 ms | 50 ms |
| explicit PD update | 100 Hz | **1 kHz** |
| actuator `control_dt` | 0.01 s | **0.001 s** |
| torque-rate limit | 1000 Nm/s | 1000 Nm/s |
| step당 최대 torque 변화 | 10 Nm / 10 ms | **1 Nm / 1 ms** |
| raw action | `tanh(z)`, `(-1, 1)` | 동일 |
| processed offset | 최대 약 `+-0.1 rad` | 동일 |
| PD gain | Kp=80, Kd=4 | 동일 |
| torque magnitude | joint1~4: 87 Nm, joint5~7: 12 Nm | 동일 |
| observation/action 차원 | 58 / 14 | 동일 |
| episode 길이 | 8 s = 160 policy steps | 동일 |
| RSL-RL 로그 | `dual_fr3_reach_sim2real_v1` | `dual_fr3_reach_sim2real_v2` |

v1과 v2 모두 50 ms마다 새 policy action을 받는다. 차이는 그 사이에 v1은
5회의 10 ms integration/PD update를 수행하고, v2는 50회의 1 ms
integration/PD update를 수행한다는 점이다.

v2의 torque 처리 순서는 다음과 같다.

```text
explicit PD torque (1 kHz)
  -> magnitude saturation (87/12 Nm)
  -> torque-rate limit (+-1 Nm per 1 ms)
  -> PhysX (1 kHz)
```

이 구조에서는 10 ms 동안 허용되는 총 torque 변화량이 최대 10 Nm로 v1과
같지만, 그 변화가 10 ms 시점에 한 번 발생하지 않고 1 ms마다 최대 1 Nm씩
나뉘어 발생한다. 실제 FR3/libfranka의 1 kHz torque-rate limiting에 더 가까운
시간 이산화다.

## Action과 observation 순서

v2의 action과 joint position/velocity observation 순서는 v1과 동일하게
다음 순서로 고정한다.

```text
left_fr3_joint1 ... left_fr3_joint7,
right_fr3_joint1 ... right_fr3_joint7
```

action 식은 다음과 같다.

```text
a_t        = tanh(actor_output)             # 14D, (-1, 1)
Delta q_t  = clip(0.1 * a_t, -0.1, 0.1)    # rad
q_desired  = q_measured + Delta q_t         # 매 1 ms 재계산
```

observation 58D 구성도 v1과 동일하다.

| 순서 | observation | 차원 | 비고 |
|---:|---|---:|---|
| 1 | relative joint position | 14 | Gaussian noise std 0.005 |
| 2 | relative joint velocity | 14 | Gaussian noise std 0.05 |
| 3 | reach center position | 3 | root frame, noise std 0.01 |
| 4 | target pose command | 7 | position + quaternion |
| 5 | previous action | 14 | 직전 policy action |
| 6 | plate force compatibility slot | 2 | 항상 0 |
| 7 | object orientation compatibility slot | 4 | 항상 0 |
|  | 합계 | **58** | |

`object_orientation`은 실제 EE orientation observation이 아니다. 또한 reach
reward는 두 EE의 position만 평가하므로 특정 tool orientation을 task contract로
보장하지 않는다.

## MuJoCo dynamics와 맞춘 현재 설정

현재 v2 actuator는 MuJoCo FR3 MJCF와 수치 및 단위를 가깝게 맞추기 위해 다음
passive joint parameter를 사용한다.

| 항목 | Isaac Reach v2 | MuJoCo 대응 |
|---|---:|---:|
| armature | 0.1 kg*m^2 | `armature=0.1` |
| static friction effort | 0.2 Nm | `frictionloss=0.2`에 대한 근사 |
| dynamic friction effort | 0.2 Nm | `frictionloss=0.2`에 대한 근사 |
| viscous friction | 0.003 Nm*s/rad | `damping=0.003` |
| robot gravity | disabled | MuJoCo gravity disabled |

PhysX의 static/dynamic friction과 MuJoCo의 constraint 기반 `frictionloss`는
solver 구현이 다르므로 위 값이 같아도 stick-slip 거동까지 완전히 같지는 않다.
armature와 friction 값은 현재 MuJoCo 모델과의 sim-to-sim 정합을 위한 값이며,
실제 FR3에서 관절별로 식별된 물리 파라미터는 아니다.

중력 OFF는 실제 로봇에 중력이 존재하지 않는다는 뜻이 아니다. 실제 Franka
hardware의 내부 중력보상 이후 외부 torque controller가 보는 관절 동역학을
근사하기 위한 설정이다. payload 오차, 내부 보상 오차와 motor dynamics는
여전히 sim-to-real 차이로 남는다.

주의할 점은 검증된 v1 reference run
`2026-09-03_18-57-37`의 저장된 `env.yaml`에는 armature/friction override가
없다는 것이다. 이 passive dynamics 설정은 그 checkpoint 이후 현재 actuator
configuration에 추가되었으므로, 과거 v1 checkpoint와 현재 v2 dynamics는
동일하지 않다. v2는 반드시 처음부터 재학습한다.

## Policy와 PPO 구조

v2는 v1의 actor, critic과 PPO 설정을 그대로 상속한다.

```text
Actor : 58 -> 256 -> 128 -> 64 -> 14 -> tanh
Critic: 58 -> 256 -> 128 -> 64 -> 1
activation: ELU
observation normalization: off
actor distribution: tanh-squashed Gaussian (init_std=1.0, std_type=log)
```

PPO log-probability에는 tanh Jacobian correction이 포함된다. entropy는 latent
Gaussian이 아닌 squashed distribution에서 추정하며, `clip_actions=1.0`은
actor 범위를 만드는 수단이 아니라 수치적 safety guard로 유지한다.

주요 PPO 설정은 `num_steps_per_env=24`, `gamma=0.98`, `lambda=0.95`,
`learning_rate=1e-4`, `entropy_coef=0.006`, `max_iterations=5000`이다.
policy rate가 v1과 같은 20 Hz이므로 rollout 24 step의 물리 시간은 두 버전 모두
1.2초다. 다만 v2는 각 policy step에 physics substep이 10배 많아 계산 비용이
증가한다.

## 구현 위치

- v2 environment와 1 kHz 설정: `tasks/manager_based/reach/joint_pos_env_cfg.py`
- v2 Gym/Play 등록: `tasks/manager_based/reach/__init__.py`
- PPO actor 설정과 v2 로그 이름: `tasks/manager_based/reach/agents/rsl_rl_ppo_cfg.py`
- tanh-squashed distribution: `tasks/manager_based/reach/agents/tanh_squashed_gaussian.py`
- torque saturation/rate limiter: `dual_fr3_lab/actuators/torque_rate_limited_pd.py`
- armature/friction 및 v2 actuator dt: `dual_fr3_lab/robots/dual_fr3_plate_ee.py`
- PhysX open-loop 비교 도구: `scripts/tools/record_reach_v2_torque_response.py`

## 학습

v2는 physics/actuator dt와 passive joint dynamics가 v1 checkpoint의 학습환경과
다르므로 v1 checkpoint에서 resume하지 않고 처음부터 학습한다.

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Reach-DualFR3-Sim2Real-v2 \
  --headless \
  --max_iterations 5000
```

기본 병렬 환경 수는 4096이다. 1 kHz physics로 GPU 메모리나 처리 시간이
부족하면 `--num_envs 2048` 등으로 줄인다.

학습 결과는 다음 위치에 저장된다.

```text
logs/rsl_rl/dual_fr3_reach_sim2real_v2/<run>/model_XXXX.pt
```

## 영상 검증

v2의 완료된 5000-iteration checkpoint에 대한 play 영상 검증은 아직 수행되지
않았다. 다음 항목을 확인한 뒤 reference checkpoint를 결정한다.

- 여러 target에서 양팔 EE가 모두 도달하는가
- target 도달 후 1 kHz PD에서 고주파 진동이 발생하지 않는가
- wrist와 plate orientation이 과도하게 뒤틀리지 않는가
- goal resampling과 episode reset을 지속 진동으로 잘못 판단하지 않았는가
- action/torque trace에서 limiter saturation이 반복되지 않는가

orientation 안정성은 영상으로 관찰할 수 있지만 현재 position-only reward가
이를 보장하지는 않는다.

## Play와 고정 goal sequence

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Reach-DualFR3-Sim2Real-Play-v2 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/dual_fr3_reach_sim2real_v2/<run>/model_4999.pt \
  --viz kit
```

고정된 goal로 v1, v2 또는 MuJoCo 결과를 반복 비교할 때만 다음 옵션을 추가한다.

```text
--reach_goal_sequence config/reach_goal_sequence.example.json
```

## 배포 주의사항

- v1 checkpoint를 v2 reference checkpoint 대신 사용하지 않는다.
- 배포 NPZ는 actor 마지막의 tanh를 포함해야 한다.
- 배포 노드의 `[-1, 1]` clipping은 수치적 safety guard로 유지한다.
- policy action은 20 Hz로 생성하고 같은 offset을 50 ms 동안 유지한다.
- controller는 1 kHz마다 `q_measured + held_offset` target을 다시 계산한다.
- explicit PD torque도 1 kHz마다 다시 계산하며 100 Hz torque gating은 사용하지 않는다.
- torque는 PD 이후 joint1~4 87 Nm, joint5~7 12 Nm로 제한한다.
- 1000 Nm/s torque-rate limiter는 1 kHz에서 최대 1 Nm/update로 적용한다.
- Isaac과 MuJoCo의 solver/integrator 및 실제 FR3 motor dynamics 차이는 여전히 남는다.
- 실로봇 실행 전 shadow mode에서 observation, action joint 순서와 target 범위를 검증한다.
