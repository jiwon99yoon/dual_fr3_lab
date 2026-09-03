# Dual FR3 Reach Sim2Real v1

`Isaac-Reach-DualFR3-Sim2Real-v1`은 Reach v0에서 확인한 실로봇
power-limit 문제를 줄이기 위해 action 범위와 actuator 안전 제약을 명시한
재학습용 환경이다. tanh-squashed actor로 처음부터 재학습한
`2026-09-03_18-57-37/model_4999.pt`에서 목표 도달과 안정적인 유지 동작을
확인했으며, 현재 simulation 기준 v1 reference checkpoint로 사용한다.

## 설계 목표

v1은 기존 v0의 observation(58D), goal 범위, ready pose, 20 Hz policy와
다음 relative action 의미를 유지한다.

```text
q_target = q_measured + 0.1 * tanh(latent_action)
```

policy action은 20 Hz로 갱신되며, 같은 processed offset을 유지한 채
`RelativeJointPositionAction`이 각 100 Hz physics substep에서 현재 관절 위치를
기준으로 target을 다시 계산한다.

실로봇과 학습환경 모두에서 별도의 velocity/acceleration trajectory limiter는
사용하지 않는다. Limiter의 내부 reference state가 policy가 학습하지 않은
추가 동역학을 만드는 것을 피하기 위해서다.

대신 다음 제한을 학습환경과 배포환경에 공통으로 적용한다.

- tanh-squashed Gaussian actor output in `(-1, 1)`
- wrapper clipping as a numerical safety guard
- FR3 관절별 torque magnitude limit
- libfranka와 같은 torque-rate limit

## v0 대비 변경점

| 항목 | Reach v0 | Reach Sim2Real v1 |
|---|---|---|
| Gym ID | `Isaac-Reach-DualFR3-v0` | `Isaac-Reach-DualFR3-Sim2Real-v1` |
| raw action | 별도 제한 없음 | `tanh(z)`, 항상 `(-1, 1)` |
| processed offset | `0.1 * raw_action` | `0.1 * tanh(z)`, 최대 약 `±0.1 rad` |
| policy distribution | unbounded Gaussian | tanh-squashed Gaussian + corrected log-prob/entropy |
| 14-D joint 순서 | asset 내부 순서 사용 | `left joint1~7, right joint1~7`로 명시 고정 |
| PD | PhysX implicit PD | explicit PD |
| PD gain | Kp=80, Kd=4 | Kp=80, Kd=4 |
| torque magnitude | PhysX `effort_limit_sim` 87/12 Nm | actuator와 solver 양쪽에서 87/12 Nm |
| torque rate | 제한 없음 | 1000 Nm/s |
| velocity/acceleration trajectory limiter | 없음 | 없음 |
| observation/action 차원 | 58 / 14 | 58 / 14 |
| RSL-RL 로그 | `dual_fr3_reach` | `dual_fr3_reach_sim2real_v1` |

v1의 torque-rate 제한 순서는 다음과 같다.

```text
PD torque
  → magnitude saturation
  → torque-rate limit
  → PhysX
```

관절 1~4는 `±87 Nm`, 관절 5~7은 `±12 Nm`로 제한한다. Torque-rate
limit은 모든 관절에 `1000 Nm/s`를 적용한다.

현재 simulation physics는 100 Hz이므로 torque 변화량은 한 physics step당
최대 10 Nm이다. 실로봇 libfranka는 1 kHz에서 약 1 Nm/step으로 적용한다.
10 ms 구간 전체의 최대 변화량은 같지만 구간 내부 동역학은 완전히 동일하지
않다는 점에 유의해야 한다.

v1 actor의 PPO log-probability는 tanh Jacobian correction을 포함하며,
entropy도 latent Gaussian이 아닌 squashed distribution에서 추정한다. 따라서
hard clipping 후에는 동일한 action을 내보내면서 Gaussian 표준편차만 키우는
entropy loophole을 막는다. `clip_actions=1.0`은 actor 범위를 정의하는 수단이
아니라 수치적 safety guard로만 남겨 둔다.

## Action과 observation 순서

v1의 action과 joint position/velocity observation은 모두 다음 순서로 고정한다.

```text
left_fr3_joint1 ... left_fr3_joint7,
right_fr3_joint1 ... right_fr3_joint7
```

Isaac Lab 3.0이 전체 관절 선택을 asset 내부 순서로 최적화할 수 있으므로,
observation에는 명시적인 joint index를 사용한다. 배포 NPZ를 만들기 전에
`scripts/tools/dump_obs_layout.py`로 실제 순서를 다시 확인해야 한다.

## 구현 위치

- 환경과 processed-action safety clipping: `tasks/manager_based/reach/joint_pos_env_cfg.py`
- Gym 등록: `tasks/manager_based/reach/__init__.py`
- PPO actor 설정과 로그 이름: `tasks/manager_based/reach/agents/rsl_rl_ppo_cfg.py`
- tanh-squashed distribution: `tasks/manager_based/reach/agents/tanh_squashed_gaussian.py`
- actuator 모델: `dual_fr3_lab/actuators/torque_rate_limited_pd.py`
- robot actuator 설정: `dual_fr3_lab/robots/dual_fr3_plate_ee.py`

## 학습

v1은 action 분포와 actuator dynamics가 v0와 다르므로 v0 checkpoint에서
이어 학습하지 않고 처음부터 학습한다.
기존 hard-clipped Gaussian으로 학습한 초기 v1 checkpoint 또한 distribution
parameterization이 다르므로 resume하거나 신규 actor에서 재생하지 않는다.

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Reach-DualFR3-Sim2Real-v1 \
  --headless --max_iterations 5000
```

## 검증된 학습 결과 (2026-09-03)

현재 기준 run은 다음과 같다.

```text
logs/rsl_rl/dual_fr3_reach_sim2real_v1/2026-09-03_18-57-37/model_4999.pt
```

run에 저장된 `params/agent.yaml`에서 tanh-squashed distribution,
`std_type=log`, `clip_actions=1.0`을 확인했고, `params/env.yaml`에서 explicit
PD, 87/12 Nm torque limit, 1000 Nm/s torque-rate limit, 20 Hz policy와
left-7/right-7 action 순서를 확인했다. 즉 이 run의 저장 설정은 현재 v1 코드와
일치한다.

아래 값은 TensorBoard scalar의 마지막 50 iteration 평균이다.

| 지표 | v0 `08-31_17-11-10` | 초기 v1 `17-28-54` | tanh v1 `18-57-37` |
|---|---:|---:|---:|
| mean reward | 47.34 | 20.92 | **37.74** |
| reach coarse | 1.900 | 1.289 | **1.663** |
| reach fine | 4.098 | 1.446 | **3.083** |
| action-rate penalty | -0.0739 | -0.1235 | **-0.0246** |
| joint-velocity penalty | -0.00702 | -0.00141 | **-0.000834** |
| undesired-contact penalty | 0.0 | -0.00217 | **-0.000001** |
| policy mean std | 0.993 → 0.250 | 0.997 → 20.471 | **0.990 → 0.297** |
| entropy | 19.82 → -1.69 | 19.85 → 61.98 | **9.36 → -3.53** |

초기 v1과 비교하면 mean reward는 약 80% 증가했고, fine reward는 약 113%
증가했다. action-rate penalty의 절댓값은 약 80% 감소했다. 특히 초기 v1에서
`mean_std=20.58`, `entropy=62.05`까지 계속 증가하던 saturation collapse가
사라졌으며, 새 checkpoint의 14개 joint std는 평균 0.296으로 수렴했다.
entropy 값은 초기 v1의 latent Gaussian과 새 v1의 squashed distribution에서
정의가 다르므로 절댓값을 직접 비교하기보다 폭주 여부와 추세를 본다.

새 v1은 v0보다 최종 reward와 위치 정밀도는 낮지만, explicit torque 제약을
포함한 상태에서 목표 도달에 성공했고 action-rate 지표는 v0보다도 작다. 따라서
v1의 목적에 맞는 안정적인 bounded-action policy로 판정한다.

## 영상 검증

`Screencast from 2026-09-03 19-49-51.webm`은 25.98초 길이의 play 결과다.
여러 목표 위치에 대해 양팔 EE가 반복적으로 접근하고 안정적으로 유지되며,
초기 v1 영상에서 보였던 지속적인 고주파 진동과 큰 wrist/plate orientation
뒤틀림은 관찰되지 않았다. goal resampling(4~8초)과 episode timeout/reset(8초)에
따른 의도된 자세 전환은 진동과 구분한다.

다만 현재 `dual_ee_reach` reward는 EE position만 사용하고 command quaternion을
무시한다. `object_orientation` observation도 실제 EE orientation이 아니라 0으로
채워진 호환 슬롯이다. 이번 영상에서 orientation이 안정된 것은 확인했지만,
orientation 유지가 task contract로 보장된 것은 아니다. 실로봇 배포에서 특정
tool orientation이 필수라면 orientation observation/reward 또는 joint posture
regularization을 별도로 추가해야 한다.

## Play와 고정 goal sequence

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Reach-DualFR3-Sim2Real-Play-v1 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/dual_fr3_reach_sim2real_v1/2026-09-03_18-57-37/model_4999.pt \
  --viz kit
```

고정된 goal로 반복 비교할 때만 다음 옵션을 추가한다.

```text
--reach_goal_sequence config/reach_goal_sequence.example.json
```

## 배포 주의사항

- v0 checkpoint는 v1 실행 계약과 호환되지 않는다.
- simulation 검증 기준은 `2026-09-03_18-57-37/model_4999.pt`이다.
- 배포 artifact는 actor 내부의 tanh를 포함해야 한다.
- 배포 노드의 `[-1, 1]` clipping은 수치적 safety guard로 유지한다.
- 배포 controller는 100 Hz마다 measured-relative target을 direct PD에 전달한다.
- 배포 torque는 PD 이후 87/12 Nm로 제한한다.
- `franka_hardware`의 기본 1000 Nm/s torque-rate limiter를 유지해야 한다.
- 실로봇 실행 전에는 shadow mode에서 action, joint 순서와 목표 범위를 검증한다.
