# Dual FR3 Cube Lift — Sim2Real 전략과 배포 계약

> 대상 태스크: `Isaac-Lift-Cube-DualFR3-v3`
>
> 현재 기준: **2026-08-08, v3 실물 배포 후보**
>
> 후보 checkpoint: `logs/rsl_rl/dual_fr3_lift_v3/2026-08-03_16-52-40/model_4999.pt`
>
> 역할: 이 문서는 cube-lift의 **현재 실물 I/O, 남은 gap, calibration, 후속 학습과 안전 배포**를 관리한다.
> reward 설계사는 [`DUAL_FR3_LIFT_ENV_ANALYSIS.md`](DUAL_FR3_LIFT_ENV_ANALYSIS.md), 버전별 결과는
> [`dual_fr3_lift_v1.md`](dual_fr3_lift_v1.md)~[`dual_fr3_lift_v3.md`](dual_fr3_lift_v3.md),
> 공통 sim2real 원칙은 [`sim2real_gap.md`](sim2real_gap.md)를 본다.

---

## 0. 결론과 plate task와의 분리 이유

Cube-v3는 plate box-lift와 다른 배포 문제다.

- 오른팔 7 joint + binary gripper만 제어한다. 왼팔은 ready pose를 유지한다.
- 물체는 50 × 50 × 100 mm 세로 직육면체이고 기구적 gripper cage로 잡는다.
- actor는 object center position만 보고 orientation/contact force/`tau_ext`는 보지 않는다.
- 정책은 20 Hz이고, 가장 큰 gap은 **ArUco의 시간구조와 실물 gripper/arm 명령 응답**이다.

따라서 plate의 squeeze-force·양팔 접촉 대칭·box yaw reward를 이 문서에 섞지 않는다. Cube-v3의 우선순위는:

1. 실제 cube 질량·CoM, ArUco frame/rate/error, arm/gripper timing 측정.
2. 현재 v3 checkpoint를 저장 당시 env cfg로 재현해 held-out sim 평가.
3. pose sample-hold/latency와 action substep semantics/delay를 각각 검증.
4. 실물 placement가 현재 zero-shot 평가에서 확인한 ±10°를 벗어날 때만 orientation observation과 yaw DR 추가.
5. image/tactile policy와 external disturbance는 state-policy 1차 배포 실패 근거가 있을 때만 검토.

---

## 1. v3 배포 계약

### 1-1. Task와 현재 검증 상태

| 항목 | v3 기준 |
|---|---|
| 물체 | 50 × 50 × 100 mm nominal, x/y 4.5~5.5 cm, z 9.5~10.5 cm size DR |
| spawn | table 위, x ±8 cm / y ±12 cm; yaw 학습은 0° |
| policy | 20 Hz (physics 100 Hz, decimation 5) |
| episode | 8 s = 160 policy steps |
| actor/critic | `[256,128,64]` ELU, normalization OFF, history 없음 |
| sim 평가 | yaw 0/5/10° success 1.000/1.000/0.984, drop 0/384 episodes |

v2 정육면체 정책은 tall cube zero-shot에서 파지는 했지만 160/128 episode drop이 발생했고, v3 형상으로
재학습한 뒤 drop 0으로 회복했다. 이 결과는 **형상 mismatch를 broad DR나 reward로 덮지 말고 실제 물체를
학습 분포에 넣어야 한다**는 프로젝트 내부 근거다.

현재 등록된 v1/v2 task는 수정된 공통 base를 상속하므로, 과거 DexCube 4.8 cm 고정 환경을 그대로
재구성하지 않는다. v1/v2 historical checkpoint 비교에는 저장된 `params/env.yaml`과 당시 code revision이
필수다.

주의: 기존 평가 스크립트 일부는 50 Hz를 하드코딩해 v3의 time-to-success를 잘못 표시한다. 20 Hz에서는
50 policy steps가 1 s가 아니라 2.5 s다. 성공률·drop과 별개로 시간 지표는 env step dt로 다시 계산한다.

### 1-2. Actor observation: 36D

| 순서 | 관측 | Dims | 실물 소스 |
|---|---|---:|---|
| 0 | right arm 7 + finger 2 `joint_pos`, ready 대비 상대값 | 9 | FR3 + gripper state |
| 1 | 같은 joint의 `joint_vel` | 9 | FR3 + gripper state |
| 2 | object center position in robot-root frame | 3 | ArUco/perception |
| 3 | goal position 3 + identity quaternion 4 | 7 | experiment command; quaternion은 dead input이지만 유지 |
| 4 | previous raw action | 8 | policy runtime 내부 상태 |
| 합계 |  | **36** | |

Actor에는 object orientation/velocity, contact truth, joint torque, `tau_ext`가 없다. v1 ablation에서 torque
관측 제거가 채택됐으므로 실기 actor에 다시 추가하지 않는다. 현재 corruption은 joint position 0.005 rad,
joint velocity 0.05 rad/s, object position 0.01 m iid Gaussian이다.

실물 gripper API가 total width와 width velocity 한 개씩만 주면, policy가 요구하는 두 finger position·velocity로
변환해야 한다. 후보 매핑은 각 finger에 `width/2`, `width_dot/2`를 복제하는 것이지만, 실제 ROS joint-state의
joint 순서·부호·단위와 함께 unit test로 확정한다.

### 1-3. Action: 8D

```text
policy @ 20 Hz: processed_delta = 0.1 * a[0:7]         # 50 ms 동안 같은 delta
physics @ 100 Hz, substep k: q_target[k] = q[k] + processed_delta
a[7]: negative -> close, non-negative -> open
```

`RelativeJointPositionAction`은 policy tick에서 delta를 한 번 처리하지만, 다섯 physics substep마다 최신 q에
그 delta를 다시 더한다. 따라서 현재 sim action은 20 Hz absolute target hold가 아니며, `0.1`도 hard rate
limit이 아니다. 실제 속도는 PD·관성·effort/rate saturation에 따라 달라지고 actor output은 hard-bound되지 않는다.

- Exact parity를 택하면 실물 adapter가 raw delta를 50 ms hold하면서 100 Hz마다 최신 q에 다시 더해야 한다.
- Policy tick에서 snapshot target을 한 번 만들고 hold/interpolate하려면 action semantics가 바뀐다. 이 경우
  sim action term도 같은 방식으로 바꿔 재학습 또는 검증된 fine-tuning을 해야 한다.
- 어느 방식이든 이전 target에 누적하거나 neutral pose에 더하지 않으며, action clamp와 joint별
  position/velocity/acceleration/jerk hard limit을 별도로 둔다.
- 다음 관측의 `previous action`에는 clamp 이후 명령이 아니라 학습의 `mdp.last_action`과 같은 **raw policy
  output**을 넣는다. clamped/executed action은 안전·tracking log에 별도로 저장한다.
- 기존 raw action 평가에서 gripper는 `open -> close` 전환을 1회만 하고 유지했다. 채터링은 없었지만,
  실물 command latency·grasp completion·width feedback은 sim의 즉시 binary action과 다르다.

### 1-4. Ready pose와 frame

- 오른팔 ready pose(rad): `(+0.5236, -0.7854, 0, -2.3562, 0, +1.5708, +0.7854)`.
- finger default: 각 0.04 m.
- object position 기준은 dual-FR3 robot root, 즉 팔 장착 상판 frame이다.
- 문서상 world에서 root z는 0.405 m이므로, world와 robot-root 좌표를 혼동하지 않는다.

---

## 2. 현재 강건화와 남은 gap

### 2-1. 이미 반영된 것

- v3 size DR는 `prestartup`으로 실제 env별 적용됨을 검증했다.
- static friction 0.4~1.2, dynamic friction 0.3~1.0.
- density 1000 kg/m³ 기반 질량과 mass multiplier 0.7~1.3.
- joint/object-position observation white noise.
- reward 전용 bilateral finger contact gate와 성능 기반 penalty curriculum.
- robot-link gravity OFF로 실물 FR3 gravity-compensated control을 근사.
- 50 Hz v1에서 20 Hz v2/v3로 재학습해 perception rate와 가까워짐.

`finger_table_impact`는 현재 GPU filtered `force_matrix_w`에서 static table force가 0으로 나와 학습 중
실질적으로 no-op이었다. 따라서 v3가 finger-table 충격 회피를 배웠다고 가정하지 않으며, 실물에서는 독립
supervisor를 쓰고 sim reward를 고칠 때는 filter 없는 net-force 또는 별도 충돌 판정을 검증한다.

Cube mass는 size DR에 따른 부피 변화 뒤 density와 mass multiplier로 결정된다. 따라서 nominal 0.25 kg에서
단순히 0.7~1.3만 곱한 범위를 최종 질량 범위로 쓰지 않는다. 실제 runtime mass 분포를 dump하고 실물
cube 질량·CoM과 비교한다.

또한 v3 center z는 world 0.80 m로 고정된 반면 height는 9.5~10.5 cm로 변하므로, table z=0.75 m 기준
초기 바닥이 약 +2.5 mm gap부터 -2.5 mm penetration까지 달라질 수 있다. size DR runtime 검증에는 초기
contact/settling transient와 성공률의 상관도 포함한다.

### 2-2. 남은 gap

| 축 | 남은 항목 | 우선순위 |
|---|---|---:|
| Perception | camera-to-base bias, timestamp, sample-and-hold, frame age, dropout/occlusion, motion blur | 1 |
| Arm actuation | current 100 Hz substep rebase의 실물 재현 여부, command delay/hold, tracking bandwidth, joint friction/gain, rate/acc/jerk limit | 1 |
| Gripper | open/close latency, width trajectory, grasp success, contact/retry timing | 1 |
| Object | 실제 mass·CoM, shape tolerance, surface friction | 1 |
| Orientation | yaw 학습 DR와 actor orientation 관측 없음; 현재 ±10° 평가만 통과 | 2 |
| Memory | history/recurrent policy 없음 | 2 |
| Value learning | asymmetric critic 없음 | 3 |

프로젝트 내부에서 observation noise와 physics DR는 각각 단독으로 괜찮아도 함께 적용할 때 slip/drop을
만들 수 있었다. 따라서 각 축 단독 평가 뒤 최종적으로 **noise × physics × timing 조합**을 평가한다.

---

## 3. Calibration 계획

### 3-1. Arm command tracking

박스 없이 안전한 작은 step/multisine target으로 다음을 기록한다.

```text
t_state, t_policy, t_command
q_des, q, dq, command accepted/rejected
```

측정 항목:

- command→motion delay와 jitter.
- sim physics-substep의 `q`, processed delta, resulting target trace.
- 실물 adapter/controller의 rebase 또는 snapshot/hold 방식.
- joint별 rise time, steady-state error, velocity/acceleration saturation.
- 20 Hz raw delta를 hold하고 선택한 adapter로 전송할 때의 tracking residual `q_des-q`.

### 3-2. Gripper

- open, close, grasp 명령부터 width가 안정될 때까지의 시간.
- object 없음/정상 파지/실패 파지에서 width·current·hardware success flag.
- 연속 command를 보낼 때 중복 요청 처리와 timeout/fault.
- 실제 50 mm 두께 cube를 잡았을 때 안정 width 범위와 slip 시 width 변화.

Grasp 판정은 close command만으로 만들지 않는다. `close AND width/object grasp success`를 사용하고,
운반 중 width가 빈 그리퍼 수준으로 닫히면 즉시 grasp 해제 상태로 복귀한다.

### 3-3. Perception

다음을 정지와 이동 조건으로 나누어 측정한다.

```text
camera capture timestamp, pose publish timestamp, policy consume timestamp
object position, valid/stale flag, visible marker IDs, frame name
```

- 보고된 30 Hz가 capture 기준인지 publish 기준인지 확인한다.
- 정지 bias와 per-frame jitter를 분리한다.
- 팔이 빠르게 움직일 때 motion blur, marker occlusion, stale-frame 연속 길이를 측정한다.
- cube 기준 위치에서 robot-root 좌표가 예상값과 일치하는지 확인한다.
  v3 nominal center world `(0.45,-0.25,0.80)`은 root z=0.405 m 기준 약
  `(0.45,-0.25,0.395)`다. 이 수치도 실제 frame chain으로 다시 검증한다.

### 3-4. Object

- 실제 x/y/z, 질량과 CoM을 측정한다.
- 간단한 tilt/slip test로 finger-pad와 cube 사이 friction 범위를 얻는다.
- sim startup 후 env별 size/mass/material을 dump해 설정이 실제 적용됐는지 확인한다.

v0~v2에서 size DR event mode 오류가 오랫동안 조용히 미작동했던 이력이 있으므로, 앞으로는 cfg 값만
보고 DR이 동작했다고 판정하지 않는다.

---

## 4. Perception 배포 전략

### 4-1. 1차: ArUco position + zero-order hold

현재 actor가 필요한 object 정보는 중심 위치 3D뿐이다.

1. ArUco 6DoF에서 object center를 계산한다.
2. camera frame에서 robot-root frame으로 변환한다.
3. 20 Hz policy tick에서 가장 최근 valid position을 사용한다.
4. frame age와 stale timeout은 actor 밖 safety supervisor가 항상 확인한다.

단순히 pose noise를 5~10 cm로 키워 calibration bias를 덮지 않는다. 정지 상태에 큰 상수 오차가 있으면
camera/base calibration을 수정한다.

### 4-2. 2차 fallback: grasp 후 FK position

파지 후 ArUco가 gripper/arm에 가려지거나 motion blur가 크면 object position source를 FK로 바꿀 수 있다.

- 전환: close command **AND** valid grasp width/hardware grasp success.
- FK pose: robot-root→TCP + 실제 grasp offset으로 cube center 계산.
- 복귀: grasp flag 상실, width 붕괴, slip/fault 감지 시 ArUco로 즉시 복귀.

이 방식은 marker 가림을 해결하지만 object가 gripper 안에서 미끄러지는 동안 FK가 오차를 숨길 수 있다.
따라서 width/pose consistency와 timeout을 함께 본다.

### 4-3. Vision/tactile policy는 후순위

ArUco position + FK fallback이 실제로 부족하다는 데이터가 나오기 전에는 image-to-action distillation,
point cloud, tactile observation을 추가하지 않는다. 현재 task는 state-based 준정적 lift이며, 새 sensor는
학습 비용과 새로운 sim2real gap을 동시에 만든다.

---

## 5. 후속 학습 순서

현재 v3 reward와 policy output을 고정하고 다음을 한 축씩 수행한다. 먼저 exact 100 Hz rebase를 재현할지,
snapshot target branch를 새로 학습할지 결정한다. Snapshot 전환 자체와 delay DR를 한 실험에 섞지 않는다.

| 단계 | 변경 | 판정 질문 |
|---|---|---|
| Cube-S1 | measured pose sample-hold/latency/dropout/bias | 실제 perception 시간구조에서 파지·운반이 유지되는가? |
| Cube-S2 | selected action adapter + measured delay/hold/tracking limit | 실물 arm target semantics를 넣어도 성공하는가? |
| Cube-S3 | gripper latency/width transition 모델 | binary sim gripper gap을 견디는가? |
| Cube-S4 | S1+S2+S3 결합 | gap 조합에서 slip/drop이 생기는가? |
| Cube-S5 | short history 또는 작은 GRU | stale pose/hidden response에 memory가 실제 이득인가? |

History는 측정된 최대 delay를 덮도록 정한다. actor에 history를 넣기 전 `frame_age`, `q_des-q`, previous
actions처럼 deployable한 temporal signal을 우선한다.

실물 초기 yaw가 ±10° 안에 관리되지 않을 때만 별도 branch에서:

1. object orientation을 actor observation에 추가한다.
2. yaw DR을 실물 placement 범위로 확장한다.
3. checkpoint 비호환을 인정하고 from-scratch 또는 명시적 observation-adapter로 학습한다.

학습 효율이 병목이면 actor는 실물 관측을 유지하고 critic에 object velocity/orientation, exact contact,
mass/friction, sampled delay를 주는 asymmetric critic을 검토한다. Temporal gap 자체는 critic만으로 해결되지
않으므로 history와 분리해 평가한다.

External wrench는 실물에서 실제 slip/흔들림이 관찰될 때만, 먼저 friction posterior를 보정한 뒤 추가한다.

---

## 6. 실물 inference loop

```text
policy @ 20 Hz:
  1. q_arm, q_finger, dq를 읽는다.
  2. 최신 valid object center를 robot-root frame으로 얻는다.
  3. obs36 = [q-q_ready, dq, object_pos, goal_pos+(1,0,0,0), prev_action]
  4. deterministic actor mean raw_action을 계산하고 actor feedback용으로 보존한다.
  5. raw arm action을 clamp하고 processed_delta=0.1*a_executed로 50 ms hold한다.
  6. gripper sign을 상태 머신으로 처리하고 grasp width/success를 감시한다.
  7. prev_action은 raw_action으로 업데이트하고 executed action은 별도 log에 남긴다.

arm adapter @ 100 Hz (exact-parity 선택 시):
  8. 최신 q를 읽고 q_target=q+processed_delta를 매 10 ms 갱신한다.
  9. position/velocity/acceleration/jerk limiter와 controller watchdog을 적용한다.
```

위 loop는 current sim semantics를 재현하는 선택이다. Snapshot-action policy를 새로 학습했다면 그 정책의
adapter 계약을 따른다. 실물 loop에서 input normalization을 새로 넣지 않는다. 현재 policy는 raw
observation으로 학습됐다.

---

## 7. 평가와 안전

### 7-1. Simulation

- 저장 checkpoint의 `params/env.yaml`, `params/agent.yaml`과 코드 버전을 재현한다.
- true nominal, Play-v3, Play-Noisy-v3, one-factor timing sweep, physics×noise×timing 조합을 분리한다.
- `Play-v3`는 observation corruption만 끄며 size/mass/friction DR은 유지한다. 현재 evaluator의
  `--no_physics_dr`도 mass/friction만 끄고 size DR은 남기므로, **true nominal은 size·mass·friction을 모두
  고정한 별도 cfg/override**로 평가한다.
- fixed held-out seeds와 충분한 startup dynamics draw를 사용한다.
- 마지막 checkpoint만 선택하지 말고 저장 checkpoint sweep을 수행한다.

필수 지표:

- grasp, lifted, strict goal hold, full success.
- drop/slip, time-to-grasp, 실제 초 단위 time-to-success.
- object tilt, gripper width/grasp loss.
- action jerk, q tracking residual, saturation/rejected command 횟수.
- pose stale/dropout 상태별 성공률.

### 7-2. Real ladder

1. object 없는 저속 free-space trajectory.
2. cube를 두고 contact 직전까지 접근.
3. gripper close와 grasp 확인만 수행.
4. 수 cm low lift 후 즉시 내려놓기.
5. nominal goal lift/hold.
6. 초기 position/yaw와 날짜·조명 조건을 점진 확대.

### 7-3. Independent safety supervisor

- pose stale/invalid timeout.
- joint workspace·position·velocity·acceleration·jerk limit.
- gripper timeout/fault와 unexpected width loss.
- table/finger collision, drop, object out-of-workspace abort.
- command heartbeat, controller fault, physical E-stop.
- payload/model bias를 tare한 `tau_ext` joint-space envelope는 actor input이 아니어도 collision safety에 쓸 수
  있다. 이를 per-finger contact force와 동일시하지 않는다.

---

## 8. 배포 전 체크리스트

- [ ] checkpoint와 함께 저장된 env/agent cfg로 v3 평가 재현.
- [ ] 20 Hz 기준 평가 시간 계산 오류 수정.
- [ ] actor 36D 순서·단위·ready offset unit test.
- [ ] gripper total width/velocity를 두 finger 관측으로 바꾸는 순서·부호·단위 unit test.
- [ ] sim physics-substep target trace와 실물 100 Hz rebase 또는 snapshot-action branch의 의미를 일치시킴.
- [ ] chosen action adapter와 gripper sign unit test.
- [ ] raw previous action과 clamped/executed action을 분리해 unit test·logging.
- [ ] actual cube x/y/z, mass, CoM 측정.
- [ ] arm command latency/interpolation/tracking limit 측정.
- [ ] gripper timing·width·success flag 측정.
- [ ] ArUco frame, stationary bias, moving jitter, delay/dropout 측정.
- [ ] pose/action/gripper one-factor 및 결합 sim 평가 통과.
- [ ] independent safety supervisor dry-run.
- [ ] §7-2 real ladder 순차 통과.

실측값과 실물 trial은 [`log_per_task.txt`](log_per_task.txt)에 기록하고, 환경/reward 변경이 생기면
[`DUAL_FR3_LIFT_ENV_ANALYSIS.md`](DUAL_FR3_LIFT_ENV_ANALYSIS.md)에도 설계 이력을 남긴다.
