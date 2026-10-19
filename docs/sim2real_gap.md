# Dual FR3 Sim2Real — 공통 원칙과 문서 허브

> 대상 저장소: `~/dual_fr3_lab`
>
> 갱신: **2026-09-17**
>
> 역할: cube-lift와 plate box-lift가 공유하는 **calibration, 시간구조, 평가, 안전 원칙**만 관리한다.
> 태스크별 observation/action/DR/reward/roadmap은 아래 전용 문서를 기준으로 한다.
>
> **[2026-09-09 상태]** dual_fr3_lab 학습 당시엔 실물 제어 프레임워크가 없었으나, 현재 실로봇
> 배포/실험은 **`~/fr3_husky_etri`** (ROS 2, branch `etri`, 실로봇 `husky_rcu:js_ws/etri_ws`에 빌드)로
> 수행한다. 프레임워크 구조·sim↔real 계약·real-only 요소·정합성 검증은 **§8** 참조.

---

## 0. 문서 라우팅

| 궁금한 내용 | 기준 문서 |
|---|---|
| Plate-EE box-lift P38 기준 sim2real, 관련 문헌, P39 이후 실험 | [`DUAL_FR3_LIFT_PLATE_SIM2REAL.md`](DUAL_FR3_LIFT_PLATE_SIM2REAL.md) |
| Plate reward와 P1~P38 설계·실험 이력 | [`DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md`](DUAL_FR3_LIFT_PLATE_ENV_ANALYSIS.md) |
| Cube-v3 기준 sim2real와 실물 inference | [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md) |
| Cube reward/환경 설계 이력 | [`DUAL_FR3_LIFT_ENV_ANALYSIS.md`](DUAL_FR3_LIFT_ENV_ANALYSIS.md) |
| Cube v1/v2/v3의 고정 스펙과 checkpoint | [`dual_fr3_lift_v1.md`](dual_fr3_lift_v1.md), [`dual_fr3_lift_v2.md`](dual_fr3_lift_v2.md), [`dual_fr3_lift_v3.md`](dual_fr3_lift_v3.md) |
| 모든 run·평가·실측 결과 | [`log_per_task.txt`](log_per_task.txt) |
| PC 구성, workspace 구조, sim2sim→sim2real 로드맵 | [`system_pc.md`](system_pc.md) |
| **실물 배포 프레임워크(`fr3_husky_etri`) 구조·계약·real-only 요소·정합성 검증** | **본 문서 §8** |
| **plate EE Isaac-real SysID 실행·real q0 재생·로그 판독** | [`../scripts/sim2real/README.md`](../scripts/sim2real/README.md) |

이 파일의 2026-07-28 버전은 P5 plate 정책과 cube 항목을 한 문서에 섞고 있었다. 당시 내용은 git 이력에
남아 있지만 현재 기준으로는 plate force actor 관측, P5 gravity A/B, 50 Hz 공통 가정 등이 오래됐다.

---

## 1. 두 태스크를 분리해서 봐야 하는 이유

| 축 | Cube lift v3 | Plate box-lift v2/P38 |
|---|---|---|
| 물체 | 50×50×100 mm 세로 cube | 555×475×275 mm, 4 kg 박스 |
| 조작 | 오른팔 7 joint + binary gripper | 양팔 14 joint + flat plates |
| 파지 | gripper cage | 양 plate의 normal force × friction |
| actor object 관측 | center position 3D | position 3D + orientation quaternion |
| actor force 관측 | 없음 | 없음(V1/V2 56D) |
| policy rate | 20 Hz | 50 Hz |
| 핵심 gap | gripper timing, pose hold/latency, orientation 미관측 | 양팔 timing skew, pose latency, contact/TCP/CoM, compliance |
| 배포 안전 | gripper fault·drop·table contact | squeeze/collision·one-side contact·drop/yaw |

> **Reach task** (`Isaac-Reach-DualFR3-v0`): 물체 없는 양팔 EE reach. obs 58D = plate 구조 호환, policy 20Hz = cube와 동일(카메라 30Hz 기준). sim2real gap 측정·LagDR 검증용 전초 task.

공통 robot과 PPO를 사용하더라도 동일한 DR와 배포 전략을 복사하지 않는다. 특히 plate의 외란·접촉
품질 reward를 cube에 넣거나, cube의 binary grasp 가정을 plate에 적용하지 않는다.

---

## 2. 공통 sim2real 원칙

### 2-1. Calibration before randomization

1. 실물 nominal dynamics와 timing을 먼저 측정한다.
2. simulator nominal을 그 측정에 맞춘다.
3. 측정 오차와 환경 변화 범위만큼 distribution을 만든다.
4. zero-shot 실패 뒤 필요할 때만 real rollout으로 posterior를 갱신한다.

넓은 uniform DR는 gap을 자동으로 해결하지 않으며 학습을 느리게 하고 지나치게 보수적인 policy를 만들 수
있다. 다른 robot 논문의 mass/friction/delay 숫자를 Dual FR3에 그대로 복사하지 않는다.

### 2-2. Deployable actor, privileged training signal

- Actor 입력은 실물 runtime에서 같은 의미·단위·주기로 생성할 수 있어야 한다.
- Exact contact, object velocity, mass/friction, sampled delay 같은 sim truth는 reward 또는 critic에만 둔다.
- Asymmetric critic은 value learning을 돕지만 memoryless actor에 online adaptation 능력을 자동으로 주지 않는다.
- `tau_ext`를 actor에서 빼더라도 독립 safety supervisor와 calibration log에는 사용할 수 있다.

### 2-3. Timing is part of the dynamics

White noise만으로 다음 gap을 대신할 수 없다.

- camera capture→pose publish→policy consume delay.
- stale frame, sample-and-hold, dropout/occlusion.
- policy inference jitter와 command transport delay.
- 저수준 controller의 interpolation/hold와 actuator tracking bandwidth.
- 다관절·양팔·gripper channel 사이의 differential skew.

현재 두 task의 Isaac Lab `RelativeJointPositionAction`은 policy tick에서 delta를 처리한 뒤, 같은 delta를
각 **physics substep마다 최신 joint position에 다시 더해** target을 만든다. 따라서 policy rate만 보고
20/50 Hz absolute target hold로 구현하면 다른 controller가 된다. Physics-substep target trace를 기준으로
실물 adapter를 재현하거나, snapshot-target action term으로 sim을 바꾸고 재학습해야 한다.

Delay를 넣은 뒤에는 actor가 stale 상태를 구분할 `frame_age/dt` 또는 측정된 최대 지연을 덮는 history를
검토한다. History 길이는 관습적으로 정하지 않고 실측 delay로 정한다.

### 2-4. Actuator 추종 지연을 측정하고 DR로 모델링한다 — HDR35-20 사례

> **배경**: `~/remove_hook` 프로젝트에서 HDR35-20 + snu_hdr_controller를 대상으로 수행한 actuator
> system identification 결과. 접근법 자체는 robot에 무관하게 재활용 가능하므로 여기에 정리한다.

#### 문제 구조

60 Hz RL policy가 joint target `q_target[t]`을 출력해도, 실제 관절 위치 `q_real[t]`는 즉각 따라가지 않는다.
원인은 크게 두 계층이다.

1. **Rate limiter** — `snu_hdr_controller` (`rl_action_server.cpp`)는 200 Hz 루프에서
   `rl_max_joint_vel = 0.4 rad/s` 제한을 적용한 뒤 servo에 전달한다.
   policy step(16.7 ms) 동안 최대 추종 가능량 = 0.4 × 0.0167 ≈ **0.0067 rad/step**이다.
   policy가 큰 delta를 요구하면 rate limiter가 포화되어 `q_ref`가 `q_target`보다 뒤처진다.

2. **Servo 내부 지연** — rate limiter를 통과한 `q_ref`조차 servo 물리 응답(관성·마찰·PD 제어 대역)
   때문에 즉각 달성되지 않는다. 관찰된 추종 패턴은 1차 IIR로 근사할 수 있다.

   ```
   q_real[t+1] ≈ q_real[t] + α × (q_ref[t] − q_real[t])
   ```

   alpha를 1에 가깝게 설정할수록 이상 서보(즉각 추종)에 가깝다.

#### 측정 방법 — step-response 실측

자유 공간(free-space) task에서 RL policy를 실로봇 배포한 뒤 다음 데이터를 수집한다.

```text
columns: t, q_target(j1..j6), q_real(j1..j6), q_ref(j1..j6)
logging: 60 Hz (policy step마다 1행)
tool:    rl_inference.py CSV 로그 (~/ros2_ws/src/snu_hdr_total/rl_logs/)
```

각 policy step에서 아래 비율을 계산한다.

```
α_measured[t] = (q_real[t+1] − q_real[t]) / (q_ref[t] − q_real[t])
               단, |q_ref[t] − q_real[t]| < ε 인 step은 제외
```

HDR35-20 HI6 servo 실측 결과: **α ≈ 0.17 ~ 0.33** (j4/j5/j6 공통, 관절별 차이 소).

#### 두 단계 sim actuator 모델

측정 결과를 바탕으로 sim actuator를 두 단계로 교체한다.

**Stage 1 — LagDR** (rate limiter 없는 서보 지연만 모델링)

```python
# per-episode alpha를 실측 범위에서 무작위로 뽑는다
alpha ~ Uniform(0.17, 0.33)
q_actual[t+1] = q_actual[t] + alpha × (q_target[t] − q_actual[t])
```

IsaacLab 구현: `LagDRRelJointPosActionCfg(alpha_range=(0.17, 0.33))`

- 단순하고 학습이 빠르다.
- Rate limiter 포화가 없는 구간(작은 delta)에서는 실로봇과 잘 맞는다.
- HDR35-20 ep7000 free-space task 실로봇 배포 성공 (SR ≈ 99% sim → 실로봇 자유공간 통과).

**Stage 2 — RateLimitedLagDR** (rate limiter + 서보 지연 모두 반영)

```python
# 1) rate limiter (실로봇 RLActionServer와 동일 v_max)
delta_max = v_max × dt          # v_max=0.4 rad/s, dt=1/60 s → 0.00667 rad/step
q_ref[t]  = q_ref[t-1] + clip(q_target[t] − q_ref[t-1], −delta_max, +delta_max)

# 2) IIR lag (alpha DR)
alpha ~ Uniform(0.17, 0.33)
q_actual[t+1] = q_actual[t] + alpha × (q_ref[t] − q_actual[t])
```

IsaacLab 구현: `LagDRRelJointPosActionCfg(alpha_range=(0.17, 0.33), v_max=0.4)`

- **v_max는 배포 시 `rl_max_joint_vel` 파라미터와 반드시 일치시켜야 한다.**
- Rate limiter가 dominant한 관절(큰 delta 요구)에서 sim-real 괴리를 추가로 해소한다.
- EasyReach free-space task 실로봇 배포 성공 (자유공간 궤적 sim-real 정합 확인).

#### 어떤 모델을 언제 쓰나

| 조건 | 권장 모델 |
|---|---|
| 제어 코드에 rate limiter 없음 / 미확인 | LagDR (Stage 1) |
| rate limiter 존재 확인, v_max 측정 완료 | RateLimitedLagDR (Stage 2) |
| 물체 접촉·환경 DR이 actuator 모델보다 dominant | LagDR로 충분 (complexity 절감) |

#### HDR35-20에서 확인한 physics Hz 의존성

- **900 Hz** (dt=1/900, decimation=15): 기본 설정. 세밀한 contact 시뮬레이션에 유리.
- **120 Hz** (dt=1/120, decimation=2): chassis task 일부 실험. LagDR과 조합 시 실로봇 통과.
- 어느 Hz든 **LagDR 없이는 실로봇 자유공간조차 실패**했다. Physics Hz보다 actuator 모델 유무가 더 critical.

#### dual_fr3 적용 전 확인 사항

1. **FR3 제어 루프 분석**: ~~`fr3_ws/`~~ → **완료. `~/fr3_husky_etri` 분석됨(§8).**
   Reach 경로는 policy node(20Hz, 상대 offset) → policy_control_action_server(1kHz,
   Isaac-style PD `τ=Kp(q_des−q)+Kd(qd_des−qd)`)로 확인. rate limiter는 관절 위치가 아니라
   **토크 rate limit 1000 Nm/s** 형태이며 sim엔 없다(§8-3). LiftCube 경로는 legacy 1kHz
   velocity/accel/step-limited trajectory limiter를 사용해 형태가 다르다.

2. **step-response 실측**: 자유 공간 궤적에서 `q_target`, `q_real`을 같은 timestamp로 수집.
   joint별 alpha 추정. HDR35-20의 0.17~0.33을 FR3에 그대로 쓰지 않는다.

3. **policy rate 확인**: cube-lift 20 Hz / plate 50 Hz → 각 task마다 `delta_max = v_max × dt`가
   달라지므로 rate limiter 포화 구간이 다르다. task별로 alpha 범위와 v_max를 독립 측정한다.

4. **action rate penalty 병행 검토**: HDR35-20에서 `action_rate_l2 weight = −0.015`(기본 −0.005의 3배)가
   실로봇 oscillation 억제에 효과적이었다. LagDR 도입 시 함께 적용 권장.

### 2-5. Bias, drift, white noise를 분리

- camera-to-base calibration residual은 episode-persistent SE(3) bias다.
- encoder/pose frame jitter는 per-step noise다.
- 온도·조명·mechanical settling은 느린 drift일 수 있다.
- dropout은 값에 큰 Gaussian noise를 더하는 것이 아니라 invalid/stale observation이다.

Quaternion에는 componentwise Gaussian을 직접 더하기보다 tangent-space(axis-angle) perturbation 후
normalize하는 방식이 실제 orientation error와 맞는다.

### 2-6. 한 실험에 한 축

Reward, reset, latency, history, critic, physical DR를 동시에 바꾸지 않는다. 최소 순서는:

```text
baseline -> one-factor ablation -> one-factor ablation -> combined stress test
```

학습 도중 environment semantics를 바꾸지 않으며, fine-tuning 시 custom curriculum state가 checkpoint와
함께 저장되는지 확인한다.

---

## 3. 공통 calibration 데이터 계약

### 3-1. Robot/control log

```text
t_state, t_policy_start, t_policy_end, t_command
q_des, q, dq, tau, tau_ext
controller mode, command result, saturation/fault/watchdog
```

얻어야 할 값:

- command→motion delay와 jitter.
- joint별 tracking bandwidth와 steady-state residual.
- velocity/acceleration/jerk limit과 reject 조건.
- policy rate 사이 저수준 interpolation/hold 방식.
- plate task는 left/right differential skew, cube task는 gripper timing을 추가.

### 3-2. Perception log

```text
camera capture timestamp, pose publish timestamp, policy consume timestamp
pose, frame_id, covariance/quality, visible marker IDs, valid/stale flag
```

정지/이동/가림 조건을 분리해 frame rate, bias, jitter, dropout length, recovery를 측정한다. Position과
orientation은 동일한 source timestamp와 validity를 사용해야 한다.

### 3-3. Object/contact log

- 실제 크기, 질량, CoM/inertia 근사.
- surface material과 static/dynamic friction 시험.
- representative contact에서 command, tracking residual, `tau_ext`, object motion.
- sim startup 후 실제 env별 mass/material/scale dump.

Config에 DR event가 등록됐다는 사실만으로 적용을 믿지 않는다. Cube size DR가 오랫동안 잘못된 event mode로
미작동했던 이력을 공통 회귀시험으로 삼는다.

---

## 4. 공통 개발·평가 파이프라인

### Stage 0 — Reproducible simulation baseline

- checkpoint가 학습될 때의 resolved env/agent cfg와 code revision을 복원한다.
- checkpoint sweep과 fixed held-out seeds를 사용한다.
- nominal, observation-only, physics-only, combined 조건을 분리한다.

`scripts/rsl_rl/play.py`는 현재 등록된 task cfg로 env를 만들 수 있으므로, 최신 코드로 과거 checkpoint를
재생한 결과를 historical policy 성능으로 오해하지 않는다.

### Stage 1 — Real-to-sim calibration

- object 없는 safe trajectory로 actuation/frame/timestamp를 확인한다.
- 저하중 contact와 static object로 contact/perception을 확인한다.
- 측정값과 불확실성을 nominal + distribution으로 문서화한다.

### Stage 2 — Matched simulation robustness

- action timing, perception timing, physical posterior를 one-factor로 검증한다.
- 마지막에 factor combination과 distribution corners를 평가한다.
- 평균 return보다 task stage, pose error, drop/slip, tracking/safety metric을 본다.

### Stage 3 — Real safety ladder

```text
no object/free-space
  -> approach only
  -> contact/grasp only
  -> low lift
  -> nominal lift/hold
  -> initial-condition and disturbance robustness
```

각 단계가 실패하면 이후 단계를 진행하지 않는다. 실물에서 관찰한 gap만 다음 training branch의 변경 근거로
사용한다.

### Stage 4 — Posterior update

Zero-shot가 실패한 뒤에만 소수 real rollout으로 sim/real trajectory를 비교한다. SimOpt/BayesSim류의
parameter posterior나 task-driven adaptation을 쓸 수 있지만, 추정 parameter를 물리적 진실로 단정하지
않고 real task performance와 held-out trajectory match로 판정한다.

---

## 5. 공통 평가·재현성 규칙

- 마지막 checkpoint만 평가하지 않고 저장 checkpoint 전체 또는 대표 sweep을 평가한다.
- random seed, initial pose, physics draw, perception corruption을 고정 가능한 suite로 만든다.
- startup-only DR는 episode 수가 아니라 **서로 다른 env/dynamics draw 수**를 센다.
- success는 task별 full condition과 hold duration을 포함해 구현한다.
- time-to-success는 하드코딩한 Hz가 아니라 `step_dt`로 초를 계산한다.
- 결과에는 mean뿐 아니라 median, p95/worst tail, confidence interval을 남긴다.
- run마다 command, git revision/diff, env/agent cfg, checkpoint, evaluation cfg를 함께 기록한다.
- 모든 실행·실측 결과는 `log_per_task.txt`에 남긴다.

---

## 6. 공통 safety supervisor

Safety는 policy reward나 robot 내부 reflex 하나에 맡기지 않는다.

- command heartbeat와 controller fault.
- joint position/velocity/acceleration/jerk hard limit.
- workspace, self/inter-arm/table collision.
- perception stale/invalid timeout.
- object drop/out-of-workspace abort.
- task별 `tau_ext`/gripper force·width threshold.
- physical E-stop과 저속·저하중 초기 시험.

Actor에 force observation이 없는 것과 force safety를 사용하지 않는 것은 다른 결정이다.

---

## 7. 공통 참고 문헌

- [Lin et al., Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids](https://proceedings.mlr.press/v305/lin25c.html)
- [Kataoka et al., Bi-Manual Block Assembly via Sim-to-Real Reinforcement Learning](https://arxiv.org/abs/2303.14870)
- [Dao et al., Sim-to-Real Learning for Humanoid Box Loco-Manipulation](https://arxiv.org/abs/2310.03191)
- [DeXtreme: Transfer of Agile In-Hand Manipulation from Simulation to Reality](https://arxiv.org/html/2210.13702)
- [DemoStart](https://arxiv.org/html/2409.06613)
- [SimOpt](https://arxiv.org/abs/1810.05687)
- [AdaptSim](https://proceedings.mlr.press/v229/ren23b.html)

Plate contact/reward 문헌과 각 논문의 직접 적용점은 plate 전용 문서 §3에서 관리한다.

---

## 8. 실물 배포 프레임워크 — `fr3_husky_etri`

> dual_fr3_lab을 학습하던 시점(~2026-08)엔 실물 제어 프레임워크가 없었다. 현재 실로봇 배포는
> 연구실에서 별도로 작성한 ROS 2 프레임워크로 수행한다.
>
> - **repo**: `https://github.com/iamjoseph1/fr3_husky_etri` (branch `etri`)
> - **로컬 분석 clone**: `~/fr3_husky_etri` (fork: `jiwon99yoon/fr3_husky_etri`, upstream=원본)
> - **실로봇 빌드**: `husky_rcu:js_ws/etri_ws` (`colcon build --packages-up-to fr3_husky_nn_policy fr3_husky_controller`)
> - 배포 정책은 PyTorch 없이 **NPZ + NumPy**로 추론(`NumpyMLPActor`, `allow_pickle=False`). GPU 불필요.
> - 기준 문서: `fr3_husky_nn_policy/README.md`.

### 8-1. 지원 태스크와 실행 모드

| Task | 제어 관절 | 정책 | actuator 실행 경로 |
|---|---|---|---|
| LiftCube | 오른팔 7 + gripper | 36→8, `dual_fr3_lift_v3_actor.npz` | 절대 `q + 0.1·action` target을 legacy 1kHz velocity/accel/step-limited trajectory limiter로 추종 |
| Reach | 양팔 14 | 58→14, `dual_fr3_reach_actor.npz` | 상대 `0.1·action` offset을 50ms hold, 1kHz마다 `q_target=measured_q+offset` 재계산 + Isaac-style PD 토크 |

- 두 정책 모두 20Hz. launch 기본은 **`shadow_mode:=true`** (추론만, 로봇에 명령 미전송) — 정합성/관측 디버깅 안전.
- 배포 정책은 Isaac 학습본에서 `export_ppo_actor`로 NPZ 변환. **Reach는 tanh-squash 정책이라 `--output-activation tanh` 필수**이며, 노드가 identity-output(unsquashed) artifact를 거부한다.

### 8-2. Reach 제어 구조 (명령 흐름)

```
[Policy Node · Python · 20Hz]                    [policy_control_action_server · C++ · 1kHz]
 /joint_states ──이름 매핑──▶ q,qd(14)             (isaac_relative_control=true)
   └▶ build 58D obs ─▶ NumpyMLPActor ─▶ a(14), tanh, clip(±1)
        └▶ delta = 0.1·a  (50ms hold)             ┌─ 매 1kHz:
             └ PolicyJointCommand{target=delta,   │  q_desired = clamp(q_measured + delta)
               relative_position_offsets=true} ──▶│  τ = Kp·(q_des−q) + Kd·(qd_des−qd)
               RunPolicyControl{isaac_relative}    │  τ = clamp(τ, ±effort_limit[j])
                                                   │  τ = torque_rate_limit(τ, 1000 Nm/s)  ← 학습 implicit 경로에는 없음
                                                   └  writeCommand(τ)   (중력보상은 Franka HW 별도)
```

- `/joint_states`는 **이름 기반 매핑**(`DUAL_ARM_JOINT_NAMES` = left j1..7 → right j1..7)이라 메시지 순서에 안전.
- PD 상수는 학습쪽과 동일: `kIsaacStiffness=80`, `kIsaacDamping=4`, effort limit `87(j1-4)/12(j5-7) Nm` → sim `dual_fr3.py` ImplicitActuator와 일치.
- 정책은 절대각이 아니라 **50ms 유지되는 상대 offset**만 던지고, 폐루프 PD는 1kHz 컨트롤러가 담당(Isaac `RelativeJointPositionAction` + implicit PD를 C++로 재현).
- Reach 노드는 `start_policy` 시 20Hz로 `eef_trajectory.csv` + target + PNG 4장을 자동 로깅 → sim↔real EEF 궤적 비교 도구로 바로 활용 가능(EEF 정의 = link7 + [0,0,0.132]).

### 8-3. sim↔real 계약 정합성 (Test A, `scripts/sim2real/`)

**오프라인 검증 완료** (`reach_obs_contract_check.py`, 로봇/Isaac 불필요, ✅ ALL PASS):

| 항목 | Sim (dual_fr3_lab) | Real (etri) | 상태 |
|---|---|---|---|
| Obs 레이아웃 | 58D: jpos_rel(14)+jvel(14)+center(3)+cmd(7)+prev_act(14)+0(2)+0(4) | `build_reach_observation` fuzz 5000, max\|Δ\|<1e-6 | ✅ |
| 기본자세(ready) | `init_state` left j1 −0.5236 / right +0.5236 … | `DEFAULT_DUAL_ARM_JOINT_POSITION` 정확 일치 | ✅ |
| Quaternion | UniformPoseCommand (w,x,y,z), zero RPY | `[1,0,0,0]` wxyz | ✅ |
| NPZ actor | 58→256→128→64→14, elu, **tanh** out | 동일, source_iteration=4999 | ✅ |
| Action | `RelativeJointPositionAction` scale=0.1 | `delta=0.1·action`, 1kHz measured_q+offset 재계산 | ✅(구조) |
| Actuator | ImplicitPD Kp80/Kd4, effort 87/12 | `kIsaac*` 동일 | ✅ |

**Isaac 런타임 필요 — 미완(가장 중요)**: `reach_joint_order_check.py`를 `env_isaaclab`에서 실행해
Isaac articulation의 실제 joint 순서가 real 가정(left j1..7 → right j1..7)과 같은지 확인해야 한다.
여기가 어긋나면 58D 계약이 통과해도 정책 입력이 permute되어 실물에서 전혀 다르게 움직인다.
배포 정확본(`dual_fr3_reach_sim2real_v2/2026-09-09_11-05-39/model_4999.pt`)은 이 PC에 없어(로컬은
`2026-08-30` run만 존재) npz↔.pt 수치 일치는 학습 PC에서 재확인 필요.

### 8-4. 학습 sim 경로와 real의 차이 (sysID로 계측·모델링 대상)

Reach 학습의 implicit PD 법칙 형태는 real PD와 비슷하지만, 아래 요소는 **학습 sim 경로에는
직접 재현되지 않아** dynamics gap을 만든다. 단, `run_dual_fr3_sysid.py` 전용 runner는 real
비교를 위해 항목 1·2의 1 kHz torque 경로를 코드로 재현한다. 이는 학습 환경 자체를 바꾼 것이 아니다.

1. **토크 rate limit 1000 Nm/s** (`kIsaacTorqueRateLimit`, `policy_control_action_server.cpp`)
   — 학습 implicit PD엔 없음. 1kHz에서 1 Nm/update. SysID runner에서는 동일 제한을 적용한다.
   큰 error transient에서 real이 더 느리고 부드럽게 반응할 수 있다.
2. **제어 주기 1kHz(real) vs 학습 physics 100Hz(sim)** — PD 적분 스텝이 달라 동일 Kp/Kd라도
   응답이 미세하게 다르다. SysID runner의 physics step은 real과 맞춘 1 ms다.
3. **`qd_desired` 취급** — isaac-relative 모드의 `refreshIsaacRelativeTarget`은 `q_desired`만 갱신.
   Isaac은 position target 시 velocity target=0으로 Kd가 measured qd를 감쇠. **real에서 `qd_desired`가
   0으로 유지되는지 확인 필요**(아니면 Kd 항이 sim과 달라짐).
4. **안전 projection** — `max_policy_target_delta_rad`, joint-limit margin, action clip 등으로 target을
   safe band에 투영. sim엔 이런 후처리 없음(정책 raw delta가 그대로 physics로).
5. **관절 마찰·중력보상 오차** — sim은 robot-link `disable_gravity=True`로 Franka 내장 g(q)가 완벽하다 가정.
   실물 마찰(특히 저 effort j5-7)·보상 오차는 정적 편차/추종 지연으로 나타남.
6. **관측 지연·필터링** — sim은 obs Gaussian noise만, latency 없음. 실물 `/joint_states`는 속도가
   필터링돼 있고 파이프라인 지연이 20Hz 폐루프 위상에 영향.

이들에 대한 계측·모델링 절차는 §2-3(timing), §2-4(actuator step-response·LagDR), §3-1(control log)을 따른다.
LiftCube는 위 Isaac-style PD 경로가 아니라 legacy trajectory limiter를 쓰므로 rate/accel/step 포화 형태가 다르며 별도 계측한다.

### 8-5. sysID 실험 로드맵 (Reach 우선)

- **Phase 0 — 정합성**: Test A. 오프라인 완료, joint-order(Isaac) 확인 남음.
- **Phase 1 — sysID 계측**: 정책 대신 결정적 명령(스텝/사인 sweep)을 `/policy_joint_command`(relative offset)로
  주입 → `/joint_states` 녹화. 동일 시퀀스를 sim reach env에 action으로 주입 → q/qd/τ 로깅. joint-space에서
  `q_sim(t)` vs `q_real(t)` 오버레이로 정착시간·overshoot·정상오차·위상지연 = gap 정량화. 여기서 마찰·유효
  Kp/Kd·지연·토크 rate limit 영향을 추정. 현재 `run_dual_fr3_sysid.py`는 real CSV의 정확한
  `q0_*`를 재사용하고, 1 kHz custom torque path와 A/B/마찰 PhysX readback을 검증하는 replay
  runner까지 준비됐다(GPU smoke 통과). 새 real rich log를 받은 뒤 overlay/파라미터 비교를 수행한다.
- **Phase 2 — 중력보상 A/B**: `disable_gravity` on/off 학습본을 실물 transfer 비교(코드 주석에 예정된 실험).
- **Phase 3 — gap 반영 재학습**: 계측값으로 actuator/latency/DR 갱신 후 재학습(한 실험에 한 축, §2-6).

#### Controller-1 vs controller-3 action semantics ablation (2026-09-17 준비)

기존 `Isaac-Reach-DualFR3-v0`는 controller-3: policy delta를 20 Hz hold하고 매 physics step에
`q_cmd(t)=q_meas(t)+delta[k]`를 계산한다. 따라서 nonzero delta 동안 position error는 구조적으로
delta에 가깝고, 이는 velocity/지속 torque 성격이다.

별도 `Isaac-Reach-DualFR3-Snapshot-v0`는 controller-1: policy tick에만
`q_cmd=q_meas(t_k)+delta[k]`를 snapshot으로 만들고 다음 50 ms 동안 hold한다. 두 task는 action
target semantics 외 robot/plant, obs 58D, reward, 20 Hz policy, 100 Hz physics, PD/effort/DR을 같다.
기존 checkpoint를 교차 적용하지 않고 각각 from-scratch로 학습한다. 상세 실행·real log 판독은
[`scripts/sim2real/README.md`](../scripts/sim2real/README.md)를 따른다.

### 8-6. sysID Phase 1 실측 결과 — real / MuJoCo / IsaacSim 3자 비교 (2026-09-11, 당시 baseline)

**실험 구성.** `fr3_husky_etri/fr3_sysid_probe_node.py`(q0-reference 모드)로 관절별 ramp
excitation(amp 0.25rad, prehold 1s→ramp_up 1s→hold 2s→ramp_back 1s)을 주입.
- **real / mujoco**: 실물 ROS2 스택 + C++ `policy_control_action_server`(effort, Kp=80/Kd=4,
  effort limit 87/12, torque-rate 1000Nm/s) 경유. `use_mujoco:=true`는 physics 백엔드만 MuJoCo.
- **isaac**: 당시 `scripts/sim2real/sysid_isaac_recorder.py` — 동일 q0-reference 제어 법칙을 파이썬으로
  이식해 `DUAL_FR3_PLATE_EE_CFG`(reach 학습 자산 그대로: implicit 80/4, 중력 OFF, plate EE, DOF
  friction 0.2) articulation에 effort로 직접 주입. gym env·정책·.pt 불필요한 open-loop plant 비교.
- 데이터: `data/sysid/sysid_logs_260911_1422/sysid_logs/{real,mujoco,isaac,...}/`
  (2026-09-15 `~/Downloads`에서 repo로 이동, `.gitignore data/sysid/`로 로컬 보관),
  overlay PNG: `.../overlay/` (`scripts/sim2real/sysid_overlay_plot.py`, real=빨강/mujoco=파랑/isaac=초록).

**정량 지표 (ramp_amp0.250, 14관절).**

| | steady(hold, 목표 0.25) | ramp 추종 lag(ref−q, 작을수록↑) | 여기 토크 크기 |
|---|---|---|---|
| real | ≈0.25 (손목 소폭 undershoot) | 0.02~0.03 | 노이지, j2/j4는 **중력보상 DC ~15–25Nm** |
| mujoco | ≈0.25 | **0.014~0.018 (가장 타이트)** | 깨끗, 작음 |
| isaac | ≈0.25 | **j1/j3/j6=0.08~0.10, j2/j4=0.04, j5/j7=0.015** | j1/j3/j6=8~11Nm, j5/j7 정상 |

**판독.**
1. **정상상태는 3자 일치** → 기하·Kp·정착 정상. 위치가 clean signal(토크는 real에 중력 DC가
   섞여 raw 비교 불가 — 향후 prehold DC 빼거나 excitation delta로 비교할 것).
2. **MuJoCo ≈ real (오히려 더 타이트).** MuJoCo가 램프를 "못 따라오는" 게 아니라 near-ideal이라
   더 잘 따라옴. real이 약간 더 처지고 노이지한 건 실물 마찰·모터/드라이버 대역폭·`/joint_states`
   필터·지연·중력보상 오차 때문(=정상적인 소규모 sim2real gap).
3. **IsaacSim이 유일한 outlier** — 과도구간 S자 지연 + 큰 매끈한 토크 험프. 근위·특정축
   (j1,j3,j6) 심하고 원위(j5,j7)는 real/mujoco와 붙음.

**Isaac 굼뜸의 원인 진단 (`scripts/sim2real/sysid_inertia_dump.py`).**
- 확정: 명령 토크가 관절에 완전히 인가됨(`applied_torque`=commanded), 정적 generalized mass
  matrix 정상(M_j1≈0.43, M_j2≈3.4 kg·m²), **armature=0**. → 정적 관성/토크주입은 정상.
- 그럼에도 **동적 응답이 정적 관성 대비 5~10배 굼뜸** = 정적/동적 불일치. 남은 용의자:
  1. **PhysX DOF friction = 0.2** (관절 구속력에 비례 → 고하중 근위축이 더 나쁨: j1/j3 worst,
     j5/j7 ok 패턴과 일치). MuJoCo는 마찰 0/작을 가능성. **1순위 가설.**
  2. **PhysX 솔버 iteration 부족** → articulation 구속 해가 soft/laggy. 매끈한 토크험프가 이를 시사.
  3. USD 추가 joint damping 가능성.

**IsaacSim 디벨롭 플랜 (한 항목씩 바꿔 sweep→overlay 재비교, 목표: isaac lag→~0.02, 여기토크→mujoco 수준).**
> ⚠️ **2026-09-15 §8-7에서 실행 완료. 아래 1·2번 가설은 반증됐다** — 진짜 원인은 빠져 있던 **armature(A)**였다.
1. ~~**friction 검증(1순위)**~~ → **반증**. `--joint_friction 0`이 baseline과 byte-identical(무영향). τ_dry=0.2는
   baseline USD에도 이미 있던 값이라 gap과 무관.
2. ~~**솔버 iteration**~~ → 상향 시 오히려 발산(joint1 lag 0.11→0.24, torque 8.4→19). 무효.
3. 관성·링크질량을 franka_description과 대조(이미 정상 범위, 낮은 우선순위).
4. 정합 후 amp 0.05/0.25·pulse 교차검증, real 일정 지연 시 소량 transport delay 추가.

### 8-7. IsaacSim outlier 해결 — armature/viscous 주입 (2026-09-15)

§8-6의 "friction 1순위" 가설은 **반증**됐다. `sysid_isaac_recorder.py`에 `--armature`(A), `--joint_damping`
(viscous B), 기존 `--joint_friction`(τ_dry)을 추가해 이미지의 dynamics 모델
`[M(q)+A]q̈ + C q̇ + B q̇ + τ_dry(q̇) = τ_app` 를 실제 PhysX 관절 물성으로 주입(`write_joint_armature_to_sim`,
`write_joint_viscous_friction_coefficient_to_sim`, `write_joint_friction_coefficient_to_sim`)하고 검증했다.
MuJoCo/real이 쓰는 값 **A=0.1 kg·m², B=0.003 Nms, τ_dry=0.2 Nm** 를 Isaac에 넣는다.

**Ablation (left_joint1, peak|ref−q| / max τ).**

| config | peak lag | max τ | 판정 |
|---|---|---|---|
| real | 0.037 | 1.48 | 기준 |
| mujoco | 0.028 | 1.38 | 기준 |
| isaac_base (A0 / B0 / τ0.2) | **0.111** | **8.44** | outlier |
| isaac A-only (A=0.1) | 0.038 | 2.23 | 정상화 |
| isaac B-only (B=0.003) | 0.021 | 0.83 | 정상화 |
| isaac ABtau (0.1/0.003/0.2) | 0.023 | 1.02 | real/mujoco 대역 |

**14관절 sweep (ABtau, peak|ref−q| 평균).**

| | real | mujoco | isaac_base | isaac_ABtau |
|---|---|---|---|---|
| MEAN peak lag | 0.037 | 0.028 | **0.080** | **0.023** |

baseline 최악축(j1 0.11 / j3 0.145 / j6 0.129)이 ABtau에서 전부 ~0.03/0.015로 붕괴 → **전 14관절이 real/mujoco 대역으로 수렴**. torque 험프(8~11Nm)도 1~2Nm로 정상화.

**결론 및 현재 반영 상태.**
1. **진짜 원인은 armature(A) 부재.** baseline Isaac은 저관성 관절에 Kp=80 stiff PD를 물리 1ms로 적분해
   under-damped(과도 험프·8배 토크)였다. armature 0.1 또는 viscous 0.003 **어느 하나만 넣어도** 정칙화돼
   real/mujoco 대역으로 붙는다(둘 다 넣은 ABtau가 이미지 통합모델과 일치). **friction(τ_dry)은 무영향.**
2. B-only가 수치상 가장 타이트(0.021)하나 real보다도 타이트해 과대감쇠 소지 → **이미지대로 A/B/τ 셋 다
   넣는 ABtau가 물리적으로 정합적**이다.
3. **2026-09-16 반영 완료**: A/B/τdry를 `DUAL_FR3_CFG`에 넣었고, `DUAL_FR3_PLATE_EE_CFG`는 이를
   상속한다. 학습용 actuator는 계속 implicit이지만 PhysX plant 물성은 적용된다. 새 SysID runner에서
   14관절 모두 A=0.1 / τdry=0.2 / B=0.003 readback을 확인했다.
4. **미결**: (a) 정상상태 torque는 real에 중력보상 DC가 섞여 raw 비교 불가 — excitation delta로 재비교.
   (b) 변경된 plant로 **정책 재학습은 아직 하지 않았다**. 기존 정책의 transfer 영향은 재학습/평가로
   별도 검토해야 한다(한 실험에 한 축, §2-6). (c) amp 0.05/pulse 교차검증은 아직.

**데이터·그림.** ablation: `.../isaac_ablation/`, sweep: `.../isaac_ABtau_sweep/`,
비교 PNG: `.../overlay/ABtau_vs_base/cmp4_*.png`(real/mujoco/base/ABtau 4자), 14관절: `.../overlay/overlay_*_ABtau.png`.

**도구 (이 repo, `scripts/sim2real/`).**
- [`README.md`](../scripts/sim2real/README.md) — plate EE Isaac-real SysID의 목적, real 로그 요구사항,
  q0 replay 실행 순서와 결과 판독.
- `run_dual_fr3_sysid.py` — 현재 권장 runner. `--ref-dir`의 real CSV에서 profile/진폭이 같은 최신
  파일을 골라 14개 `q0_*`를 정확히 재사용한다. 학습 implicit position drive는 SysID 실행 중에만
  끄고, 1 kHz PD·effort/slew limit 기반의 custom torque path를 사용한다. plant A/B/마찰 readback이
  기대값과 다르면 실패하며, CSV에는 재구성 torque와 실제로 적용한 torque를 함께 남긴다. 자동 파라미터
  fitting 기능은 아직 없다.
- `sysid_isaac_recorder.py` — q0-reference 제어 이식 + probe 동일 63열 CSV 기록. `--armature`(A)·
  `--joint_damping`(viscous B)·`--joint_friction`(τ_dry)로 dynamics 물성 주입, `--ref_dir`로 real/mujoco
  settle에서 관절별 q0 로드, `--sweep` 14관절. (viscous B는 Isaac Sim 5.0+ 필요.)
- `sysid_inertia_dump.py` — armature/friction/mass-matrix 덤프.
- `sysid_overlay_plot.py` — real/mujoco/isaac 3자 overlay PNG (`--joint all`, `--isaac_dir`로 변형폴더 지정).
- 주의: Isaac Sim `simulation_app.close()`가 자주 hang → `완료. CSV` 출력 후 프롬프트 안 돌아오면 `Ctrl+C`
  (스크립트 자동화 시 **PID로만 kill**; `pkill -f sysid_isaac_recorder`는 래퍼 쉘 자기자신까지 죽여 금물).
