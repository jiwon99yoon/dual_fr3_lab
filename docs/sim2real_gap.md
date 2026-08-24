# Dual FR3 Sim2Real — 공통 원칙과 문서 허브

> 대상 저장소: `~/dual_fr3_lab`
>
> 갱신: **2026-08-08**
>
> 역할: cube-lift와 plate box-lift가 공유하는 **calibration, 시간구조, 평가, 안전 원칙**만 관리한다.
> 태스크별 observation/action/DR/reward/roadmap은 아래 전용 문서를 기준으로 한다.

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

### 2-4. Bias, drift, white noise를 분리

- camera-to-base calibration residual은 episode-persistent SE(3) bias다.
- encoder/pose frame jitter는 per-step noise다.
- 온도·조명·mechanical settling은 느린 drift일 수 있다.
- dropout은 값에 큰 Gaussian noise를 더하는 것이 아니라 invalid/stale observation이다.

Quaternion에는 componentwise Gaussian을 직접 더하기보다 tangent-space(axis-angle) perturbation 후
normalize하는 방식이 실제 orientation error와 맞는다.

### 2-5. 한 실험에 한 축

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
