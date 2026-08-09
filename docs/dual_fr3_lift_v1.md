# dual_fr3_lift_v1 — 배포 후보 환경 스펙 & sim2real 가이드

> **2026-08-08 정정:** 실물 I/O와 action adapter의 현재 기준은
> [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md)다. 아래의 “policy rate마다
> `q_current+0.1a` target 생성·hold” 설명은 실제 Isaac Lab 동작과 다르다. Sim에서는 같은 delta를
> 100 Hz physics substep마다 최신 q에 다시 더하며, gripper total width→두 finger 관측 매핑도 별도 검증한다.

> **이 문서의 역할**: `Isaac-Lift-Cube-DualFR3-v1` 환경과 그 학습 정책(배포 후보)의 **현재 스펙 요약서**.
> 설계 이력·근거는 `DUAL_FR3_LIFT_ENV_ANALYSIS.md`(왜 이렇게 됐나), run 기록은 `log_per_task.txt`.
> 여기는 "실로봇에 옮기려면 뭘 알아야 하나"에 답하는 문서다.
> 작성: 2026-07-29 (v1 ablation 판정 직후)

## 0. 정책 신상명세

| 항목 | 값 |
|---|---|
| 배포 후보 checkpoint | `logs/rsl_rl/dual_fr3_lift_v1/2026-07-29_12-38-27/model_4999.pt` |
| 학습 | PPO (rsl-rl-lib **3.0.1**), 4096 envs, 5000 iter, from-scratch, seed 42, [PC2: IsaacLab 2.3.2/Isaac Sim 5.1] |
| 네트워크 | Actor/Critic 모두 MLP [256, 128, 64], ELU. 입력 36 → 출력 8 |
| **obs 정규화** | **없음** (`actor_obs_normalization=False`) — **실물에서도 raw 값 그대로 입력** |
| 제어 주기 | **50 Hz** (sim 100Hz PhysX × decimation 2) — ⚠ FR3 zero-shot 문헌 표준은 **10~15Hz** (perception 주기에 맞춤). 우리가 빠른 편 → §9 참조 |
| 검증 성능 | 성공률 **100%** (goal 5cm·1초 유지; strict 2.5cm도 100%), yaw ±10°까지 동일, time-to-success 1.33s, drop 0/384ep — Play-Noisy(학습 분포) 기준 |

## 1. Task / Scene

- **과제**: 테이블 위 cube를 오른팔로 파지 → 공중 목표 위치로 이동 → 유지. 에피소드 5초(250 step)
- **테이블**: 얇은 상판 0.6×1.0×0.02 m, **윗면 world z=0.75**, 중심 x=0.6 (로봇 원점 기준), 양옆 다리 y=±0.49
- **cube**: DexCube, 한 변 4.5~5.5 cm (env별 DR), 스폰 world (0.45±0.08, −0.25±0.12, 0.78), yaw 고정(0°)
- **goal**: base frame 기준 x 0.35~0.55 / y −0.45~−0.15 / z 0.45~0.60 (world z 0.855~1.005), 에피소드당 1회 샘플링

## 2. Robot

- Husky 베이스(고정) + 양팔 FR3. **오른팔 7 joint + 오른손 그리퍼만 제어**, 왼팔은 PD가 ready 유지
- base frame(팔 장착 상판) = **world z 0.405**. 모든 base-frame 값 변환 시 이 오프셋 주의
- **중력보상 모델링**: sim에서 로봇 링크 `disable_gravity=True` = 실물 FR3의 g(q) 보상과 등가.
  **실물은 중력보상이 켜진 임피던스/위치 제어 모드**여야 학습 조건과 일치
- 관절 게인: stiffness 80 / damping 4 (Panda 공식 게인, 실물 임피던스와 정합 의도), 그리퍼 2e3/1e2
- **ready pose (rad)** — obs의 joint_pos가 이 값 "대비 상대값"이므로 실물 구현에 필수:
  `joint1 +0.5236, joint2 −0.7854, joint3 0, joint4 −2.3562, joint5 0, joint6 1.5708, joint7 0.7854, finger 0.04`

## 3. Observation (36 dims, 순서 중요)

| # | 항목 | dims | 의미 / 실물 소스 |
|---|---|---|---|
| 0–8 | `joint_pos` | 9 | 오른팔 7 + finger 2, **ready pose 대비 상대값** (q − q_default). 실물: 엔코더 |
| 9–17 | `joint_vel` | 9 | 같은 joint 속도 (절대값). 실물: 엔코더 미분/제공값 |
| 18–20 | `object_position` | 3 | cube 위치, **robot base frame**. 실물: perception (요구 정확도 ±1cm) |
| 21–27 | `target_object_position` | 7 | goal pos 3 (base frame) + quat 4 (**항상 identity(1,0,0,0) — 죽은 입력이지만 반드시 채워서 입력**) |
| 28–35 | `actions` | 8 | **직전 스텝의 raw 정책 출력** (아래 action 정의 그대로) |

- **τ_ext/토크 없음** — v1 ablation으로 제거 확정 (실물 τ_ext 스트리밍 불필요)
- 학습 시 noise (실물 오차 허용치로 해석): joint_pos σ0.005 rad, joint_vel σ0.05 rad/s,
  object_position **σ1 cm**, 전부 additive Gaussian. 이 수준 이내 오차면 정책이 흡수함 (검증됨)
- ✅ **joint 순서 실측 완료 (2026-07-30, `dump_obs_layout.py`)**: obs[0:9] joint_pos 순서 =
  **right_joint1, 2, 3, 4, 5, 6, 7, right_finger1, right_finger2** (joint_vel도 동일).
  주의: IsaacLab 전체 관절은 좌우 인터리브+깊이순(left_j1,right_j1,left_j2,...)으로 정렬되지만,
  cfg가 오른팔(`right_fr3_joint[1-7]` + finger)만 필터링해 결과적으로 joint1~7 순서가 됨.
  → Arm은 libfranka의 `q[0:7]`(joint1~7) 순서와 맞는다. Gripper total width/velocity를 맨 뒤의
  finger 2개 position/velocity로 바꾸는 매핑은 cube Sim2Real 기준 문서대로 별도 unit test한다.

## 4. Action (8 dims, 50 Hz)

| # | 항목 | 의미 |
|---|---|---|
| 0–6 | arm delta | **목표 관절각 증분**: q_target = q_current + **0.1 × a** (rad). 정책 출력은 클리핑 없음 — 실물에서는 안전을 위해 a를 ±3σ(≈±2) 수준에서 클램프 권장 |
| 7 | gripper | **이진**: `a < 0 → close(0.0m)`, `a ≥ 0 → open(0.04m)` — **부호 규약 주의 (코드 검증됨)** |

- delta 방식이므로 실물 명령 스트림 = "현재 관절각 읽기 → +0.1a → 위치 명령 전송"을 50Hz 반복
- 그리퍼는 실물에서 상태 머신으로 매핑 필요 (연속 토글 방지) — **toggle 빈도 점검이 선행 TODO**

## 5. 학습 분포 (DR) = 실물 커버리지

| 축 | 학습 범위 | 실물 해석 |
|---|---|---|
| ~~cube 크기~~ | ~~4.5~5.5 cm~~ **버그로 미작동 — 실제 4.8cm 고정** (mode="usd" 무효, 2026-08-03 발견·소급 정정. v3부터 prestartup으로 실작동) | 4.8cm 근방 정육면체만 검증됨 |
| cube 질량 | 기본 ×0.7~1.3 | 가벼운 목재~플라스틱 블록 수준 |
| 마찰 | static 0.4~1.2 / dyn 0.3~1.0 | 매끈한 플라스틱~고무까지 |
| 스폰 위치 | x ±8cm, y ±12cm (중심 world 0.45, −0.25) | 파란 존 안에 놓기 |
| 스폰 yaw | 학습 0° 고정, **평가로 ±10°까지 성공률 100% 검증** | 손으로 가지런히(±10° 이내) 놓기 |
| perception 오차 | ±1cm Gaussian | 요구 인식 정확도 |
| **미커버 (알려진 갭)** | obs **latency**(지연) 미모델, 그리퍼 동역학(즉시 토글 가정), yaw >10°, joint 게인 오차 | 배포 시 주의 목록 |

## 6. Reward / Curriculum (학습 전용 — 배포와 무관, 참고용)

grasp gate(양 finger pad 0.5N 동시 접촉) × 높이 gate(상판+4cm)로 lifting(15)/tracking(16)/fine(5)을
게이팅, + upright(3) + reaching(1) − hold_still/action_l2/action_rate/joint_vel/충돌/finger-table 충격
페널티. 페널티 4종은 성능 기반 EMA curriculum(lifting>10/s)으로 지연 발동.
전체 수식·근거: `DUAL_FR3_LIFT_ENV_ANALYSIS.md` §2-4.

## 7. 실물 배포 절차 (초안)

```
50Hz 루프:
  1. 읽기: q(9), q̇(9)  [오른팔 7 + finger 2]
  2. perception: cube 위치 → base frame 변환 (±1cm 이내)
  3. obs 조립(36) = [q − q_ready, q̇, cube_pos_b, goal_pos_b + (1,0,0,0), prev_action]  # 정규화 없음
  4. a = policy(obs)                       # MLP forward, deterministic (mean)
  5. 팔: q_target = q_current + 0.1 × a[0:7] → 위치/임피던스 명령 (중력보상 ON)
  6. 그리퍼: a[7] < 0 → close / ≥ 0 → open. 히스테리시스 latch N=3 step(~60ms) 연속 시 전송
     (§8-1: sim에선 계단 1개라 불필요하나, 실물 노이즈 대비 안전 여유. 그리퍼 채널 전용)
  7. prev_action ← a
```
- goal은 실험자가 base frame으로 지정 (학습 범위 내: §1)
- 정책 export: `play.py`가 `exported/policy.onnx` 생성 — 실물 파이프라인에서 onnx 사용 가능

## 8. TODO (v1 전용)

- [x] **1. 그리퍼 toggle 빈도 점검 (2026-07-30 완료, `eval_gripper_toggle.py`)** — v1 정책 raw 시퀀스
      덤프 결과: **64개 env 전부 전환 딱 1회** (전환 분포 `{1: 64}`, 예외 0). 명령 패턴 =
      `open 접근 → close 1회 → 끝까지 유지` 완벽한 계단 하나. **채터링·2연타·헛짚음 전부 없음.**
      (초기 집계 지표의 "평균 1.75회/채터링 100%/N=13"은 카운터 버그 착시였음 — raw 시퀀스가 진실.
       이 사례로 "집계 숫자보다 raw 덤프 먼저" 원칙 확인.)
      **결론: 히스테리시스 사실상 불필요. 배포 안전 여유로 작은 latch N=3(~60ms)만 걸면 충분**
      (실물 노이즈로 파지 경계에서 1~2 step 흔들릴 가능성 대비 — 계단 1개는 무조건 통과).
      그리퍼 채널 전용 (팔 delta는 연속 명령이라 latch 무관).
- [x] **2. obs joint 순서 실측 완료 (2026-07-30)** — obs[0:9]=right_joint1~7+finger1,2 (§3). libfranka q[0:7] 그대로 사용 OK
- [ ] **3. perception 파이프라인** — 카메라/마커 선정, base frame 변환, ±1cm 검증
- [ ] **4. 실물 인터페이스 구현** — §7 루프 (τ_ext 불필요 확정으로 단순화됨)
- [ ] **5. 실물 1차 실험 프로토콜** — cube 가지런히(±10°) 놓기, goal 3~5개 지점, 성공/실패 기록
- [ ] 관찰 카드: 운반 기울기 (upright 2.43 — play에서 눈에 띄면 weight 3→4 재학습 검토)
- [ ] 이후: Run E (yaw ±45° + orientation obs), 평가 seed 고정/env 128


## 9. 제어 주기 & sim2real 표준 대비 (2026-07-30 조사)

FR3 zero-shot transfer 문헌의 **3층 주기 구조** (여러 논문·IsaacLab 이슈에서 일관):
```
[1kHz]     실물 joint impedance 컨트롤러 (libfranka FCI 고정, Ruckig 보간) — 정책과 무관하게 항상 1kHz
[10~15Hz]  RL 정책: 목표 관절각 발행 — perception 주기에 맞춤
[10~30Hz]  perception (ArUco/FoundationPose): object pose 갱신
```
→ **정책은 1kHz로 돌 필요 없음**. 10~15Hz로 목표 관절각만 던지고 실물 1kHz 컨트롤러가 그 사이를 보간.
"인식이 60Hz 안 됨"은 문제가 아니라 표준 (다들 정책을 인식 주기에 맞춤).

**우리 현황 vs 표준:**

| 항목 | 우리 (v1) | FR3 표준 | 조치 |
|---|---|---|---|
| action space | joint delta (0.1) | joint position/delta | ✓ 일치 (task-space보다 sim2real 유리) |
| 저수준 컨트롤러 | PD stiffness 80 | joint impedance @1kHz | ✓ 대응 (Run B 의도) |
| 중력보상 | disable_gravity | 실물 g(q) 보상 | ✓ 일치 |
| **정책 주기** | **50 Hz** | **10~15 Hz** | ⚠ 우리가 빠름 = "action 너무 빠름"의 정체 |
| **obs latency DR** | **없음** | 대부분 있음 (1~3 step 지연) | ⚠ 미커버 |

**핵심 시사점**: "action이 빠르다"(사용자 관찰) + "perception이 느리다"(10~30Hz)는 **별개 문제가 아니라
같은 손잡이** — `decimation 2→10` 재학습이 둘을 동시에 해결 (정책 10Hz + action 5배 느려짐).
이게 마침 FR3 zero-shot 표준 세팅이기도 함. 단, decimation은 정책이 학습 중 그 시간구조에 맞춰
최적화한 값이라 **배포 때 바꾸면 안 되고 재학습 필수** (5초÷10Hz=50 step으로 태스크 난이도 상승 유의).

**단계적 계획** (지금 바꾸지 않음, 실물 1차 결과 보고 결정):
- ① (재학습 없이) 비동기 hold: 팔 50Hz 유지 + cube pose는 perception 최신값 zero-order hold.
  파지 후 cube 정적이라 근사 잘 통함. **제일 싸고 흔한 1차 시도.**
- ② (재학습) decimation 10 + latency DR: 표준 주기로 정렬 + 인식 지연 강건화. ①로 부족하면.

### 9-1. v2 태스크 계획 확정 (2026-07-30, 사용자 결정 — decimation 재학습 트랙)

v1(50Hz)이 "action 너무 빠름 + perception 정합 안 됨" 갭 → **decimation 재학습으로 정면 대응** (②트랙).
비동기 hold(①)는 건너뜀 (사용자: from-scratch 재학습으로 감).

**v2 스펙** (= v1과 완전 동일, 변경점 3개):
- `decimation 2 → 5` : 정책 50Hz → **20Hz** (물리 100Hz 불변). perception 20~30Hz 예상과 정합,
  문헌 표준(10~15Hz)보단 약간 빠르나 예산 여유 확보. perception 실측(팀원 작업 중) 후 10Hz(dec 10)로
  더 낮추는 카드 유지.
- `episode_length_s 5 → 8` : 20Hz × 8s = **160 step** (현 250의 64%). dec 10이었으면 80 step이라
  빡빡 → dec 5 + 8s가 균형점.
- `resampling_time_range 5 → 8` : goal 재샘플링을 episode에 맞춤. 베이스가 5초 고정이라 8초
  에피소드면 goal이 5초에 한 번 더 뽑혀 "에피소드당 goal 2개"가 됨(최초 v2 run에서 play로 발견,
  중단·수정). episode = goal 1개로 유지 (실물 프로토콜=goal 하나 지정과 정합).
- 나머지(obs 36, delta 0.1, reward/DR/curriculum) 전부 v1 동일. from-scratch, 로그 `dual_fr3_lift_v2/`.
- **decimation 값 근거**: 문헌은 "정책 주기 Hz"로 보고(decimation은 물리Hz 종속). 매니퓰레이션
  zero-shot은 10~15Hz 수렴(perception 정합). 우리 물리 100Hz → 20Hz=dec5, 14Hz=dec7, 10Hz=dec10.
  dec5로 시작(예산 안전) → 되면 dec10으로 낮춰 표준 하한 도달.
- 리스크: 160 step이 파지+운반+유지에 모자라면 → episode 10s 또는 dec 재검토. latency DR은 v2
  성공 후 별도 축으로 (변인 분리, C 시리즈 교훈).
