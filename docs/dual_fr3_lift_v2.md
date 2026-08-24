# dual_fr3_lift_v2 — 20Hz 배포 후보 환경 스펙 & sim2real 가이드

> **2026-08-08 정정:** 실물 action/observation 계약은
> [`DUAL_FR3_LIFT_CUBE_SIM2REAL.md`](DUAL_FR3_LIFT_CUBE_SIM2REAL.md)를 우선한다. 아래 20 Hz target-hold
> 초안과 v1의 배포 loop는 실제 100 Hz physics-substep current-q rebase를 생략한 historical 설명이다.

> **v1(50Hz)의 후속 — 정책 주기를 실물 perception에 맞춘 버전.** v1.md와 대부분 동일하며,
> **차이점(제어 주기·에피소드)만** 여기 명시하고 나머지는 v1.md를 참조한다.
> 설계 근거: `DUAL_FR3_LIFT_ENV_ANALYSIS.md` §5-12, §9, §9-1. run 기록: `log_per_task.txt`.
> 작성: 2026-08-03 (v2 eval 판정 직후)

## 0. 정책 신상명세

| 항목 | 값 |
|---|---|
| 배포 후보 checkpoint | `logs/rsl_rl/dual_fr3_lift_v2/2026-07-31_14-33-32/model_4999.pt` |
| 학습 | PPO (rsl-rl-lib 3.0.1), 4096 envs, 5000 iter, from-scratch, seed 42, [5090-v2: IsaacLab 2.2.1/Isaac Sim 5.0] |
| 네트워크 | Actor/Critic MLP [256,128,64] ELU, 입력 **36** → 출력 8 (v1과 동일) |
| obs 정규화 | 없음 — raw 값 그대로 |
| **제어 주기** | **20 Hz** (물리 100Hz × decimation 5) ← v1(50Hz)과의 핵심 차이 |
| 에피소드 | **8초 = 160 step** (v1은 5초/250 step) |
| 검증 성능 | 성공률 **100%** (5cm·strict 2.5cm 모두), yaw ±10°까지 동일, precise-hold 0.94, drop 0/384ep, hold 0.96 — Play-Noisy-v2 기준 |

## 1. v1 대비 차이점 (이것만 다름, 나머지는 v1.md 참조)

| 항목 | v1 | **v2** |
|---|---|---|
| decimation | 2 | **5** (정책 50Hz → **20Hz**) |
| episode_length_s | 5 | **8** (160 step) |
| resampling_time_range | 5 | **8** (goal 1개/에피소드 유지) |
| obs / action / reward / DR | — | **v1과 완전 동일** (obs 36, delta 0.1, grasp gating, upright, 물성 DR, obs noise …) |

**변경 이유** (§9): v1 50Hz가 "action 너무 빠름(사용자 관찰) + 실물 perception(~10~30Hz)과 정합 안 됨".
FR3 zero-shot 문헌은 정책주기 10~15Hz로 수렴 → 20Hz(dec5)로 낮춤. 물리 100Hz는 불변(안정성).
resampling도 8초로 맞춰야 goal이 에피소드당 1개 (베이스 5초 고정이라 안 맞추면 goal 2개 버그).

## 2. v1 대비 성능 (동급 이상)

| 지표 | v2 (20Hz) | v1 (50Hz) |
|---|---|---|
| 성공률 (yaw 0/5/10°) | **1.000 / 1.000 / 1.000** (strict 동일) | 1.000 (동일) |
| precise-hold (<5cm) | **0.94** | 0.92 |
| hold (파지+들림) | **0.96** | 0.95 |
| drop | **0 ×3** (yaw별) | 0 (일부 run 널뜀) |
| upright (학습 tail) | **2.75** | 2.43 |
| lifting 발견 | **it55** | it76 |
| pos err | 0.213 | 0.209 |

→ **20Hz로 성능 손실 없음.** 오히려 precise-hold·drop 안정성·upright·발견속도에서 우위
(느린 주기가 탐색을 안정화 + 운반 자세 유지에 유리). 실물 perception 주기 정합은 덤.

## 3. 배포 절차 (v1.md §7과 동일하되 주기만)

v1.md §7 루프 그대로. **차이: 루프를 50Hz가 아니라 20Hz(50ms 주기)로 실행.**
```
20Hz 루프 (v1.md §3 obs 36, §4 action 8, §7 조립 동일):
  · 팔: q_target = q_current + 0.1 × a[0:7]  (실물 1kHz joint impedance가 50ms 구간 보간)
  · 그리퍼: a[7] 부호 + 히스테리시스 latch (v1.md §8-1, 20Hz면 N≈2 step으로 충분)
  · perception: 실물 인식 주기(~10~30Hz)와 정책 20Hz가 근접 → hold 근사 부담 적음 (v1보다 유리)
```
- obs joint 순서, ready pose, action 규약, DR 커버리지 = **v1.md와 100% 동일** (§2~§5 참조)

### 3-1. object_position 소스 전략 (2026-08-03 논의 — ArUco 오차 대응)

**팀원 실측**: ArUco 오차 = 통상 ~5cm / 팔 고속 이동 시 최대 10cm (motion blur 추정).
**우리 학습 커버리지**: σ1cm Gaussian → **1~2cm(3σ 3cm) 지터만** 커버. 5cm는 분포 밖.
단 오차의 "시점"이 관건 — cube가 **테이블에 정지한 파지 전** 구간(정밀 필요)은 blur 없어
오차가 훨씬 작을 것으로 추정. 운반 중(오차 큼)은 위치 정밀도 요구가 낮음 (사용자 통찰).

**배포 전략 (점진)**:
- **1차: A안 (ArUco만)** — 팀원의 "정지 시 오차" 실측이 1~2cm 이내면 파지까지 학습 분포 안.
  운반 중 흔들림은 눈으로 관찰.
- **B안 (파지 후 FK 전환)을 코드에 스위치로 내장** — object_position 소스를 ArUco → TF 조회
  (base→TCP frame + cube 오프셋)로 전환. **if문 + TF 한 줄** (robot_state_publisher가 FK를 TF로
  이미 제공). 운반 중 5~10cm 오차와 그리퍼 가림을 동시 해결. 흔들림 관찰되면 플래그만 켬.
  **⚠ grasped 판정은 close 명령만으로 절대 트리거 금지** (2026-08-03 논의): 파지를 놓치면 손가락이
  허공에서 0cm까지 닫히는데, 명령 기준이면 "유령 cube를 든" FK로 전환돼 영영 복귀 못 함.
  판정 = close 명령 **AND 그리퍼 폭이 cube 두께(0.03~0.07m)에서 멈춤**. 운반 중에도 매 스텝
  폭 감시 — 0 근처로 무너지면(슬립) 즉시 해제 → ArUco 복귀. libfranka `grasp(width, epsilon)`의
  성공 bool을 쓰면 하드웨어가 이 판정을 대신해줌. (정책도 obs의 finger 관절값으로 같은 신호를
  보므로 일관적.)
- **금지: σ5cm 재학습** — 정밀도 파괴 + bias는 zero-mean DR로 못 덮음. 정지 상태에서도 5cm면
  그건 calibration bias → 캘리브레이션을 고칠 것.

**frame 정합 검증 (실물 투입 전 필수)**: 우리 object_position 기준 = dual_fr3 URDF **root link
(팔 장착 상판, world z=0.405)** frame. 팀원 추정도 "URDF base frame"이라 하나 **같은 링크인지**
확인 필요. 검증법: cube를 스폰존 중앙(world 0.45, −0.25, 0.775)에 놓고 팀원 추정값이
base 기준 **(0.45, −0.25, ≈0.37)** 나오는지 대조 — frame 정합 + calibration bias 동시 검증.

**팀원 확인 요청 중**: ① 정지 상태 오차 별도 실측 ② 정지에서도 5cm면 calibration 재수행.

## 4. TODO / 다음 카드

- [ ] play 육안 재확인 (사용자: "1~2s에 완료, 8s는 여유" — 확인됨)
- [ ] 실물 perception 실측 주기 확인 (팀원 작업 중) → 20Hz가 정합인지 최종 판단
- [ ] (선택) dec 10(10Hz) 추가 실험 — 표준 하한, 8초÷10Hz=80 step 예산으로 더 빡빡
- [ ] latency DR 추가 (v2 성공 확인됨 → 이제 별도 축으로 붙일 수 있음)
- [ ] 관찰 카드 (vision/tactile 직접 입력): **현재 불필요.** cube lift는 좌표(object_position) 기반
      준정적 파지 — vision을 정책에 직접 넣거나(point cloud) tactile sim(TacSL)을 붙일 이유 없음.
      실제 인식 특성은 이미 DR로 흉내 중: obs noise ±1cm(인식 오차) + 20Hz(주기 정합) + latency DR(예정).
      IsaacLab에 실제 ArUco 렌더+detection을 넣는 것도 가능하나 좌표 기반엔 과함(렌더링이 학습 무겁게).
      **실물에서 좌표만으로 부족(가림/조명/각도로 마커 안 보임)해지면** → 그때 vision 직접 입력 검토,
      레시피 참고: VT-Refine (arXiv:2510.14930 — point cloud + sim 노이즈 주입 + real2sim2real,
      단 그건 contact-rich 조립용이라 tactile은 우리 태스크에 여전히 불필요).
- [ ] 관찰 카드 (외란 주입): cube lift는 준정적 태스크라 외란 DR 우선순위 낮음(현재 없음).
      **실물에서 "파지 후 슬립/흔들림"이 실제 관찰되면** → (b) 마찰 하한 낮추기 먼저(태스크 직결·기존
      DR 범위 조정이라 쌈) → 그래도 부족하면 (a) 박스에 random external force/torque 이벤트. 순차 도입
      (한 축씩, C 시리즈 교훈). plate(contact-rich)는 외란 필요도가 더 높음 — cube와 성격 다름.
- [ ] eval tts 표시 버그: 스크립트가 50Hz 하드코딩이라 v2 tts를 1.16s로 표시(실제 2.9s). 성공률·hold·drop은 정확
- 배포 checkpoint 선택: v1(50Hz) vs v2(20Hz) — **실물 perception 주기 나오면 그에 가까운 쪽** (현재 v2 유력)
