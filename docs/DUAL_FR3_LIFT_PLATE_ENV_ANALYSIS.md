# Dual FR3 Plate-EE Box Lift — 환경 · Reward 정밀 분석

> 대상: `Isaac-Lift-Box-DualFR3-Plate-v0` (`lift/config/dual_fr3_plate/joint_pos_env_cfg.py` + 베이스 `lift/lift_env_cfg.py`)
> 알고리즘: PPO (rl_games / rsl_rl 양쪽 등록), 기본 4096 envs
> 작성 기준: **2026-07-27** — v4(중심 파지 유도)까지 반영. run 이력·결과는 `logs/rsl_rl/log_per_task.txt` 참조
> (P1=v1 모서리 걸침 → v2/v3 → **P2 성공** (129.6, 모서리 몰림 잔존) → v4 → **P3 실패: 정지 국소해** → **v5 근접 게이팅**)
> P3 교훈: "제자리에서 딸 수 있는 shaping + 움직임 페널티" 조합은 정지 국소해를 만든다 — 파지 품질 항은 근접 게이팅 필수
> 자매 문서: `DUAL_FR3_LIFT_ENV_ANALYSIS.md` (오른팔 cube lift — gating/curriculum 교훈의 출처)

---

## 1. Task Description

- **작업**: 바닥(z=0) 위 박스(30~50cm, env별 고정)를 **그리퍼 없이 양팔의 판(plate) EE로 조여(squeeze)** 마찰력만으로 들어올려, 매 에피소드 샘플링되는 목표 위치(공중)로 운반
- **로봇**: Husky 베이스 + FR3 양팔 + **140×140×25mm 판 EE** (`DUAL_FR3_PLATE_EE_CFG`, USD `dual_fr3_plate_ee.usd`). 그리퍼 없음 → **양팔 14 joint 전부 제어**
- **EE 정의**: `left/right_fr3_link7` + 판 작업면 오프셋 `PLATE_TCP_OFFSET_POS = (0,0,0.132)` (플랜지 0.107 + 판 두께 0.025). 판 collider는 link7 body 소속 (자기 링크와 비충돌, 외부 물체와만 접촉)
- **파지 = 힘 제어**: 그리퍼가 없으므로 "잡기"는 **양 판의 법선력 × 마찰**로만 성립.
  힘은 명시적 action이 아니라 **PD 위치 오차(침투 깊이)로 암묵 제어** — 임피던스 제어와 동일 원리.
  rule-based 데모(`scripts/tutorials/05_controllers/run_dual_fr3_plate_lift.py`)로 2~3cm 침투 명령이 2kg 박스를 드는 것 검증 완료
- **에피소드**: 5.0 s × 50 Hz 제어 (sim 100 Hz, decimation 2) = **250 step**

---

## 2. MDP 구조 (2026-07-27 기준)

### 2-1. Observation (policy, 총 54 dims, history 없음)

| Term | 내용 | Dims | 비고 |
|------|------|------|------|
| `joint_pos` | 양팔 14 joint, default 대비 상대값 | 14 | |
| `joint_vel` | 위와 동일 | 14 | |
| `object_position` | 박스 위치, robot root frame | 3 | |
| `target_object_position` | goal pose command (pos 3 + quat 4) | 7 | quat 4 dims는 identity → dead dims (cube task와 동일) |
| `actions` | 직전 action | 14 | |
| `plate_forces` | **양 판 접촉력 크기 (L, R)** × scale 0.02 | 2 | cube task와 달리 obs에 포함 — 근거는 §3-2 |

**obs noise (v3, 2026-07-27 추가)**: joint_pos σ0.005 rad / joint_vel σ0.05 / object_position σ0.01 m
/ plate_forces σ0.04 (2N 상당, τ_ext 추정 오차 대응) — cube task Run C와 동일 근거
(noise 단독·물성 DR 단독은 drop 0, "조합"만 슬립 드랍 발생하는 상호작용 확인).

### 2-2. Action (14 dims) — **delta position**

| | 구현 | Scale |
|---|---|---|
| arms (14) | `RelativeJointPositionActionCfg` — `target = q_current + 0.1·a` | 0.1 |

- cube task의 절대+offset(scale 0.5) 대신 **delta 채택** (2026-07-27 합의). 근거:
  1. contact-rich task — 스텝당 0.1rad 제한이 접촉 중 명령을 부드럽게 함 (공식 dexsuite와 동일 설정)
  2. 절대+offset은 ready ±0.5rad hard-bound라 **바닥 박스 접촉 자세에 도달 못 할 위험**
  3. 실로봇 이식 시 rate limit 내장 효과
- 부작용: `action_rate` 페널티가 "가속도 페널티" 성격으로 변함, drift는 obs의 joint_pos로 policy가 보정

### 2-3. Sensors (squeeze 판정 + obs)

```python
# 판별로 분리 (PhysX 제약: filter는 sensor body당 1:1 — cube task에서 확인된 것과 동일)
left_plate_object_contact  = ContactSensorCfg(prim_path=".../left_fr3_link7",
                                              filter_prim_paths_expr=[".../Object"])
right_plate_object_contact = ContactSensorCfg(prim_path=".../right_fr3_link7",
                                              filter_prim_paths_expr=[".../Object"])
```

- 판 collider가 link7 body에 속하므로 link7 센서가 판-박스 접촉력을 그대로 보고
- `filter_prim_paths_expr`로 **박스와의 접촉력만** 측정 (바닥·자기충돌 제외)
- reward gate와 obs(`plate_forces`) 양쪽에 사용

### 2-4. Reward

```
G      = 𝟙[min(‖F_left‖, ‖F_right‖) > 1.0 N]        # squeeze gate: 양 판이 동시에 박스 접촉
L      = 𝟙[z_obj > half_env + 0.10]                  # 높이 gate — env별 (안착 중심 + LIFT_DELTA, v3)

r_total = 2.0  · mean_i(1 − tanh(d_ee_i-face_i / 0.3))    # reaching (양 "측면 중심점"까지, 비gated, v2)
        + 3.0  · mean_i(1 − tanh(d_ee_i-face_i / 0.08))   # reaching_fine (중심 정밀 접근, v4)
P      = (1−tanh(d_L/0.15))·(1−tanh(d_R/0.15))       # 근접 factor — 양 판 모두 중심점 근처일 때만 1 (v5)

        + 1.0  · P · mean(clamp(−n_i·ŷ_box, 0, 1))        # plate_alignment (면-면 자세, v2, v5 근접 게이팅)
        − 4.0  · P · (|x_L−x_R| + |z_L−z_R|)              # plates_misalignment (v4 강화, v5 근접 게이팅)
        + 2.0  · G                                        # squeezing (접촉 자체 보상, 발견 가속)
        + 15.0 · L · G                                    # lifting        (squeeze-gated)
        + 16.0 · L · G · (1 − tanh(d_obj-goal / 0.3))     # goal_tracking  (squeeze-gated)
        +  5.0 · L · G · (1 − tanh(d_obj-goal / 0.05))    # fine_grained   (squeeze-gated)
        − 1e-4 · N_{link[1-6] 접촉}  (curriculum 발동 후 −1.0)   # undesired_contact_penalty (v3)
        − 1e-4 · ‖a_t − a_{t−1}‖²   (curriculum 발동 후 −0.1)   # action_rate
        − 1e-4 · ‖q̇‖²              (curriculum 발동 후 −0.1)   # joint_vel
```

- **cube task 대비 변경점**:
  - `reaching`: `object_dual_ee_face_distance` — 양 판→박스 **양 측면 중심점**(yaw·env별
    반너비 반영) 거리 tanh 평균, std 0.3, weight 2. v1은 박스 "중심" 기준이라 위/모서리
    어디로 접근해도 같은 보상 → Run P1 play에서 **모서리 걸침 파지 발현**, v2에서 교체
  - `plate_alignment`(w+1.0) / `plates_misalignment`(w−2.0) 신설 (v2): 판 법선↔박스 면
    법선 반평행 보상 + 두 판 높이/전후 어긋남 페널티 — "면-면으로 마주 보고 꼬집기" 유도.
    env별 반너비는 `xformOp:scale` startup 1회 캐싱(`_box_half_sizes`)으로 계산
  - `squeezing` 신설 (w=2): reaching → lifting 사이 간극이 cube보다 커서
    ("접근"과 "조여서 들기" 사이에 "양팔 동시 접촉"이라는 중간 단계 존재) 사다리 보상 추가
  - gate threshold 0.5→**1.0 N**: 판은 면접촉이라 스침 접촉 오탐 여지가 커서 상향
- **G를 lifting/tracking 전부에 곱는 이유**: cube task §3에서 실증된 exploit 방어 구조 그대로 —
  높이 gate만으로는 "팔뚝에 얹기/밀어올리기"가 이득이 됨

### 2-5. Termination / Curriculum / DR

| 항목 | 내용 |
|---|---|
| time_out | 5 s (success termination / sparse bonus 없음 — cube task와 동일 약점) |
| object_dropping | **제거 (None)** — 바닥 위 박스는 낙하 개념이 없음 |
| curriculum | `modify_reward_weight_on_lifting_learned` 재사용 — lifting > 10/s 시 action_rate/joint_vel −1e-4→−0.1, **contact_penalty −1e-4→−1.0** (latch, 성능 기반) |
| DR: 박스 크기 | **0.30~0.50 m** (`scale DR 0.75~1.25` × 기준 0.4), `mode="usd"` per-env 고정, `replicate_physics=False` |
| DR: 박스 밀도 | 기준 밀도 30 kg/m³ + **질량 scale 0.6~1.4** (`randomize_rigid_body_mass`, startup, 관성 재계산) → 실질 질량 0.5~5.3 kg. 스모크 테스트 실측: 8 envs에서 1.2~2.7 kg 분포 확인 |
| DR: 박스 스폰 | x −0.05~+0.10, y ±0.10, **yaw ±0.3 rad** (기준 world (0.5, 0, 0.26) — 최대 박스 기준 스폰 후 작은 박스는 낙하 안착) |
| DR: 박스 마찰 | **static 0.4~1.2 / dynamic 0.3~1.0** (v3에서 cube Run C와 같은 폭으로 확대. startup, 64 buckets, make_consistent). 판은 USD에 재질 명시(0.6/0.55) → 유효 static(average) ≈ 0.5~0.9 |
| DR: joint 물성 | **액추에이터 PD 게인 ±15% scale** (v3, `randomize_actuator_gains`, startup, 양팔 14 joint) |
| DR: goal | base 기준 x 0.45~0.60, y ±0.10, z **0.15~0.35** (world 0.555~0.755 — 초기값 0.25~0.45는 최대 0.7m 상승이라 과도해 하향, 데모 검증 높이 포함) |
| 시각화 | DR 존 마커: 파란 판=박스 스폰 범위(바닥), 초록 판 2장=goal 범위 상/하한 (시각 전용, cube task와 동일 방식) |
| **없는 DR** | obs latency (검토 보류) |

---

## 3. 설계 논거

### 3-1. Squeeze는 왜 위치 명령으로 되는가 (rule-based 데모의 교훈)

- PD 게인 하에서 **접촉에 막힌 위치 명령의 잔여 오차 = 법선력**. 데모에서 침투 2cm 명령 시
  잔여 오차가 정확히 침투량만큼 남으며(err 0.021) 박스가 유지됨을 확인
- **결정적 함정 (데모에서 실증)**: 이동 중 침투 명령이 풀리면 즉시 미끄러짐.
  데모의 보간이 시작점을 "현재 TCP(면에 막힌 위치)"로 잡는 바람에 lift 시작 순간 조임 명령이
  0으로 리셋 → 판이 박스 면을 타고 허공으로 (1차 시도 실패 원인). `start_b`로 조임 명령을
  고정해 해결. **RL에서도 같은 원리**: policy는 상승 중에도 조임 방향 delta를 계속 출력해야
  하며, 이를 실패 시 즉각적인 보상 손실(G=0 → 초당 ~35 손실)로 학습하게 됨
- delta action은 이 "계속 밀기"와 궁합이 좋음: 접촉에 막힌 상태에서 조임 방향 성분을
  유지하면 자연스럽게 침투 명령이 유지됨

### 3-2. plate_forces를 obs에 넣는 이유 (cube task와 반대 결정)

- cube task는 "실로봇 그리퍼에 F/T 없음"이 근거로 contact을 reward 전용으로 제한했다
- **이 셋업에는 FT 센서가 없다** (mount_to_ft CAD는 참고용이었음). 대신 **FR3 내장
  관절 토크 센서 기반 외력 추정치로 채운다**: libfranka의 `tau_ext_hat_filtered`(외력
  관절 토크) → `O_F_ext_hat_K`(EE 외력 추정, 사실상 가상 F/T). 판 접촉력 크기 ≈ 이 값의 norm
- 힘 조절이 과제의 본질(밀도 DR로 필요 조임력이 env마다 다름)이므로, 접촉력 피드백이
  없으면 policy가 박스 미끄러짐(관측 지연)으로만 힘 부족을 알 수 있어 학습이 느려짐
- **배포 전제 조건**: ① FR3 payload 설정에 판+볼트(~1.4 kg) 질량·무게중심 등록 (판 무게가
  외력으로 잡히는 것 방지, FT tare에 해당) ② τ_ext 추정은 모델 오차·관절 마찰로 수 N급
  오차/드리프트가 있으므로 **sim에서 plate_forces에 noise + bias DR 필수** (§4)
- 보험: 추정치 품질이 부족하면 obs에서 제거하고 재학습해도 task는 성립 — delta action +
  joint_pos obs 조합은 "명령 대비 관절 미동 = 접촉"이라는 암묵적 힘 신호를 제공한다

### 3-3. lifting 판정 높이 0.35의 트레이드오프 (→ v3에서 env별 판정으로 해결)

> **2026-07-27 해결**: `_box_half_sizes`(xformOp:scale 캐싱)로 env별 안착 중심을 알 수 있게
> 되어, 판정을 `z > half_env + 0.10`으로 교체 (`object_is_lifted_perenv_and_squeezed` 등).
> 아래는 v1 고정값 시절의 논의 기록.

- 고정값 0.35는 **최대 박스(안착 중심 0.25) 기준 +10cm**. 큐브가 모서리로 구르는 순간의
  중심 상승(최대 0.25·√2≈0.35)과 겹치지만 G(양판 접촉) 동시 조건이 오탐을 차단
- 대신 **최소 박스(중심 0.15)는 +20cm 들어야** 판정 통과 — 작은 박스일수록 과제가 어려움.
  per-env 판정 높이(스폰 중심 + Δ)가 정석이나, scale DR 값을 reward 함수에서 참조하는
  배관이 필요해 v1은 고정값. 학습이 작은 박스에서만 정체되면 여기부터 의심할 것

---

## 4. 남은 구조적 약점 (심각도순)

1. **Success 개념 부재** — cube task와 동일 (sparse bonus/종료/성공률 지표 없음)
2. ~~작은 박스 불리한 고정 lifting 높이~~ → **2026-07-27 반영 (v3)** — env별 판정(안착 중심+0.10)
3. ~~박스 yaw DR 없음~~ → **2026-07-27 반영** (yaw ±0.3 rad)
4. ~~obs noise 부재~~ → **2026-07-27 반영 (v3)** (§2-1 참조. 남은 것: obs latency, τ_ext 바이어스/드리프트 모델)
5. ~~마찰 DR 부재~~ → **2026-07-27 반영** (박스 static 0.5~1.0 DR + 판 재질 0.6 명시)
6. ~~충돌 페널티 없음~~ → **2026-07-27 반영 (v3)**: `arms_contact` net-force 센서(양팔 link1~6,
   판=link7 제외) + `undesired_contacts` 페널티 + 성능 curriculum (cube Run B 패턴).
   남은 구멍: **판(link7)이 바닥을 치는 것은 미감지** — link7 센서는 박스 filter 전용이라
   분리 불가(PhysX body당 filter 1개). 문제 시 판-바닥 filtered 센서 추가 검토
7. **판 질량 미반영**: 실물 판+볼트 ~1.4kg가 link7 질량에 없음 — 실배포 전 USD/URDF 반영 필요
8. **양 판 접촉 높이 자유**: 한 판은 위쪽·한 판은 아래쪽을 눌러 모멘트로 박스가 회전하는
   실패 모드 가능 — 학습에서 자연 해결이 안 되면 판 높이차 페널티 고려

---

## 5. 로드맵

- [x] **5-1. 환경 구현 + 스모크 테스트** — 2026-07-27 완료 (obs 54 / action 14 / DR 분포 확인)
- [x] **5-2. rule-based 실현 가능성 검증** — 데모로 squeeze-lift 물리 성립 확인 (사용자 GUI 검증)
- [x] **5-3. 마찰 DR** — 2026-07-27 반영 (박스 static 0.5~1.0 / dynamic 0.4~0.9, 판 USD 재질 0.6/0.55 명시)
- [x] **5-4. 박스 yaw DR** — 2026-07-27 반영 (±0.3 rad)
- [x] **5-7. DR 존 시각화** — 2026-07-27 추가 (스폰=파랑, goal 상/하한=초록, cube task와 동일 방식)
- [x] **5-0. 첫 학습 run (Run P1)**: `logs/rsl_rl/dual_fr3_plate_lift/2026-07-27_16-00-04`
   (v1 config). 들기는 발견했으나 **판 모서리 걸침 파지 발현** → v2 보상 개편의 근거가 됨
- [~] **5-8. Run P2 (진행 중)**: `logs/rsl_rl/dual_fr3_plate_lift/2026-07-27_18-38-18`
   (v2 꼬집기 유도 + v3 강건성/충돌 페널티 전부 반영, 4096 envs, 5000 iter).
   관찰 포인트 — ① play에서 면-면 꼬집기 발현 ② noise+DR로 인한 수렴 지연 정도
   ③ `squeezing`→`lifting` 사다리 ④ 커리큘럼 3종(action_rate/joint_vel/contact) 발동 후 생존.
   run 이력은 `logs/rsl_rl/log_per_task.txt` 참조
- [ ] **5-5. Success 정의** + sparse bonus / 성공률 지표 (cube task 5-3과 공용 설계 가능)
- [ ] **5-6. sim2real 대비**: obs noise (특히 plate_forces에 FT급 노이즈), 판 질량 반영,
   env별 lifting 판정 높이

---

## 6. Cube task(오른팔 그리퍼)와의 구조 대비

| 축 | cube lift (`dual_fr3`) | box lift (`dual_fr3_plate`) | 비고 |
|---|---|---|---|
| 파지 원리 | 그리퍼 기구적 파지 | **양팔 squeeze 마찰 파지** | 힘 제어가 본질 |
| action | 오른팔 7 절대+offset(0.5) + 이진 그리퍼 | **양팔 14 delta(0.1)** | contact-rich라 delta |
| obs 접촉력 | 없음 (실물 센서 없음) | **판 접촉력 2 dims** (FR3 τ_ext 외력 추정으로 대응) | §3-2 |
| gate | 양 finger pad 0.5 N | 양 판 1.0 N | 동일 철학, threshold 상향 |
| 사다리 보상 | reaching → lifting | reaching → **squeezing** → lifting | 중간 단계 추가 |
| 물체 DR | 크기 4.5~5.5cm | **크기 30~50cm + 밀도(질량 0.6~1.4×)** | |
| 물체 위치 | 테이블 위 (z=0.6) | **바닥 (z=0)** — 하향 도달 필요 | delta 채택 사유 중 하나 |
| object_dropping | 상판 −10cm | 제거 (무의미) | |
| curriculum | 성능 기반 (공유) | 동일 함수 재사용 | |
