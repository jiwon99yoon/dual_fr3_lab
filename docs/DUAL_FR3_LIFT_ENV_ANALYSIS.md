# Dual FR3 Cube Lift — 환경 · Reward 정밀 분석

> **이 문서 = 설계 서사 + 근거 (v0~Run D 아크가 왜 이렇게 됐나).** 계속 최신화됨.
> **배포용 완성 스펙(현재 후보 정책)은 별도 문서**: `dual_fr3_lift_v1.md`(50Hz),
> `dual_fr3_lift_v2.md`(20Hz 정육면체), `dual_fr3_lift_v3.md`(20Hz 세로 50×50×100 — **실물 배포 후보**).
> "실물에 뭘 올리나"는 그쪽을 볼 것. 최신 파생: **v1**=v0에서 tau_ext obs 제거(§5-11),
> **v2**=v1에서 decimation 5·20Hz(§5-12), **v3**=v2에서 세로 cube + 크기 DR 버그 수정(§5-13).

> 대상: `Isaac-Lift-Cube-DualFR3-v0` (`lift/config/dual_fr3/joint_pos_env_cfg.py` + 베이스 `lift/lift_env_cfg.py`)
> 알고리즘: PPO (rsl_rl), 4096 envs
> 작성 기준: **2026-07-27 14:42** — grasp gating(ContactSensor) 구현·검증 완료 시점.
> **최종 갱신 2026-07-28 저녁 (Run D 구현 완료)**: §2는 Run D 기준 (obs 43, delta action,
> action_l2, upright 보상, finger-table 충격 페널티). yaw DR + orientation obs는 Run E로 분리
> (거동 교정과 난이도 상승 분리 — 사용자 결정). 이하 이력 헤더:
> **갱신 2026-07-27 16시경 (Run B 준비 완료)**: MJCF 테이블(상판 0.75) + 중력보상(disable_gravity)
> + 충돌 페널티 + 측정 joint 토크 obs(43 dims) 반영. §2는 Run B 기준, Run A까지의 상태는 git 이력 참조.
> 이력: run 4개 (`logs/rsl_rl/dual_fr3_lift/`) TensorBoard + `params/env.yaml` 대조 분석 포함.
> Run A(v5, grasp gating만): `2026-07-27_14-59-58`, 3000 iter 완료 — 분석은 §5-0 판정 기준 참조

---

## 1. Task Description

- **작업**: MJCF 테이블(얇은 상판 z=0.75 + 양옆 다리, 선배 MuJoCo 세팅 `dual_fr3_table.md`, 단 x=0.6 배치) 위 DexCube(4.5~5.5cm)를 오른팔로 잡아 들어올려, 매 에피소드 샘플링되는 목표 위치(공중, world z 0.855~1.005)로 이동
- **로봇**: Husky 베이스 + FR3 양팔. **오른팔 7 joint + 오른손 그리퍼만 제어**, 왼팔은 PD가 ready pose 유지 (action 없음). **로봇 링크 disable_gravity=True** — 실물 FR3의 중력보상 모델링 (payload 중력은 유지, stiffness 80과 함께 실물 임피던스 정합)
- **EE 정의**: `right_fr3_link7` + TCP 오프셋 (z+0.2104, yaw −45°) — FrameTransformer로 계산, rule-based 데모에서 검증된 값
- **에피소드**: 5.0 s × 50 Hz 제어 (sim 100 Hz, decimation 2) = **250 step**. 목표는 에피소드당 1회 샘플링 (resampling 5 s = episode 길이)

---

## 2. MDP 구조 (2026-07-27 14:42 기준)

### 2-1. Observation (v1 기준 총 36 dims — v0는 43, history 없음)

| Term | 내용 | Dims | 비고 |
|------|------|------|------|
| `joint_pos` | 오른팔 7 + 오른손가락 2, default 대비 상대값 | 9 | 왼팔 9개 제외 (차원 축소) |
| `joint_vel` | 위와 동일 joint | 9 | |
| `object_position` | cube 위치, **robot root frame** | 3 | |
| `target_object_position` | goal pose command (pos 3 + quat 4) | 7 | **quat 4 dims는 항상 identity → dead dims** (orientation 명령 미사용) |
| `actions` | 직전 action | 8 | arm 7 + gripper 1 |
| ~~`joint_torques`~~ | 오른팔 측정 joint 토크 | ~~7~~ | **v1 ablation으로 제거 확정 (2026-07-29, §5-11)** — 이 태스크에선 순수 잡음 주입이었음 (critic loss 절반 감소로 실증). v0(-v0 태스크)에만 잔존. 접촉 의존 태스크로 확장 시 재고려 |
| ~~`object_orientation`~~ (Run E 예약) | cube 자세 quat (root frame, w≥0) | (4) | **yaw DR과 세트로 연기** — yaw 없는 동안 cube 기울기=그리퍼 기울기라 joint_pos로 추론 가능. mdp 함수는 구현 완료 |

**obs noise (Run C부터 학습에 포함)**: joint_pos σ0.005 rad, joint_vel σ0.05, object_position σ0.01 m
(perception 모사), joint_torques σ0.5 Nm — 전부 additive Gaussian, `enable_corruption=True`로 학습 시 활성.
PLAY cfg는 corruption off(깨끗한 baseline 평가), `Play-Noisy-v0`가 재활성(강건성 평가, §5-6).
finger pad 접촉력은 obs에 넣지 않는다 — 실로봇에 F/T·촉각 센서가 없기 때문 (contact은 reward 전용, §3).
관절 토크는 실물에 있으므로 예외.

### 2-2. Action (8 dims) — Run C 3차부터 delta 전환 (2026-07-28)

| | 구현 | Scale |
|---|---|---|
| arm (7) | `RelativeJointPositionActionCfg` — **현재 자세 기준 delta 명령** (plate와 동일) | 0.1 |
| gripper (1) | `BinaryJointPositionActionCfg` — open 0.04 / close 0.0 **이진** | — |

**절대(scale 0.5) → delta(0.1) 전환 근거** (Run C 1·2차 실패 분석):

| 조건 | action | obs noise | 결과 |
|---|---|---|---|
| franka 원본 / cube v2·A·B | 절대 0.5 | 없음 | 성공 |
| cube C1 / C2 | 절대 0.5 | 있음 | **noise_std 5.5 / 12.8 폭주** |
| plate P2 / P4 | delta 0.1 | 있음 | 성공 (std 1.98 피크 후 수축) |

절대 action에서는 탐색 노이즈가 곧 목표 자세 점프(±0.5 rad)라 obs noise로 advantage가 흐려지면
"entropy 우세 → 더 큰 휘두름 → drop → 더 흐린 신호"의 양성 피드백이 열린다. delta는 스텝당
이동이 ±0.1 rad로 **구조적으로 캡**되어 이 고리가 못 열림. 문헌 근거: Demystifying Action Space
Design (arXiv:2602.23408 — delta 일관 우위, joint-space 안정 우위), Benchmarking Action Spaces
(arXiv:2606.18594 — increment 계열 우위), gSDE (Raffin et al. — 스텝별 Gaussian 탐색의 저크 문제).
주의: obs `actions` 항의 의미가 바뀌므로 이전 checkpoint와 호환 불가. delta 특성상 zero-action이
"현 자세 유지"라 리셋 직후 래칫 처짐류 거동은 정상 (plate에서 기확인).

### 2-3. Sensors (grasp 판정용, 2026-07-27 추가)

```python
# joint_pos_env_cfg.py — PhysX 제약: filter는 sensor body당 1:1이라 finger마다 분리
# Run D부터 filter 2개: [0]=Object(파지 gate), [1]=Table(충격 페널티)
leftfinger_object_contact  = ContactSensorCfg(prim_path=".../right_fr3_leftfinger",
                                              filter_prim_paths_expr=[".../Object", ".../Table"])
rightfinger_object_contact = ContactSensorCfg(prim_path=".../right_fr3_rightfinger",
                                              filter_prim_paths_expr=[".../Object", ".../Table"])
```

- `filter_prim_paths_expr`로 접촉 상대를 분리 측정. **주의 (Run D): `object_grasped`는 반드시
  filter index 0(Object)만 봐야 함** — 전체 filter에 amax를 취하면 finger로 테이블을 누르는
  것도 파지로 오인된다 (구현은 `filter_idx=0` 기본값으로 처리, plate는 filter 1개라 무영향)
- 전제: `DUAL_FR3_CFG.spawn.activate_contact_sensors=True` (`dual_fr3.py`, 2026-07-27 변경)
- **주의**: ContactSensor 하나에 finger 2개를 regex로 묶으면 PhysX filter가
  `expected 2N, found N` 에러를 냄 → 반드시 finger당 sensor 분리
- 추가 (Run B): `right_arm_contact` = `right_fr3_link[1-7]` net force 센서 (filter 없음 →
  body 수 제약 없음). 충돌 페널티용 — finger는 제외 (물체를 만져야 하는 부위)

### 2-4. Reward (per-second 정규화 값 기준, TensorBoard `Episode_Reward/*`)

```
G      = 𝟙[min(‖F_left‖, ‖F_right‖) > 0.5 N]        # grasp gate: 양 finger pad가 동시에 Object 접촉
L      = 𝟙[z_obj > 0.64]                             # 높이 gate: 상판 + 4cm

r_total = 1.0  · (1 − tanh(d_ee-obj / 0.1))              # reaching_object (비gated, 접근 gradient)
        + 15.0 · L · G                                    # lifting_object        (grasp-gated)
        + 16.0 · L · G · (1 − tanh(d_obj-goal / 0.3))     # object_goal_tracking  (grasp-gated)
        +  5.0 · L · G · (1 − tanh(d_obj-goal / 0.05))    # fine_grained          (grasp-gated)
        − 0.1  · ‖a_t − a_{t−1}‖²   (curriculum 발동 후; 초기 −1e-4)   # action_rate
        − 0.1  · ‖q̇‖²              (curriculum 발동 후; 초기 −1e-4)   # joint_vel
        − 1.0  · Σ 𝟙[‖F_link_i‖ > 1 N]  (curriculum 발동 후; 초기 −1e-4)  # undesired_contact_penalty (Run B)
        − 0.5  · ‖v_obj‖² 𝟙[d_goal < 0.1]                                  # object_hold_still (Run C)
        − 0.005· ‖a‖²  (curriculum 발동 후; 초기 −1e-4)                    # action_l2 (C4, delta 전용 — null-space 유영 차단)
        + 3.0  · (ẑ_cube · ẑ_world)⁺ · L · G                               # object_upright (Run D, 운반 중 윗면 유지)
        − 1.0  · Σ_finger 𝟙[F_finger-table > 10 N]  (curriculum 후)        # finger_table_impact (Run D, "콩 박기" 대응)
```

- 구현: `lift/mdp/rewards.py`의 `object_grasped` / `object_is_lifted_and_grasped` / `object_goal_distance_grasped`
- **설계 의도**: 들기·운반 보상 전부에 G를 곱해 "실제 파지 중"일 때만 지급.
  높이 gate(L)만으로는 손등에 얹어 나르는 exploit이 이득이었음 (§3, 2026-07-25 run에서 실증)
- G를 lifting에도 곱는 이유: tracking만 gate하면 얹어 나르기가 여전히 15/s를 벌어 exploit이 절반만 죽는다
- fine_grained weight는 5 (2026-07-25 run에서 10으로 올린 것이 exploit 발현에 기여 → gating 검증 후 재상향 검토)
- reaching은 gate하지 않음: 파지 전 접근 단계의 유일한 gradient

### 2-5. Termination / Curriculum / DR

| 항목 | 내용 |
|---|---|
| time_out | 5 s (성공해도 계속 — **success termination / sparse bonus 없음**) |
| object_dropping | cube z < 0.65 (상판 −10cm) |
| curriculum | `modify_reward_weight_on_lifting_learned` — lifting 보상(만렙~15/s)이 **10/s 초과 시** action_rate·joint_vel·contact 페널티 강화 (latch, 성능 기반) |
| DR: cube 크기 | 한 변 4.5~5.5cm, `mode="usd"` per-env 고정 (`replicate_physics=False`) |
| DR: cube 스폰 | x ±0.08, y ±0.12 (상판 위). **yaw ±45°는 Run E 예약** |
| DR: goal | base 기준 x 0.35~0.55, y −0.45~**−0.15** (Run B~C에서 −0.05 시도 후 원복 — 중앙선 goal은 실익 없이 왼팔 충돌 위험만 증가, 2026-07-27), z 0.45~0.60 |
| DR: cube 물성 (Run C) | 마찰 static 0.4~1.2 / dynamic 0.3~1.0 (make_consistent), 질량 ×0.7~1.3 — startup, per-env |
| DR: obs noise (Run C) | §2-1 참조 (Gaussian, 학습 시 활성) |
| **없는 DR** | joint stiffness/damping, obs latency, 그리퍼 동역학 |

---

## 3. Grasp gating의 배경 — "높이 gate만으로는 안 된다"

### 3-1. 문제 (2026-07-25 run에서 실증)

gating 이전의 `object_goal_distance`는 gate가 `𝟙[z_obj > 0.64]` 하나뿐이었다. cube가 *어떻게*
공중에 있는지 — TCP로 파지했는지, 손등·팔뚝에 얹었는지 — 를 구분하는 항이 없어서,
**cube를 손등에 얹고 균형 잡아 나르는 정책**(서커스 거동)이 정상 파지와 거의 같은 보상을 받았다.

exploit의 기회비용은 `reaching_object` 1/s뿐 (전체 ~37/s의 3%). fine_grained를 5→10으로 올린
2026-07-25 run에서 "그리퍼 자세 제약 없이 cube 중심을 goal에 붙이는 정밀도 이득 > reaching 손해"가
확정되며 exploit이 지배 전략이 됐다.

### 3-2. 로그 증거 (4 run 비교, tail-50 평균)

| Run | iter | reaching (max 1) | lifting | pos err [m] | 거동 (play 관찰) |
|---|---|---|---|---|---|
| 07-24_13-24-51 | 1.5k | 0.24 | **0.00** | 0.34 | 학습 실패 (안 움직임) |
| 07-24_14-46-05 | 1.9k | **0.83** | 13.96 | 0.21 | **정상 파지** ✅ |
| 07-24_16-07-02 | 3k | **0.86** | 14.04 | 0.22 | 파지하나 왼팔 충돌 |
| 07-25_18-40-22 | 5k | **0.07** | 13.62 | **0.12** | 서커스 (얹어 나르기) |

`reaching_object` tail이 사실상 **"grasp 유지율" 지표**: 0.83~0.86(파지) vs 0.07(exploit).
**gated 재학습의 성공 판정도 이 지표 — tail ≥ 0.8 복귀 여부.**

### 3-3. Run별 config 차이 (env.yaml diff 확정본)

| | v1 (13-24-51) | v2 (14-46-05) | v3 (16-07-02) | v4 (18-40-22) |
|---|---|---|---|---|
| lift 판정 높이 | 상판+8cm | **+4cm** | +4cm | +4cm |
| 페널티 curriculum | 10k step | **30k step** | 30k step | **성능 기반 (lifting>10)** |
| cube 크기 DR | — | — | **4.8~5.2cm** | **4.5~5.5cm** |
| goal y 범위 | ~−0.05 | ~−0.05 | ~−0.05 | **~−0.15** (왼팔 충돌 방지) |
| fine_grained w | 5 | 5 | 5 | **10** |
| 결과 | 붕괴 | 정상 ✅ | 충돌 | 서커스 |

교훈:
- v1→v2: **페널티가 들기 발견보다 먼저 세지면 "안 움직이는" local optimum으로 붕괴**
  (성능 기반 curriculum은 이것의 올바른 일반화)
- v3: goal이 몸 중앙선(−0.05)까지 가면 오른팔이 왼팔 영역 침범 — **충돌은 goal 샘플링 문제**
- v4: fine_grained 2배 + 장기 학습(5k)이 grasp 없는 reward 구멍을 노출

### 3-4. Gate 설계 논의 — 왜 contact이고, 왜 obs가 아닌가

- **Reward는 배포와 무관**: reward는 학습 중에만 존재. 실로봇에 F/T·촉각 센서가 없어도
  sim의 contact 정보를 reward에 쓰는 데는 아무 대가가 없다.
- **Contact gate > EE 거리 gate**: 손등에 얹으면 finger *pad*가 아니라 hand 몸체에 닿으므로
  "양 pad 동시 접촉" 조건은 진짜 파지만 통과시킨다. EE 거리 gate는 얹은 위치가 TCP에서
  가까우면 뚫릴 수 있다.
- **Actor obs에 contact은 넣지 않는다**: 실로봇 배포 시 없는 신호. 대신 finger joint_pos가
  암묵적 파지 센서 역할 (close 명령 후 손가락이 cube 폭 ~2.2–2.8cm에서 멈추면 파지 성공,
  0까지 닫히면 실패 — FR3 실물 그리퍼도 width 피드백 제공).
- critic-only privileged obs(asymmetric AC)는 선택사항으로 보류 — 이 태스크는 partial
  observability가 크지 않음.

### 3-5. Gate 검증 결과 (2026-07-27)

v2 정상 파지 정책(model_1850)을 gated 환경에서 replay (16 envs × 250 steps):

```
step  25: lifted=0.94  gate=0.94
step  75~: lifted=1.00  gate=1.00   (이후 끝까지 동일)
```

- 모든 측정 지점에서 **gate == lifted** → 진짜 파지에서 false negative ≈ 0, threshold 0.5N 적절
- 검증 중 발견한 운영 팁: headless 스크립트가 `simulation_app.close()`에서 무한 대기하는
  현상 있음 (작업은 수 초 만에 완료된 상태) — 멈춘 듯 보이면 `kill -USR1`로 스택 확인

---

## 4. 남은 구조적 약점 (심각도순, 2026-07-27 Run C 기준 갱신)

> 해결됨 (이력): ~~충돌 무페널티~~ → Run B에서 right_arm_contact 페널티 + 성능 curriculum으로 해결
> (goal y −0.05 복원 후에도 contact ~0 확인). ~~obs noise / 물성 DR 부재~~ → Run C 학습 cfg에 포함
> (§2-1, §2-5; 근거는 §5-6 상호작용 진단).

1. **Success 개념 — 학습엔 없음(의도), 평가엔 구현됨 (2026-07-28~)**: reward에는 여전히 sparse
   bonus·success termination이 없다 (grasp-gated tracking 연속값만으로 학습 — 의도된 설계).
   단 **평가용 에피소드 성공률 지표는 eval 스크립트에 구현 완료**: 파지∧들림∧goal ε 이내를 N step
   연속 유지 시 성공 latch (ε=5cm 표준/2.5cm strict 병기, N=success_hold_steps 기본 50).
   D/v1/v2 모두 성공률 99~100% 확인. sparse bonus 추가는 불필요로 판정 (이미 정밀도 충분).
   ⚠ 주의: N step 고정이라 정책 주기가 다르면 유지 "시간"이 달라짐 (50Hz의 50step=1s ≠ 20Hz의 2.5s).
   주기 다른 정책 비교 시 --success_hold_steps로 실시간 맞출 것.
2. **그리퍼 인터페이스 갭**: sim은 50Hz 이진 토글이 가능하지만 실물 Franka 그리퍼는
   명령당 수백 ms — 정책이 토글을 남용하면 실물 전이 불가. play에서 토글 빈도 확인 필요,
   심하면 toggle 페널티 또는 명령 latch 도입.
3. **Cube 자세 — 두 층위로 구분 (2026-07-27 재논의, 2026-07-28 Run D 확정)**:
   - *goal orientation* (목표 지점에서의 cube 자세): 대칭 cube 위치 이동 태스크라 불필요 — 의도된 미사용.
     obs의 goal quat 4 dims는 죽은 입력, `Metrics/orientation_error`는 무시할 지표.
     **plate도 동일** — plate의 orientation 항들(plate_alignment 등)은 goal이 아니라 전부 파지 층위.
     stacking/insertion으로 태스크가 확장될 때만 열 것.
   - *파지 시점 자세 정합* (그리퍼 yaw ↔ cube 면 정렬): **실환경 필수인데 현재 미커버.**
     스폰 yaw DR이 없어(x,y만 랜덤) 항상 축 정렬 스폰 → 정책이 고정 접근 방향만 학습
     → 실물에서 임의 yaw cube에 모서리 파지로 실패할 것. obs에도 cube 자세 정보 없음.

   **→ Run D 계획 확정 (2026-07-28, delta 검증(Run C 3차) 성공 후 실행. plate 검증 레시피 이식):**

   | 단계 | 내용 | plate 대응물 (검증됨) |
   |---|---|---|
   | ① 스폰 yaw DR ±45° | `reset_object_position` pose_range에 yaw 추가. 정사각 대칭이라 ±45°로 전 구성 커버 | 스폰 yaw ±0.3 (P1부터) |
   | ② obs에 cube 자세 추가 | "DR을 넣으면 관측도 줘야 한다" (plate P5 교훈 — 아니면 정합이 관측 불가능한 목표가 됨). 실물은 6D pose 인식으로 획득 가능. obs 차원 변경 → checkpoint 비호환, 별도 run | `object_orientation_in_robot_root_frame` (quat w≥0 부호 고정) 재사용 |
   | ③ 정합 shaping **보류** | 2-finger가 cube 모서리를 물면 grasp gate는 통과 가능하나 불안정 — **play에서 모서리 파지가 실제 관찰되면** 그때 "그리퍼 yaw ↔ cube 면" 정렬 항 추가 | plate도 P1에서 모서리 걸침 관찰 후 v2에서 alignment 추가. 단 P3 교훈: 정합 shaping은 근접 게이팅 없이 넣으면 "제자리 정렬" 국소해 유발 → 넣게 되면 반드시 근접/접촉 게이팅과 함께 |
4. **obs latency 미모델링**: 실물 perception은 노이즈뿐 아니라 지연(수십 ms~프레임 단위)도
   있음. Run C 검증 후 1~2 step 지연 랜덤화 검토.
5. **이진 그리퍼**: 파지력 조절 불가. force 제어가 필요해지면 한계.

---

## 5. 로드맵

- [x] **5-1. Grasp-gated reward** — 2026-07-27 구현·검증 완료 (§2-3, §2-4, §3-5)
- [x] **5-2. fine_grained 5 원복** — 2026-07-27 완료
- [x] **5-0. Run A (gating만 재학습)**: `2026-07-27_14-59-58`, 3000 iter — **판정 성공**: reaching tail **0.853** (기준 0.8, v4 0.07), lifting 14.04, tracking 14.29 / fine 3.68 (역대 최고), drop ~0%. 서커스 exploit 소멸
- [x] **5-4. 충돌 페널티** — Run B에 포함 (right_arm_contact + undesired_contacts, 성능 curriculum). goal y −0.05 복원
- [x] **Run B 학습 실행 중**: `2026-07-27_15-59-59`, 3000 iter (2026-07-27 16시경 시작). 체크: contact_penalty가 curriculum 발동 후에도 ~0 유지 / reaching ≥ 0.8 / tracking 절대값은 goal 상향 탓에 Run A보다 낮게 시작 가능
- [x] **5-7. Run D (`2026-07-28_18-55-30` [PC2]) — 성공, Run A~D 아크 완결**: upright 2.70/3,
  finger_impact 0, fine 3.53 최고, drop 0.2%. 강건성 재평가: baseline≈noisy (hold 0.95,
  precise 0.92, err 1.1cm, drop 0) — B 정책의 468~750 drop이 0으로 (시리즈 목적 달성).
  **yaw 민감도 sweep: 성공률 0~±10° 평평 99~100%, tts 1.4s → "가지런히 놓기" 실로봇에
  재학습 불필요 확정** (수치·발견은 log_per_task 해당 섹션 참조). 원래 구현 기록:
  ① object_upright 보상 (w3.0, C4 play의 운반 중 기울어짐 대응) ② finger_table_impact 페널티
  (>10N, "콩 박기" 대응; finger 센서 filter [Object, Table] 확장 — **object_grasped는 filter_idx=0
  필수**, 아니면 테이블 누르기가 파지로 오인). 판정: upright ≥ 2.5/3, finger_impact → 0, C4 지표 유지
- [x] **5-11. v1 = tau_ext obs ablation — 판정: 제거 확정 (2026-07-29, `2026-07-29_12-38-27` [PC2])**:
  토크 제거(obs 36)가 **전면 우세** — reward 172.5(D 167.7), fine 3.84 신기록, lifting 발견 2배
  빠름(it76), critic loss 절반(0.40 vs 0.90 — σ0.5Nm 7dim 토크 채널이 순수 잡음 주입이었다는
  결정적 증거). eval: yaw 0/5/10° 전부 SUCCESS 1.000(strict 동일), drop 0×3, tts 1.33s.
  → **배포 후보 = v1 checkpoint**, 실물 인터페이스에서 tau_ext 스트리밍 생략 (§5-9 ③ 단순화).
  교훈: 토크/힘 obs는 태스크의 접촉 의존도에 비례해서만 (contact-rich인 remove_hook/plate와 달리
  cube lift엔 잉여). 유일 열세 upright 2.43(기준 2.5 미달)은 drop 0.1%라 실질 무해 — 관찰 카드
- [x] **5-12. v2 = decimation 재학습 (정책 50→20Hz) — 판정: 성공 (2026-07-31, `2026-07-31_14-33-32` [PC1])**:
  v1(obs 36) 동일 + 변경 3개: decimation 2→5(정책 20Hz), episode 5→8s(160 step),
  resampling 5→8s(goal 1개/ep 유지 — 8초 에피소드에서 goal 5초 재샘플로 2개 되는 버그 수정).
  목적: "action 너무 빠름"(사용자 관찰) + perception 느림(~10~30Hz) 동시 해결 (§9). 근거: FR3
  zero-shot 문헌 정책주기 10~15Hz 수렴. 결과: 20Hz로 **성능 손실 없음** — reaching 0.861, lifting
  14.44, upright **2.75**(v1 2.43보다 우위!), drop 0.3%, lifting 발견 it55(v1 it76보다 빠름 —
  느린 주기가 탐색 안정화). 미세 열세 fine 3.50(v1 3.84)이나 pos err 0.213 동급. play: 1~2s에
  파지·운반·유지 완료(8s는 여유). eval 3종 진행 중. → **20Hz 배포 후보** (실물 perception 정합).
  다음 카드: 되면 dec 10(10Hz), latency DR은 별도 축. 상세 스펙: `dual_fr3_lift_v2.md`.
- [x] **5-13. v3 = 세로 cube 50×50×100 재학습 + ⚠크기 DR 버그 수정 (2026-08-03, `2026-08-03_16-52-40` [PC1])**:
  실물 cube가 50×50×100mm 확정 → v2(정육면체) 10cm zero-shot에서 drop 160(CoM 높아 슬립) → 세로
  물체 재학습. **개발 중 크기 DR이 v0~v3 내내 무효였음을 발견**: `randomize_cube_size`가 mode="usd"
  였는데 env는 "prestartup"만 호출(manager_based_env.py:155) → 이벤트 침묵, 실제 4.80cm 고정 학습.
  (play에서 사용자가 "세로 아닌 것 같다"로 포착 → sim 실측 전 env 균일.) 물성 DR(마찰·질량)은
  startup이라 정상이었음. 수정: object를 primitive(MeshCuboidCfg)로 + mode="prestartup" + scale
  재계산 → 실측 검증 per-env 랜덤. 결과: **drop 0**(eval yaw 0/5/10° SR 1.0/1.0/0.984), 세로 cube
  안정 파지 달성. 잔여: upright 2.38 낮음(숙인 CoM 파지 — drop 0을 만든 적응이나 운반 기울 여지) →
  v3.1(upright 3→5) 선택 카드, 실물 1차 후 판단. finger_table_impact는 static Table의 GPU filter
  미지원으로 무효(no-op) — 콩 박기 관찰 시 locomotion식 net-force로 대응. 상세: `dual_fr3_lift_v3.md`.
- [ ] **5-9. 실로봇 inference 준비 (현재 최우선, 2026-07-28 결정)**: ① perception — 실물에서
  `object_position`(±1cm noise로 학습됨) 생성 파이프라인 (ArUco/6D pose, 최대 갭)
  ② 그리퍼 toggle 빈도 점검 (play 로그 — 실물 그리퍼는 명령당 수백 ms) ③ 50Hz delta joint
  명령 스트리밍 인터페이스 (FR3, τ_ext obs 포함) ④ (선택) eval에 drop 후 회복률 지표 추가
- [ ] **5-10. 평가 방법론 강화 (논문 수치용)**: 평가 seed 고정 + env 128 (또는 3-seed 평균,
  ICROS식) — drop 카운트 run간 0/250 분산(startup 물성 추첨 소표본) 해소
- [ ] **5-8. Run E 예약 (실로봇 1차 검증 후로 연기, 2026-07-28 사용자 결정)**: 스폰 yaw ±45° DR + orientation obs (43→47) 세트.
  mdp 함수 구현 완료, cfg 배선만 남음. 난이도 카드: 심하게 늦으면 yaw ±20°부터.
  정합 shaping은 모서리 파지 관찰 시에만 (근접 게이팅 필수 — P3 교훈)
- [ ] **5-3. Success 정의**: `𝟙[d < 0.05, N step 유지]` sparse bonus(+50) 및/또는 종료 + eval 성공률
- [x] **5-5. Run C 1차 (`2026-07-27_18-38-13`, 5000 iter) — 실패, 원인 규명 완료**:
  구성 = obs noise + cube 물성 DR + `object_hold_still`(goal 10cm 내 속도² w−0.5).
  실패 사슬: ① obs noise로 advantage 흐려짐 → entropy 우세 → noise_std 1.0→5.5 폭주
  ② iter ~1520 curriculum 조기 latch — **drop 종료 env는 "들다 놓친" 것이라 lifting 합이
  구조적으로 높은 편향 표본**인데 소수 drop 배치 평균으로 판정해버림 → 휘젓는 정책에
  action_rate −59/s 직격, 보상 −288 붕괴 ③ 이후 페널티를 쓴 채 재학습되어 나쁜 국소해 수렴
  (최종: reward 80 vs B 143, pos err 0.32, **contact −1.64/s = 왼팔 상시 충돌**, fine 0.16)
- [x] **curriculum EMA 수정 + goal y 원복** (2026-07-27, 스모크 통과): 배치 크기 가중 EMA
  (α = min(0.5, 3·batch/num_envs), term별 분리 저장) — drop 배치는 α~0.001로 무력화,
  timeout 배치도 2회 연속 초과 필요. goal y 상한 −0.05→**−0.15 원복** (중앙선 goal은
  실익 없이 충돌 위험만 키움)
- [x] **Run C 2차 (`2026-07-27_20-57-10`, 5000 iter) — 실패, 새 모드**: EMA curriculum은
  오발동 0회로 정상 작동했으나 noise_std가 1.0→**12.8**까지 단조 폭주 (C1의 5.5보다 심함,
  C1에선 오발동 페널티가 우연히 std를 눌렀던 것). 정책 평균 자체가 붕괴해 play에서 팔 휘두름.
  hold_still(−0.04/s)은 무죄. → **재해석: plate P2/P4가 같은 noise+DR에서 from-scratch 성공
  → 진범은 "절대 action(scale 0.5) × obs noise" 조합의 탐색 불안정** (§2-2 ablation 표 참조)
- [x] **Run C 3차 준비 완료 (2026-07-28, 스모크 통과)**: action을 절대(0.5)→**delta
  (`RelativeJointPositionActionCfg`, scale 0.1, plate와 통일)**로 전환. 나머지 세팅 동일 —
  단일 변인 검증 실험. 판정: noise_std ~2 피크 후 수축(1000 iter 내) + reaching ≥ 0.8 +
  강건성 재평가 drop ≈ 0. 예비 카드: entropy_coef 0.006→0.002, asymmetric critic.
  **from-scratch 유지 (사용자 결정 — plate 선례로 정당화됨)**

### 5-6. 강건성 진단 (2026-07-27, Run B 정책 model_4999, 32env×4ep/조건)

`Isaac-Lift-Cube-DualFR3-Play-Noisy-v0` + `scripts/tools/eval_dual_fr3_lift_robustness.py`
(축별 스위치 `--no_obs_noise` / `--no_physics_dr`)로 재학습 없이 취약점 측정:

| 조건 | hold | precise(<5cm) | drop/128ep |
|---|---|---|---|
| baseline (학습 분포) | 0.957 | 0.688 | 0 |
| obs noise만 | 0.920 | 0.592 | 0 |
| 물성 DR만 | 0.951 | 0.690 | 0 |
| **둘 다** (2회 반복) | 0.93 | 0.62~0.64 | **468~750** |

**결론**: 단독 축은 무해하나 **"perception 떨림 × 낮은 마찰" 상호작용**이 이동 중 슬립 드랍을
유발 — 실환경에서 항상 공존하는 조합이므로 zero-shot 불가의 주 원인. obs noise는 정밀도(−14%)를,
조합은 파지 유지(드랍)를 깎는다. → Run C는 두 축을 반드시 **함께** 학습에 포함 (구현 완료).
학습 후 같은 스크립트·같은 지표로 재평가하여 drop이 0 근처로 복귀하는지 확인할 것.

---

## 6. remove_hook (ICROS) 환경과의 구조 대비

| 축 | remove_hook (fixed_multi) | dual_fr3 lift (현재) | 시사점 |
|---|---|---|---|
| 순서 강제 | multiplicative r_ori × r_pos | **L · G · r_track (동일 철학 적용됨)** | 완료 |
| coarse+fine | 0.7·tanh(d/0.4) + 0.3·tanh(d/0.05) | 16·tanh(d/0.3) + 5·tanh(d/0.05) | 구조 동일 |
| sparse bonus | +100 (4cm 도달) | 없음 | §5-3 |
| 접촉 안전 | FT tare 페널티 + threshold DR | **충돌 페널티 (Run B, undesired_contacts + 성능 curriculum)** | 완료 |
| obs noise | pos/vel/FT 전부 | **pos/vel/obj_pos/토크 Gaussian (Run C)** | 완료 |
| history | ×5 stack (175 dims) | 없음 (47 dims) | 정적 태스크라 당장 불필요 |
| 성공 판정 | 4cm threshold, eval 3 seeds×100ep | 없음 | §5-3 |

---

## 7. TODO (TL;DR — 지금 해야 할 것만, 우선순위순)

> 갱신: 2026-07-29. 상세 근거는 위 §5 해당 항목 참조. **완료되면 여기서 지우고 §5에 [x]로 기록.**
> **배포(v1) 관련 스펙·TODO는 `dual_fr3_lift_v1.md`가 원본** — 여기는 환경 설계 관점 항목만 유지.

**현 상태 요약: Run A~D + v1 ablation 완결. 배포 후보 = v1 정책
(`dual_fr3_lift_v1/2026-07-29_12-38-27/model_4999.pt`, obs 36) — 성공률 100% (yaw ±10°까지,
strict 2.5cm 포함), tts 1.33s, drop 0. 토크 obs 제거 확정 (§5-11).**

- [ ] **1. 그리퍼 toggle 빈도 점검** — D 정책 play에서 open/close 명령 전환 빈도 측정.
      실물 그리퍼는 명령당 수백 ms라 파닥거리면 배포 불가. sim에서 10분. (§4-2)
- [ ] **2. perception 파이프라인 설계** — 실물 `object_position` 소스 결정 (ArUco vs 6D pose).
      요구 정확도: 학습 noise 허용치 ±1cm 이내. 실험실 카메라 셋업 확인부터. (§5-9, 최대 갭)
- [ ] **3. 50Hz delta joint 스트리밍 인터페이스** — FR3 실물 명령 전송. **τ_ext 읽기 불필요
      (v1 ablation 결과)** → obs 36 전부가 joint state + perception + goal로 구성. (§5-9)
- [ ] **4. (선택) eval에 drop 후 회복률 지표** — 재파지 능력 정량 확인. (2026-07-28 논의)
- [ ] **5. (논문 시점) 평가 seed 고정 + env 128 / 3-seed** — drop 카운트 분산 해소. (§5-10)
- [ ] **6. (실로봇 1차 후) Run E**: yaw ±45° DR + orientation obs 43→47. 배선만 남음. (§5-8)
- [ ] **7. (보류 카드)** success sparse bonus(불필요 판정 유력), obs latency DR, 정합 shaping
      (모서리 파지 관찰 시에만), IsaacLab fork 정리 (dual_fr3 잔재 삭제)
